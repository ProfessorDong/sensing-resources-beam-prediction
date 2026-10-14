"""
Cache DeepSense Sc.36 to a single npz so training does not re-run preprocessing.

Preprocessing is ~0.081 s/frame (pandas LiDAR path), so the full 24,799 frames
takes ~36 min once. Output is ~2.8 GB uncompressed.

SPLIT POLICY -- read this before using the cache.
Consecutive frames are near-duplicates (a moving vehicle sampled densely), so a
RANDOM split leaks the test set into training and would manufacture exactly the
kind of inflated number this project is trying to stop reporting. This script
therefore writes contiguous TEMPORAL blocks with a GAP between them:

    train = first 70%        gap = 2%        test = last 28%

Because unit1 travels ~18 km over the scenario, a temporal split is also a
geographic split. The gap prevents boundary frames from straddling the split.

Usage:
    python build_sc36_cache.py                 # full set
    python build_sc36_cache.py --stride 4      # quarter, for a fast iteration
"""
import os, sys, csv, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from preprocess_deepsense_sc36 import (
    BASE, CSV, MAT, GRID, EXTENT, Z_BAND, lidar_to_bev, radar_to_measurement,
    _load_beam_tables, N_RADAR_BOARDS,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
TRAIN_FRAC, GAP_FRAC = 0.70, 0.02


def main():
    import scipy.io
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(CSV)))
    idxs = list(range(0, len(rows), a.stride))
    if a.limit:
        idxs = idxs[:a.limit]
    pw_tab, ob_tab = _load_beam_tables()
    if pw_tab is None:
        print("WARNING: scenario36.mat beam tables unavailable; falling back to text files")

    sc, me, hb, lb, keep = [], [], [], [], []
    t0 = time.time()
    for k, i in enumerate(idxs):
        r = rows[i]
        try:
            occ, hgt = lidar_to_bev(os.path.join(BASE, r["unit1_lidar1"].lstrip("./")))
            if occ is None or occ.sum() < 5:
                continue
            boards = [scipy.io.loadmat(os.path.join(BASE, r[f"unit1_radar{b}"].lstrip("./")))["data"]
                      for b in range(1, N_RADAR_BOARDS + 1)]
            v = pw_tab[i].reshape(-1)
            mx = v.max()
            hb.append((v / mx if mx > 0 else v).astype(np.float32))
            lb.append(float(ob_tab[i]))
            sc.append(np.stack([occ, hgt, np.zeros_like(occ)], 0))
            me.append(radar_to_measurement(boards))
            keep.append(i)
        except Exception as e:
            if len(keep) < 5:
                print(f"  row {i}: skipped ({type(e).__name__}: {e})")
            continue
        if (k + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(idxs)} rows, {len(sc)} kept, {el/60:.1f} min, "
                  f"eta {(el/(k+1)*(len(idxs)-k-1))/60:.1f} min", flush=True)

    sc = np.asarray(sc, np.float32); me = np.asarray(me, np.float32)
    hb = np.asarray(hb, np.float32); lb = np.asarray(lb, np.float32)
    keep = np.asarray(keep, np.int64)

    n = len(sc)
    n_tr = int(TRAIN_FRAC * n)
    n_gap = int(GAP_FRAC * n)
    split = {"train": [0, n_tr], "gap": [n_tr, n_tr + n_gap], "test": [n_tr + n_gap, n]}

    os.makedirs(OUT_DIR, exist_ok=True)
    out = a.out or os.path.join(OUT_DIR, f"sc36_stride{a.stride}.npz")
    np.savez(out, scenes=sc, meas=me, hbeam=hb, beamlabel=lb, row_index=keep)
    meta = {"n": int(n), "stride": a.stride, "grid": GRID, "extent_m": EXTENT,
            "z_band_m": list(Z_BAND), "split": split,
            "split_policy": "contiguous temporal blocks with a gap; NEVER split randomly",
            "occupancy_fill": float((sc[:, 0] > 0.5).mean()),
            "beamlabel_range": [float(lb.min()), float(lb.max())],
            "elapsed_min": (time.time() - t0) / 60,
            "channels": {"scene": ["lidar_occupancy", "height_proxy", "reserved_zero"],
                         "meas": ["RA_all", "RA_static", "RA_dynamic", "range_doppler"],
                         "hbeam": "4 arrays x 64 beams, per-frame max-normalized",
                         "beamlabel": "unit1_overall-beam"}}
    json.dump(meta, open(out.replace(".npz", "_meta.json"), "w"), indent=2)
    print(f"\nkept {n}/{len(idxs)}  train {n_tr} | gap {n_gap} | test {n - n_tr - n_gap}")
    print(f"occupancy fill {100*meta['occupancy_fill']:.2f}%   elapsed {meta['elapsed_min']:.1f} min")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
