"""
Cache View-of-Delft to a single npz: scenes once, measurements at all three
radar sparsity levels (1/3/5 accumulated frames).

The scene target is identical across sparsity levels -- only the measurement
changes -- so building it once keeps the sparsity sweep a strictly controlled
comparison: the same frames, the same ground truth, the same split, differing in
measurement density alone.

SPLIT: the OFFICIAL ImageSets splits are used verbatim (train 5139, val 1296).
Do not invent a split. VoD frames are sequential, so the official split already
handles the temporal-leakage problem; test.txt is the unlabeled benchmark
holdout and is not used here.

Usage:  python build_vod_cache.py            # full train+val
        python build_vod_cache.py --limit 400
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_vod import (load_vod, split_ids, transforms, lidar_to_bev,
                            radar_to_bev, labels_to_bev, LIDAR, ROOT, RADAR_DIR, GRID)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def build(split, ids, R2L, C2L, grid=GRID, verbose=True):
    scenes, m1, m3, m5, frames = [], [], [], [], []
    rdirs = {k: os.path.join(ROOT, v, "training", "velodyne") for k, v in RADAR_DIR.items()}
    t0 = time.time()
    for k, fid in enumerate(ids):
        lp = os.path.join(LIDAR, "velodyne", f"{fid}.bin")
        rp = {n: os.path.join(d, f"{fid}.bin") for n, d in rdirs.items()}
        if not os.path.exists(lp) or not all(os.path.exists(p) for p in rp.values()):
            continue
        L = np.fromfile(lp, dtype=np.float32).reshape(-1, 4)
        occ, hgt, ref, gz = lidar_to_bev(L, grid)
        if occ.sum() < 5:
            continue
        R = {n: np.fromfile(p, dtype=np.float32).reshape(-1, 7) for n, p in rp.items()}
        obj, nobj = labels_to_bev(os.path.join(LIDAR, "label_2", f"{fid}.txt"), C2L, grid)
        scenes.append(np.stack([occ, hgt, ref], 0))
        m1.append(radar_to_bev(R[1], R2L, grid))
        m3.append(radar_to_bev(R[3], R2L, grid))
        m5.append(radar_to_bev(R[5], R2L, grid))
        frames.append({"id": fid, "ground_z": gz, "n_labels": nobj,
                       "n_radar": {str(n): int(len(v)) for n, v in R.items()},
                       "n_lidar": int(len(L))})
        if verbose and (k + 1) % 500 == 0:
            el = time.time() - t0
            print(f"    {split}: {len(scenes)}/{k+1}  {el/60:.1f} min  "
                  f"eta {(el/(k+1)*(len(ids)-k-1))/60:.1f} min", flush=True)
    return (np.asarray(scenes, np.float32), np.asarray(m1, np.float32),
            np.asarray(m3, np.float32), np.asarray(m5, np.float32), frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    R2L, C2L = transforms()
    os.makedirs(OUT, exist_ok=True)
    meta = {"grid": GRID, "split_source": "official lidar/ImageSets", "splits": {}}
    store = {}
    for split in ("train", "val"):
        ids = split_ids(split)
        if a.limit:
            ids = ids[:a.limit]
        print(f"building {split}: {len(ids)} ids")
        sc, m1, m3, m5, fr = build(split, ids, R2L, C2L)
        store[f"{split}_scenes"] = sc
        store[f"{split}_meas_r1"] = m1
        store[f"{split}_meas_r3"] = m3
        store[f"{split}_meas_r5"] = m5
        meta["splits"][split] = {
            "n": int(len(sc)), "n_ids": len(ids),
            "occupancy_fill": float((sc[:, 0] > 0.5).mean()),
            "mean_radar_pts": {n: float(np.mean([f["n_radar"][str(n)] for f in fr]))
                               for n in (1, 3, 5)},
            "mean_lidar_pts": float(np.mean([f["n_lidar"] for f in fr])),
            "meas_fill": {n: float((store[f'{split}_meas_r{n}'][:, 0] > 0).mean())
                          for n in (1, 3, 5)},
            "frames": fr}
        print(f"  {split}: {len(sc)} frames, occ fill "
              f"{100*meta['splits'][split]['occupancy_fill']:.2f}%, radar "
              f"{ {n: round(v) for n,v in meta['splits'][split]['mean_radar_pts'].items()} } pts")
    p = os.path.join(OUT, "vod.npz")
    np.savez(p, **store)
    json.dump(meta, open(p.replace(".npz", "_meta.json"), "w"), indent=2)
    print(f"\nsaved: {p}  ({os.path.getsize(p)/1073741824:.2f} GB)")
    print("Scenes are shared across r1/r3/r5, so the sparsity sweep is a controlled comparison.")


if __name__ == "__main__":
    main()
