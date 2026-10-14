"""
Beam prediction, retried: does a LARGER BEV footprint make the beam predictable?

WHAT THIS TESTS
---------------
`exp_beam_pilot.py` found the beam unpredictable from BEV occupancy on Sc.36, with
`gt_scene` (the ceiling on what geometry offers) sitting exactly on the modal-beam
floor. Three causes were diagnosed; only the first is fixable by changing the grid:

  1. The beam points at unit2, but unit2 lies inside the +-32 m footprint in only
     43.7% of frames (median separation 38 m, p95 211 m). At +-100 m it would be
     80.3%.                                                    <-- TESTED HERE
  2. BEV occupancy is ANONYMOUS: it marks where objects are, not which is the
     communication partner.                                    <-- not fixable by grid
  3. The 256-way label pools 4 arrays with different boresights (pooled
     bearing-beam spearman +0.020).                            <-- not fixable by grid

TWO THINGS THIS DOES THAT THE FIRST PILOT DID NOT
-------------------------------------------------
**Separates extent from resolution.** Simply enlarging to +-100 m at 64x64 makes
cells 3.125 m, so a car occupies well under one cell -- that would confound "bigger
footprint" with "coarser objects". The sweep therefore varies both, and a 128x128
variant keeps ~1.56 m cells at the largest extent.

**Adds a GPS oracle.** This is the decisive control. It feeds the readout unit2's
heading-compensated relative position directly, so it bounds the PREMISE rather than
the representation: if knowing exactly where unit2 is fails to predict the beam, the
beam is not determined by unit2's bearing and the whole idea is void. If the oracle
succeeds while `gt_scene` fails, the conclusion becomes precise and much more
interesting -- the information is present in the geometry, and BEV occupancy fails
specifically because it cannot say WHICH object is the partner (cause 2).

Prior evidence that the oracle should work: heading-compensated bearing to unit2
correlates with beam-within-array at spearman +0.827 for array 1.

No diffusion inference is needed here -- only LiDAR/radar rasterization and small
readouts -- so this costs ~1 h instead of the ~3 h the full experiment needs. The
full experiment runs only if a grid configuration clears both gates.

Output: experiments/results/beam_grid_sweep.json
"""
import os, sys, json, time, collections, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_beam_prediction import train_readout
from preprocess_deepsense_sc36 import BASE, CSV, MAT, lidar_to_bev, radar_to_measurement

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "sc36_stride1.npz")
# (half-extent metres, grid cells) -> cell size
CONFIGS = [(32.0, 64), (64.0, 64), (100.0, 64), (100.0, 128)]
EPOCHS = 30
Z_BAND = (-2.5, 7.0)


def heading_relative_position(rows, idx):
    """unit2 position in unit1's travel frame: (forward, lateral, range).

    unit1 drives ~18 km so the antenna boresight rotates; absolute bearing is
    meaningless (pooled spearman with beam +0.02) while heading-compensated
    bearing reaches +0.827 on array 1.
    """
    lat1 = np.array([float(r["unit1_gps1_lat"]) for r in rows])
    lon1 = np.array([float(r["unit1_gps1_lon"]) for r in rows])
    lat2 = np.array([float(r["unit2_gps1_lat"]) for r in rows])
    lon2 = np.array([float(r["unit2_gps1_lon"]) for r in rows])
    R = 6371000.0
    cl = np.cos(np.radians(np.nanmean(lat1)))
    e1, n1 = np.radians(lon1) * cl * R, np.radians(lat1) * R
    e2, n2 = np.radians(lon2) * cl * R, np.radians(lat2) * R
    k = 5
    de = np.full(len(rows), np.nan); dn = np.full(len(rows), np.nan)
    de[k:-k] = e1[2 * k:] - e1[:-2 * k]
    dn[k:-k] = n1[2 * k:] - n1[:-2 * k]
    moving = np.hypot(de, dn) > 1.0
    hd = np.where(moving, np.arctan2(de, dn), np.nan)          # travel direction
    brg = np.arctan2(e2 - e1, n2 - n1)
    rel = brg - hd
    d = np.hypot(e2 - e1, n2 - n1)
    fwd, lat = d * np.cos(rel), d * np.sin(rel)
    out = np.stack([fwd, lat, d, np.cos(rel), np.sin(rel)], 1).astype(np.float32)
    out = out[idx]
    # scale and make NaNs explicit rather than silently zero
    valid = np.isfinite(out).all(1)
    out[~valid] = 0.0
    sc = np.array([100.0, 100.0, 200.0, 1.0, 1.0], np.float32)
    return np.concatenate([out / sc, valid[:, None].astype(np.float32)], 1), valid


def build_bev(row_index, rows, extent, grid, verbose=True):
    """Rasterize LiDAR (target) and radar (measurement) at one (extent, grid)."""
    import scipy.io
    import preprocess_deepsense_sc36 as P
    old_e, old_g = P.EXTENT, P.GRID
    scenes, meas = [], []
    t0 = time.time()
    for k, i in enumerate(row_index):
        r = rows[i]
        occ, hgt = P.lidar_to_bev(os.path.join(BASE, r["unit1_lidar1"].lstrip("./")),
                                  grid=grid, extent=extent, z_band=Z_BAND)
        if occ is None:
            occ = np.zeros((grid, grid), np.float32); hgt = occ.copy()
        boards = [scipy.io.loadmat(os.path.join(BASE, r[f"unit1_radar{b}"].lstrip("./")))["data"]
                  for b in range(1, 5)]
        scenes.append(np.stack([occ, hgt, np.zeros_like(occ)], 0))
        meas.append(P.radar_to_measurement(boards, grid=grid))
        if verbose and (k + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"      {k+1}/{len(row_index)}  {el/60:.1f} min  "
                  f"eta {(el/(k+1)*(len(row_index)-k-1))/60:.1f} min", flush=True)
    P.EXTENT, P.GRID = old_e, old_g
    return np.asarray(scenes, np.float32), np.asarray(meas, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    device = torch.device("cuda")
    import csv as _csv
    rows = list(_csv.DictReader(open(CSV)))
    z = np.load(CACHE)
    meta = json.load(open(CACHE.replace(".npz", "_meta.json")))
    ridx = z["row_index"]
    hbeam = z["hbeam"]
    sp = meta["split"]
    if a.limit:
        keep = np.arange(0, len(ridx), max(1, len(ridx) // a.limit))[:a.limit]
        ridx, hbeam = ridx[keep], hbeam[keep]
        ntr = int(0.7 * len(ridx)); split = {"train": [0, ntr], "test": [ntr, len(ridx)]}
    else:
        split = {"train": sp["train"], "test": sp["test"]}
    print(f"frames {len(ridx)}   train {split['train']}  test {split['test']}")

    beam = hbeam.reshape(len(hbeam), -1).argmax(1)
    arr = beam // 64
    tr = slice(*split["train"]); te = slice(*split["test"])
    dom = collections.Counter(arr[tr].tolist()).most_common(1)[0][0]
    print(f"dominant array in train: {dom+1}")

    gps, gps_valid = heading_relative_position(rows, ridx)
    print(f"GPS oracle features: {gps.shape}  valid in {100*gps_valid.mean():.1f}% of frames")
    # how often is unit2 inside each footprint?
    d_m = gps[:, 2] * 200.0
    for ext, g in CONFIGS:
        fwd, lat = gps[:, 0] * 100.0, gps[:, 1] * 100.0
        inside = gps_valid & (np.abs(fwd) < ext) & (np.abs(lat) < ext)
        print(f"  unit2 inside +-{ext:g} m grid: {100*inside.mean():5.1f}%   "
              f"(cell {2*ext/g:.2f} m)")

    save = os.path.join(RESULTS_DIR, "beam_grid_sweep.json")
    out = json.load(open(save)) if os.path.exists(save) else {}
    out.setdefault("cells", {})
    out["unit2_coverage"] = {f"{ext:g}m": float((gps_valid &
                             (np.abs(gps[:, 0]*100) < ext) & (np.abs(gps[:, 1]*100) < ext)).mean())
                             for ext, _ in CONFIGS}

    tasks = {}
    tasks["A_256way"] = (256, False, beam, np.ones(len(beam), bool))
    tasks[f"B_array{dom+1}_64way"] = (64, True, beam % 64, arr == dom)

    cfgs = CONFIGS if a.configs == "all" else [CONFIGS[int(i)] for i in a.configs.split(",")]
    for ext, grid in cfgs:
        tag = f"e{ext:g}_g{grid}"
        bev_cache = os.path.join(RESULTS_DIR, f"bev_{tag}.npz")
        if os.path.exists(bev_cache):
            zz = np.load(bev_cache); S, M = zz["scenes"], zz["meas"]
            print(f"\n[{tag}] BEV cached {S.shape}")
        else:
            print(f"\n[{tag}] rasterizing BEV at +-{ext:g} m, {grid}x{grid} "
                  f"(cell {2*ext/grid:.2f} m) ...")
            S, M = build_bev(ridx, rows, ext, grid)
            np.savez(bev_cache, scenes=S, meas=M)
            print(f"  saved {bev_cache}  {S.shape}")
        print(f"  scene occupancy fill {100*(S[:,0]>0.5).mean():.2f}%")

        for tname, (ncls, ordinal, y, mask) in tasks.items():
            mtr = np.zeros(len(y), bool); mtr[tr] = True; mtr &= mask
            mte = np.zeros(len(y), bool); mte[te] = True; mte &= mask
            if mtr.sum() < 200 or mte.sum() < 50:
                print(f"  {tname}: too few frames, skipped"); continue
            mb = collections.Counter(y[mtr].tolist()).most_common(1)[0][0]
            floor = float((y[mte] == mb).mean())
            # THE NO-INFORMATION BOUND -- see exp_beam_prediction.py for the full note.
            # `floor` is the TRAIN-modal beam's share in test, which is the right baseline
            # predictor but NOT the right reference for information: train and test are
            # different stretches of an 18 km drive with different dominant beams, so a
            # CONSTANT predictor reaches the TEST-modal share (0.1876 vs a 0.0631 floor on
            # Task A). The first version of this script gated on `floor` and therefore
            # reported a spurious gt_scene "+0.1246 ceiling" at +-100 m that was exactly
            # 0.1876 - 0.0631, i.e. one readout seed landing on the test-modal class.
            cnt = collections.Counter(y[mte].tolist())
            bmb, bn = cnt.most_common(1)[0]
            bound = bn / int(mte.sum())
            print(f"\n  == {tag} / {tname}  n_class={ncls} "
                  f"train {int(mtr.sum())} test {int(mte.sum())}  floor {floor:.4f}  "
                  f"NO-INFO BOUND {bound:.4f} (const class {bmb}) ==")
            print(f"    {'input':<12}{'top1':>9}{'top5':>9}{'within2':>10}{'mae':>7}"
                  f"  vs floor  vs BOUND")
            res = {"floor": floor, "bound_const": bound, "bound_class": int(bmb),
                   "n_train": int(mtr.sum()), "n_test": int(mte.sum())}
            for name, X in (("hbeam", hbeam), ("gps_oracle", gps),
                            ("gt_scene", S), ("radar", M)):
                key = f"{tag}/{tname}/{name}"
                if key in out["cells"]:
                    m = out["cells"][key]
                else:
                    Xtr = torch.from_numpy(np.ascontiguousarray(X[mtr])).float()
                    Xte = torch.from_numpy(np.ascontiguousarray(X[mte])).float()
                    m, _ = train_readout(Xtr, torch.from_numpy(y[mtr]).long(),
                                         Xte, torch.from_numpy(y[mte]).long(),
                                         ncls, 42, a.epochs, device, ordinal)
                    out["cells"][key] = m
                    json.dump(out, open(save, "w"), indent=2)
                w2 = m.get("within2", float("nan")); ma = m.get("mae_beams", float("nan"))
                print(f"    {name:<12}{m['top1']:>9.4f}{m['top5']:>9.4f}"
                      f"{w2:>10.4f}{ma:>7.2f}  {m['top1']-floor:>+8.4f}"
                      f"  {m['top1']-bound:>+8.4f}", flush=True)
                res[name] = m
            # Gains are measured against the NO-INFORMATION BOUND, not the floor.
            g1 = res["gps_oracle"]["top1"] - bound
            g2 = res["gt_scene"]["top1"] - bound
            print(f"    PREMISE  gps_oracle {g1:+.4f} vs BOUND -> "
                  f"{'beam IS determined by unit2 bearing' if g1 > 0.05 else 'beam is NOT determined by unit2 bearing -- premise fails'}")
            print(f"    CEILING  gt_scene   {g2:+.4f} vs BOUND -> "
                  f"{'BEV geometry carries it' if g2 > 0.05 else 'BEV geometry does NOT carry it'}")
            if g1 > 0.05 and g2 <= 0.05:
                print(f"    => information is PRESENT in geometry but BEV occupancy cannot "
                      f"expose it (cause 2: occupancy is anonymous)")
            out["cells"][f"{tag}/{tname}/_verdict"] = {
                "floor": floor, "bound_const": bound, "bound_class": int(bmb),
                "gps_oracle_gain_vs_bound": float(g1), "gt_scene_gain_vs_bound": float(g2),
                "premise_holds": bool(g1 > 0.05), "bev_sufficient": bool(g2 > 0.05)}
            json.dump(out, open(save, "w"), indent=2)

    print(f"\n{'='*84}\nSUMMARY: gt_scene gain over the NO-INFORMATION bound, by grid\n{'='*84}")
    print(f"{'config':>14}{'cell m':>8}{'unit2 in':>10}   task            gps_oracle   gt_scene")
    for ext, grid in cfgs:
        tag = f"e{ext:g}_g{grid}"
        for tname in tasks:
            v = out["cells"].get(f"{tag}/{tname}/_verdict")
            if v:
                print(f"{tag:>14}{2*ext/grid:>8.2f}{100*out['unit2_coverage'][f'{ext:g}m']:>9.1f}%"
                      f"   {tname:<16}{v['gps_oracle_gain_vs_bound']:>+10.4f}"
                      f"{v['gt_scene_gain_vs_bound']:>+11.4f}")
    any_ok = any(v.get("bev_sufficient") for k, v in out["cells"].items() if k.endswith("_verdict"))
    print(f"\nOVERALL: {'a grid configuration CLEARS the ceiling gate -> run the full experiment' if any_ok else 'no grid configuration makes the beam predictable from BEV occupancy'}")
    out["any_config_passes"] = bool(any_ok)
    json.dump(out, open(save, "w"), indent=2)
    print(f"Saved: {save}")


if __name__ == "__main__":
    main()
