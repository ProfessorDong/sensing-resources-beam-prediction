"""Does Paper A's mode 4 survive a validated image readout? Re-run of Figure F6's arms.

WHY THIS EXISTS
---------------
F6 claims that bird's-eye occupancy cannot expose the best beam: a GPS oracle clears the
no-information bound while ground-truth LiDAR occupancy, the reconstruction and the raw radar
do not. Every one of those arms used `exp_beam_prediction.ImgReadout`, which ends in a global
average pool.

On DeepSense Sc.31 that readout under-read a spatially distributed input badly: radar scored
0.0672 pooled against 0.1741 with the pool replaced by a flatten, i.e. from below the bound to
+0.0840 above it. So F6's verdict may be a statement about the readout rather than about
occupancy, and mode 4 of Paper A's taxonomy rests on it.

WHAT THIS RUNS
--------------
Same cache, same contiguous split and same 256-way task as `beam_prediction_e100.json`, with
two additions the original lacked:

  1. an INSTRUMENT CHECK for the image path. unit 2's known heading-relative position is
     rasterized into a blob on the same +-100 m grid, with the same channel count as the scene
     arms. The beam is recoverable from that image by construction, so if a readout cannot
     clear the bound on it, nothing else that readout reports may be read.
  2. every arm under BOTH readouts, pooled and spatial, so the architecture's contribution is
     measured rather than assumed.

Everything is scored against the best constant on the TEST split (0.1876), never the
train-modal floor.

Reading:
  occupancy still below the bound with a validated spatial readout -> mode 4 SURVIVES and is
      now much better supported, because the alternative explanation has been eliminated.
  occupancy clears the bound -> mode 4 must be withdrawn, F6 redrawn, and the white paper's
      taxonomy paragraph rewritten.

Output: results/sc36_spatial_readout_audit.json
"""
import os, sys, csv, json, collections, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_beam_prediction import train_readout
import exp_beam_prediction as EBP
from audit_readout_spatial import ImgReadoutSpatial
from exp_beam_grid_sweep import heading_relative_position

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEV_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets",
                       "deepsense6g", "Scenario36", "scenario36.csv")
GRID, EXTENT = 64, 100.0


def gps_blob(fwd, lat, valid, n_ch=3, sigma=1.6):
    """Heading-relative unit2 position -> (n_ch, GRID, GRID) blob on the +-100 m grid."""
    gx = (lat + EXTENT) / (2 * EXTENT) * (GRID - 1)
    gy = (fwd + EXTENT) / (2 * EXTENT) * (GRID - 1)
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(np.float32)
    out = np.zeros((len(fwd), n_ch, GRID, GRID), np.float32)
    for i in range(len(fwd)):
        if not valid[i] or not (0 <= gx[i] < GRID and 0 <= gy[i] < GRID):
            continue
        b = np.exp(-((xx - gx[i]) ** 2 + (yy - gy[i]) ** 2) / (2 * sigma ** 2))
        out[i, :] = b.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    dev = torch.device("cuda")

    z = np.load(os.path.join(CACHE, "sc36_e100.npz"))
    meta = json.load(open(os.path.join(CACHE, "sc36_e100_meta.json")))
    tr_sl = slice(*meta["split"]["train"]); te_sl = slice(*meta["split"]["test"])
    scenes, meas, hbeam = z["scenes"], z["meas"], z["hbeam"]
    ridx = z["row_index"]

    beams = hbeam.reshape(len(hbeam), -1).argmax(1)          # 256-way, verified label
    arr = beams // 64
    dom = collections.Counter(arr[tr_sl].tolist()).most_common(1)[0][0]
    print(f"dominant array in train: {dom + 1}")

    rows = list(csv.DictReader(open(DEV_CSV)))
    gps, gps_valid = heading_relative_position(rows, ridx)
    fwd, lat = gps[:, 0] * 100.0, gps[:, 1] * 100.0
    valid = gps_valid
    inside = float(np.mean(valid & (np.abs(fwd) < EXTENT) & (np.abs(lat) < EXTENT)))
    print(f"  GPS features {gps.shape}, unit2 inside the grid in {100*inside:.1f}% of frames")
    blob = gps_blob(fwd, lat, valid, n_ch=scenes.shape[1])

    recon = np.load(os.path.join(RESULTS_DIR, "recon_cache_seed0_e100.npz"))
    ALL = {
        "gps_blob_INSTRUMENT": (blob[tr_sl], blob[te_sl]),
        "gt_scene": (scenes[tr_sl], scenes[te_sl]),
        "recon": (recon["tr"], recon["te"]),
        "radar": (meas[tr_sl], meas[te_sl]),
        "gps_vec_reference": (gps[tr_sl], gps[te_sl]),
    }
    out = {"unit2_inside_grid": inside, "epochs": a.epochs, "tasks": {}}

    # Task A is the 256-way operational beam; Task B restricts to the dominant array, where
    # the beam-to-bearing map is consistent. F6 shows both, so both need a validated
    # instrument, not just the one.
    tasks = [("A_256way", beams, 256, np.ones(len(beams), bool)),
             (f"B_array{dom+1}_64way", beams % 64, 64, arr == dom)]

    for tname, lab, ncls, mask in tasks:
        ytr, yte = lab[tr_sl][mask[tr_sl]], lab[te_sl][mask[te_sl]]
        bmb, bn = collections.Counter(yte.tolist()).most_common(1)[0]
        bound = bn / len(yte)
        floor = float((yte == collections.Counter(ytr.tolist()).most_common(1)[0][0]).mean())
        print(f"\n=== {tname}: train {len(ytr)}  test {len(yte)}")
        print(f"  train-modal floor {floor:.4f}   NO-INFORMATION BOUND {bound:.4f} (class {bmb})")
        rec = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "n_class": ncls,
               "bound_const": bound, "bound_class": int(bmb), "train_modal_floor": floor,
               "results": {}}
        arms = {k: (v[0][mask[tr_sl]], v[1][mask[te_sl]]) for k, v in ALL.items()}
        yt = torch.from_numpy(ytr).long(); ye = torch.from_numpy(yte).long()
        print(f"  {'arm':<22}{'readout':<10}{'top1':>8}{'top5':>8}{'vs bound':>10}")
        for name, (Xtr, Xte) in arms.items():
            readouts = [("vector", None)] if Xtr.ndim == 2 else \
                       [("pooled", None), ("spatial", ImgReadoutSpatial)]
            for rname, cls in readouts:
                if cls is not None:
                    orig = EBP.ImgReadout; EBP.ImgReadout = cls
                try:
                    m, _ = train_readout(torch.from_numpy(np.ascontiguousarray(Xtr)).float(), yt,
                                         torch.from_numpy(np.ascontiguousarray(Xte)).float(), ye,
                                         256, 42, a.epochs, dev, False)
                finally:
                    if cls is not None:
                        EBP.ImgReadout = orig
                rec["results"][f"{name}/{rname}"] = {
                    **{k: float(v) for k, v in m.items()},
                    "vs_bound": float(m["top1"] - bound)}
                print(f"  {name:<22}{rname:<10}{m['top1']:>8.4f}{m['top5']:>8.4f}"
                      f"{m['top1']-bound:>+10.4f}", flush=True)

        R = rec["results"]
        instr = max(R["gps_blob_INSTRUMENT/pooled"]["vs_bound"],
                    R["gps_blob_INSTRUMENT/spatial"]["vs_bound"])
        rec["instrument_valid"] = bool(instr > 0.05)
        occ = max(R["gt_scene/pooled"]["vs_bound"], R["gt_scene/spatial"]["vs_bound"])
        rad = max(R["radar/pooled"]["vs_bound"], R["radar/spatial"]["vs_bound"])
        rcn = max(R["recon/pooled"]["vs_bound"], R["recon/spatial"]["vs_bound"])
        rec["occupancy_best_vs_bound"] = occ
        rec["radar_best_vs_bound"] = rad
        rec["recon_best_vs_bound"] = rcn
        rec["instrument_vs_bound"] = instr
        rec["mode4_survives"] = bool(rec["instrument_valid"] and occ <= 0.05)
        print(f"  INSTRUMENT (gps blob, best readout) {instr:+.4f} -> "
              f"{'VALID' if rec['instrument_valid'] else 'INVALID, nothing below may be read'}")
        print(f"  occupancy best {occ:+.4f}   radar best {rad:+.4f}   recon best {rcn:+.4f}")
        print(f"  MODE 4 {'SURVIVES' if rec['mode4_survives'] else 'DOES NOT SURVIVE'} on {tname}")
        out["tasks"][tname] = rec

    out["mode4_survives_both_tasks"] = all(t["mode4_survives"] for t in out["tasks"].values())
    print(f"\nMODE 4 across both tasks: "
          f"{'SURVIVES' if out['mode4_survives_both_tasks'] else 'DOES NOT SURVIVE'}")
    p = os.path.join(RESULTS_DIR, "sc36_spatial_readout_audit.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
