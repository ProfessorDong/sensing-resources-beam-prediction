"""Does Figure F6 survive balanced-accuracy checkpoint selection?

WHY
---
`exp_beam_prediction.train_readout` keeps the epoch with the best validation ACCURACY. On
DeepSense Sc.33 that rule was shown to prefer collapsed epochs: switching to balanced accuracy
cut collapses from 5/10 to 2/10 and moved the median from -0.0013 to +0.0161 relative to the
bound (`collapse_fix_audit.json`). The bias is toward collapse, and it bites hardest when the
no-information bound is high, because a constant predictor then scores well on validation.

F6's Task A bound is 0.1876, higher than either Sc.31 (0.0901) or Sc.33 (0.1371), so F6's arms
sit in the regime where this rule is worst. Every F6 arm failed its bound, and a
collapse-biased selector cannot manufacture a false negative into a false positive, so the
direction of the error is safe. But on Sc.33 the fix was enough to flip a sign, so "safe in
direction" is not the same as "unchanged", and Paper A's mode 4 rests on these numbers.

This re-runs F6's arms with balanced-accuracy selection and the spatial readout, including the
position-image instrument check, and reports median and collapse count per arm.

Output: results/sc36_balanced_select_audit.json
"""
import os, sys, csv, json, collections, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from audit_collapse_fix import train
from audit_sc36_spatial_readout import gps_blob, CACHE, DEV_CSV, EXTENT
from exp_beam_grid_sweep import heading_relative_position


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    dev = torch.device("cuda")

    z = np.load(os.path.join(CACHE, "sc36_e100.npz"))
    meta = json.load(open(os.path.join(CACHE, "sc36_e100_meta.json")))
    tr_sl = slice(*meta["split"]["train"]); te_sl = slice(*meta["split"]["test"])
    scenes, meas, hbeam, ridx = z["scenes"], z["meas"], z["hbeam"], z["row_index"]
    beams = hbeam.reshape(len(hbeam), -1).argmax(1)
    arr = beams // 64
    dom = collections.Counter(arr[tr_sl].tolist()).most_common(1)[0][0]

    rows = list(csv.DictReader(open(DEV_CSV)))
    gps, gps_valid = heading_relative_position(rows, ridx)
    blob = gps_blob(gps[:, 0] * 100.0, gps[:, 1] * 100.0, gps_valid, n_ch=scenes.shape[1])
    recon = np.load(os.path.join(RESULTS_DIR, "recon_cache_seed0_e100.npz"))
    ALL = {"gps_blob_INSTRUMENT": (blob[tr_sl], blob[te_sl]),
           "gt_scene": (scenes[tr_sl], scenes[te_sl]),
           "recon": (recon["tr"], recon["te"]),
           "radar": (meas[tr_sl], meas[te_sl])}

    out = {"selection": "balanced accuracy", "readout": "ImgReadoutSpatial",
           "seeds": a.seeds, "epochs": a.epochs, "tasks": {}}
    for tname, lab, ncls, mask in (("A_256way", beams, 256, np.ones(len(beams), bool)),
                                   (f"B_array{dom+1}_64way", beams % 64, 64, arr == dom)):
        ytr, yte = lab[tr_sl][mask[tr_sl]], lab[te_sl][mask[te_sl]]
        bmb, bn = collections.Counter(yte.tolist()).most_common(1)[0]
        bound = bn / len(yte)
        floor = float((yte == collections.Counter(ytr.tolist()).most_common(1)[0][0]).mean())
        yt = torch.from_numpy(ytr).long(); ye = torch.from_numpy(yte).long()
        print(f"\n=== {tname}: train {len(ytr)} test {len(yte)} bound {bound:.4f}")
        print(f"  {'arm':<22}{'median':>9}{'vs bound':>10}{'collapsed':>11}")
        rec = {"bound": bound, "floor": floor, "arms": {}}
        for name, (Xtr_np, Xte_np) in ALL.items():
            Xtr = torch.from_numpy(np.ascontiguousarray(Xtr_np[mask[tr_sl]])).float()
            Xte = torch.from_numpy(np.ascontiguousarray(Xte_np[mask[te_sl]])).float()
            v = np.array([train(Xtr, yt, Xte, ye, ncls, 42 + s, dev,
                                lr=1e-3, epochs=a.epochs, balanced=False, select="bal")[0]
                          for s in range(a.seeds)])
            coll = int(sum(abs(x - floor) < 0.005 or abs(x - bound) < 0.005 for x in v))
            rec["arms"][name] = {"seeds": [float(x) for x in v],
                                 "median": float(np.median(v)), "n_collapsed": coll,
                                 "median_vs_bound": float(np.median(v) - bound)}
            print(f"  {name:<22}{np.median(v):>9.4f}{np.median(v)-bound:>+10.4f}"
                  f"{coll:>8}/{a.seeds}", flush=True)
        A = rec["arms"]
        rec["instrument_valid"] = bool(A["gps_blob_INSTRUMENT"]["median_vs_bound"] > 0.05)
        rec["mode4_survives"] = bool(rec["instrument_valid"] and
                                     max(A[k]["median_vs_bound"]
                                         for k in ("gt_scene", "recon", "radar")) <= 0.05)
        print(f"  instrument {A['gps_blob_INSTRUMENT']['median_vs_bound']:+.4f} -> "
              f"{'VALID' if rec['instrument_valid'] else 'INVALID'};  "
              f"mode 4 {'SURVIVES' if rec['mode4_survives'] else 'DOES NOT SURVIVE'}")
        out["tasks"][tname] = rec
    out["mode4_survives_both"] = all(t["mode4_survives"] for t in out["tasks"].values())
    print(f"\nMODE 4 under balanced selection: "
          f"{'SURVIVES' if out['mode4_survives_both'] else 'DOES NOT SURVIVE'}")
    p = os.path.join(RESULTS_DIR, "sc36_balanced_select_audit.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
