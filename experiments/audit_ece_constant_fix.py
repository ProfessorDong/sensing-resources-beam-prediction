"""
Correct the constant-predictor arm of ece_reestimation.json.

audit_ece_estimator.py computed the constant predictor as the mean of the TEST
split (load_*() returns test data only). A test-set mean is perfectly
calibrated on that same test set by construction, which is why it reported
ECE = 0.0000 at every bin count and scheme. That is an oracle, not a result.

This script recomputes that one arm honestly: per-cell mean of the TRAIN split,
evaluated against the TEST split, with the same four estimators and bin grid.
It merges the corrected record back into ece_reestimation.json under
"constant" and records the erroneous values under "constant_TESTMEAN_INVALID"
so the mistake stays visible in the artifact.

Splits match the canonical pipeline exactly:
  ISAC   train Sc.42+43  -> test Sc.44
  V2I    train Sc.31     -> test Sc.33
  RADIal train first 80% -> test last 20%
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from audit_ece_estimator import calib, BIN_GRID

SAVE = os.path.join(RESULTS_DIR, "ece_reestimation.json")
DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")


def splits():
    from train_all_real import preprocess_isac_scenario, preprocess_radial
    from preprocess_deepsense import preprocess_scenario as pv2i

    ps, pm = [], []
    for sc in (42, 43):
        a, b = preprocess_isac_scenario(os.path.join(DS, "deepsense6g", f"Scenario{sc}"),
                                        sc, max_frames=1500)
        ps.append(a); pm.append(b)
    te, _ = preprocess_isac_scenario(os.path.join(DS, "deepsense6g", "Scenario44"),
                                     44, max_frames=1500)
    yield "DeepSense_ISAC", np.concatenate(ps)[:, 0], te[:, 0]

    tr, _, _ = pv2i(os.path.join(DS, "deepsense6g", "Scenario31"))
    ts, _, _ = pv2i(os.path.join(DS, "deepsense6g", "Scenario33"))
    yield "DeepSense_V2I", tr[:, 0], ts[:, 0]

    seq = sorted(glob.glob(os.path.join(DS, "radial", "extracted", "RECORD*")))
    s, _ = preprocess_radial(seq[0], os.path.join(DS, "radial", "labels_CVPR.csv"),
                             os.path.join(DS, "radial", "miscellaneous", "RADAR_CONFIG.m"),
                             max_frames=500)
    n = int(0.8 * len(s))
    yield "RADIal", s[:n, 0], s[n:, 0]


def main():
    out = json.load(open(SAVE))
    for name, occ_tr, occ_te in splits():
        if name not in out:
            print(f"  {name}: not in results file, skipping"); continue
        pred = np.repeat(occ_tr.mean(0, keepdims=True), len(occ_te), 0)
        p, y = pred.ravel(), occ_te.ravel()
        rec = {"n_cells": int(p.size), "mean_pred": float(p.mean()),
               "base_rate": float(y.mean()), "fitted_on": "train split",
               "sweep": {}}
        for nb in BIN_GRID:
            e1w = calib(p, y, nb, "width")[0]
            e1m, e2p, e2d = calib(p, y, nb, "mass")
            rec["sweep"][str(nb)] = {"E1_width": e1w, "E1_mass": e1m,
                                     "E2_plugin": e2p, "E2_debias": e2d}
        if "constant" in out[name]:
            out[name]["constant_TESTMEAN_INVALID"] = out[name]["constant"]
            out[name]["constant_TESTMEAN_INVALID"]["note"] = (
                "oracle: fitted on the test split; retained only to document the error")
        out[name]["constant"] = rec
        s15 = rec["sweep"]["15"]
        print(f"  {name:<16} constant(train mean)  E1_width@15={s15['E1_width']:.4f}  "
              f"E1_mass@15={s15['E1_mass']:.4f}  E2_debias@15={s15['E2_debias']:.4f}  "
              f"-> E1_width@500={rec['sweep']['500']['E1_width']:.4f}")
    json.dump(out, open(SAVE, "w"), indent=2)
    print(f"\nSaved: {SAVE}")


if __name__ == "__main__":
    main()
