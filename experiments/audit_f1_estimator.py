"""Figure 1 audit: are the model and its baselines scored with the SAME IoU estimator?

THE PROBLEM
-----------
Two IoU reductions are in use in this project and they are not the same number.

  mean-per-frame  `s2s_diff_v3.compute_iou`: per-frame intersection/union, then mean over
                  frames. This is what `evaluate_full` reports, and it is the estimator
                  behind EVERY model number in Figure 1.
  pooled          `audit_trivial_baselines.iou` and `exp_vod_sparsity.iou_masked`: sum the
                  intersection and the union over all frames and all cells, then divide.
                  This is the estimator behind the CONSTANT and THRESHOLD baselines for the
                  three original benchmarks, and behind the THRESHOLD baseline for Sc.36 and
                  View-of-Delft.

They coincide only when frames are homogeneous. On View-of-Delft the analogous pair already
measured 2.1% apart. Figure 1's smallest margin is DeepSense indoor ISAC at 0.0087, which is
1.0% of the score, so the sign of that comparison could be an artifact of mixing estimators
rather than a fact about the model.

WHAT THIS SCRIPT DOES
---------------------
Recomputes every Figure 1 baseline under BOTH estimators on the exact test frames, using the
hyperparameters already fitted on train, and reports the gap. Baselines are closed-form, so
no retraining is involved and the model numbers are untouched.

Output: results/f1_estimator_audit.json
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")


def iou_pooled(p, g):
    p, g = p > 0.5, g > 0.5
    u = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / u) if u else 0.0


def iou_perframe(p, g):
    """Matches s2s_diff_v3.compute_iou: per-frame IoU, then mean over frames."""
    p, g = p > 0.5, g > 0.5
    inter = np.logical_and(p, g).sum(axis=(-2, -1)).astype(np.float64)
    union = np.logical_or(p, g).sum(axis=(-2, -1)).astype(np.float64)
    return float((inter / (union + 1e-8)).mean())


def both(p, g):
    return {"pooled": iou_pooled(p, g), "per_frame": iou_perframe(p, g)}


def main():
    tb = json.load(open(os.path.join(RESULTS_DIR, "trivial_baselines.json")))
    out = {"estimators": {
        "per_frame": "s2s_diff_v3.compute_iou -- per-frame IoU then mean; ALL model numbers",
        "pooled": "audit_trivial_baselines.iou / iou_masked -- pooled over frames and cells"}}

    from train_all_real import preprocess_isac_scenario, preprocess_radial
    from preprocess_deepsense import preprocess_scenario as pv2i

    # ---- DeepSense indoor ISAC: train Sc.42+43, test Sc.44
    ps, pm = [], []
    for sc in (42, 43):
        a, b = preprocess_isac_scenario(os.path.join(DS, "deepsense6g", f"Scenario{sc}"), sc,
                                        max_frames=1500)
        ps.append(a); pm.append(b)
    tr_s = np.concatenate(ps); tr_m = np.concatenate(pm)
    te_s, te_m = preprocess_isac_scenario(os.path.join(DS, "deepsense6g", "Scenario44"), 44,
                                          max_frames=1500)
    sets = {"DeepSense_ISAC": (tr_s, tr_m, te_s, te_m)}

    a_s, a_m, _ = pv2i(os.path.join(DS, "deepsense6g", "Scenario31"))
    b_s, b_m, _ = pv2i(os.path.join(DS, "deepsense6g", "Scenario33"))
    sets["DeepSense_V2I"] = (a_s, a_m, b_s, b_m)

    seq = sorted(glob.glob(os.path.join(DS, "radial", "extracted", "RECORD*")))
    s_, m_ = preprocess_radial(seq[0], os.path.join(DS, "radial", "labels_CVPR.csv"),
                               os.path.join(DS, "radial", "miscellaneous", "RADAR_CONFIG.m"),
                               max_frames=500)
    n = int(0.8 * len(s_))
    sets["RADIal"] = (s_[:n], m_[:n], s_[n:], m_[n:])

    for name, (trs, trm, tes, tem) in sets.items():
        gt_tr, gt_te = trs[:, 0], tes[:, 0]
        # constant: per-cell train occupancy mean, exactly as the published baseline
        const = np.repeat(gt_tr.mean(0, keepdims=True), len(gt_te), 0)
        c = int(tb[name]["threshold"]["channel"]); t = float(tb[name]["threshold"]["tau"])
        thr = (tem[:, c] > t).astype(np.float32)
        rec = {"n_test": int(len(gt_te)),
               "constant": both(const, gt_te),
               "threshold": both(thr, gt_te),
               "threshold_rule": f"ch{c} > {t:g}",
               "published_pooled_constant": tb[name]["constant"]["IoU"],
               "published_pooled_threshold": tb[name]["threshold"]["test_IoU"]}
        for k in ("constant", "threshold"):
            rec[k]["gap_perframe_minus_pooled"] = rec[k]["per_frame"] - rec[k]["pooled"]
        out[name] = rec
        print(f"{name}: n_test={rec['n_test']}")
        for k in ("constant", "threshold"):
            v = rec[k]
            print(f"   {k:<10} pooled {v['pooled']:.4f}   per-frame {v['per_frame']:.4f}   "
                  f"gap {v['gap_perframe_minus_pooled']:+.4f}")

    p = os.path.join(RESULTS_DIR, "f1_estimator_audit.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
