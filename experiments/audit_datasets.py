"""
Construction audit of the three benchmarks.

Checks, per dataset:
  D1  Are the measurement channels distinct, or duplicated?
  D2  Is the dynamic channel D actually populated? (manuscript Sec. III claims
      "{0,1} ground truth from temporal differencing")
  D3  Is occupancy binary? (bears on the singular-target problem in Theorem 1)
  D4  Is occupancy a deterministic per-frame monotone threshold of some
      measurement channel? If so the "reconstruction" task is pointwise
      thresholding, not an ill-posed inverse problem.
  D5  Scene diversity: mean pairwise IoU between distinct test frames. High
      values mean the scenes are near-identical and a constant predictor wins.
  D6  Trivial baselines evaluated with the paper's own metrics:
        - per-cell training-mean predictor (the prior-only baseline)
        - best fixed global threshold on each measurement channel
      Compared against the published S2S-Diff numbers.

Output: experiments/results/dataset_audit.json
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

SAVE = os.path.join(RESULTS_DIR, "dataset_audit.json")
DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

PUBLISHED = {  # S2S-Diff, from v3_results.json
    "DeepSense_ISAC": {"IoU": 0.8533, "ECE": 0.0061, "NLL": 0.0766, "Brier": 0.017912},
    "DeepSense_V2I":  {"IoU": 0.1849, "ECE": 0.2636, "NLL": 1.4901, "Brier": 0.266422},
    "RADIal":         {"IoU": 0.3866, "ECE": 0.0525, "NLL": 0.0736, "Brier": 0.011883},
}


def iou(pred_bin, gt_bin):
    i = np.logical_and(pred_bin, gt_bin).sum()
    u = np.logical_or(pred_bin, gt_bin).sum()
    return float(i / u) if u else 0.0


def metrics(p, g, n_bins=15):
    """p,g: [N,H,W] float in [0,1]; g binary. Mirrors the paper's metrics."""
    pb, gb = p > 0.5, g > 0.5
    out = {"IoU": iou(pb, gb)}
    tp = np.logical_and(pb, gb).sum()
    out["F1"] = float(2 * tp / (pb.sum() + gb.sum())) if (pb.sum() + gb.sum()) else 0.0
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    out["NLL"] = float(-(g * np.log(pc) + (1 - g) * np.log(1 - pc)).mean())
    out["Brier"] = float(((p - g) ** 2).mean())
    pf, gf = p.ravel(), g.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for k in range(n_bins):
        m = (pf >= edges[k]) & (pf < edges[k + 1])
        if m.sum():
            ece += m.sum() / len(pf) * abs(pf[m].mean() - gf[m].mean())
    out["ECE"] = float(ece)
    return out


def audit(name, tr_s, tr_m, te_s, te_m):
    print(f"\n{'='*74}\n{name}\n{'='*74}")
    r = {"n_train": int(len(tr_s)), "n_test": int(len(te_s))}
    occ_tr, occ_te = tr_s[:, 0], te_s[:, 0]

    # D1 duplicated measurement channels
    C = te_m.shape[1]
    dup = [[int(c1), int(c2)] for c1 in range(C) for c2 in range(c1 + 1, C)
           if np.array_equal(te_m[:, c1], te_m[:, c2])]
    r["D1_duplicate_meas_channel_pairs"] = dup
    r["D1_distinct_meas_channels"] = int(C - len({c2 for _, c2 in dup}))
    print(f"  [D1] measurement channels: {C} declared, "
          f"{r['D1_distinct_meas_channels']} distinct   duplicates={dup}")

    # D2 dynamic channel
    r["D2_dynamic_all_zero"] = bool(np.all(tr_s[:, 2] == 0) and np.all(te_s[:, 2] == 0))
    print(f"  [D2] dynamic channel identically zero: {r['D2_dynamic_all_zero']}")

    # D3 binary occupancy
    u = np.unique(occ_te)
    r["D3_occupancy_unique_values"] = int(len(u))
    r["D3_occupancy_is_binary"] = bool(len(u) <= 2)
    r["D3_occupancy_fill_test"] = float((occ_te > 0.5).mean())
    print(f"  [D3] occupancy unique values={len(u)}  binary={r['D3_occupancy_is_binary']}"
          f"  fill={100*r['D3_occupancy_fill_test']:.2f}%")

    # D4 is occ a per-frame monotone threshold of a measurement channel?
    d4 = {}
    for c in range(C):
        viol = 0
        for i in range(len(te_s)):
            yo = te_m[i, c][occ_te[i] > 0.5]
            yn = te_m[i, c][occ_te[i] <= 0.5]
            if len(yo) and len(yn) and yn.max() > yo.min():
                viol += 1
        d4[f"ch{c}_frames_violating"] = int(viol)
        d4[f"ch{c}_is_exact_threshold"] = bool(viol == 0)
    r["D4_threshold_structure"] = d4
    exact = [c for c in range(C) if d4[f"ch{c}_is_exact_threshold"]]
    print(f"  [D4] channels for which occupancy is EXACTLY a per-frame threshold: "
          f"{exact if exact else 'none'}")

    # D5 scene diversity
    rng = np.random.RandomState(0)
    idx = rng.choice(len(te_s), size=min(120, len(te_s)), replace=False)
    vals = [iou(occ_te[a] > .5, occ_te[b] > .5)
            for ii, a in enumerate(idx) for b in idx[ii + 1:]]
    r["D5_mean_pairwise_IoU_between_distinct_test_frames"] = float(np.mean(vals))
    print(f"  [D5] mean pairwise IoU between distinct test frames = "
          f"{np.mean(vals):.4f}   (1.0 = all scenes identical)")

    # D6 trivial baselines
    mean_scene = occ_tr.mean(axis=0, keepdims=True)
    const = np.repeat(mean_scene, len(occ_te), axis=0)
    r["D6_constant_train_mean"] = metrics(const, occ_te)
    print(f"  [D6] constant train-mean predictor : "
          f"IoU={r['D6_constant_train_mean']['IoU']:.4f}  "
          f"ECE={r['D6_constant_train_mean']['ECE']:.4f}  "
          f"NLL={r['D6_constant_train_mean']['NLL']:.4f}  "
          f"Brier={r['D6_constant_train_mean']['Brier']:.6f}")

    best = {"IoU": -1}
    for c in range(C):
        for t in np.arange(0.05, 0.96, 0.01):
            v = iou(te_m[:, c] > t, occ_te > 0.5)
            if v > best["IoU"]:
                best = {"IoU": float(v), "channel": int(c), "tau": float(round(t, 2))}
    r["D6_best_fixed_threshold"] = best
    print(f"  [D6] best fixed threshold on Y      : "
          f"IoU={best['IoU']:.4f}  (ch{best['channel']}, tau={best['tau']:.2f})")

    pub = PUBLISHED[name]
    r["published_S2S_Diff"] = pub
    print(f"  [--] published S2S-Diff             : IoU={pub['IoU']:.4f}  "
          f"ECE={pub['ECE']:.4f}  NLL={pub['NLL']:.4f}  Brier={pub['Brier']:.6f}")
    r["verdict_constant_beats_S2S_IoU"] = bool(r["D6_constant_train_mean"]["IoU"] >= pub["IoU"])
    r["verdict_threshold_beats_S2S_IoU"] = bool(best["IoU"] >= pub["IoU"])
    return r


def main():
    from train_all_real import preprocess_isac_scenario, preprocess_radial
    from preprocess_deepsense import preprocess_scenario as prep_v2i
    out = {}

    # ISAC: Sc.42+43 -> 44
    parts_s, parts_m = [], []
    for sc in (42, 43):
        s, m = preprocess_isac_scenario(os.path.join(DS, "deepsense6g", f"Scenario{sc}"),
                                        sc, max_frames=1500)
        parts_s.append(s); parts_m.append(m)
    te_s, te_m = preprocess_isac_scenario(os.path.join(DS, "deepsense6g", "Scenario44"),
                                          44, max_frames=1500)
    out["DeepSense_ISAC"] = audit("DeepSense_ISAC",
                                  np.concatenate(parts_s), np.concatenate(parts_m),
                                  te_s, te_m)

    # V2I: Sc.31 -> 33
    a_s, a_m, _ = prep_v2i(os.path.join(DS, "deepsense6g", "Scenario31"))
    b_s, b_m, _ = prep_v2i(os.path.join(DS, "deepsense6g", "Scenario33"))
    out["DeepSense_V2I"] = audit("DeepSense_V2I", a_s, a_m, b_s, b_m)

    # RADIal: 80/20
    seq = sorted(glob.glob(os.path.join(DS, "radial", "extracted", "RECORD*")))
    s, m = preprocess_radial(seq[0], os.path.join(DS, "radial", "labels_CVPR.csv"),
                             os.path.join(DS, "radial", "miscellaneous", "RADAR_CONFIG.m"),
                             max_frames=500)
    n = int(0.8 * len(s))
    out["RADIal"] = audit("RADIal", s[:n], m[:n], s[n:], m[n:])

    os.makedirs(os.path.dirname(SAVE), exist_ok=True)
    json.dump(out, open(SAVE, "w"), indent=2)
    print(f"\nSaved: {SAVE}")


if __name__ == "__main__":
    main()
