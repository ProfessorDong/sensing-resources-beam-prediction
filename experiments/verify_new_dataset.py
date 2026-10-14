"""
Degeneracy screen for a new dataset. RUN THIS BEFORE WRITING A LOADER OR TRAINING.

This project lost a submission cycle because nobody ran trivial baselines on the
benchmarks before training on them. All three turned out to be degenerate:
ISAC indoor is solved better by a constant, RADIal's ground truth is a threshold
of its own measurement, and V2I is unsolved by everything including the model.

Four screens, a couple of minutes, no GPU:

  S1  Channel duplication  — are the "N-channel" measurements N distinct channels?
                             (RADIal's four were identical copies.)
  S2  Threshold circularity — is occupancy a per-frame monotone function of some
                             measurement channel? (RADIal: 0 violations in 500 frames.)
  S3  Scene diversity      — mean pairwise IoU between distinct frames. Above ~0.7
                             and a constant predictor is hard to beat. (ISAC: 0.9135.)
  S4  Trivial baselines    — constant (TRAIN mean) and best TRAIN-fitted threshold,
                             evaluated on test with the paper's metrics.

VERDICT: a dataset failing S1, S2 or S4 must not be used as a reconstruction
benchmark, whatever its own paper calls it.

Use as a library:

    from verify_new_dataset import screen
    screen("MyDataset", train_scenes, train_meas, test_scenes, test_meas)

or adapt main() below for a new loader.

Output: experiments/results/verify_<name>.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

DIVERSITY_WARN = 0.70          # above this, a constant predictor is competitive


def iou(pred_bin, gt_bin):
    i = np.logical_and(pred_bin, gt_bin).sum()
    u = np.logical_or(pred_bin, gt_bin).sum()
    return float(i / u) if u else 0.0


def metrics(p, g, n_bins=15):
    """Paper's metric suite on the occupancy channel. p,g: [N,H,W] in [0,1]."""
    pb, gb = p > 0.5, g > 0.5
    tp = np.logical_and(pb, gb).sum()
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    pf, gf = p.ravel(), g.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for k in range(n_bins):
        m = (pf >= edges[k]) & (pf < edges[k + 1])
        if m.sum():
            ece += m.sum() / len(pf) * abs(pf[m].mean() - gf[m].mean())
    return {"IoU": iou(pb, gb),
            "F1": float(2 * tp / (pb.sum() + gb.sum())) if (pb.sum() + gb.sum()) else 0.0,
            "NLL": float(-(g * np.log(pc) + (1 - g) * np.log(1 - pc)).mean()),
            "Brier": float(((p - g) ** 2).mean()),
            "ECE_equalwidth15": float(ece)}


def screen(name, tr_scenes, tr_meas, te_scenes, te_meas, model_IoU=None, save=True):
    """tr_*/te_*: [N,C,H,W] float arrays; channel 0 of scenes is occupancy."""
    print(f"\n{'='*78}\nDEGENERACY SCREEN: {name}\n{'='*78}")
    r = {"name": name, "n_train": int(len(tr_scenes)), "n_test": int(len(te_scenes))}
    otr, ote = tr_scenes[:, 0], te_scenes[:, 0]
    C = te_meas.shape[1]
    fails = []

    # -- S1 channel duplication -------------------------------------------
    dup = [[c1, c2] for c1 in range(C) for c2 in range(c1 + 1, C)
           if np.array_equal(te_meas[:, c1], te_meas[:, c2])]
    distinct = C - len({c2 for _, c2 in dup})
    r["S1_channels_declared"] = C
    r["S1_channels_distinct"] = int(distinct)
    r["S1_duplicate_pairs"] = dup
    s1_fail = distinct < C
    print(f"  S1 channels      : {C} declared, {distinct} distinct"
          f"{'   *** FAIL: duplicated ***' if s1_fail else '   ok'}")
    if s1_fail: fails.append("S1_duplicate_channels")

    # -- S2 threshold circularity -----------------------------------------
    exact = []
    viol = {}
    for c in range(C):
        v = 0
        for i in range(len(te_scenes)):
            yo = te_meas[i, c][ote[i] > 0.5]
            yn = te_meas[i, c][ote[i] <= 0.5]
            if len(yo) and len(yn) and yn.max() > yo.min():
                v += 1
        viol[f"ch{c}"] = int(v)
        if v == 0:
            exact.append(c)
    r["S2_frames_violating_monotone_threshold"] = viol
    r["S2_channels_exactly_threshold"] = exact
    s2_fail = len(exact) > 0
    print(f"  S2 circularity   : channels where GT is EXACTLY a per-frame threshold: "
          f"{exact if exact else 'none'}"
          f"{'   *** FAIL: target is a function of the input ***' if s2_fail else '   ok'}")
    if s2_fail: fails.append("S2_threshold_circular")

    # -- S3 scene diversity ------------------------------------------------
    rng = np.random.RandomState(0)
    idx = rng.choice(len(te_scenes), size=min(120, len(te_scenes)), replace=False)
    vals = [iou(ote[a] > .5, ote[b] > .5)
            for ii, a in enumerate(idx) for b in idx[ii + 1:]]
    div = float(np.mean(vals)) if vals else float("nan")
    r["S3_mean_pairwise_IoU"] = div
    s3_warn = div > DIVERSITY_WARN
    print(f"  S3 diversity     : mean pairwise IoU between distinct frames = {div:.4f}"
          f"{'   *** WARN: scenes nearly identical ***' if s3_warn else '   ok'}")
    if s3_warn: fails.append("S3_low_diversity(warn)")

    # -- S4 trivial baselines ---------------------------------------------
    const = np.repeat(otr.mean(0, keepdims=True), len(ote), 0)
    mc = metrics(const, ote)
    best = (-1.0, None, None)
    for c in range(C):
        for t in np.arange(0.02, 0.99, 0.01):
            v = iou(tr_meas[:, c] > t, otr > 0.5)      # fit on TRAIN
            if v > best[0]:
                best = (v, c, round(float(t), 2))
    _, bc, bt = best
    thr_test = iou(te_meas[:, bc] > bt, ote > 0.5)     # apply to TEST
    r["S4_constant_train_mean"] = mc
    r["S4_threshold"] = {"channel": int(bc), "tau": bt,
                         "train_IoU": float(best[0]), "test_IoU": float(thr_test)}
    print(f"  S4 constant      : IoU={mc['IoU']:.4f}  ECE={mc['ECE_equalwidth15']:.4f}  "
          f"NLL={mc['NLL']:.4f}  Brier={mc['Brier']:.6f}")
    print(f"  S4 threshold     : IoU={thr_test:.4f}  (ch{bc}, tau={bt:.2f}, train-fitted)")
    if model_IoU is not None:
        r["model_IoU"] = float(model_IoU)
        beaten = [n for n, v in (("constant", mc["IoU"]), ("threshold", thr_test))
                  if v >= model_IoU]
        r["S4_trivial_beats_model"] = beaten
        print(f"  S4 vs model      : model IoU={model_IoU:.4f}  "
              f"beaten by: {beaten if beaten else 'neither'}"
              f"{'   *** FAIL ***' if beaten else '   ok'}")
        if beaten: fails.append("S4_trivial_baseline_wins")

    r["fails"] = fails
    r["verdict"] = "USABLE" if not [f for f in fails if "warn" not in f] else "DEGENERATE"
    print(f"\n  VERDICT: {r['verdict']}" + (f"   ({', '.join(fails)})" if fails else ""))

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        p = os.path.join(RESULTS_DIR, f"verify_{name}.json")
        json.dump(r, open(p, "w"), indent=2)
        print(f"  saved: {p}")
    return r


def main():
    print(__doc__)
    print("No dataset specified. Import screen() and pass your arrays, e.g.:\n"
          "    from verify_new_dataset import screen\n"
          "    screen('VoD', tr_s, tr_m, te_s, te_m, model_IoU=0.42)\n")


if __name__ == "__main__":
    main()
