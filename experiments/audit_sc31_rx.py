"""Is rx=2 genuinely worse than rx=1, or was that an unstable run?

THE ANOMALY
-----------
The Scenario 31 resource ablation reported, at 3 seeds:

    rx=1   top-1 0.1385   s.d. 0.0086
    rx=2   top-1 0.1141   s.d. 0.0642      <- seven times the next largest s.d., and BELOW rx=1
    rx=4   top-1 0.1875   s.d. 0.0095      (the full-aperture reference)

More antennas should not hurt. Either the 3-seed rx=2 mean is corrupted by collapsed runs, or
the 2-element angle FFT produces something genuinely unhelpful.

WHAT THIS RUNS
--------------
1. **Matched seed counts.** rx=1, 2 and 4 all at the same larger seed count, so no mean is
   compared against another at a different n. Every per-seed value is recorded, because a mean
   and an s.d. hide whether a distribution is bimodal, which is what a collapse looks like.
2. **A collapse detector.** A run whose top-1 lands within tolerance of the train-modal floor
   or of the no-information bound has predicted a constant class; those are reported separately
   rather than averaged in silently.
3. **A Welch test** on rx=1 against rx=2, since the question is whether the ordering is real.
4. **An information check on the input itself.** With one antenna the angle FFT of a length-1
   sequence is flat, so the range-azimuth channel carries no bearing at all; with two it is a
   single ambiguous spatial frequency. Measuring the azimuthal structure of each setting
   separates "this input carries less" from "this run trained badly".

Output: results/sc31_rx_audit.json
"""
import os, sys, json, collections, argparse
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_beam_prediction import train_readout
import exp_beam_prediction as EBP
from audit_readout_spatial import ImgReadoutSpatial

TRAIN_FRAC, GAP_FRAC = 0.70, 0.02
RX = (1, 2, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    dev = torch.device("cuda")
    z = np.load(os.path.join(RESULTS_DIR, "cache_sc31_ablation.npz"))
    beams = z["beams"]
    n = len(beams); ntr = int(TRAIN_FRAC * n); gap = int(GAP_FRAC * n)
    tr, te = slice(0, ntr), slice(ntr + gap, n)
    ytr, yte = beams[tr], beams[te]
    bmb, bn = collections.Counter(yte.tolist()).most_common(1)[0]
    bound = bn / len(yte)
    tmb = collections.Counter(ytr.tolist()).most_common(1)[0][0]
    floor = float((yte == tmb).mean())
    print(f"train {ntr}  test {len(yte)}   bound {bound:.4f}   train-modal floor {floor:.4f}")
    yt = torch.from_numpy(ytr).long(); ye = torch.from_numpy(yte).long()

    out = {"bound_const": bound, "train_modal_floor": floor, "seeds": a.seeds,
           "epochs": a.epochs, "settings": {}}
    EBP.ImgReadout, orig = ImgReadoutSpatial, EBP.ImgReadout
    try:
        for r in RX:
            X = z[f"{r}|256|250"]
            # how much azimuthal structure does this input actually carry?
            ra = X[:, 0]                                   # range-azimuth channel
            az_std = float(ra.std(axis=2).mean())          # variation ACROSS azimuth
            rg_std = float(ra.std(axis=1).mean())          # variation across range, reference
            Xtr = torch.from_numpy(np.ascontiguousarray(X[tr])).float()
            Xte = torch.from_numpy(np.ascontiguousarray(X[te])).float()
            vals = []
            for s in range(a.seeds):
                m, _ = train_readout(Xtr, yt, Xte, ye, 64, 42 + s, a.epochs, dev, False)
                vals.append(float(m["top1"]))
                print(f"  rx={r} seed {42+s}: {m['top1']:.4f}", flush=True)
            v = np.array(vals)
            collapsed = [float(x) for x in v
                         if abs(x - floor) < 0.005 or abs(x - bound) < 0.005]
            good = v[[not (abs(x - floor) < 0.005 or abs(x - bound) < 0.005) for x in v]]
            ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
            out["settings"][f"rx={r}"] = {
                "n_rx": r, "seeds": vals, "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
                "ci95": [float(ci[0]), float(ci[1])],
                "n_collapsed": len(collapsed), "collapsed_values": collapsed,
                "mean_excluding_collapsed": float(good.mean()) if len(good) else None,
                "azimuth_std": az_std, "range_std": rg_std,
                "azimuth_to_range_ratio": az_std / rg_std if rg_std else 0.0}
            print(f"  rx={r}: mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
                  f"range [{v.min():.4f},{v.max():.4f}]  collapsed {len(collapsed)}/{len(v)}  "
                  f"azimuthal structure {az_std:.4f}\n", flush=True)
    finally:
        EBP.ImgReadout = orig

    A, B = out["settings"]["rx=1"], out["settings"]["rx=2"]
    t, p = stats.ttest_ind(A["seeds"], B["seeds"], equal_var=False)
    out["rx1_vs_rx2"] = {"delta": B["mean"] - A["mean"], "t": float(t), "p": float(p),
                         "rx2_worse_significantly": bool(p < 0.05 and B["mean"] < A["mean"])}
    print(f"rx=1 {A['mean']:.4f} vs rx=2 {B['mean']:.4f}: "
          f"delta {B['mean']-A['mean']:+.4f}, Welch t={t:.2f}, p={p:.4f}")
    print(f"azimuthal structure  rx=1 {A['azimuth_std']:.4f}  rx=2 {B['azimuth_std']:.4f}  "
          f"rx=4 {out['settings']['rx=4']['azimuth_std']:.4f}")
    p_ = os.path.join(RESULTS_DIR, "sc31_rx_audit.json")
    json.dump(out, open(p_, "w"), indent=2)
    print(f"\nSaved: {p_}")


if __name__ == "__main__":
    main()
