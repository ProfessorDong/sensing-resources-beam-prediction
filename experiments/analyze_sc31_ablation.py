"""Robust analysis of the Sc.31 resource ablation, given a ~10-20% collapse rate.

WHY NOT THE MEAN
----------------
The readout collapses onto a constant class in a fraction of runs that varies by setting
(0/10 at 16-125 chirps, 2/10 at 8, 3/10 at 250). A collapsed run is a failed optimization, not
a measurement of what the input carries, and averaging it in drags a setting's mean toward the
bound in proportion to how often it happened. That is why the 10-seed MEAN is non-monotone in
chirps while the MEDIAN is not.

Collapse is defined by a rule fixed before the run: a top-1 within 0.005 of the train-modal
floor or of the no-information bound, i.e. the predictor is constant. The rule is about the
predictor's behaviour, not about its score being low.

Both statistics are reported. Uncertainty on the median is bootstrapped rather than assumed,
and every claim is tested rather than read off the ordering.

Output: results/sc31_ablation_analysis.json
"""
import os, sys, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

RNG = np.random.RandomState(0)
B = 20000


def boot_ci(x, fn=np.median, b=B):
    x = np.asarray(x)
    s = np.array([fn(RNG.choice(x, len(x), replace=True)) for _ in range(b)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot_diff(x, y, fn=np.median, b=B):
    """P(fn(y) > fn(x)) and a CI on the difference, by independent resampling."""
    x, y = np.asarray(x), np.asarray(y)
    d = np.array([fn(RNG.choice(y, len(y), replace=True)) -
                  fn(RNG.choice(x, len(x), replace=True)) for _ in range(b)])
    return float(d.mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))), \
        float((d > 0).mean())


def main():
    import sys as _s
    sc_tag = _s.argv[1] if len(_s.argv) > 1 else "31"
    d = json.load(open(os.path.join(RESULTS_DIR, f"sc{sc_tag}_resource_ablation.json")))
    S, bound = d["settings"], d["bound_const"]
    out = {"bound_const": bound, "n_seeds": d["seeds"], "settings": {}}

    print(f"bound {bound:.4f}\n")
    print(f"{'setting':<14}{'mean':>8}{'median':>9}{'clean':>9}{'coll':>6}"
          f"{'median 95% CI':>22}")
    for k, v in S.items():
        x = np.asarray(v["top1_seeds"])
        clean = v.get("top1_mean_excluding_collapsed")
        lo, hi = boot_ci(x)
        out["settings"][k] = {
            "n_chirps": v["n_chirps"], "n_samples": v["n_samples"], "n_rx": v["n_rx"],
            "sweep": v["sweep"], "seeds": v["top1_seeds"],
            "mean": v["top1_mean"], "median": v["top1_median"], "sd": v["top1_sd"],
            "mean_excluding_collapsed": clean, "n_collapsed": v["n_collapsed"],
            "median_ci95": [lo, hi], "median_vs_bound": v["top1_median"] - bound,
            "utilization": v["utilization"], "top5_mean": v["top5_mean"]}
        print(f"{k:<14}{v['top1_mean']:>8.4f}{v['top1_median']:>9.4f}"
              f"{(clean if clean else float('nan')):>9.4f}{v['n_collapsed']:>4}/10"
              f"   [{lo:.4f}, {hi:.4f}]")

    ch = sorted([v for v in out["settings"].values() if v["sweep"] == "chirps"],
                key=lambda v: v["n_chirps"])
    c = [v["n_chirps"] for v in ch]
    med = [v["median"] for v in ch]
    print(f"\nCHIRP SWEEP, median: {' -> '.join(f'{m:.4f}' for m in med)}")
    rho, prho = stats.spearmanr(c, med)
    out["chirp_trend"] = {"n_chirps": c, "median": med,
                          "spearman_rho": float(rho), "spearman_p": float(prho),
                          "monotone_median": bool(all(a <= b + 1e-9 for a, b in zip(med, med[1:])))}
    print(f"  Spearman(chirps, median) = {rho:+.4f}, p = {prho:.4f}   "
          f"monotone: {out['chirp_trend']['monotone_median']}")

    # the two claims that matter, each tested rather than read off the ordering
    i8, i64, i250 = c.index(8), c.index(64), c.index(250)
    tests = {}
    for tag, a_, b_ in (("8_to_64", ch[i8], ch[i64]),
                        ("64_to_250", ch[i64], ch[i250]),
                        ("8_to_250", ch[i8], ch[i250])):
        dm, (lo, hi), pgt = boot_diff(a_["seeds"], b_["seeds"])
        tests[tag] = {"median_delta": dm, "ci95": [lo, hi], "p_greater": pgt,
                      "distinguishable_from_zero": bool(lo > 0 or hi < 0)}
        print(f"  {tag:<10} median delta {dm:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(gain>0) = {pgt:.3f}  -> "
              f"{'distinguishable' if tests[tag]['distinguishable_from_zero'] else 'NOT distinguishable'} from zero")
    out["marginal_value_tests"] = tests

    # does utilization still track the gain when both are robust statistics?
    allv = list(out["settings"].values())
    u = [v["utilization"] for v in allv]
    g = [v["median"] - bound for v in allv]
    r, p = stats.pearsonr(u, g); rs, ps = stats.spearmanr(u, g)
    out["utilization_vs_gain"] = {"pearson_r": float(r), "pearson_p": float(p),
                                  "spearman_rho": float(rs), "spearman_p": float(ps),
                                  "n": len(allv), "statistic": "median gain over bound"}
    print(f"\nutilization vs median gain over bound, all {len(allv)} settings:")
    print(f"  Pearson r = {r:+.4f} (p={p:.2e})   Spearman = {rs:+.4f} (p={ps:.2e})")

    p_ = os.path.join(RESULTS_DIR, f"sc{sc_tag}_ablation_analysis.json")
    json.dump(out, open(p_, "w"), indent=2)
    print(f"\nSaved: {p_}")


if __name__ == "__main__":
    main()
