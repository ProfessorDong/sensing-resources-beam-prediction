#!/usr/bin/env python
"""Corrected statistics for Paper B, following the 2026-10-06 audit.

Six defects, each reproducible:

1. The quoted chirp slope was fitted to MEANS while the surrounding text discusses MEDIANS.
2. Partial-correlation p-values used an ordinary Pearson reference, ignoring the degree of
   freedom spent on the control variable. With n settings and one control, df = n - 3.
3. The Scenario-33 trend quoted the asymptotic Spearman p for six points. With 6! = 720
   orderings the exact two-sided permutation p is available and is what should be reported.
4. The exponent's point estimate was fitted to squared MEDIAN RMSE while its bootstrap
   resampled squared MEAN RMSE, so the interval did not describe the reported statistic.
5. One near-equal-payload contrast was reported as a null on the strength of a percentile
   interval while the saved permutation p was 0.015. Interval and test are not dual; report both.
6. The two "equal payload" pairs are not both exact: 256*125 = 128*250 = 32000, but
   256*64 = 16384 against 64*250 = 16000, a 2.4% mismatch.

Output: results/corrected_statistics.json
"""
import itertools, json, os
import numpy as np
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CH = [8, 16, 32, 64, 125, 250]


def main():
    out = {}
    # ---- 1 & 3: slopes and exact trend tests ----
    for sc in ("31", "33"):
        d = json.load(open(os.path.join(RES, f"sc{sc}_resource_ablation_G_max.json")))
        S = d["settings"]
        med = [S[f"chirps={c}"]["top1_median"] for c in CH]
        mean = [S[f"chirps={c}"]["top1_mean"] for c in CH]
        rho = stats.spearmanr(CH, med)
        null = [stats.spearmanr(CH, p).statistic for p in itertools.permutations(med)]
        exact = float(np.mean(np.abs(null) >= abs(rho.statistic) - 1e-12))
        out[f"sc{sc}"] = {
            "slope_of_medians_per_doubling": float(np.polyfit(np.log2(CH), med, 1)[0]),
            "slope_of_means_per_doubling": float(np.polyfit(np.log2(CH), mean, 1)[0]),
            "stored_slope_was_means": d["chirp_trend"]["slope_top1_per_log2_chirp"],
            "spearman_rho": float(rho.statistic),
            "spearman_asymptotic_p": float(rho.pvalue),
            "spearman_exact_two_sided_p": exact,
        }
    # ---- 2: partial correlations with the correct reference distribution ----
    m = json.load(open(os.path.join(RES, "sc31_mediation_analysis_max.json")))
    n_set = len([r for r in m["per_setting"] if r["R"] > 0])
    df = n_set - 3                      # 2 variables + 1 control
    out["partial_correlations"] = {"n_settings": n_set, "df": df}
    for k in ("partial_logR_gain_given_rmse", "partial_rmse_gain_given_logR"):
        r, reported = m[k]
        t = abs(r) * np.sqrt(df / (1 - r * r))
        out["partial_correlations"][k] = {
            "r": r, "reported_p": reported,
            "corrected_p": float(2 * stats.t.sf(t, df))}
    # ---- 4: exponent, bootstrap consistent with the point estimate ----
    B = json.load(open(os.path.join(RES, "sc31_bearing_mediation_max.json")))["settings"]
    sel = [v for v in B.values() if v["n_rx"] == 4]
    R = np.array([v["resource_product_R"] for v in sel], float)
    x = R / R.max()
    seeds = np.array([v["rmse_deg_seeds"] for v in sel])
    f = lambda z, a, b, c: a * z ** (-b) + c

    def fit(msq, p0=None):
        hi = [np.inf, 6.0, msq.max()]
        lo = [0.0, 0.05, 0.0]
        # A seed p0 taken from another resample can sit outside this resample's bounds,
        # which makes curve_fit raise before it starts; clip it back inside.
        start = list(p0) if p0 is not None else [msq.max(), 1.0, msq.min() / 2]
        start = [min(max(v, l), h if np.isfinite(h) else v) for v, l, h in zip(start, lo, hi)]
        return optimize.curve_fit(f, x, msq, p0=start, bounds=(lo, hi), maxfev=200000)[0]
    point = fit(np.median(seeds, 1) ** 2)
    rng = np.random.default_rng(20261006)
    for tag, paired in (("independent_per_setting", False), ("joint_seed", True)):
        est = []
        for _ in range(4000):
            ids = rng.integers(seeds.shape[1],
                               size=(seeds.shape[1],) if paired else seeds.shape)
            smp = seeds[:, ids] if paired else np.take_along_axis(seeds, ids, 1)
            try:
                est.append(fit(np.median(smp, 1) ** 2, point)[1])
            except Exception:
                pass
        out.setdefault("exponent", {})[tag] = {
            "resamples": 4000, "converged": len(est),
            "ci95": np.percentile(est, [2.5, 97.5]).tolist()}
    out["exponent"]["point_b_median_based"] = float(point[1])
    out["exponent"]["floor_deg"] = float(np.sqrt(point[2]))
    # ---- 5 & 6: the near-equal-payload contrasts ----
    P = json.load(open(os.path.join(RES, "sc31_pooling_comparison.json")))
    iso = {v["comparison"]: v for v in P["verdicts"]}
    out["equal_payload"] = {}
    for nm, key, pay in (("exact", "chirps=125 -> samples=128", (256 * 125, 128 * 250)),
                         ("approximate", "chirps=64 -> samples=64", (256 * 64, 64 * 250))):
        v = iso[key]["max"]
        out["equal_payload"][nm] = {
            "comparison": key, "payloads": list(pay),
            "exactly_matched": pay[0] == pay[1],
            "mismatch_pct": 100 * abs(pay[0] - pay[1]) / max(pay),
            "delta": v["delta"], "ci95": v["ci"], "permutation_p": v["p"],
            "interval_excludes_zero": bool(v["ci"][0] > 0 or v["ci"][1] < 0)}
    # Which settings actually clear the no-information bound, by bootstrap interval rather
    # than by point estimate. The point estimate alone overstates this at low resources.
    out["clears_bound"] = {}
    for sc in (31, 33):
        d = json.load(open(os.path.join(RES, f"sc{sc}_resource_ablation_G_max.json")))
        bound = d["bound_const"]
        rg = np.random.default_rng(20261006)
        bad, n = [], 0
        for k, v in d["settings"].items():
            x = np.asarray(v["top1_seeds"], float)
            bt = np.median(rg.choice(x, (20000, x.size), True), axis=1)
            n += 1
            if np.percentile(bt, 2.5) <= bound:
                bad.append(k)
        out["clears_bound"][str(sc)] = {
            "bound": bound, "n_settings": n, "n_clearing": n - len(bad),
            "not_clearing": bad, "rule": "2.5th bootstrap percentile of the median > bound",
            "resamples": 20000, "seed": 20261006}
    # The azimuth-peak lookup is near-constant on Scenario 33; the share of its most common
    # prediction says that more honestly than a count of distinct classes.
    out["peak_predictor_concentration"] = {
        str(sc): {k: {"distinct_classes": v["distinct_top1_classes"],
                      "max_class_share": v["max_class_share"]}
                  for k, v in json.load(
                      open(os.path.join(RES, f"sc{sc}_reference_predictors.json")))["predictors"].items()
                  if k.startswith("radar-peak")}
        for sc in (31, 33)}

    p = os.path.join(RES, "corrected_statistics.json")
    json.dump(out, open(p, "w"), indent=2)
    for sc in ("31", "33"):
        o = out[f"sc{sc}"]
        print(f"Sc.{sc}: slope medians {o['slope_of_medians_per_doubling']:.6f} vs means "
              f"{o['slope_of_means_per_doubling']:.6f} (stored {o['stored_slope_was_means']:.6f})")
        print(f"       Spearman rho {o['spearman_rho']:.4f}, asymptotic p "
              f"{o['spearman_asymptotic_p']:.6f}, EXACT p {o['spearman_exact_two_sided_p']:.6f}")
    pc = out["partial_correlations"]
    print(f"\npartials (n={pc['n_settings']}, df={pc['df']}):")
    for k in ("partial_logR_gain_given_rmse", "partial_rmse_gain_given_logR"):
        print(f"  {k}: r={pc[k]['r']:+.4f} reported p={pc[k]['reported_p']:.4f} -> "
              f"corrected p={pc[k]['corrected_p']:.5f}")
    e = out["exponent"]
    print(f"\nexponent b={e['point_b_median_based']:.4f}, floor {e['floor_deg']:.2f} deg")
    for t in ("independent_per_setting", "joint_seed"):
        print(f"  {t}: CI95 [{e[t]['ci95'][0]:.3f}, {e[t]['ci95'][1]:.3f}] "
              f"({e[t]['converged']}/{e[t]['resamples']} converged)")
    print("\nequal-payload contrasts:")
    for nm, v in out["equal_payload"].items():
        print(f"  {nm}: payload {v['payloads']} (mismatch {v['mismatch_pct']:.1f}%), "
              f"delta {v['delta']:+.4f}, CI {np.round(v['ci95'],4).tolist()}, "
              f"perm p={v['permutation_p']:.4f}, CI excludes 0: {v['interval_excludes_zero']}")
    print("\nsettings whose bootstrap median interval excludes the no-information bound:")
    for sc, v in out["clears_bound"].items():
        print(f"  Sc.{sc}: {v['n_clearing']} of {v['n_settings']}; "
              f"not clearing: {', '.join(v['not_clearing']) or 'none'}")
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
