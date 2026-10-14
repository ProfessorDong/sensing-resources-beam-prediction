#!/usr/bin/env python
"""The mediation claim, tested the way the methods literature recommends testing it.

Section VIII previously argued the mediation from two partial correlations: the
resource-to-gain path collapses once bearing error is controlled, while the bearing-to-gain
path survives. That is the causal-steps logic of Baron and Kenny, and MacKinnon, Lockwood,
Hoffman, West and Sheets (Psychological Methods 7(1):83-104, 2002) find in simulation that it
has "low Type I error rates and the LOWEST statistical power in all conditions studied". So a
collapsing direct path is weak evidence in either direction and should not carry the claim.

What they recommend instead is the JOINT SIGNIFICANCE TEST of the two constituent paths:

  alpha  the resource changes the mediator            log R  ->  bearing RMSE
  beta   the mediator is associated with the outcome  bearing RMSE -> beam gain, given log R

"The joint significance test of alpha and beta appears to be the best test in this group as it
has the most power and the most accurate Type I error rates in all cases", and they "strongly
recommend this test" on the grounds that what matters is establishing (a) and (b) rather than
the disappearance of a direct path.

TWO LIMITS WE STATE RATHER THAN HIDE.
1. Their simulations are over independent observations in a single-mediator regression model.
   Our nine units are RESOURCE SETTINGS, each a median over seeds, so their tabulated error
   rates do not transfer. What transfers is which hypothesis the test addresses.
2. "No parameter estimate or standard error of the intervening variable effect is available for
   the joint test", so no effect size or interval follows from it. We therefore claim the
   presence of both links and no magnitude for the mediated effect.

Settings with m = 1 are excluded throughout: their resource product is identically zero, so
log R is undefined there.

Output: results/joint_significance.json
"""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def pcorr(x, y, z):
    """Partial correlation of x and y given z, with its df and two-sided p."""
    x, y, z = map(lambda v: np.asarray(v, float), (x, y, z))
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = len(x) - 3                       # 2 variables + 1 control
    t = abs(r) * np.sqrt(df / (1 - r * r))
    return r, df, float(2 * stats.t.sf(t, df))


def main():
    M = json.load(open(os.path.join(RES, "sc31_bearing_mediation_max.json")))
    A = json.load(open(os.path.join(RES, "sc31_resource_ablation_G_max.json")))
    bound = A["bound_const"]
    rows = []
    for k, v in M["settings"].items():
        R = v["resource_product_R"]
        if R <= 0:                        # m = 1: log R undefined
            continue
        rows.append((k, np.log10(R), v["rmse_deg_median"],
                     A["settings"][k]["top1_median"] - bound))
    rows.sort(key=lambda r: r[1])
    lab = [r[0] for r in rows]
    logR = np.array([r[1] for r in rows])
    rmse = np.array([r[2] for r in rows])
    gain = np.array([r[3] for r in rows])
    n = len(rows)

    # alpha: resource -> mediator. A simple correlation; the resource is the manipulated
    # variable, so nothing needs controlling for.
    a_r, a_p = stats.pearsonr(logR, rmse)
    a_df = n - 2
    # beta: mediator -> outcome, controlling for the resource.
    b_r, b_df, b_p = pcorr(rmse, gain, logR)
    # the Baron and Kenny step, reported only to say we are not relying on it
    d_r, d_df, d_p = pcorr(logR, gain, rmse)

    out = {
        "n_settings": n, "settings": lab,
        "excluded": [k for k, v in M["settings"].items() if v["resource_product_R"] <= 0],
        "exclusion_reason": "m = 1 gives m(m^2-1) = 0, so R = 0 and log R is undefined",
        "source": "MacKinnon, Lockwood, Hoffman, West and Sheets, Psychological Methods "
                  "7(1):83-104, 2002, Statistical Recommendations",
        "alpha_resource_to_mediator": {
            "pair": "log10 R -> bearing RMSE", "r": float(a_r), "df": a_df,
            "p": float(a_p), "significant": bool(a_p < 0.05)},
        "beta_mediator_to_outcome_given_resource": {
            "pair": "bearing RMSE -> beam gain | log10 R", "r": b_r, "df": b_df,
            "p": b_p, "significant": bool(b_p < 0.05)},
        "joint_significance": bool(a_p < 0.05 and b_p < 0.05),
        "max_of_the_two_p": float(max(a_p, b_p)),
        "baron_kenny_direct_path_given_mediator": {
            "pair": "log10 R -> beam gain | bearing RMSE", "r": d_r, "df": d_df, "p": d_p,
            "note": "reported for completeness only; MacKinnon et al. find the causal-steps "
                    "criterion has the lowest power of the 14 tests they compare, so its "
                    "non-significance is not treated as evidence"},
        "no_effect_size": "the joint test supplies no estimate or interval for the mediated "
                          "effect, so none is claimed",
        "caveat": "nine units are resource settings, not independent observations, so the "
                  "Type I error rates simulated by MacKinnon et al. do not transfer",
    }
    print(f"n = {n} settings (excluded: {out['excluded']})\n")
    print(f"  alpha  log10 R -> bearing RMSE            r = {a_r:+.4f}  df = {a_df}  "
          f"p = {a_p:.4f}  {'SIGNIFICANT' if a_p < 0.05 else 'not significant'}")
    print(f"  beta   bearing RMSE -> gain | log10 R     r = {b_r:+.4f}  df = {b_df}  "
          f"p = {b_p:.4f}  {'SIGNIFICANT' if b_p < 0.05 else 'not significant'}")
    print(f"\n  JOINT SIGNIFICANCE TEST: {'PASSES' if out['joint_significance'] else 'FAILS'}"
          f"  (both paths significant; max p = {max(a_p, b_p):.4f})")
    print(f"\n  for completeness, the Baron and Kenny step we do NOT rely on:")
    print(f"         log10 R -> gain | bearing RMSE     r = {d_r:+.4f}  df = {d_df}  "
          f"p = {d_p:.4f}")
    p = os.path.join(RES, "joint_significance.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
