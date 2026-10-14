#!/usr/bin/env python
"""Test the chain resources -> bearing error -> beam accuracy, and the CRB exponent.

THE MODEL, AND WHY IT IS NOT A LOG-LOG SLOPE
--------------------------------------------
The CRB governs the bearing estimate the radar can support, sigma_theta^2 ~ R^(-1) with
R = N_samp*N_chirp*m(m^2-1). What is measured is a REGRESSOR's RMSE, which also carries an
irreducible floor: GPS error on the label, unit2's own position uncertainty, and the
readout's finite capacity. Those add in variance, not in RMSE, so

    RMSE^2(R) = a * R^(-b) + c,     with the CRB predicting b = 1 EXACTLY.

Regressing log RMSE on log R instead would absorb the floor c into the slope and bias it
toward zero, so the exponent is estimated with the floor in the model. b is reported with a
bootstrap CI over seeds, and the test is whether that CI covers 1.

The second link is a plain correlation between measured bearing RMSE and the beam-prediction
gain over the no-information bound, both of which are measured, neither of which is a
function of the other by construction.
"""
import json, os, sys
import numpy as np
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
rng = np.random.default_rng(20261004)


def fit_b(R, msq, p0=None):
    """Fit RMSE^2 = a*R^-b + c. Returns (a, b, c) or None."""
    def f(x, a, b, c):
        return a * np.power(x, -b) + c
    sc = R / R.max()
    try:
        p, _ = optimize.curve_fit(f, sc, msq, p0=p0 or [msq.max(), 1.0, msq.min() * 0.5],
                                  bounds=([0, 0.05, 0], [np.inf, 6.0, msq.max()]),
                                  maxfev=200000)
        return p
    except Exception:
        return None


def main():
    pool = sys.argv[1] if len(sys.argv) > 1 else "decimate"
    sc = sys.argv[2] if len(sys.argv) > 2 else "31"
    sfx = "" if pool == "decimate" else f"_{pool}"
    med = json.load(open(os.path.join(RES, f"sc{sc}_bearing_mediation{sfx}.json")))
    beam = json.load(open(os.path.join(RES, f"sc{sc}_resource_ablation_G{sfx}.json")))
    M, Bm = med["settings"], beam["settings"]
    bd, bb = med["bound_rmse_deg"], beam["bound_const"]
    print("=" * 78)
    print(f"MEDIATION, Scenario{sc}, pool={pool}.  bearing bound {bd:.3f} deg   "
          f"beam bound {bb:.4f}")
    print(f"bearing seeds {med['seeds']}   beam seeds {beam['seeds']}")
    print("=" * 78)
    keys = [k for k in M if k in Bm]
    print(f"\n  {'setting':<13}{'R':>10}{'RMSE':>8}{'red%':>7}{'beam med':>10}"
          f"{'gain':>9}{'coll':>7}")
    rows = []
    for k in keys:
        m, b = M[k], Bm[k]
        red = 100 * (1 - m["rmse_deg_median"] / bd)
        rows.append({"k": k, "R": m["resource_product_R"], "sweep": m["sweep"],
                     "rmse": m["rmse_deg_median"], "red_pct": red,
                     "rmse_seeds": m["rmse_deg_seeds"],
                     "beam": b["top1_median"], "gain": b["top1_median"] - bb,
                     "n_coll": m["n_collapsed"]})
        print(f"  {k:<13}{m['resource_product_R']:>10}{m['rmse_deg_median']:>8.3f}"
              f"{red:>7.1f}{b['top1_median']:>10.4f}{b['top1_median']-bb:>+9.4f}"
              f"{m['n_collapsed']:>5}/{m['n_seeds']}")

    # ---------------- link 1: the CRB exponent ----------------
    print("\n" + "=" * 78)
    print("LINK 1  resources -> bearing error.  Fit RMSE^2 = a*R^-b + c; CRB predicts b = 1.")
    print("=" * 78)
    for name, sel in (("all settings with m >= 2", [r for r in rows if r["R"] > 0]),
                      ("chirp sweep only (m, bandwidth fixed)",
                       [r for r in rows if r["sweep"] == "chirps"])):
        R = np.array([r["R"] for r in sel], float)
        msq = np.array([r["rmse"] ** 2 for r in sel])
        p = fit_b(R, msq)
        if p is None:
            print(f"\n  {name}: fit did not converge"); continue
        a, b, c = p
        pred = a * np.power(R / R.max(), -b) + c
        resid = np.sqrt(np.maximum(pred, 0)) - np.sqrt(msq)
        bs = []
        for _ in range(4000):
            mq = np.array([np.mean(rng.choice(r["rmse_seeds"], len(r["rmse_seeds"]), True)) ** 2
                           for r in sel])
            q = fit_b(R, mq, p0=[a, b, c])
            if q is not None: bs.append(q[1])
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"\n  {name}  (n = {len(sel)})")
        print(f"    b = {b:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   "
              f"irreducible floor sqrt(c) = {np.sqrt(c):.2f} deg")
        print(f"    residual s.d. {resid.std(ddof=1):.3f} deg")
        verdict = ("CONSISTENT with the CRB exponent b = 1" if lo <= 1.0 <= hi
                   else "INCONSISTENT with b = 1")
        print(f"    -> {verdict}")

    # ---------------- link 2: bearing error -> beam accuracy ----------------
    print("\n" + "=" * 78)
    print("LINK 2  bearing error -> beam accuracy.  Both measured; neither is a function of")
    print("        the other by construction.")
    print("=" * 78)
    x = np.array([r["rmse"] for r in rows]); y = np.array([r["gain"] for r in rows])
    pr = stats.pearsonr(x, y); sp = stats.spearmanr(x, y)
    print(f"\n  bearing RMSE vs beam gain over bound, n = {len(rows)}")
    print(f"    Pearson  r = {pr.statistic:+.4f}  p = {pr.pvalue:.2e}")
    print(f"    Spearman   = {sp.statistic:+.4f}  p = {sp.pvalue:.2e}")
    print(f"    (negative is the expected sign: a smaller bearing error buys a larger gain)")

    # ---------------- does bearing error MEDIATE the resource effect? -------
    print("\n" + "=" * 78)
    print("MEDIATION  partial correlations among log R, bearing RMSE and beam gain,")
    print("           over the settings with m >= 2.")
    print("=" * 78)
    sel = [r for r in rows if r["R"] > 0]
    lR = np.log10([r["R"] for r in sel]); br = np.array([r["rmse"] for r in sel])
    gn = np.array([r["gain"] for r in sel])
    def pcor(u, v, w):
        ru = u - np.polyval(np.polyfit(w, u, 1), w)
        rv = v - np.polyval(np.polyfit(w, v, 1), w)
        return stats.pearsonr(ru, rv)
    print(f"\n  n = {len(sel)}")
    print(f"    log R  vs gain                      r = {stats.pearsonr(lR, gn).statistic:+.4f}"
          f"  p = {stats.pearsonr(lR, gn).pvalue:.3f}")
    print(f"    log R  vs bearing RMSE              r = {stats.pearsonr(lR, br).statistic:+.4f}"
          f"  p = {stats.pearsonr(lR, br).pvalue:.3f}")
    print(f"    bearing RMSE vs gain                r = {stats.pearsonr(br, gn).statistic:+.4f}"
          f"  p = {stats.pearsonr(br, gn).pvalue:.3f}")
    p1 = pcor(lR, gn, br); p2 = pcor(br, gn, lR)
    print(f"    log R vs gain | bearing RMSE        r = {p1.statistic:+.4f}  p = {p1.pvalue:.3f}")
    print(f"    bearing RMSE vs gain | log R        r = {p2.statistic:+.4f}  p = {p2.pvalue:.3f}")
    print(f"\n  Reading: if bearing error mediates, controlling for it should shrink the")
    print(f"  log R -> gain partial toward zero while bearing RMSE -> gain survives.")
    print(f"  With n = {len(sel)} settings this is underpowered; report the direction, and do")
    print(f"  not claim mediation from the partials alone.")

    out = {"scenario": f"Scenario{sc}", "pool": pool, "bearing_bound_deg": bd, "beam_bound": bb,
           "bearing_seeds": med["seeds"], "beam_seeds": beam["seeds"],
           "per_setting": rows,
           "link2_pearson_r": float(pr.statistic), "link2_pearson_p": float(pr.pvalue),
           "link2_spearman_r": float(sp.statistic), "link2_spearman_p": float(sp.pvalue),
           "partial_logR_gain_given_rmse": [float(p1.statistic), float(p1.pvalue)],
           "partial_rmse_gain_given_logR": [float(p2.statistic), float(p2.pvalue)]}
    p = os.path.join(RES, f"sc{sc}_mediation_analysis{sfx}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
