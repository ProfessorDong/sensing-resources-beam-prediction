#!/usr/bin/env python
"""Does removing the zero-beat-frequency term change any Scenario 31 conclusion?

`validate_radar_frontend.py` found that the brightest cell of the Scenario 31 range-azimuth map
lies in the first range bin in 100.0% of frames, because the chain as published applies no
leakage removal and transmit leakage arrives at zero beat frequency. `audit_dc_removal.py`
shows that subtracting the window-weighted fast-time mean removes it exactly: the artifact rate
goes to 0.0%, the share of range-transform energy in the first two bins falls from 45.7% to
0.3%, and the azimuth of the peak starts tracking the partner's true bearing.

That repaired front end feeds the readout differently at every setting, so the sweep has to be
re-run rather than argued about. This compares the two runs setting by setting and re-tests
every verdict the paper states, in the same form as the resampler comparison, so a conclusion
that holds only on the leakage-dominated input is caught here.

One factor differs between the two runs: the mean subtraction. Same recipe, same seeds, same
split, same resampler, same bound.

Output: results/sc31_dc_comparison.json
"""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
rng = np.random.default_rng(20261007)

PAIRS = [("chirps=8", "chirps=64"), ("chirps=64", "chirps=125"),
         ("chirps=125", "chirps=250"), ("chirps=8", "chirps=250"),
         ("samples=64", "samples=128"), ("samples=128", "chirps=250"),
         ("rx=1", "rx=2"), ("rx=2", "chirps=250")]
ISO = [("chirps=125", "samples=128"), ("chirps=64", "samples=64")]
CH = [8, 16, 32, 64, 125, 250]


def perm(a, b, stat=np.median, n=20000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = stat(a) - stat(b); pool = np.concatenate([a, b]); c = 0
    for _ in range(n):
        p = rng.permutation(pool)
        c += abs(stat(p[: a.size]) - stat(p[a.size:])) >= abs(obs) - 1e-12
    bs = np.array([stat(rng.choice(a, a.size, True)) - stat(rng.choice(b, b.size, True))
                   for _ in range(n)])
    return obs, np.percentile(bs, 2.5), np.percentile(bs, 97.5), (c + 1) / (n + 1)


def main():
    A = json.load(open(os.path.join(RES, "sc31_resource_ablation_G_max.json")))
    p_dc = os.path.join(RES, "sc31_resource_ablation_G_max_dcmean.json")
    if not os.path.exists(p_dc):
        print(f"missing {p_dc}; run exp_sc31_resource_ablation.py --pool max --dc mean")
        return 1
    B = json.load(open(p_dc))
    assert abs(A["bound_const"] - B["bound_const"]) < 1e-12, "bounds differ; not comparable"
    assert A["pool"] == B["pool"] == "max" and A["recipe"] == B["recipe"]
    assert A.get("dc_removal", "none") == "none" and B["dc_removal"] == "mean"
    bound = A["bound_const"]
    out = {"bound": bound, "pool": A["pool"], "recipe": A["recipe"],
           "seeds": {"leakage_retained": A["seeds"], "leakage_removed": B["seeds"]},
           "factor": "window-weighted fast-time mean subtraction before the range transform",
           "n_adc": None}
    print("=" * 92)
    print(f"ZERO-BEAT-FREQUENCY REMOVAL: retained vs removed.  bound {bound:.4f} "
          f"(identical; labels untouched)")
    print(f"seeds: retained {A['seeds']}, removed {B['seeds']}")
    print("=" * 92)
    print(f"\n{'setting':<13}{'retained':>10}{'removed':>10}{'delta':>10}{'p':>9}"
          f"{'top5 ret':>10}{'top5 rem':>10}{'coll':>12}")
    rows = []
    for k in A["settings"]:
        if k not in B["settings"]:
            continue
        a, b = A["settings"][k], B["settings"][k]
        o, lo, hi, p = perm(b["top1_seeds"], a["top1_seeds"])
        rows.append({"setting": k,
                     "retained_median": a["top1_median"], "removed_median": b["top1_median"],
                     "delta": float(o), "ci": [float(lo), float(hi)], "p": float(p),
                     "retained_top5": a["top5_mean"], "removed_top5": b["top5_mean"],
                     "retained_collapsed": a["n_collapsed"], "removed_collapsed": b["n_collapsed"],
                     "removed_clears_bound": bool(b["top1_median"] > bound)})
        r = rows[-1]
        print(f"{k:<13}{r['retained_median']:>10.4f}{r['removed_median']:>10.4f}"
              f"{o:>+10.4f}{p:>9.4f}{r['retained_top5']:>10.4f}{r['removed_top5']:>10.4f}"
              f"{r['retained_collapsed']:>6}/{A['seeds']}{r['removed_collapsed']:>4}/{B['seeds']}")
    d = np.array([r["delta"] for r in rows])
    sig = [r for r in rows if r["p"] < 0.05]
    out["per_setting"] = rows
    out["mean_delta"] = float(d.mean())
    out["n_settings"] = len(rows)
    out["n_higher"] = int((d > 0).sum())
    out["n_significant"] = len(sig)
    out["max_p"] = float(max(r["p"] for r in rows))
    print(f"\n  removal is higher in {out['n_higher']}/{len(rows)} settings, "
          f"mean change {d.mean():+.4f}, significant at p<0.05 in {len(sig)}/{len(rows)}")

    # Trend statistics on the chirp axis, recomputed on the repaired input.
    import itertools
    med = [B["settings"][f"chirps={c}"]["top1_median"] for c in CH]
    rho = stats.spearmanr(CH, med)
    null = [stats.spearmanr(CH, q).statistic for q in itertools.permutations(med)]
    out["chirp_trend_removed"] = {
        "n_chirps": CH, "median": med,
        "spearman_rho": float(rho.statistic),
        "spearman_exact_two_sided_p": float(np.mean(np.abs(null) >= abs(rho.statistic) - 1e-12)),
        "slope_of_medians_per_doubling": float(np.polyfit(np.log2(CH), med, 1)[0]),
        "monotone": bool(all(x <= y + 1e-9 for x, y in zip(med, med[1:])))}
    t = out["chirp_trend_removed"]
    print(f"\n  chirp sweep, leakage removed: {' -> '.join(f'{x:.4f}' for x in med)}")
    print(f"    Spearman {t['spearman_rho']:+.4f}, exact p {t['spearman_exact_two_sided_p']:.4f}, "
          f"slope {t['slope_of_medians_per_doubling']:+.4f} per doubling, "
          f"monotone {t['monotone']}")

    # Which settings clear the bound by bootstrap interval, under each front end.
    out["clears_bound"] = {}
    for tag, D in (("leakage_retained", A), ("leakage_removed", B)):
        g = np.random.default_rng(20261007)
        bad = []
        for k, v in D["settings"].items():
            x = np.asarray(v["top1_seeds"], float)
            bt = np.median(g.choice(x, (20000, x.size), True), axis=1)
            if np.percentile(bt, 2.5) <= bound:
                bad.append(k)
        out["clears_bound"][tag] = {"n_settings": len(D["settings"]),
                                    "n_clearing": len(D["settings"]) - len(bad),
                                    "not_clearing": bad}
        c = out["clears_bound"][tag]
        print(f"  {tag:<18} clears the bound by bootstrap interval in "
              f"{c['n_clearing']}/{c['n_settings']}; exceptions: {', '.join(bad) or 'none'}")

    print("\n" + "=" * 92)
    print("DO THE VERDICTS SURVIVE? Every comparison the paper states, re-tested on both inputs.")
    print("=" * 92)
    print(f"\n  {'comparison':<28}{'leakage retained':>24}{'leakage removed':>24}{'same?':>8}")
    verd = []
    for lo_k, hi_k in PAIRS + ISO:
        o = {}
        for tag, D in (("retained", A), ("removed", B)):
            if lo_k not in D["settings"] or hi_k not in D["settings"]:
                o[tag] = None; continue
            dl, l, h, pv = perm(D["settings"][hi_k]["top1_seeds"],
                                D["settings"][lo_k]["top1_seeds"])
            o[tag] = {"delta": float(dl), "ci": [float(l), float(h)], "p": float(pv),
                      "verdict": "rise" if l > 0 else ("fall" if h < 0 else "flat")}
        same = bool(o["retained"] and o["removed"]
                    and o["retained"]["verdict"] == o["removed"]["verdict"])
        f = lambda z: f"{z['delta']:+.4f} {z['verdict']:<4} p={z['p']:.3f}" if z else "n/a"
        print(f"  {lo_k + ' -> ' + hi_k:<28}{f(o['retained']):>24}{f(o['removed']):>24}"
              f"{('yes' if same else 'NO'):>8}")
        verd.append({"comparison": f"{lo_k} -> {hi_k}", "iso_payload": (lo_k, hi_k) in ISO,
                     **o, "verdict_unchanged": same})
    flipped = [v for v in verd if not v["verdict_unchanged"]]
    out["verdicts"] = verd
    out["n_verdicts"] = len(verd)
    out["n_verdicts_unchanged"] = len(verd) - len(flipped)
    print(f"\n  {out['n_verdicts_unchanged']}/{len(verd)} verdicts unchanged.")
    if flipped:
        print("  LEAKAGE-DEPENDENT, must be reported as such:")
        for v in flipped:
            print(f"    {v['comparison']}: retained {v['retained']['verdict']}, "
                  f"removed {v['removed']['verdict']}")
    # ---- the one non-learned predictor the repair changes: the azimuth peak ----
    # Section VI reports that the brightest cell carries little beam information. On the
    # leakage-dominated input that peak was not a scatterer at all, so the comparison was
    # between a lookup on an artifact and the bound. It has to be rescored on the repaired map.
    import csv, collections
    sys.path.insert(0, HERE)
    from audit_reference_predictors import lookup_table, topk_scores, TRAIN_FRAC, GAP_FRAC, NBEAM
    rowsc = list(csv.DictReader(open(os.path.join(
        HERE, "..", "datasets", "deepsense6g", "Scenario31", "scenario31_dev.csv"))))
    nn = len(rowsc)
    ntr, gap = int(TRAIN_FRAC * nn), int(GAP_FRAC * nn)
    tr, te = slice(0, ntr), slice(ntr + gap, nn)
    yb = np.array([int(float(r["unit1_beam"])) for r in rowsc]) - 1
    brg = np.load(os.path.join(RES, "sc31_bearing_labels.npz"))["bearing_deg"].astype(float)
    peak = {}
    print("\n" + "=" * 92)
    print("THE AZIMUTH-PEAK REFERENCE PREDICTOR, rescored on the repaired map")
    print("=" * 92)
    print(f"\n  {'front end':<34}{'top-1':>8}{'top-5':>8}{'cls':>6}{'maxshr':>9}"
          f"{'peak az vs bearing':>21}")
    for tag, cname, drop in (("leakage retained, bin 0 kept", "cache_sc31_ablation_max.npz", 0),
                             ("leakage retained, bin 0 dropped", "cache_sc31_ablation_max.npz", 1),
                             ("leakage removed", "cache_sc31_ablation_max_dcmean.npz", 0)):
        cp = os.path.join(RES, cname)
        if not os.path.exists(cp):
            continue
        ra = np.load(cp)["4|256|250"][:, 0]           # (n, range, azimuth)
        sub = ra[:, drop:, :]
        f_ = sub.reshape(nn, -1).argmax(1)
        _, az = np.unravel_index(f_, sub.shape[1:])
        tab = lookup_table(az[tr], yb[tr], 64, 5)
        t1, t5 = topk_scores(tab[az[te]], yb[te])
        cnt = collections.Counter(tab[az[te]][:, 0].tolist())
        pr = stats.pearsonr(az.astype(float), brg)
        peak[tag] = {"cache": cname, "first_range_bin_dropped": bool(drop),
                     "top1": t1, "top5": t5, "clears_bound": bool(t1 > bound),
                     "distinct_top1_classes": len(cnt),
                     "max_class_share": float(max(cnt.values()) / len(yb[te])),
                     "peak_azimuth_vs_bearing_pearson": float(pr.statistic),
                     "peak_azimuth_vs_bearing_p": float(pr.pvalue),
                     "distinct_peak_azimuths": int(len(set(az.tolist())))}
        v = peak[tag]
        print(f"  {tag:<34}{t1:>8.4f}{t5:>8.4f}{v['distinct_top1_classes']:>6}"
              f"{v['max_class_share']:>9.3f}{v['peak_azimuth_vs_bearing_pearson']:>+13.4f}"
              f" p={v['peak_azimuth_vs_bearing_p']:.1e}")
    out["azimuth_peak_predictor"] = peak
    print(f"\n  no-information bound {bound:.4f}; the peak lookup clears it in "
          f"{sum(v['clears_bound'] for v in peak.values())} of {len(peak)} front ends")

    p = os.path.join(RES, "sc31_dc_comparison.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
