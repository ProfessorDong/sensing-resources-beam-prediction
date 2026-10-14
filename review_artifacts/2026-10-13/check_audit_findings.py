#!/usr/bin/env python
"""Independent check of the 2026-10-13 audit's findings A, C, D, E and the k-grid claim.

Imports no project analysis module. Rebuilds the greedy sets, the swap search, the matched-k
comparison, the all-label accuracy and the expanded selection from the raw CSVs, the raw
power files and the saved rankings.
"""
import collections
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "results")
DS = os.path.join(ROOT, "datasets", "deepsense6g")
NBEAM, TSEC = 64, 5.0
KS_PUB = (1, 2, 3, 5, 8, 16)
out = {}


def load(sc):
    rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}",
                                                 f"scenario{sc}_dev.csv"))))
    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    P = np.array([np.loadtxt(os.path.join(DS, f"Scenario{sc}",
                                          r["unit1_pwr_60ghz"].lstrip("./"))) for r in rows])
    return y, P, np.isfinite(P).all(1)


def greedy(P, n=16):
    """Greedy by the LITERAL regret objective, no log identity."""
    out_, taken = [], set()
    for _ in range(n):
        cand = [b for b in range(NBEAM) if b not in taken]
        r = [float(np.mean(10.0 * np.log10(P.max(1) / P[:, out_ + [b]].max(1))))
             for b in cand]
        b = cand[int(np.argmin(r))]
        out_.append(b); taken.add(b)
    return out_


def mean_regret(P, S):
    return float(np.mean(10.0 * np.log10(P.max(1) / P[:, list(S)].max(1))))


def swap_search(P, S, rounds=6):
    """Single-beam replacement local search. Any improvement DISPROVES optimality."""
    S = list(S)
    best = mean_regret(P, S)
    for _ in range(rounds):
        improved = False
        for i in range(len(S)):
            for b in range(NBEAM):
                if b in S:
                    continue
                T = S[:i] + [b] + S[i + 1:]
                v = mean_regret(P, T)
                if v < best - 1e-12:
                    S, best, improved = T, v, True
        if not improved:
            break
    return S, best


for sc in (31, 33):
    y, P, fin = load(sc)
    z = np.load(os.path.join(RES, f"beam_shortlists_sc{sc}.npz"))
    n, ntr, nte = len(y), int(z["n_train"]), int(z["n_test"])
    nval = max(1, ntr // 10)
    val = np.arange(ntr - nval, ntr)
    fit = np.arange(0, ntr - nval)
    te = np.arange(n - nte, n)
    ff, fv, ft = fit[fin[fit]], val[fin[val]], te[fin[te]]
    r = {"n_test_all": int(len(te)), "n_test_finite": int(len(ft))}

    order = greedy(P[ff])
    hind = greedy(P[ft])
    r["order"] = order
    r["hindsight_order"] = hind

    # ---- A: is the test-fitted greedy set optimal? ----
    r["A_counterexamples"] = []
    for k in (3, 5, 8, 16):
        g = hind[:k]
        gv = mean_regret(P[ft], g)
        S, bv = swap_search(P[ft], g)
        r["A_counterexamples"].append({
            "k": k, "greedy_set": sorted(g), "greedy_mean_db": gv,
            "better_set": sorted(S), "better_mean_db": bv,
            "improvement_db": gv - bv, "greedy_is_optimal": bv >= gv - 1e-12})
        # and how the TRAIN-fitted set compares with a demonstrated feasible set
        if k == 16:
            tv = mean_regret(P[ft], order[:k])
            r["A_train_vs_better"] = {"train_fitted_db": tv, "better_db": bv,
                                      "train_minus_better_db": tv - bv,
                                      "train_minus_greedy_db": tv - gv}

    # ---- C: matched-k radar vs static, mean AND tail ----
    Q = json.load(open(os.path.join(RES, "beam_power_quality.json")))
    pts = {(p["setting"], p["k"]): p for p in Q["scenarios"][str(sc)]["frontier"]["points"]}
    r["C_matched"] = []
    for k in KS_PUB:
        rad = pts[("chirps=250", k)]["test"]
        L = 10.0 * np.log10(P[ft].max(1) / P[ft][:, order[:k]].max(1))
        r["C_matched"].append({
            "k": k, "radar_mean_db": rad["mean_regret_db"], "static_mean_db": float(L.mean()),
            "radar_wins_mean": rad["mean_regret_db"] < float(L.mean()),
            "radar_exc3": rad["exceed_3dB"], "static_exc3": float((L > 3.0).mean()),
            "static_wins_tail": float((L > 3.0).mean()) <= rad["exceed_3dB"],
            "static_n_exceed_3dB": int((L > 3.0).sum())})

    # ---- D: static accuracy on ALL test labels vs the finite subset ----
    bound = float(collections.Counter(y[te].tolist()).most_common(1)[0][1]) / len(te)
    for k, tag in ((1, "top1"), (5, "top5")):
        S = order[:k]
        r[f"D_static_{tag}_all"] = float(np.isin(y[te], S).mean())
        r[f"D_static_{tag}_finite"] = float(np.isin(y[ft], S).mean())
    r["D_bound_all"] = bound
    r["D_bound_beam"] = int(collections.Counter(y[te].tolist()).most_common(1)[0][0])
    r["D_static_top1_all_exceeds_bound"] = r["D_static_top1_all"] > bound + 1e-12
    r["D_static_top1_finite_exceeds_bound"] = r["D_static_top1_finite"] > bound + 1e-12

    # ---- E: the two frequency priors ----
    prior_fit = [b for b, _ in collections.Counter(y[ff].tolist()).most_common(NBEAM)]
    prior_full = [b for b, _ in collections.Counter(y[np.arange(ntr)].tolist()).most_common(NBEAM)]
    r["E_prior_differs"] = prior_fit[:16] != prior_full[:16]
    r["E_prior_k16_finite_rows"] = {
        "fitted_on_optimization_rows": mean_regret(P[ft], prior_fit[:16]),
        "fitted_on_full_training_region": mean_regret(P[ft], prior_full[:16])}

    # ---- section 4: every integer k from 1 to 16 ----
    tgt = Q["scenarios"][str(sc)]["frontier"]["selection"]["dedicated"]["validation_target_db"]
    r["target_db"] = float(tgt)
    for grid_name, KS in (("published", KS_PUB), ("full_1_to_16", tuple(range(1, 17)))):
        cand = []
        for k in KS:
            cand.append(("power-aware static", k, TSEC * k,
                         mean_regret(P[fv], order[:k]), mean_regret(P[ft], order[:k]),
                         float((10.0 * np.log10(P[ft].max(1)
                                / P[ft][:, order[:k]].max(1)) > 3.0).mean())))
            cand.append(("fixed training prior", k, TSEC * k,
                         mean_regret(P[fv], prior_fit[:k]), mean_regret(P[ft], prior_fit[:k]),
                         float((10.0 * np.log10(P[ft].max(1)
                                / P[ft][:, prior_fit[:k]].max(1)) > 3.0).mean())))
        for (s_, k), p in pts.items():
            if k in KS:
                cand.append((f"radar, {s_}", k, p["cost_dedicated_us"],
                             p["val"]["mean_regret_db"], p["test"]["mean_regret_db"],
                             p["test"]["exceed_3dB"]))
        ok = [c for c in cand if c[3] <= tgt + 1e-12]
        mc = min(c[2] for c in ok)
        tied = sorted([c for c in ok if abs(c[2] - mc) < 1e-9], key=lambda c: (c[3], c[1], c[0]))
        b = tied[0]
        r[f"sel_{grid_name}"] = {"n_candidates": len(cand), "policy": b[0], "k": int(b[1]),
                                 "cost_us": float(b[2]), "val_db": b[3], "test_db": b[4],
                                 "test_exc3": b[5]}
    out[str(sc)] = r

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "audit_findings_check.json"), "w"), indent=1)

print("=" * 78)
for sc in ("31", "33"):
    r = out[sc]
    print(f"\n--- Scenario {sc} ---")
    print("A  greedy-on-test optimality:")
    for c in r["A_counterexamples"]:
        print(f"    k={c['k']:2d}  greedy {c['greedy_mean_db']:.6f} -> swap "
              f"{c['better_mean_db']:.6f}  improvement {c['improvement_db']:.6f} dB"
              f"   optimal={c['greedy_is_optimal']}")
    if "A_train_vs_better" in r:
        a = r["A_train_vs_better"]
        print(f"    k=16 train-fitted {a['train_fitted_db']:.6f}: "
              f"{a['train_minus_greedy_db']:+.6f} vs greedy, "
              f"{a['train_minus_better_db']:+.6f} vs a demonstrated better set")
    print("C  matched k (radar full frame vs static):")
    for m in r["C_matched"]:
        print(f"    k={m['k']:2d}  mean {m['radar_mean_db']:.4f} / {m['static_mean_db']:.4f}"
              f"  radar wins mean={m['radar_wins_mean']!s:5}  "
              f"exc3 {m['radar_exc3']*100:.3f}% / {m['static_exc3']*100:.3f}%"
              f"  static n>3dB={m['static_n_exceed_3dB']}")
    print(f"D  static top1 all-labels {r['D_static_top1_all']:.7f}  "
          f"finite {r['D_static_top1_finite']:.7f}  bound {r['D_bound_all']:.7f}")
    print(f"   static top5 all-labels {r['D_static_top5_all']:.7f}  "
          f"finite {r['D_static_top5_finite']:.7f}")
    print(f"   exceeds bound on all labels: {r['D_static_top1_all_exceeds_bound']}  "
          f"on finite rows: {r['D_static_top1_finite_exceeds_bound']}")
    print(f"E  priors differ in first 16: {r['E_prior_differs']}  "
          f"k=16 regret opt-rows {r['E_prior_k16_finite_rows']['fitted_on_optimization_rows']:.7f}"
          f" vs full-region {r['E_prior_k16_finite_rows']['fitted_on_full_training_region']:.7f}")
    for g in ("published", "full_1_to_16"):
        s = r[f"sel_{g}"]
        print(f"   selection[{g:13}] n={s['n_candidates']:3d}  {s['policy']} k={s['k']}"
              f"  cost {s['cost_us']} us  val {s['val_db']:.6f}  test {s['test_db']:.6f}")
print("=" * 78)
