#!/usr/bin/env python
"""The selection over EVERY shortlist length 1..16 for radar AND static, not just the six.

The audit proposes expanding the static grid. Expanding only the static side would hand it
sixteen lengths against the radar's six, so this expands both from the saved rankings, which
already carry all sixteen positions. Radar quality is the median over seeds of the per-seed
mean, which is the statistic the paper's own selector uses.
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
KS_ALL = tuple(range(1, 17))
out = {}


def load(sc):
    rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}",
                                                 f"scenario{sc}_dev.csv"))))
    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    P = np.array([np.loadtxt(os.path.join(DS, f"Scenario{sc}",
                                          r["unit1_pwr_60ghz"].lstrip("./"))) for r in rows])
    return y, P, np.isfinite(P).all(1)


def greedy(P, n=16):
    o, taken = [], set()
    for _ in range(n):
        c = [b for b in range(NBEAM) if b not in taken]
        r = [float(np.mean(10.0 * np.log10(P.max(1) / P[:, o + [b]].max(1)))) for b in c]
        b = c[int(np.argmin(r))]
        o.append(b); taken.add(b)
    return o


def mr(P, S):
    return float(np.mean(10.0 * np.log10(P.max(1) / P[:, list(S)].max(1))))


def exc(P, S, d=3.0):
    return float((10.0 * np.log10(P.max(1) / P[:, list(S)].max(1)) > d).mean())


def radar_seedmedian(P_rows, sl, mask, k):
    """median over seeds of the per-seed mean regret, on the finite rows of this block"""
    per = [float(np.mean(10.0 * np.log10(
        P_rows.max(1) / np.take_along_axis(
            P_rows, sl[s][mask][:, :k].astype(np.int64), 1).max(1))))
        for s in range(sl.shape[0])]
    return float(np.median(per))


def radar_exc_seedmedian(P_rows, sl, mask, k, d=3.0):
    per = [float((10.0 * np.log10(
        P_rows.max(1) / np.take_along_axis(
            P_rows, sl[s][mask][:, :k].astype(np.int64), 1).max(1)) > d).mean())
        for s in range(sl.shape[0])]
    return float(np.median(per))


for sc in (31, 33):
    y, P, fin = load(sc)
    z = np.load(os.path.join(RES, f"beam_shortlists_sc{sc}.npz"))
    n, ntr, nte = len(y), int(z["n_train"]), int(z["n_test"])
    nval = max(1, ntr // 10)
    val = np.arange(ntr - nval, ntr); fit = np.arange(0, ntr - nval); te = np.arange(n - nte, n)
    ff, fv, ft = fit[fin[fit]], val[fin[val]], te[fin[te]]
    vmask, tmask = fin[val], fin[te]
    Q = json.load(open(os.path.join(RES, "beam_power_quality.json")))
    F = Q["scenarios"][str(sc)]["frontier"]
    pts = {(p["setting"], p["k"]): p for p in F["points"]}
    tgt = float(F["selection"]["dedicated"]["validation_target_db"])
    order = greedy(P[ff])
    prior = [b for b, _ in collections.Counter(y[ff].tolist()).most_common(NBEAM)]
    # the fixed frame charge of each setting, recovered from a stored point
    frame = {}
    for (s_, k), p in pts.items():
        frame[s_] = p["cost_dedicated_us"] - TSEC * k
    r = {"target_db": tgt, "grids": {}}
    for gname, KS in (("published_6", KS_PUB), ("all_16", KS_ALL)):
        cand = []
        for k in KS:
            cand.append(dict(policy="power-aware static", k=k, cost_ded=TSEC * k,
                             cost_reu=TSEC * k, val=mr(P[fv], order[:k]),
                             test=mr(P[ft], order[:k]), exc=exc(P[ft], order[:k])))
            cand.append(dict(policy="fixed training prior", k=k, cost_ded=TSEC * k,
                             cost_reu=TSEC * k, val=mr(P[fv], prior[:k]),
                             test=mr(P[ft], prior[:k]), exc=exc(P[ft], prior[:k])))
        for s_ in sorted(frame):
            sl_t = z[f"spatial|{s_}|shortlist"]
            sl_v = z[f"spatial|{s_}|val_shortlist"]
            for k in KS:
                cand.append(dict(policy=f"radar, {s_}", k=k,
                                 cost_ded=frame[s_] + TSEC * k, cost_reu=TSEC * k,
                                 val=radar_seedmedian(P[fv], sl_v, vmask, k),
                                 test=radar_seedmedian(P[ft], sl_t, tmask, k),
                                 exc=radar_exc_seedmedian(P[ft], sl_t, tmask, k)))
        g = {"n_candidates": len(cand)}
        for regime, ck in (("dedicated", "cost_ded"), ("reuse", "cost_reu")):
            ok = [c for c in cand if c["val"] <= tgt + 1e-12]
            mc = min(c[ck] for c in ok)
            tied = sorted([c for c in ok if abs(c[ck] - mc) < 1e-9],
                          key=lambda c: (c["val"], c["k"], c["policy"]))
            b = tied[0]
            g[regime] = {"n_meeting": len(ok), "min_cost_us": mc, "n_tied": len(tied),
                         "selected": b["policy"], "k": b["k"], "val": b["val"],
                         "test": b["test"], "exc3": b["exc"],
                         "uses_radar": b["policy"].startswith("radar"),
                         "tied_names": [f"{c['policy']} k={c['k']}" for c in tied[:4]]}
        r["grids"][gname] = g
    # cross-check the published point reproduces under this independent radar statistic
    dsel = F["selection"]["dedicated"]["selected"]
    sl_t = z[f"spatial|{dsel['setting']}|shortlist"]
    r["published_radar_point_test_db"] = radar_seedmedian(P[ft], sl_t, tmask, dsel["k"])
    r["published_radar_point_stored_db"] = pts[(dsel["setting"], dsel["k"])]["test"]["mean_regret_db"]
    out[str(sc)] = r

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "full_grid_check.json"), "w"), indent=1)
for sc in ("31", "33"):
    r = out[sc]
    print(f"\n=== Scenario {sc}   validation target {r['target_db']:.6f} dB ===")
    print(f"  reproduction of the published radar point: "
          f"{r['published_radar_point_test_db']:.6f} vs stored "
          f"{r['published_radar_point_stored_db']:.6f}")
    for gname, g in r["grids"].items():
        print(f"  grid {gname:12s} ({g['n_candidates']:3d} candidates)")
        for regime in ("dedicated", "reuse"):
            d = g[regime]
            print(f"     {regime:10s} {d['n_meeting']:3d} meet | cheapest {d['min_cost_us']:7.2f} us"
                  f" | {d['selected']} k={d['k']} | val {d['val']:.6f} test {d['test']:.6f}"
                  f" exc3 {d['exc3']*100:.3f}% | radar={d['uses_radar']}")
