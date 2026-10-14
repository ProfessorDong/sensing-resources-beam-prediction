#!/usr/bin/env python
"""Independent re-derivation of the audit's power-aware static shortlist claim.

Written from the raw CSVs and power files. It imports no project analysis module and reads
nothing from the auditor's own script, so agreement between the two is a reproduction and
not a shared bug. Read-only.
"""
import collections, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "datasets", "deepsense6g")
RES = os.path.join(ROOT, "experiments", "results")
NBEAM, KS = 64, (1, 2, 3, 5, 8, 16)
TSEC = 5.0
FS_MSPS = 100.0


def load(sc):
    rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}",
                                                 f"scenario{sc}_dev.csv"))))
    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    P = np.array([np.loadtxt(os.path.join(DS, f"Scenario{sc}",
                                          r["unit1_pwr_60ghz"].lstrip("./"))) for r in rows])
    seq = np.array([int(r["seq_index"]) for r in rows])
    return y, P, seq


def greedy(P):
    """Ordered beam set minimising mean dB regret on these rows, added one beam at a time."""
    L = np.log10(P)                      # maximising mean log10(got) minimises mean regret
    best = np.full(len(P), -np.inf)
    order, chosen = [], np.zeros(NBEAM, bool)
    for _ in range(max(KS)):
        gain = np.array([np.maximum(best, L[:, b]).mean() if not chosen[b] else -np.inf
                         for b in range(NBEAM)])
        b = int(np.argmax(gain))         # np.argmax takes the smallest index on a tie
        order.append(b); chosen[b] = True
        best = np.maximum(best, L[:, b])
    return order


def score(P, order, k):
    got = P[:, order[:k]].max(1)
    L = 10.0 * np.log10(P.max(1) / got)
    return {"k": k, "mean_regret_db": float(L.mean()),
            "exceed_3dB": float((L > 3.0).mean()), "exceed_1dB": float((L > 1.0).mean()),
            "inclusion": float((L <= 1e-12).mean()), "n": int(len(L))}


out = {}
for sc in (31, 33):
    y, P, seq = load(sc)
    z = np.load(os.path.join(RES, f"beam_shortlists_sc{sc}.npz"))
    n, ntr = len(y), int(z["n_train"])
    nval = max(1, ntr // 10)
    fit = np.arange(0, ntr - nval)
    val = np.array(z["val_row_index"])
    te = np.arange(n - int(z["n_test"]), n)
    assert np.array_equal(val, np.arange(ntr - nval, ntr)), "validation block is not the tail"
    fin = np.isfinite(P).all(1)
    ff, fv, ft = fit[fin[fit]], val[fin[val]], te[fin[te]]
    order = greedy(P[ff])
    # a second, independently written objective: direct mean-dB regret, no log identity
    chk, cs = [], []
    for _ in range(5):
        cand = [b for b in range(NBEAM) if b not in cs]
        r = [np.mean(10 * np.log10(P[ff].max(1) / P[ff][:, cs + [b]].max(1))) for b in cand]
        cs.append(cand[int(np.argmin(r))]); chk.append(cs[-1])
    assert chk == order[:5], (chk, order[:5])
    rec = {"n_fit_finite": len(ff), "n_val_finite": len(fv), "n_test_finite": len(ft),
           "order16": order, "test": {}, "val": {}}
    for k in KS:
        rec["test"][str(k)] = score(P[ft], order, k)
        rec["val"][str(k)] = score(P[fv], order, k)
    # the frequency prior over the same fit rows, for a like-for-like static comparison
    pri = [b for b, _ in collections.Counter(y[ff].tolist()).most_common(NBEAM)]
    rec["prior_order16"] = pri[:16]
    rec["prior_test"] = {str(k): score(P[ft], pri, k) for k in KS}
    rec["prior_val"] = {str(k): score(P[fv], pri, k) for k in KS}
    out[str(sc)] = rec
    print(f"Sc{sc}: fit {len(ff)}  val {len(fv)}  test {len(ft)}")
    print(f"  greedy order (first 16): {order}")
    for k in KS:
        t, v = rec["test"][str(k)], rec["val"][str(k)]
        p = rec["prior_test"][str(k)]
        print(f"   k={k:2d}  static test regret {t['mean_regret_db']:.5f} dB  "
              f">3dB {100*t['exceed_3dB']:.4f}%  incl {t['inclusion']:.4f}   "
              f"| val {v['mean_regret_db']:.5f} | prior test {p['mean_regret_db']:.5f}")
json.dump(out, open(os.path.join(HERE, "static_policy.json"), "w"), indent=1)
print("\nwrote static_policy.json")
