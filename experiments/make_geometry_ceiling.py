#!/usr/bin/env python
"""The geometry ceiling: the BEST position-only predictor we can build, not merely one.

A ceiling that is not maximized over the predictors available flatters the method it is meant
to bound. Ours did: a k-nearest-neighbour predictor on (range, bearing) gave 0.4534 on
Scenario 33, but a 64-bin quantile lookup on bearing alone reaches 0.4892, so the reported
"fraction of available information recovered" was too generous. This sweeps both families and
takes the maximum, and records which one won.

Both families use only unit 2's GPS position, through the same label and the same split as
every learned run, so the ceiling is a CONTROL: position-based beam selection is established
prior art (Va et al., IEEE TVT 2018).

Output: results/geometry_ceiling.json
"""
import collections, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TRAIN_FRAC, GAP_FRAC = 0.70, 0.02


def knn(rng_, brg, beam, tr, te, k):
    sc = np.array([rng_.std(), brg.std()])
    A = np.c_[rng_[tr], brg[tr]] / sc
    B = np.c_[rng_[te], brg[te]] / sc
    D = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    order = np.argsort(D, axis=1)[:, :k]
    nb = beam[tr][order]
    t1 = t5 = 0
    for i in range(len(nb)):
        c = np.bincount(nb[i], minlength=64)
        t1 += c.argmax() == beam[te][i]
        t5 += beam[te][i] in np.argsort(-c)[:5]
    return t1 / len(nb), t5 / len(nb)


def lookup(vals, beam, tr, te, nbin):
    q = np.quantile(vals[tr], np.linspace(0, 1, nbin + 1))
    key = np.clip(np.digitize(vals, q[1:-1]), 0, nbin - 1)
    glob = collections.Counter(beam[tr].tolist()).most_common(5)
    g1, g5 = glob[0][0], [b for b, _ in glob]
    p1 = np.empty(len(te_idx := np.arange(len(vals))[te]), int)
    p5 = np.empty((len(te_idx), 5), int)
    tab1, tab5 = {}, {}
    for b in range(nbin):
        sel = beam[tr][key[tr] == b]
        if len(sel) == 0:
            tab1[b], tab5[b] = g1, (g5 + [g1] * 5)[:5]
        else:
            c = collections.Counter(sel.tolist()).most_common(5)
            f = [x for x, _ in c]
            tab1[b], tab5[b] = c[0][0], (f + [c[0][0]] * 5)[:5]
    for i, b in enumerate(key[te]):
        p1[i], p5[i] = tab1[b], tab5[b]
    yt = beam[te]
    return float((p1 == yt).mean()), float((p5 == yt[:, None]).any(1).mean())


out = {}
for tag, best_radar in (("sc31", 0.2357), ("sc33", 0.1828)):
    z = np.load(os.path.join(RES, f"{tag}_bearing_labels.npz"))
    brg = z["bearing_deg"].astype(float); rng_ = z["range_m"].astype(float)
    beam = z["beam"].astype(int); n = len(beam)
    ntr = int(TRAIN_FRAC * n); gap = int(GAP_FRAC * n)
    tr, te = slice(0, ntr), slice(ntr + gap, n)
    bmb, c = collections.Counter(beam[te].tolist()).most_common(1)[0]
    bound = c / len(beam[te])
    cands = []
    for k in (1, 3, 5, 10, 20, 50):
        t1, t5 = knn(rng_, brg, beam, tr, te, k)
        cands.append({"family": "knn", "param": k, "top1": t1, "top5": t5})
    for nb in (16, 32, 64, 128):
        t1, t5 = lookup(brg, beam, tr, te, nb)
        cands.append({"family": "bearing-lookup", "param": nb, "top1": t1, "top5": t5})
    best = max(cands, key=lambda r: r["top1"])
    frac = (best_radar - bound) / (best["top1"] - bound)
    out[tag] = {"bound": bound, "candidates": cands,
                "best_family": best["family"], "best_param": best["param"],
                "ceiling_top1": best["top1"], "ceiling_top5": best["top5"],
                "ceiling_gain": best["top1"] - bound,
                "best_radar_median": best_radar, "radar_frac_of_ceiling": frac,
                # kept for backward compatibility with existing checks
                "best_k": best["param"] if best["family"] == "knn" else None,
                "knn": [c for c in cands if c["family"] == "knn"]}
    print(f"=== {tag.upper()}  bound {bound:.4f} ===")
    for r in cands:
        mark = "  <-- ceiling" if r is best else ""
        print(f"  {r['family']:<16}{str(r['param']):>5}{r['top1']:>9.4f}{r['top5']:>9.4f}{mark}")
    print(f"  radar {best_radar:.4f} recovers {100*frac:.1f}% of the ceiling's gain\n")
json.dump(out, open(os.path.join(RES, "geometry_ceiling.json"), "w"), indent=2)
print("Saved: results/geometry_ceiling.json")
