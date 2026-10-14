#!/usr/bin/env python
"""Non-learned baselines that USE the radar, for the Sc.31/33 beam task.

WHY THIS EXISTS
---------------
An external review asked for an explicit differentiable forward operator as
a comparator. That request targeted a reconstruction architecture the new work does not
contain, but the concern behind it is live: a method evaluated only against a constant and
against learned baselines that share its weaknesses has not been tested. The manuscript already
reports a no-information bound and a position-only ceiling. Neither of them uses the radar, so
neither answers "does the learned readout beat classical processing on the same input?"

These do, and none of them is trained:

  peak-az        azimuth index of the brightest cell of the range-azimuth map, mapped to a beam
                 through a lookup fitted on the training split only
  peak-az-rng    the same using the (range, azimuth) index pair, a 2-D lookup
  persistence    the previous frame's true beam, the standard beam-tracking baseline. Frames are
                 near-duplicates under a contiguous temporal split, so this is the trivial
                 baseline most likely to embarrass a learned predictor, and it must be reported.
  bearing-oracle the GPS bearing through the SAME lookup, which separates "the lookup is weak"
                 from "the radar peak is a poor bearing estimate"

Every lookup is fitted on train and evaluated on test, with the identical split the learned
runs use. Nothing here is stochastic, so no seeds are needed.

Output: results/sc3X_stronger_baselines.json
"""
import argparse, collections, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TRAIN_FRAC, GAP_FRAC = 0.70, 0.02
NBEAM = 64


def fit_lookup(keys_tr, beams_tr, nkey):
    """key -> (modal beam, 5 most frequent beams). Train only."""
    top1 = np.zeros(nkey, int)
    top5 = np.zeros((nkey, 5), int)
    glob = collections.Counter(beams_tr.tolist()).most_common(5)
    gl1 = glob[0][0]
    gl5 = [b for b, _ in glob] + [gl1] * (5 - len(glob))
    for k in range(nkey):
        sel = beams_tr[keys_tr == k]
        if len(sel) == 0:
            top1[k], top5[k] = gl1, gl5
            continue
        c = collections.Counter(sel.tolist()).most_common(5)
        top1[k] = c[0][0]
        f = [b for b, _ in c]
        top5[k] = (f + [c[0][0]] * (5 - len(f)))[:5]
    return top1, top5


def score(pred1, pred5, y):
    return float((pred1 == y).mean()), float((pred5 == y[:, None]).any(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="31", choices=["31", "33"])
    a = ap.parse_args()
    sc = a.scenario
    z = np.load(os.path.join(RES, f"cache_sc{sc}_ablation_max.npz"))
    X = z["4|256|250"]                      # (n, 2, range, azimuth) at full resources
    beams = z["beams"].astype(int)
    lab = np.load(os.path.join(RES, f"sc{sc}_bearing_labels.npz"))
    brg = lab["bearing_deg"].astype(float)
    n = len(beams)
    ntr = int(TRAIN_FRAC * n); gap = int(GAP_FRAC * n)
    tr, te = slice(0, ntr), slice(ntr + gap, n)
    y = beams[te]
    bmb, cnt = collections.Counter(y.tolist()).most_common(1)[0]
    bound = cnt / len(y)
    G = json.load(open(os.path.join(RES, "geometry_ceiling.json")))[f"sc{sc}"]
    ceil = G["ceiling_top1"]
    print(f"Scenario {sc}: train {ntr}, test {len(y)}, bound {bound:.4f}, ceiling {ceil:.4f}")
    print("All lookups fitted on TRAIN only. Nothing here is trained or stochastic.\n")

    ra = X[:, 0]                                            # (n, range, azimuth)
    flat = ra.reshape(len(ra), -1).argmax(1)
    rng_idx, az_idx = np.unravel_index(flat, ra.shape[1:])

    out = {"scenario": f"Scenario{sc}", "bound_const": bound, "geometry_ceiling": ceil,
           "n_train": int(ntr), "n_test": int(len(y)), "baselines": {}}

    def report(name, p1, p5, note=""):
        t1, t5 = score(p1, p5, y)
        out["baselines"][name] = {"top1": t1, "top5": t5, "vs_bound": t1 - bound,
                                  "frac_of_ceiling": (t1 - bound) / (ceil - bound),
                                  "note": note}
        print(f"  {name:<16}{t1:>8.4f}{t5:>8.4f}{t1-bound:>+10.4f}"
              f"{100*(t1-bound)/(ceil-bound):>9.1f}%")

    print(f"  {'baseline':<16}{'top-1':>8}{'top-5':>8}{'vs bound':>10}{'of ceiling':>10}")
    # 1. azimuth peak -> lookup
    l1, l5 = fit_lookup(az_idx[tr], beams[tr], 64)
    report("peak-az", l1[az_idx[te]], l5[az_idx[te]],
           "azimuth index of the brightest range-azimuth cell, train-fitted lookup")
    # 2. (range, azimuth) peak -> 2-D lookup
    key = rng_idx * 64 + az_idx
    l1b, l5b = fit_lookup(key[tr], beams[tr], 64 * 64)
    report("peak-az-rng", l1b[key[te]], l5b[key[te]],
           "range and azimuth index of the brightest cell, train-fitted 2-D lookup")
    # 3. persistence: the previous frame's true beam (and its 4 nearest beam indices)
    prev = np.concatenate([[beams[0]], beams[:-1]])[te]
    p5 = np.stack([np.clip(prev + d, 0, NBEAM - 1) for d in (0, -1, 1, -2, 2)], 1)
    report("persistence", prev, p5,
           "previous frame's TRUE beam; top-5 is that beam and its 4 nearest indices")
    # 4. the same lookup driven by GPS bearing, to separate lookup quality from peak quality
    qb = np.quantile(brg[tr], np.linspace(0, 1, 65))
    bk = np.clip(np.digitize(brg, qb[1:-1]), 0, 63)
    l1c, l5c = fit_lookup(bk[tr], beams[tr], 64)
    report("bearing-oracle", l1c[bk[te]], l5c[bk[te]],
           "GPS bearing through the same 64-bin lookup; isolates the lookup's own ceiling")

    # is the radar peak even tracking the partner?
    r = stats.pearsonr(az_idx[te], brg[te]); rs = stats.spearmanr(az_idx[te], brg[te])
    out["peak_az_vs_gps_bearing"] = {"pearson": float(r.statistic), "pearson_p": float(r.pvalue),
                                     "spearman": float(rs.statistic)}
    print(f"\n  radar azimuth peak vs GPS bearing: Pearson {r.statistic:+.4f} "
          f"(p={r.pvalue:.1e}), Spearman {rs.statistic:+.4f}")
    print("  (a weak correlation means the brightest cell is usually not the partner, which is")
    print("   the anonymity problem, not a defect of the lookup)")

    p = os.path.join(RES, f"sc{sc}_stronger_baselines.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
