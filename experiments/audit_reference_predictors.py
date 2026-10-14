#!/usr/bin/env python
"""Reference predictors for the beam task, rebuilt after the 2026-10-06 audit.

Four defects in the previous reference set are repaired here.

1. PEAK BASELINE WAS DEGENERATE. The range-azimuth argmax fell in the first pooled range
   bin in 7007 of 7012 Scenario-31 frames, so the lookup emitted a single class for every
   test frame and scored exactly the train-modal prior. That is a zero-range/DC artifact,
   not a target. Excluding the first range bin the peak uses 58 distinct azimuths. We report
   both, and we report prediction diversity so a degenerate predictor cannot be mistaken for
   an informative one again.

2. NO FIXED-PRIOR COMPARATOR. Exhaustive search is the weakest possible baseline. The
   deployable no-sensing policy is the k most frequent beams in training, which costs the
   same k probes. It is swept over k here.

3. THE GEOMETRY REFERENCE WAS SELECTED ON TEST. Family and hyperparameter are now chosen on
   a validation split carved from training only, matching the network's inner split, and the
   selected configuration is evaluated once on test. The feature scaling is also fitted on
   training rows only.

4. PERSISTENCE CROSSED RECORDING DISCONTINUITIES. Eligibility now requires the previous row
   to share the sequence index, be the consecutive raw index, and lie within a positive
   interval of at most 200 ms.

Output: results/sc3X_reference_predictors.json
"""
import argparse, collections, csv, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DS = os.path.join(HERE, "..", "datasets", "deepsense6g")
TRAIN_FRAC, GAP_FRAC, VAL_FRAC = 0.70, 0.02, 0.10
NBEAM = 64


def topk_scores(pred_k, y):
    """pred_k: (n, k) candidate lists already ordered. Returns top-1 and top-k."""
    return float((pred_k[:, 0] == y).mean()), float((pred_k == y[:, None]).any(1).mean())


def lookup_table(keys, y, nkey, k):
    """key -> k most frequent training beams; global prior fills empty cells."""
    glob = [b for b, _ in collections.Counter(y.tolist()).most_common(k)]
    glob = (glob + [glob[0]] * k)[:k]
    tab = np.tile(np.array(glob), (nkey, 1))
    for b in range(nkey):
        sel = y[keys == b]
        if len(sel):
            f = [x for x, _ in collections.Counter(sel.tolist()).most_common(k)]
            tab[b] = (f + [f[0]] * k)[:k]
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="31", choices=["31", "33"])
    a = ap.parse_args()
    sc = a.scenario
    rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}", f"scenario{sc}_dev.csv"))))
    n = len(rows)
    ntr, gap = int(TRAIN_FRAC * n), int(GAP_FRAC * n)
    start = ntr + gap
    nv = ntr // 10                       # the network's inner validation split
    fit, val = slice(0, ntr - nv), slice(ntr - nv, ntr)
    tr, te = slice(0, ntr), slice(start, n)

    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    seq = np.array([int(r["seq_index"]) for r in rows])
    idx = np.array([int(r["index"]) for r in rows])
    t = []
    for r in rows:
        hms, frac = r["time_stamp"].split("-")
        h, m, s = (int(x) for x in hms.split(":"))
        t.append((h * 3600 + m * 60 + s) * 1e6 + int(frac))
    dt_ms = np.diff(np.array(t)) / 1000.0

    lab = np.load(os.path.join(RES, f"sc{sc}_bearing_labels.npz"))
    assert np.array_equal(lab["beam"], y), "label alignment"
    brg, rng_ = lab["bearing_deg"].astype(float), lab["range_m"].astype(float)
    cache = np.load(os.path.join(RES, f"cache_sc{sc}_ablation_max.npz"))
    ra = cache["4|256|250"][:, 0]        # (n, range, azimuth)
    yte = y[te]
    out = {"scenario": f"Scenario{sc}", "n": n, "n_fit": ntr - nv, "n_val": nv,
           "n_gap": gap, "n_test": len(yte),
           "train_test_sequence_overlap": sorted(map(int, set(seq[tr]) & set(seq[te]))),
           "predictors": {}}

    def report(name, pk, note, mask=None):
        yy = yte if mask is None else yte[mask]
        pp = pk if mask is None else pk[mask]
        t1, t5 = topk_scores(pp, yy)
        cnt = collections.Counter(pp[:, 0].tolist())
        out["predictors"][name] = {
            "top1": t1, "top5": t5, "n_eval": int(len(yy)),
            "distinct_top1_classes": len(cnt),
            "max_class_share": float(max(cnt.values()) / len(yy)),
            "note": note}
        print(f"  {name:<34}{t1:>8.4f}{t5:>8.4f}{len(cnt):>7}"
              f"{max(cnt.values())/len(yy):>9.3f}{len(yy):>8}")

    print(f"Scenario {sc}: fit {ntr-nv}, val {nv}, gap {gap}, test {len(yte)}")
    print(f"  {'predictor':<34}{'top-1':>8}{'top-5':>8}{'cls':>7}{'maxshr':>9}{'n':>8}")

    # ---- 1. fixed train prior, the deployable no-sensing policy ----
    prior = [b for b, _ in collections.Counter(y[tr].tolist()).most_common(NBEAM)]
    for k in (1, 3, 5, 8, 16):
        pk = np.tile(np.array((prior + prior)[:k]), (len(yte), 1))
        report(f"train-prior top-{k}", pk, "k most frequent training beams; uses no sensing")
    # ---- 2. hindsight best constant, a reference not a predictor ----
    ce = np.bincount(yte, minlength=NBEAM)
    report("hindsight-constant top-1", np.tile(np.argsort(-ce)[:5], (len(yte), 1)),
           "best constant chosen on TEST; a hindsight reference, not deployable")
    # ---- 3. radar peak lookup, as published and repaired ----
    for drop, tag in ((0, "radar-peak (as published)"), (1, "radar-peak (first range bin excluded)")):
        sub = ra[:, drop:, :]
        f = sub.reshape(n, -1).argmax(1)
        _, az = np.unravel_index(f, sub.shape[1:])
        tab = lookup_table(az[tr], y[tr], 64, 5)
        report(tag, tab[az[te]],
               "azimuth of the range-azimuth maximum through a train-fitted lookup"
               + ("" if drop else "; dominated by a zero-range artifact"))
        # Whether the brightest cell tracks the partner at all. This MUST be recorded per
        # variant: the as-published argmax is a leakage artifact whose azimuth is nearly
        # constant, so its correlation describes the artifact and not the repaired predictor
        # the paper reports. Quoting the first beside the second was a live defect.
        pr = stats.pearsonr(az[te].astype(float), brg[te])
        sp = stats.spearmanr(az[te].astype(float), brg[te])
        out["predictors"][tag]["peak_azimuth_vs_bearing"] = {
            "pearson": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman": float(sp.statistic),
            "distinct_peak_azimuths": int(len(set(az[te].tolist()))),
            "n_eval": int(len(az[te]))}
    # ---- 4. persistence, on temporally eligible rows only ----
    prev = np.concatenate([[y[0]], y[:-1]])
    elig = np.ones(n, bool)
    elig[1:] = (seq[1:] == seq[:-1]) & (idx[1:] == idx[:-1] + 1) & (dt_ms > 0) & (dt_ms <= 200)
    pk = np.stack([np.clip(prev + d, 0, NBEAM - 1) for d in (0, -1, 1, -2, 2)], 1)[te]
    report("persistence (all test rows)", pk,
           "previous row's TRUE beam and its 4 nearest indices; needs a prior alignment")
    report("persistence (temporally eligible)", pk,
           "same sequence, consecutive index, interval in (0,200] ms", mask=elig[te])
    out["persistence_ineligible_test_rows"] = int((~elig[te]).sum())

    # ---- 5. geometry reference, selected on validation only ----
    def knn(k, fit_s, ev_s):
        sc_ = np.array([rng_[fit_s].std(), brg[fit_s].std()])      # fitted on fit rows only
        A = np.c_[rng_[fit_s], brg[fit_s]] / sc_
        B = np.c_[rng_[ev_s], brg[ev_s]] / sc_
        D = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        o = np.argsort(D, 1)[:, :k]
        nb = y[fit_s][o]
        return np.stack([np.argsort(-np.bincount(r, minlength=NBEAM))[:5] for r in nb])

    def blookup(nb_, fit_s, ev_s):
        q = np.quantile(brg[fit_s], np.linspace(0, 1, nb_ + 1))
        kf = np.clip(np.digitize(brg[fit_s], q[1:-1]), 0, nb_ - 1)
        ke = np.clip(np.digitize(brg[ev_s], q[1:-1]), 0, nb_ - 1)
        return lookup_table(kf, y[fit_s], nb_, 5)[ke]

    cands = [("knn", k) for k in (1, 3, 5, 10, 20, 50)] + \
            [("bearing-lookup", b) for b in (16, 32, 64, 128)]
    vsel = []
    for fam, par in cands:
        pk = knn(par, fit, val) if fam == "knn" else blookup(par, fit, val)
        vsel.append({"family": fam, "param": par,
                     "val_top1": float((pk[:, 0] == y[val]).mean())})
    best = max(vsel, key=lambda r: r["val_top1"])
    pk = knn(best["param"], tr, te) if best["family"] == "knn" else blookup(best["param"], tr, te)
    report(f"GPS reference ({best['family']} {best['param']}, val-selected)", pk,
           "family and hyperparameter chosen on the inner validation split; "
           "scaling fitted on training rows; evaluated once on test")
    out["gps_reference_selection"] = {"candidates": vsel, "selected": best}
    p = os.path.join(RES, f"sc{sc}_reference_predictors.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
