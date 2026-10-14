#!/usr/bin/env python
"""What a wrong beam costs, what a stale beam costs, and how much of that is seed noise.

Top-$k$ inclusion is the only beam metric in the resource sweep, and for a 64-beam codebook
formed by a 16-element array it is a blunt one: adjacent beams overlap, so missing the best
index by one is not the failure that missing it by thirty is. The recording carries the
measured received power of all 64 beams for every frame (`unit1_pwr_60ghz`, whose argmax is
the published label), so the loss from probing a shortlist rather than the best beam is
measurable. Used for EVALUATION ONLY: no predictor here sees a power vector.

For a frame $i$ and an ordered shortlist $S_k(i)$,

    L_k(i) = 10 log10 [ P_i(best) / max_{b in S_k(i)} P_i(b) ]   dB, >= 0,

the power the link gives up by probing only $S_k(i)$. Outage is the fraction of frames with
$L_k$ above a declared margin; we report 1 dB and 3 dB, the second being the same half-power
criterion the beam-coherence calculation uses. Powers are in the recording's own linear
units, so a ratio in decibels is meaningful and an absolute spectral efficiency is not. We do
not convert to rate.

FOUR THINGS THIS ANSWERS THAT THE SAVED SWEEP CANNOT
----------------------------------------------------
  1  power regret and outage against every comparator, at matched $k$
  2  the same comparison on the rows persistence is eligible for, so that the one comparator
     that beats the radar is not beating it on a different denominator
  3  a seed interval on the expected alignment overhead, which is linear in the top-5
     inclusion rate and had only a mean saved
  4  what a decision costs when it is used one, two or more frames after the measurement,
     binned by the elapsed time actually recorded, which is the measurement the ideal
     line-of-sight pointing time cannot supply

Input: results/beam_shortlists_sc{31,33}.npz from exp_beam_shortlists.py.
Output: results/beam_power_quality.json
"""
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DS = os.path.join(HERE, "..", "datasets", "deepsense6g")
sys.path.insert(0, HERE)

NBEAM, KS = 64, (1, 2, 3, 5, 8, 16)
MARGINS = (1.0, 3.0)
TSEC_US, KSHORT = 5.0, 5
NADC_RATE_MSPS = 100.0        # the faster of the two rates Table III prices, 256/2.56 us
SETTING_CFG = {f"chirps={c}": {"n_rx": 4, "n_samples": 256, "n_chirps": c}
               for c in (8, 16, 32, 64, 125, 250)}
SETTING_CFG.update({f"samples={v}": {"n_rx": 4, "n_samples": v, "n_chirps": 250}
                    for v in (64, 128)})
SETTING_CFG.update({f"rx={v}": {"n_rx": v, "n_samples": 256, "n_chirps": 250}
                    for v in (1, 2)})
MAX_GAP_MS = 200.0
LAGS = (1, 2, 3, 5)


def load_rows(sc):
    rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}",
                                                 f"scenario{sc}_dev.csv"))))
    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    seq = np.array([int(r["seq_index"]) for r in rows])
    idx = np.array([int(r["index"]) for r in rows])
    t = []
    for r in rows:
        hms, frac = r["time_stamp"].split("-")
        h, m, s = (int(x) for x in hms.split(":"))
        t.append((h * 3600 + m * 60 + s) * 1e6 + int(frac))
    dt_ms = np.diff(np.array(t, float)) / 1000.0
    P = np.array([np.loadtxt(os.path.join(DS, f"Scenario{sc}",
                                          r["unit1_pwr_60ghz"].lstrip("./"))) for r in rows])
    assert P.shape == (len(rows), NBEAM), P.shape
    # The published label IS the argmax of this vector, which is what licenses using it as the
    # evaluation reference rather than as a second, unvalidated measurement.
    ok = np.isfinite(P).all(1)
    assert np.array_equal(np.nanargmax(P[ok], 1), y[ok]), "power argmax is not the label"
    return y, seq, idx, dt_ms, P, ok


def regret_db(P, shortlist):
    """10 log10 (best power / best power inside the shortlist), in dB, per frame."""
    best = P.max(1)
    got = np.take_along_axis(P, shortlist.astype(np.int64), 1).max(1)
    return 10.0 * np.log10(best / got)


def summarize(P, shortlist, mask):
    """Inclusion, power regret and outage for one ordered shortlist, over `mask` rows."""
    k = shortlist.shape[1]
    out = {"k": int(k), "n_eval": int(mask.sum())}
    L = regret_db(P[mask], shortlist[mask][:, :k])
    out["top_k_inclusion"] = float((L <= 1e-12).mean())
    out["regret_mean_db"] = float(L.mean())
    out["regret_median_db"] = float(np.median(L))
    out["regret_p90_db"] = float(np.percentile(L, 90))
    out["regret_max_db"] = float(L.max())
    for mg in MARGINS:
        out[f"outage_{mg:g}dB"] = float((L > mg).mean())
    return out


def lookup_table(keys, y, nkey, k):
    glob = [b for b, _ in collections.Counter(y.tolist()).most_common(k)]
    glob = (glob + [glob[0]] * k)[:k]
    tab = np.tile(np.array(glob), (nkey, 1))
    for b in range(nkey):
        sel = y[keys == b]
        if len(sel):
            f = [x for x, _ in collections.Counter(sel.tolist()).most_common(k)]
            tab[b] = (f + [f[0]] * k)[:k]
    return tab


def comparators(sc, y, seq, idx, dt_ms, n, ntr, start, kmax):
    """Ordered shortlists of length kmax for every non-learned comparator, on test rows."""
    te = slice(start, n)
    nte = n - start
    out = {}
    prior = [b for b, _ in collections.Counter(y[:ntr].tolist()).most_common(NBEAM)]
    out["fixed training prior"] = np.tile(np.array(prior[:kmax]), (nte, 1))
    ce = np.bincount(y[te], minlength=NBEAM)
    out["hindsight constant"] = np.tile(np.argsort(-ce)[:kmax], (nte, 1))
    # persistence: the previous true beam, then its codebook neighbours outward
    offs = [0]
    for d in range(1, kmax):
        offs += [-d, d]
    offs = offs[:kmax]
    prev = np.concatenate([[y[0]], y[:-1]])
    out["persistence"] = np.stack([np.clip(prev + d, 0, NBEAM - 1) for d in offs], 1)[te]
    # position reference: the validation-selected family, rebuilt at this kmax
    sel = json.load(open(os.path.join(RES, f"sc{sc}_reference_predictors.json")))
    best = sel["gps_reference_selection"]["selected"]
    lab = np.load(os.path.join(RES, f"sc{sc}_bearing_labels.npz"))
    brg, rng_ = lab["bearing_deg"].astype(float), lab["range_m"].astype(float)
    tr = slice(0, ntr)
    if best["family"] == "knn":
        scl = np.array([rng_[tr].std(), brg[tr].std()])
        A = np.c_[rng_[tr], brg[tr]] / scl
        Bq = np.c_[rng_[te], brg[te]] / scl
        D = ((Bq[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        nb = y[tr][np.argsort(D, 1)[:, :best["param"]]]
        out["position reference"] = np.stack(
            [np.argsort(-np.bincount(r, minlength=NBEAM))[:kmax] for r in nb])
    else:
        nbq = best["param"]
        q = np.quantile(brg[tr], np.linspace(0, 1, nbq + 1))
        kf = np.clip(np.digitize(brg[tr], q[1:-1]), 0, nbq - 1)
        ke = np.clip(np.digitize(brg[te], q[1:-1]), 0, nbq - 1)
        out["position reference"] = lookup_table(kf, y[tr], nbq, kmax)[ke]
    out["_selection"] = best
    return out


def eligible_chain(seq, idx, dt_ms, n, start, lag):
    """Rows i in test whose frame i+lag is reachable by `lag` consecutive logged steps.

    Returns a boolean mask over test rows and the elapsed time in ms for each.
    """
    nte = n - start
    ok = np.zeros(nte, bool)
    el = np.full(nte, np.nan)
    for j in range(nte):
        i = start + j
        if i + lag >= n:
            continue
        tot = 0.0
        good = True
        for q in range(lag):
            a = i + q
            if not (seq[a + 1] == seq[a] and idx[a + 1] == idx[a] + 1
                    and 0 < dt_ms[a] <= MAX_GAP_MS):
                good = False
                break
            tot += dt_ms[a]
        ok[j], el[j] = good, tot
    return ok, el


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="31,33")
    which = ap.parse_args().scenarios.split(",")
    out = {"metric": "L_k = 10 log10(best beam power / best power in the shortlist), dB",
           "power_units": "the recording's own linear units; ratios only, never a rate",
           "margins_db": list(MARGINS), "k_values": list(KS),
           "T_sec_us": TSEC_US, "k_shortlist": KSHORT, "n_beam": NBEAM,
           "scenarios": {}}
    for sc in which:
        z = np.load(os.path.join(RES, f"beam_shortlists_sc{sc}.npz"))
        y, seq, idx, dt_ms, P, finite = load_rows(sc)
        n = len(y)
        ntr, start = int(z["n_train"]), int(z["n_train"]) + int(0.02 * n)
        assert start == n - int(z["n_test"]), (start, n, int(z["n_test"]))
        assert np.array_equal(z["beams_all"], y), "shortlist file disagrees with the CSV"
        te = slice(start, n)
        Pt, yt = P[te], y[te]
        fin = finite[te]
        # persistence is defined only on temporally eligible rows; the radar predictor is
        # defined on all of them. Scoring the two on different denominators and comparing
        # them is what this file exists to stop.
        elig = np.zeros(n, bool)
        elig[1:] = ((seq[1:] == seq[:-1]) & (idx[1:] == idx[:-1] + 1)
                    & (dt_ms > 0) & (dt_ms <= MAX_GAP_MS))
        el_te = elig[te]
        rec = {"n_test": int(len(yt)),
               "n_test_nonfinite_power": int((~fin).sum()),
               "n_nonfinite_power_all_rows": int((~finite).sum()),
               "n_eval_all": int(fin.sum()),
               "n_eval_eligible": int((fin & el_te).sum()),
               "test_sequences": sorted(map(int, set(seq[te].tolist()))),
               "n_test_sequences": len(set(seq[te].tolist())),
               "comparators": {}, "learned": {}, "delay": {}, "per_sequence": {}}
        print(f"\n=== Scenario {sc}: test {len(yt)}, {int((~fin).sum())} rows dropped for a "
              f"non-finite power entry, {rec['n_test_sequences']} test sequences")

        cmp_ = comparators(sc, y, seq, idx, dt_ms, n, ntr, start, max(KS))
        rec["position_reference_selection"] = cmp_.pop("_selection")
        for name, sl in cmp_.items():
            rec["comparators"][name] = {
                "all_rows": {str(k): summarize(Pt, sl[:, :k], fin) for k in KS},
                "eligible_rows": {str(k): summarize(Pt, sl[:, :k], fin & el_te) for k in KS}}
            a = rec["comparators"][name]["all_rows"]["5"]
            print(f"  {name:<24}k=5  incl {a['top_k_inclusion']:.4f}  "
                  f"regret mean {a['regret_mean_db']:.3f} dB  "
                  f"outage@3dB {a['outage_3dB']:.4f}")

        for key in sorted(k for k in z.files if k.endswith("|shortlist")):
            arch, label, _ = key.split("|")
            S = z[key]
            nseed = S.shape[0]
            per_k = {}
            for k in KS:
                seeds = [summarize(Pt, S[s][:, :k], fin) for s in range(nseed)]
                seeds_el = [summarize(Pt, S[s][:, :k], fin & el_te) for s in range(nseed)]
                def agg(ss, f):
                    """Median over seeds, with the empirical seed range around it.

                    `p2p5`/`p97p5` are PERCENTILES OF THE 15 SEED VALUES, not a bootstrap
                    confidence interval for the median. Naming them `lo`/`hi` invited exactly
                    that misreading, so the names say what they are.
                    """
                    v = np.array([q[f] for q in ss])
                    return {"median": float(np.median(v)),
                            "p2p5": float(np.percentile(v, 2.5)),
                            "p97p5": float(np.percentile(v, 97.5)),
                            "iqr": [float(np.percentile(v, 25)),
                                    float(np.percentile(v, 75))]}
                per_k[str(k)] = {
                    "n_eval": seeds[0]["n_eval"],
                    "regret_median_db": agg(seeds, "regret_median_db"),
                    "inclusion": agg(seeds, "top_k_inclusion"),
                    "regret_mean_db": agg(seeds, "regret_mean_db"),
                    "regret_p90_db": agg(seeds, "regret_p90_db"),
                    **{f"outage_{m:g}dB": agg(seeds, f"outage_{m:g}dB") for m in MARGINS},
                    "inclusion_eligible": agg(seeds_el, "top_k_inclusion"),
                    "regret_mean_db_eligible": agg(seeds_el, "regret_mean_db")}
            # expected alignment overhead, per seed, from this run's own top-5 inclusion
            p5 = np.array([summarize(Pt, S[s][:, :KSHORT], fin)["top_k_inclusion"]
                           for s in range(nseed)])
            ov = KSHORT * TSEC_US + (1 - p5) * (NBEAM - KSHORT) * TSEC_US
            ovq = {"median": float(np.median(ov)),
                   "p2p5": float(np.percentile(ov, 2.5)),
                   "p97p5": float(np.percentile(ov, 97.5))}
            rec["learned"][f"{arch}|{label}"] = {
                "top1_seeds": [float(x) for x in z[f"{arch}|{label}|top1"]],
                "top5_seeds": [float(x) for x in z[f"{arch}|{label}|top5"]],
                "top1_median": float(np.median(z[f"{arch}|{label}|top1"])),
                "top5_median": float(np.median(z[f"{arch}|{label}|top5"])),
                "top5_mean": float(np.mean(z[f"{arch}|{label}|top5"])),
                "overhead_us": ovq,
                "by_k": per_k}
            if label == "chirps=250":
                a = per_k["5"]
                print(f"  {arch} full frame      k=5  incl {a['inclusion']['median']:.4f}  "
                      f"regret mean {a['regret_mean_db']['median']:.3f} dB  "
                      f"outage@3dB {a['outage_3dB']['median']:.4f}  "
                      f"overhead {np.median(ov):.1f} us")

        # ---- a decision used LATER than the frame it was computed from ----
        S = z["spatial|chirps=250|shortlist"]
        for lag in LAGS:
            ok, el = eligible_chain(seq, idx, dt_ms, n, start, lag)
            jj = np.arange(len(yt)) + lag
            # the LATER frame's own power vector is the reference, so it must be usable too
            valid = ok & fin & np.r_[fin[lag:], np.zeros(lag, bool)]
            if valid.sum() < 50:
                continue
            Pl = Pt[jj[valid]]
            res = {}
            for k in (1, 5):
                # STALE: the shortlist computed at frame i, used at frame i+lag.
                # FRESH: the shortlist computed at frame i+lag itself. Both are scored on the
                # IDENTICAL future frames, which is the only way the comparison isolates age
                # rather than mixing in a different evaluation set.
                def rg(sl):
                    return 10 * np.log10(Pl.max(1) / np.take_along_axis(
                        Pl, sl.astype(np.int64), 1).max(1))
                stale = [rg(S[s][valid][:, :k]) for s in range(S.shape[0])]
                fresh = [rg(S[s][jj[valid]][:, :k]) for s in range(S.shape[0])]
                res[str(k)] = {
                    "inclusion_median": float(np.median([(r <= 1e-12).mean() for r in stale])),
                    "regret_mean_db_median": float(np.median([r.mean() for r in stale])),
                    "outage_3dB_median": float(np.median([(r > 3.0).mean() for r in stale])),
                    "fresh_inclusion_median":
                        float(np.median([(r <= 1e-12).mean() for r in fresh])),
                    "fresh_regret_mean_db_median": float(np.median([r.mean() for r in fresh])),
                    "fresh_outage_3dB_median":
                        float(np.median([(r > 3.0).mean() for r in fresh])),
                    # paired per-seed differences: the median of the differences is not the
                    # difference of the medians, and the paired figure is the right one
                    "paired_regret_increase_db_median": float(np.median(
                        [a.mean() - b.mean() for a, b in zip(stale, fresh)])),
                    "paired_inclusion_drop_median": float(np.median(
                        [(b <= 1e-12).mean() - (a <= 1e-12).mean()
                         for a, b in zip(stale, fresh)]))}
            rec["delay"][str(lag)] = {
                "n_eval": int(valid.sum()),
                "n_logged_steps": lag,
                "elapsed_ms_median": float(np.median(el[valid])),
                "elapsed_ms_min": float(el[valid].min()),
                "elapsed_ms_max": float(el[valid].max()),
                "elapsed_ms_p95": float(np.percentile(el[valid], 95)),
                **res}
            d = rec["delay"][str(lag)]
            print(f"  lag {lag} ({d['elapsed_ms_median']:.0f} ms, n={d['n_eval']}): "
                  f"top-1 {d['1']['inclusion_median']:.4f}  top-5 "
                  f"{d['5']['inclusion_median']:.4f}  "
                  f"regret@5 {d['5']['regret_mean_db_median']:.3f} dB")

        # ---- the joint resource/shortlist frontier, and a rule chosen on validation ----
        # Sensing duration and shortlist length are two knobs on the same budget, and the
        # sweep so far has varied only the first at a fixed k=5. Under a FIXED-SHORTLIST
        # policy, which probes k beams and keeps the best measured one, both the cost and the
        # quality are computable for every (setting, k) already saved, with no oracle anywhere.
        #
        #   dedicated:  A(r,k) = Nc(r) * Tobs(r) + k * Tsec,  Tobs = Ns / fs
        #   reuse:      A(k)   = k * Tsec
        #
        # THE OPERATING POINT IS CHOSEN ON THE INNER VALIDATION BLOCK, under a rule fixed
        # before any test number was read: take the paper's existing default, the full frame
        # at k=5, as the quality target, and among all (setting, k) whose VALIDATION mean
        # regret is no worse than that target, choose the one of least modeled cost. Then
        # evaluate once on test. Without this the frontier is a post-hoc selection.
        FS_MSPS = NADC_RATE_MSPS
        vrow = z["val_row_index"]
        Pv, yv = P[vrow], y[vrow]
        vfin = np.isfinite(Pv).all(1)

        def cost(label, k, regime):
            cfg = SETTING_CFG[label]
            tobs = cfg["n_samples"] / FS_MSPS            # microseconds
            probe = k * TSEC_US
            return probe if regime == "reuse" else cfg["n_chirps"] * tobs + probe

        def qual(sl, Pm, ym, mask, k):
            best = Pm[mask].max(1)
            got = np.take_along_axis(Pm[mask], sl[mask][:, :k].astype(np.int64), 1).max(1)
            L = 10 * np.log10(best / got)
            return {"mean_regret_db": float(L.mean()),
                    "exceed_3dB": float((L > 3.0).mean()),
                    "inclusion": float((L <= 1e-12).mean())}

        front = []
        for label in SETTING_CFG:
            key = f"spatial|{label}|shortlist"
            if key not in z.files:
                continue
            St, Sv = z[key], z[f"spatial|{label}|val_shortlist"]
            for k in KS:
                te_ = [qual(St[i], Pt, yt, fin, k) for i in range(St.shape[0])]
                va_ = [qual(Sv[i], Pv, yv, vfin, k) for i in range(Sv.shape[0])]
                front.append({
                    "setting": label, "k": int(k),
                    "n_chirps": SETTING_CFG[label]["n_chirps"],
                    "n_samples": SETTING_CFG[label]["n_samples"],
                    "n_rx": SETTING_CFG[label]["n_rx"],
                    "cost_dedicated_us": cost(label, k, "dedicated"),
                    "cost_reuse_us": cost(label, k, "reuse"),
                    "test": {f: float(np.median([q[f] for q in te_]))
                             for f in ("mean_regret_db", "exceed_3dB", "inclusion")},
                    "val": {f: float(np.median([q[f] for q in va_]))
                            for f in ("mean_regret_db", "exceed_3dB", "inclusion")}})
        # Every comparator that uses no radar sensing, over the same shortlist lengths. Their
        # dedicated cost IS the probing cost, since there is no frame to acquire, which is
        # what makes them directly comparable on the frontier's x axis.
        _cmp = comparators(sc, y, seq, idx, dt_ms, n, ntr, start, max(KS))
        pri = _cmp["fixed training prior"]
        prior_curve = []
        for k in KS:
            prior_curve.append({"k": int(k), "cost_us": k * TSEC_US,
                                **{f"test_{f}": v for f, v in
                                   qual(pri, Pt, yt, fin, k).items()}})
        ref_curves = {}
        for _nm, _sl, _msk in (("fixed training prior", pri, fin),
                               ("previous beam", _cmp["persistence"], fin & el_te),
                               ("partner position", _cmp["position reference"], fin)):
            ref_curves[_nm] = [{"k": int(k), "cost_us": k * TSEC_US,
                                **{f"test_{f}": v for f, v in
                                   qual(_sl, Pt, yt, _msk, k).items()}} for k in KS]
        sel = {}
        for regime in ("dedicated", "reuse"):
            default = next(r for r in front
                           if r["setting"] == "chirps=250" and r["k"] == KSHORT)
            target = default["val"]["mean_regret_db"]
            ok = [r for r in front if r["val"]["mean_regret_db"] <= target + 1e-12]
            pick = min(ok, key=lambda r: (r[f"cost_{regime}_us"], r["k"]))
            sel[regime] = {
                "rule": ("least modeled cost among settings whose validation mean regret is "
                         "no worse than the full frame at k=5"),
                "validation_target_db": target, "n_candidates": len(front),
                "n_meeting_target": len(ok),
                "selected": {kk: pick[kk] for kk in
                             ("setting", "k", f"cost_{regime}_us", "test", "val")},
                "default": {kk: default[kk] for kk in
                            ("setting", "k", f"cost_{regime}_us", "test", "val")},
                "cost_ratio_vs_default":
                    default[f"cost_{regime}_us"] / pick[f"cost_{regime}_us"],
                "test_regret_change_db":
                    pick["test"]["mean_regret_db"] - default["test"]["mean_regret_db"]}
        rec["frontier"] = {"fs_msps": FS_MSPS, "T_sec_us": TSEC_US, "points": front,
                           "prior_curve": prior_curve, "reference_curves": ref_curves,
                           "selection": sel}

        # ---- how borderline is the dedicated verdict at each end of the airtime axis? ----
        # Table III's dedicated column charges Nc*Tobs on top of the oracle-fallback overhead
        # of (5) and compares against the fixed prior, which needs no sensing. At the shortest
        # frame that margin is a fraction of a microsecond out of ~228, so the sign depends on
        # the aggregation and on the assumed probe duration. At the full frame it does not.
        pri5 = pri[:, :KSHORT]
        p5p_all = float((pri5 == yt[:, None]).any(1).mean())
        p5p_fin = float((pri5[fin] == yt[fin][:, None]).any(1).mean())
        def _ovh(p):
            return KSHORT * TSEC_US + (1 - p) * (NBEAM - KSHORT) * TSEC_US
        margins = {}
        rng = np.random.default_rng(0)
        for label in ("chirps=8", "chirps=250"):
            Sx = z[f"spatial|{label}|shortlist"]
            nc = SETTING_CFG[label]["n_chirps"]
            tobs = SETTING_CFG[label]["n_samples"] / FS_MSPS
            row = {"n_chirps": nc, "sensing_us": nc * tobs}
            for mn, msk, p5p in (("all_test", np.ones(len(yt), bool), p5p_all),
                                 ("power_valid", fin, p5p_fin)):
                p5r = np.array([(Sx[i][msk][:, :KSHORT] == yt[msk][:, None]).any(1).mean()
                                for i in range(Sx.shape[0])])
                d = nc * tobs + _ovh(p5r) - _ovh(p5p)
                row[mn] = {"mean_us": float(d.mean()), "median_us": float(np.median(d)),
                           "n_seeds_cheaper": int((d < 0).sum()), "n_seeds": int(len(d)),
                           "p5_radar_mean": float(p5r.mean()), "p5_prior": float(p5p)}
                if mn == "all_test":
                    bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                                   for _ in range(20000)])
                    row[mn]["seed_bootstrap_mean_us"] = [float(np.percentile(bs, 2.5)),
                                                         float(np.percentile(bs, 97.5))]
                    gap = row[mn]["p5_radar_mean"] - p5p
                    row[mn]["breakeven_Tsec_us"] = (float(nc * tobs / (gap * (NBEAM - KSHORT)))
                                                    if gap > 0 else None)
            margins[label] = row
        rec["dedicated_margin"] = {
            "note": ("Table III convention: dedicated sensing charged at the faster priced "
                     "rate, compared against the fixed prior, which needs no sensing. "
                     "Positive means the radar costs more. The bootstrap resamples SEEDS "
                     "only and is conditional on these recordings and this recipe."),
            "fs_msps": FS_MSPS, "T_sec_us": TSEC_US, "k": KSHORT, "settings": margins}
        mm = margins["chirps=8"]["all_test"]
        print(f"  dedicated margin at 8 chirps: mean {mm['mean_us']:+.4f} us, median "
              f"{margins['chirps=8']['all_test']['median_us']:+.4f}, "
              f"{mm['n_seeds'] - mm['n_seeds_cheaper']}/{mm['n_seeds']} seeds dearer, "
              f"bootstrap [{mm['seed_bootstrap_mean_us'][0]:+.2f}, "
              f"{mm['seed_bootstrap_mean_us'][1]:+.2f}], break-even Tsec "
              f"{mm['breakeven_Tsec_us']:.3f} us")
        print(f"  dedicated margin at the full frame: mean "
              f"{margins['chirps=250']['all_test']['mean_us']:+.1f} us, "
              f"{margins['chirps=250']['all_test']['n_seeds_cheaper']}/15 seeds cheaper")
        d = sel["dedicated"]
        print(f"  frontier: {len(front)} points; dedicated rule picks "
              f"{d['selected']['setting']}, k={d['selected']['k']}, "
              f"{d['selected']['cost_dedicated_us']:.1f} us "
              f"({d['cost_ratio_vs_default']:.1f}x cheaper than the default), test regret "
              f"{d['selected']['test']['mean_regret_db']:.4f} dB vs "
              f"{d['default']['test']['mean_regret_db']:.4f}")
        print(f"  reuse rule picks {sel['reuse']['selected']['setting']}, "
              f"k={sel['reuse']['selected']['k']}, "
              f"{sel['reuse']['selected']['cost_reuse_us']:.1f} us")

        # ---- recording-level spread, which a seed bootstrap does not measure ----
        sq = seq[te]
        rows = []
        for q in sorted(set(sq.tolist())):
            m = (sq == q) & fin
            if m.sum() < 20:
                continue
            t1 = [float((S[s][m][:, 0] == yt[m]).mean()) for s in range(S.shape[0])]
            reg = [regret_db(Pt[m], S[s][m][:, :KSHORT]).mean() for s in range(S.shape[0])]
            rows.append({"sequence": int(q), "n": int(m.sum()),
                         "top1_median": float(np.median(t1)),
                         "regret_mean_db_median": float(np.median(reg))})
        # The pooled tail comparison on Scenario 33 is dominated by one recording, so the
        # per-recording exceedance and a leave-one-recording-out range are recorded here. With
        # five clusters these are influence diagnostics, not coverage statements.
        prior5 = cmp_["fixed training prior"][:, :KSHORT] if False else None
        pr_sl = comparators(sc, y, seq, idx, dt_ms, n, ntr, start,
                            max(KS))["fixed training prior"]
        def exc(sl, mask, thr=3.0):
            return float((regret_db(Pt[mask], sl[mask][:, :KSHORT]) > thr).mean())
        tail = []
        for q in sorted(set(sq.tolist())):
            mm = (sq == q) & fin
            if mm.sum() < 20:
                continue
            r_ = float(np.median([exc(S[s], mm) for s in range(S.shape[0])]))
            p_ = exc(pr_sl, mm)
            tail.append({"sequence": int(q), "n": int(mm.sum()),
                         "radar_exceedance": r_, "prior_exceedance": p_,
                         "difference_pp": 100 * (r_ - p_)})
        pooled = float(np.median([exc(S[s], fin) for s in range(S.shape[0])])) - exc(pr_sl, fin)
        loo = []
        for t_ in tail:
            mm = fin & (sq != t_["sequence"])
            loo.append(100 * (float(np.median([exc(S[s], mm) for s in range(S.shape[0])]))
                              - exc(pr_sl, mm)))
        rec["tail_by_recording"] = {
            "threshold_db": 3.0, "k": KSHORT, "rows": tail,
            "pooled_difference_pp": 100 * pooled,
            "leave_one_out_difference_pp": {str(t_["sequence"]): d
                                            for t_, d in zip(tail, loo)},
            "loo_min_pp": min(loo), "loo_max_pp": max(loo),
            "most_influential_sequence": int(tail[int(np.argmax(
                [abs(t_["difference_pp"]) for t_ in tail]))]["sequence"])}
        v = np.array([r["top1_median"] for r in rows])
        seed_t1 = [float((S[s][fin][:, 0] == yt[fin]).mean()) for s in range(S.shape[0])]
        rec["seed_spread"] = {
            "top1_iqr": [float(np.percentile(seed_t1, 25)),
                         float(np.percentile(seed_t1, 75))],
            "top1_iqr_width": float(np.percentile(seed_t1, 75)
                                    - np.percentile(seed_t1, 25)),
            "top1_min": float(min(seed_t1)), "top1_max": float(max(seed_t1))}
        rec["per_sequence"] = {
            "rows": rows, "n_sequences_scored": len(rows),
            "top1_min": float(v.min()), "top1_max": float(v.max()),
            "top1_median": float(np.median(v)),
            "top1_iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
        print(f"  per-sequence top-1 over {len(rows)} sequences: median "
              f"{np.median(v):.4f}, range {v.min():.4f} to {v.max():.4f}")
        out["scenarios"][sc] = rec

    p = os.path.join(RES, "beam_power_quality.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
