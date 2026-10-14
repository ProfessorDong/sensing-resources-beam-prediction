#!/usr/bin/env python
"""A static probe set chosen for the power objective, and what it leaves the radar to buy.

The paper's no-sensing comparator was the fixed training prior: the k beams most often the
exact best beam in the training block. That is the right static baseline for a top-k
inclusion question and the wrong one once the objective becomes received-power regret. A
beam that is rarely the exact maximum can still deliver good power on many frames, and
several adjacent frequent winners cover nearly the same directions, so the prior spends
probes on overlapping coverage.

This builds the static set that the objective actually asks for. Greedily, with
S_0 empty and the optimization rows T,

    b_j = argmin_{b not in S_{j-1}}  (1/|T|) sum_{i in T} 10 log10 [ max_a P_i(a)
                                                   / max_{a in S_{j-1} u {b}} P_i(a) ],

ties going to the smallest beam index. It is a greedy construction and not a claim of
optimal set selection. The policy probes the same k beams on every frame and keeps the best
measured one, so at run time it consumes no radar, no position and no previous alignment;
its only input is the fixed list. Its cost is k T_sec in both regimes.

WHAT IS AND IS NOT MATCHED. The static set is fitted on the OPTIMIZATION rows only, the
same rows the networks are fitted on, with the inner validation block and the test block
excluded. It is fitted against a richer target than the networks are: the full 64-beam power
vector rather than its argmax, which is the published label. That asymmetry is in the
training supervision, not in the test-time input, and the paper states it. It also suggests
the experiment this file does not run, which is to train the readout against the same power
objective.

Output: results/static_shortlist.json
"""
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DS = os.path.join(os.path.dirname(HERE), "datasets", "deepsense6g")
NBEAM = 64
KS = (1, 2, 3, 5, 8, 16)          # the lengths the figures and the prose quote
KS_ALL = tuple(range(1, 17))      # every length the saved rankings carry
TSEC_US = 5.0
NBOOT = 20000
SEED = 20261011


def load(sc):
    rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}",
                                                 f"scenario{sc}_dev.csv"))))
    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    seq = np.array([int(r["seq_index"]) for r in rows])
    P = np.array([np.loadtxt(os.path.join(DS, f"Scenario{sc}",
                                          r["unit1_pwr_60ghz"].lstrip("./"))) for r in rows])
    assert P.shape == (len(rows), NBEAM), P.shape
    fin = np.isfinite(P).all(1)
    assert np.array_equal(np.nanargmax(P[fin], 1), y[fin]), "power argmax is not the label"
    return y, seq, P, fin


def greedy(P):
    """Ordered beam list minimizing mean dB regret on these rows, one beam at a time.

    Minimizing the mean of 10 log10(max_a P / max_{S} P) over a fixed row set is the same as
    maximizing the mean of log10(max_{S} P), since the numerator does not depend on S. The
    loop below uses that identity; `_greedy_direct` recomputes the first few steps from the
    regret itself so the identity is checked rather than trusted.
    """
    L = np.log10(P)
    best = np.full(len(P), -np.inf)
    order, taken = [], np.zeros(NBEAM, bool)
    for _ in range(max(KS)):
        g = np.where(taken, -np.inf,
                     np.array([np.maximum(best, L[:, b]).mean() for b in range(NBEAM)]))
        b = int(np.argmax(g))          # np.argmax returns the smallest index on a tie
        order.append(b)
        taken[b] = True
        best = np.maximum(best, L[:, b])
    return order


def _greedy_direct(P, nstep=5):
    out = []
    for _ in range(nstep):
        cand = [b for b in range(NBEAM) if b not in out]
        r = [float(np.mean(10.0 * np.log10(P.max(1) / P[:, out + [b]].max(1))))
             for b in cand]
        out.append(cand[int(np.argmin(r))])
    return out


def swap_search(P, S, rounds=8):
    """Single-beam replacement from S. Any improvement DISPROVES optimality of S.

    A greedy set is feasible, so its regret is an UPPER bound on the minimum over sets of
    that size. It is not a lower bound that every set must obey, and the manuscript briefly
    read it as one. This returns a demonstrated better set where one exists, so the claim is
    settled by exhibition rather than by argument. No optimality is claimed for the result.
    """
    S = list(S)
    best = float(np.mean(10.0 * np.log10(P.max(1) / P[:, S].max(1))))
    for _ in range(rounds):
        moved = False
        for i in range(len(S)):
            for b in range(NBEAM):
                if b in S:
                    continue
                T = S[:i] + [b] + S[i + 1:]
                v = float(np.mean(10.0 * np.log10(P.max(1) / P[:, T].max(1))))
                if v < best - 1e-12:
                    S, best, moved = T, v, True
        if not moved:
            break
    return S, best


def regret(P, order, k):
    return 10.0 * np.log10(P.max(1) / P[:, order[:k]].max(1))


def score(P, order, k):
    L = regret(P, order, k)
    return {"k": int(k), "n": int(len(L)), "mean_regret_db": float(L.mean()),
            "median_regret_db": float(np.median(L)),
            "exceed_1dB": float((L > 1.0).mean()), "exceed_3dB": float((L > 3.0).mean()),
            "inclusion": float((L <= 1e-12).mean())}


def main():
    out = {"policy": "fixed probe set, same k beams every frame, best measured one kept",
           "objective": "greedy minimization of mean dB power regret on the optimization rows",
           "inputs_at_run_time": "none: no radar, no position, no previous alignment",
           "training_supervision": "the full 64-beam power vector on the optimization rows, "
                                   "a richer target than the argmax label the readout is fitted to",
           "cost_model": "k * T_sec in both regimes, since there is no frame to acquire",
           "T_sec_us": TSEC_US, "k_values": list(KS), "n_beam": NBEAM, "scenarios": {}}
    Q = json.load(open(os.path.join(RES, "beam_power_quality.json")))
    for sc in (31, 33):
        y, seq, P, fin = load(sc)
        z = np.load(os.path.join(RES, f"beam_shortlists_sc{sc}.npz"))
        n, ntr, nte = len(y), int(z["n_train"]), int(z["n_test"])
        nval = max(1, ntr // 10)                      # the trainer's own inner split
        val = np.array(z["val_row_index"])
        assert np.array_equal(val, np.arange(ntr - nval, ntr)), "validation block moved"
        fit = np.arange(0, ntr - nval)
        te = np.arange(n - nte, n)
        ff, fv, ft = fit[fin[fit]], val[fin[val]], te[fin[te]]
        order = greedy(P[ff])
        assert _greedy_direct(P[ff]) == order[:5], "the log identity does not reproduce"
        # the frequency prior on the SAME rows, so the two static policies differ in their
        # objective and in nothing else
        prior = [b for b, _ in collections.Counter(y[ff].tolist()).most_common(NBEAM)]
        rec = {"n_fit": int(len(fit)), "n_fit_finite": int(len(ff)),
               "n_val": int(len(val)), "n_val_finite": int(len(fv)),
               "n_test": int(len(te)), "n_test_finite": int(len(ft)),
               "order": [int(b) for b in order],
               "prior_order": [int(b) for b in prior[:max(KS)]],
               "test": {}, "val": {}, "prior_test": {}, "prior_val": {}}
        # scored at EVERY length, since the selection now ranges over all sixteen and the
        # figure has to be able to draw whichever one it picks
        for k in KS_ALL:
            rec["test"][str(k)] = score(P[ft], order, k)
            rec["val"][str(k)] = score(P[fv], order, k)
            rec["prior_test"][str(k)] = score(P[ft], prior, k)
            rec["prior_val"][str(k)] = score(P[fv], prior, k)
        # --- is a learned shortlist a sector? ---
        # The mechanism offered for the matched-k crossover is that a ranking by
        # probability of being the exact maximum concentrates its probes, while a set
        # chosen for coverage spreads them. The first half of that is measurable on the
        # saved rankings: how often the learned top-5 is five consecutive beam indices.
        _t5 = np.sort(z["spatial|chirps=250|shortlist"][:, :, :5].astype(int), axis=2)
        _sp = _t5[:, :, -1] - _t5[:, :, 0]
        rec["learned_top5_geometry"] = {
            "frac_consecutive": float(((np.diff(_t5, axis=2) == 1).all(2)).mean()),
            "median_index_span": float(np.median(_sp)),
            "span_if_consecutive": 4,
            "static_index_span": int(max(order[:5]) - min(order[:5]))}

        # --- how much of the static set's strength is the repeated route? ---
        # The same construction refitted on the TEST rows is a HINDSIGHT REFERENCE, not an
        # optimality bound. Greedy returns one feasible set, so its regret upper-bounds the
        # minimum over sets of that size; nothing makes it a floor every set must clear.
        # The 2026-10-13 audit caught the manuscript reading it as a bound, and the swap
        # search below settles it by exhibiting better sets. What the small train-to-test
        # difference does show is that this construction TRANSFERS between the recorded
        # blocks, which is the route-repetition caveat measured on a policy; it does not
        # isolate route repetition as the cause, and it bounds nothing about an adaptive
        # policy that changes its set per frame.
        hind = greedy(P[ft])
        rec["hindsight_order"] = [int(b_) for b_ in hind]
        rec["hindsight_test"] = {str(k): score(P[ft], hind, k) for k in KS_ALL}
        rec["transfer_gap_db"] = {
            str(k): float(rec["test"][str(k)]["mean_regret_db"]
                          - rec["hindsight_test"][str(k)]["mean_regret_db"]) for k in KS_ALL}
        rec["hindsight_not_optimal"] = {}
        for k in KS_ALL:
            g = hind[:k]
            gv = float(np.mean(10.0 * np.log10(P[ft].max(1) / P[ft][:, g].max(1))))
            S_, bv = swap_search(P[ft], g)
            tv = float(np.mean(10.0 * np.log10(P[ft].max(1) / P[ft][:, order[:k]].max(1))))
            rec["hindsight_not_optimal"][str(k)] = {
                "greedy_mean_db": gv, "better_set": sorted(int(b_) for b_ in S_),
                "better_mean_db": bv, "improvement_db": gv - bv,
                "greedy_is_optimal": bool(bv >= gv - 1e-12),
                "train_minus_greedy_db": tv - gv, "train_minus_better_db": tv - bv}


        # --- accuracy on EVERY test label, which is the mask the bound uses ---
        # The static set needs powers to be FITTED, so its regret is scored on the finite
        # rows. Its exact-beam inclusion needs no power at all, only the label, so scoring
        # it on the finite subset while the no-information bound is computed on all labels
        # compares two populations. On Scenario 31 that mismatch alone put a constant-rate
        # policy above the best-constant bound, which is arithmetically impossible on
        # matched rows. Caught by the 2026-10-13 audit.
        rec["all_label_accuracy"] = {}
        bound_all = float(collections.Counter(y[te].tolist()).most_common(1)[0][1]) / len(te)
        rec["bound_all_labels"] = bound_all
        for k in KS_ALL:
            rec["all_label_accuracy"][str(k)] = {
                "n": int(len(te)),
                "static": float(np.isin(y[te], order[:k]).mean()),
                "prior": float(np.isin(y[te], prior[:k]).mean())}
        assert rec["all_label_accuracy"]["1"]["static"] <= bound_all + 1e-12, \
            "a single fixed beam cannot beat the best constant on the same rows"

        # --- the matched-k comparison against the radar, on the same test rows ---
        # The comparison itself is built below, once the all-length radar grid exists: the
        # crossover search has to range over the SAME sixteen lengths the selection does.
        # Searching six of them made the reported Scenario 33 crossover k=5 when the first
        # empirical crossing is at k=4, a length the old grid skipped. Caught 2026-10-14.
        F = Q["scenarios"][str(sc)]["frontier"]
        pts = {(p["setting"], p["k"]): p for p in F["points"]}
        full = [s for s in {k_[0] for k_ in pts} if s == "chirps=250"][0]
        # --- the selection rule, with the static policies admitted ---
        # EVERY shortlist length from 1 to 16, for the radar as well as the static families.
        # The published grid sampled six lengths, which is a sampling of the candidate set
        # and not the candidate set: on Scenario 31 the omitted k=10 meets the same
        # validation target at 50 us against the sampled grid's 80. Expanding only the
        # static side would hand it sixteen lengths against the radar's six, so both sides
        # are expanded from the saved rankings, which already carry all sixteen positions.
        # Radar quality is the median over seeds of the per-seed mean, as elsewhere.
        tgt = F["selection"]["dedicated"]["validation_target_db"]
        rec["validation_target_db"] = float(tgt)

        def _radar_med(rows, sl, mask, k, thresh=None):
            per = [10.0 * np.log10(rows.max(1) / np.take_along_axis(
                rows, sl[si][mask][:, :k].astype(np.int64), 1).max(1))
                for si in range(sl.shape[0])]
            if thresh is None:
                return float(np.median([float(q.mean()) for q in per]))
            return float(np.median([float((q > thresh).mean()) for q in per]))

        frame_us = {s_: p["cost_dedicated_us"] - TSEC_US * k_ for (s_, k_), p in pts.items()}
        radar_grid = {}
        for s_ in sorted(frame_us):
            slt, slv = z[f"spatial|{s_}|shortlist"], z[f"spatial|{s_}|val_shortlist"]
            for k in KS_ALL:
                radar_grid[(s_, k)] = {
                    "val": _radar_med(P[fv], slv, fin[val], k),
                    "test": {"mean_regret_db": _radar_med(P[ft], slt, fin[te], k),
                             "exceed_3dB": _radar_med(P[ft], slt, fin[te], k, 3.0),
                             "inclusion": 1.0 - _radar_med(P[ft], slt, fin[te], k, 1e-12)},
                    "cost_dedicated": frame_us[s_] + TSEC_US * k,
                    "cost_reuse": TSEC_US * k}
        # the stored six must reproduce under this independent recomputation
        for (s_, k), p in pts.items():
            assert abs(radar_grid[(s_, k)]["test"]["mean_regret_db"]
                       - p["test"]["mean_regret_db"]) < 1e-9, (s_, k)
        # HOW MUCH INDEPENDENT DIVERSITY THE INNER VALIDATION BLOCK ACTUALLY HAS. The
        # selection constraint is evaluated on these rows, so their recording composition
        # bounds what the constraint can mean. On Scenario 33 almost all of them come from
        # one recording, and on both scenarios one recording straddles the optimization and
        # validation boundary: the split is recording-disjoint between the TRAINING REGION
        # and the TEST block, which is the claim the paper makes, and is not
        # recording-disjoint inside the training region. Raised by the 2026-10-14 audit.
        _vs, _vc = np.unique(seq[fv], return_counts=True)
        rec["validation_recordings"] = {
            "n_recordings": int(len(_vs)),
            "n_frames_finite": int(len(fv)),
            "largest_count": int(_vc.max()),
            "largest_share": float(_vc.max() / len(fv)),
            "shared_with_optimization": int(
                len(set(seq[ff].tolist()) & set(seq[fv].tolist())))}
        rec["k_grid"] = list(KS_ALL)
        # SAVED so the frontier figure can draw the grid the selection actually searched.
        # Drawing six lengths for the radar beside sixteen for the static families made the
        # prose's "over the same shortlist lengths" false and left the two radar policies
        # the text compares off the figure entirely.
        rec["radar_grid"] = {f"{s_}|{k}": {
            "setting": s_, "k": int(k),
            "n_chirps": int(pts[(s_, KS[0])]["n_chirps"]),
            "n_samples": int(pts[(s_, KS[0])]["n_samples"]),
            "n_rx": int(pts[(s_, KS[0])]["n_rx"]),
            "cost_dedicated_us": float(g["cost_dedicated"]),
            "cost_reuse_us": float(g["cost_reuse"]),
            "val_mean_regret_db": float(g["val"]),
            "test": {kk: float(g["test"][kk]) for kk in
                     ("mean_regret_db", "exceed_3dB", "inclusion")}}
            for (s_, k), g in sorted(radar_grid.items())}
        assert len(rec["radar_grid"]) == len(frame_us) * len(KS_ALL)

        # --- the matched-k comparison, over EVERY length the selection searches ---
        # Radar quality is the median over seeds of the per-seed mean, as everywhere else,
        # and comes from radar_grid rather than the six stored frontier points, which is
        # what confines the crossover search to a sampled grid.
        rec["matched_k"] = []
        for k in KS_ALL:
            r = radar_grid[(full, k)]
            s_ = rec["test"][str(k)]
            rec["matched_k"].append({
                "k": int(k),
                "radar_full_frame": {"mean_regret_db": r["test"]["mean_regret_db"],
                                     "exceed_3dB": r["test"]["exceed_3dB"],
                                     "inclusion": r["test"]["inclusion"]},
                "static": {"mean_regret_db": s_["mean_regret_db"],
                           "exceed_3dB": s_["exceed_3dB"], "inclusion": s_["inclusion"]},
                "prior": {"mean_regret_db": rec["prior_test"][str(k)]["mean_regret_db"],
                          "exceed_3dB": rec["prior_test"][str(k)]["exceed_3dB"]},
                "static_minus_radar_db": s_["mean_regret_db"] - r["test"]["mean_regret_db"]})
        assert [m["k"] for m in rec["matched_k"]] == list(KS_ALL), \
            "the crossover must be searched over the declared grid, not a sample of it"
        # the crossover: the smallest k at which the static set is no worse than the full frame
        xs = [m["k"] for m in rec["matched_k"] if m["static_minus_radar_db"] <= 0]
        rec["k_crossover_mean"] = int(min(xs)) if xs else None
        xt = [m["k"] for m in rec["matched_k"]
              if m["static"]["exceed_3dB"] <= m["radar_full_frame"]["exceed_3dB"]]
        rec["k_crossover_tail"] = int(min(xt)) if xt else None
        # the margin at the crossing, so the prose can say how small it is rather than
        # presenting an empirical crossing as an established transition
        if rec["k_crossover_mean"] is not None:
            _m = [m for m in rec["matched_k"]
                  if m["k"] == rec["k_crossover_mean"]][0]
            rec["k_crossover_mean_margin_db"] = float(-_m["static_minus_radar_db"])
        else:
            rec["k_crossover_mean_margin_db"] = None
        for regime in ("dedicated", "reuse"):
            cand = []
            for k in KS_ALL:
                cand.append(("power-aware static", k, TSEC_US * k,
                             rec["val"][str(k)]["mean_regret_db"], rec["test"][str(k)]))
                cand.append(("fixed training prior", k, TSEC_US * k,
                             rec["prior_val"][str(k)]["mean_regret_db"],
                             rec["prior_test"][str(k)]))
            for (s_, k), g in radar_grid.items():
                cand.append((f"radar, {s_}", k, g[f"cost_{regime}"], g["val"], g["test"]))
            ok = [c for c in cand if c[3] <= tgt + 1e-12]
            mc = min(c[2] for c in ok)
            # TIES ARE BROKEN EXPLICITLY, on validation only: least cost, then least
            # validation mean regret, then fewest probes, then name. The published
            # implementation left this to dictionary order, which turned an arbitrary
            # iteration detail into a design conclusion.
            tied = sorted([c for c in ok if abs(c[2] - mc) < 1e-9],
                          key=lambda c: (c[3], c[1], c[0]))
            best = tied[0]
            # the cheapest policy of each family that meets the target, so the paper can
            # compare like with like under the expanded grid instead of quoting the
            # six-length grid's radar cost against the sixteen-length grid's static cost
            fam = {}
            for c in ok:
                key = "radar" if c[0].startswith("radar") else c[0]
                if key not in fam or c[2] < fam[key][2]:
                    fam[key] = c
            rec[f"selection_{regime}"] = {
                "n_candidates": len(cand), "n_meeting_target": len(ok),
                "cheapest_by_family": {k_: {"policy": v[0], "k": int(v[1]),
                                            "cost_us": float(v[2]),
                                            "val_mean_regret_db": float(v[3]),
                                            "test": {kk: v[4][kk] for kk in
                                                     ("mean_regret_db", "exceed_3dB")}}
                                       for k_, v in fam.items()},
                "min_cost_us": float(mc), "n_tied_at_min_cost": len(tied),
                "tie_break": "least validation mean regret, then fewest probes, then name",
                "tied": [{"policy": c[0], "k": int(c[1]),
                          "val_mean_regret_db": float(c[3])} for c in tied],
                "selected": {"policy": best[0], "k": int(best[1]), "cost_us": float(best[2]),
                             "val_mean_regret_db": float(best[3]),
                             "test": {kk: best[4][kk] for kk in
                                      ("mean_regret_db", "exceed_3dB", "inclusion")}},
                "uses_radar": best[0].startswith("radar")}
        # What the SAMPLED six-length grid would have selected, kept so the paper can say
        # what expanding the grid bought instead of asserting it. Same rule, same target,
        # same statistic; only the set of lengths differs.
        cand6 = []
        for k in KS:
            cand6.append(("power-aware static", k, TSEC_US * k,
                          rec["val"][str(k)]["mean_regret_db"]))
            cand6.append(("fixed training prior", k, TSEC_US * k,
                          rec["prior_val"][str(k)]["mean_regret_db"]))
        for (s_, k), g in radar_grid.items():
            if k in KS:
                cand6.append((f"radar, {s_}", k, g["cost_dedicated"], g["val"]))
        ok6 = [c for c in cand6 if c[3] <= tgt + 1e-12]
        mc6 = min(c[2] for c in ok6)
        b6 = sorted([c for c in ok6 if abs(c[2] - mc6) < 1e-9],
                    key=lambda c: (c[3], c[1], c[0]))[0]
        rec["selection_dedicated_sampled_grid"] = {
            "n_candidates": len(cand6), "policy": b6[0], "k": int(b6[1]),
            "cost_us": float(b6[2]),
            "saving_us": float(b6[2] - rec["selection_dedicated"]["selected"]["cost_us"])}

        # --- a paired bootstrap over whole recordings at the matched selected k ---
        # THE COMPARISON POLICY IS THE ONE THE SURROUNDING PROSE NAMES, which is the
        # cheapest radar policy of the EXPANDED grid. Reading it from the frontier's own
        # `selection` field took the six-length grid's winner (k=16 on Sc.31, k=8 on
        # Sc.33) while the paragraph had moved on to k=12 and k=7, so the published
        # intervals described a policy the text no longer discussed. Caught 2026-10-14.
        # The invariant below is the guard: a macro matched against a stale JSON field
        # cannot detect a policy mismatch, only a numeric one.
        _fam = rec["selection_dedicated"]["cheapest_by_family"]["radar"]
        rsetting = _fam["policy"].replace("radar, ", "")
        kr = int(_fam["k"])
        assert (rsetting, kr) in radar_grid, (rsetting, kr)
        assert abs(radar_grid[(rsetting, kr)]["cost_dedicated"]
                   - _fam["cost_us"]) < 1e-9, "the paired policy is not the selected one"
        Ls = regret(P[ft], order, kr)          # static at the radar's own k, for a paired test
        sl = z[f"spatial|{rsetting}|shortlist"]             # (seeds, n_test, 16)
        tmask = fin[te]
        per_seed = np.stack([
            10.0 * np.log10(P[ft].max(1) / np.take_along_axis(
                P[ft], sl[s_i][tmask][:, :kr].astype(np.int64), 1).max(1))
            for s_i in range(sl.shape[0])])
        Lr = per_seed
        rng = np.random.default_rng(SEED)
        gs = seq[ft]
        groups = np.unique(gs)
        idx = [np.where(gs == g)[0] for g in groups]
        d = []
        for _ in range(NBOOT):
            pick = rng.integers(0, len(groups), len(groups))
            sel_ = np.concatenate([idx[j] for j in pick])
            d.append(Ls[sel_].mean() - float(np.median(Lr[:, sel_].mean(1))))
        d = np.array(d)
        rec["paired_vs_selected_radar"] = {
            "k": int(kr), "radar_setting": rsetting,
            "radar_policy": _fam["policy"], "radar_cost_us": float(_fam["cost_us"]),
            "static_k": int(kr),
            "static_mean_db": float(Ls.mean()),
            "radar_median_seed_mean_db": float(np.median(Lr.mean(1))),
            "difference_db": float(Ls.mean() - float(np.median(Lr.mean(1)))),
            "ci95_whole_recording": [float(np.percentile(d, 2.5)),
                                     float(np.percentile(d, 97.5))],
            "n_recordings": int(len(groups)), "n_boot": NBOOT,
            "static_below_every_seed": bool(bool((Ls.mean() < per_seed.mean(1)).all())),
            "n_recordings_static_better": int(sum(
                1 for j in range(len(groups))
                if Ls[idx[j]].mean() < float(np.median(per_seed[:, idx[j]].mean(1)))))}
        # THE INVARIANT: the policy the paired result was computed at must be the policy
        # the selection names. This is the check whose absence let the two drift apart.
        _pv = rec["paired_vs_selected_radar"]
        _cf = rec["selection_dedicated"]["cheapest_by_family"]["radar"]
        assert _pv["radar_policy"] == _cf["policy"] and _pv["k"] == int(_cf["k"]), \
            (_pv["radar_policy"], _pv["k"], _cf["policy"], _cf["k"])
        assert abs(_pv["radar_median_seed_mean_db"]
                   - _cf["test"]["mean_regret_db"]) < 1e-9, \
            "the paired radar arm disagrees with the selection grid's own test value"
        out["scenarios"][str(sc)] = rec
        print(f"Sc{sc}: static order {order[:8]}...  crossover k "
              f"mean={rec['k_crossover_mean']} tail={rec['k_crossover_tail']}  "
              f"dedicated picks {rec['selection_dedicated']['selected']['policy']} "
              f"k={rec['selection_dedicated']['selected']['k']}  "
              f"reuse picks {rec['selection_reuse']['selected']['policy']} "
              f"k={rec['selection_reuse']['selected']['k']}")
    p = os.path.join(RES, "static_shortlist.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main())
