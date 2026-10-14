#!/usr/bin/env python
"""Independent re-derivation of the 2026-10-14 audit's P1-1 and P2-1.

Imports no project analysis module. Reads the raw per-beam power files, the development
CSVs and the saved shortlist rankings, rebuilds the greedy static set and the radar
regret from scratch, and then answers three questions:

  1. Does the STORED paired bootstrap reproduce at the old six-length policy (k=16 on
     Sc.31, k=8 on Sc.33)?  If yes, this file's implementation matches the production
     one and any difference below is a change of policy, not of arithmetic.
  2. What is the same statistic at the EXPANDED grid's cheapest radar policy, which the
     surrounding prose now describes?
  3. Over all sixteen shortlist lengths, what is the first length at which the static
     set's mean regret is no worse than the full frame's?

Run: python review_artifacts/2026-10-14/check_paired_and_crossover.py
Exit code is the result; it writes paired_and_crossover.json beside itself.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DS = os.path.join(ROOT, "datasets", "deepsense6g")
RES = os.path.join(ROOT, "experiments", "results")
NBEAM = 64
TSEC_US = 5.0
NBOOT = 20000
SEED = 20261011                    # the production script's seed, so draws are comparable
KS_ALL = tuple(range(1, 17))

FAILED = []
PASSED = 0


def check(ok, msg):
    global PASSED
    if ok:
        PASSED += 1
        print(f"  ok   {msg}")
    else:
        FAILED.append(msg)
        print(f"  FAIL {msg}")


def approx(a, b, tol, msg):
    check(abs(float(a) - float(b)) <= tol, f"{msg} ({a!r} vs {b!r}, tol {tol})")


def load(sc):
    """Raw load, independent of the project's loader."""
    with open(os.path.join(DS, f"Scenario{sc}", f"scenario{sc}_dev.csv")) as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    seq = np.array([int(r["seq_index"]) for r in rows])
    P = np.array([np.loadtxt(os.path.join(DS, f"Scenario{sc}",
                                          r["unit1_pwr_60ghz"].lstrip("./")))
                  for r in rows])
    assert P.shape == (len(rows), NBEAM), P.shape
    fin = np.isfinite(P).all(1)
    return y, seq, P, fin


def greedy_direct(P, kmax):
    """Greedy on the regret itself, with no log-identity shortcut."""
    order = []
    for _ in range(kmax):
        best_b, best_v = None, np.inf
        for b in range(NBEAM):
            if b in order:
                continue
            cols = order + [b]
            v = float(np.mean(10.0 * np.log10(P.max(1) / P[:, cols].max(1))))
            if v < best_v - 1e-15:
                best_b, best_v = b, v
        order.append(best_b)
    return order


def static_regret(P, order, k):
    return 10.0 * np.log10(P.max(1) / P[:, order[:k]].max(1))


def radar_per_seed(P, sl, mask, k):
    """Per-seed per-frame regret of a saved ranking's first k entries."""
    return np.stack([
        10.0 * np.log10(P.max(1) / np.take_along_axis(
            P, sl[s][mask][:, :k].astype(np.int64), 1).max(1))
        for s in range(sl.shape[0])])


def paired(P_ft, order, sl, tmask, k, seq_ft):
    """The manuscript's statistic: frame-weighted static mean minus the median over
    seeds of radar frame-weighted means, with a whole-recording bootstrap."""
    Ls = static_regret(P_ft, order, k)
    Lr = radar_per_seed(P_ft, sl, tmask, k)
    rng = np.random.default_rng(SEED)
    groups = np.unique(seq_ft)
    idx = [np.where(seq_ft == g)[0] for g in groups]
    d = np.empty(NBOOT)
    for t in range(NBOOT):
        pick = rng.integers(0, len(groups), len(groups))
        s_ = np.concatenate([idx[j] for j in pick])
        d[t] = Ls[s_].mean() - float(np.median(Lr[:, s_].mean(1)))
    better = sum(1 for j in range(len(groups))
                 if Ls[idx[j]].mean() < float(np.median(Lr[:, idx[j]].mean(1))))
    return {
        "k": int(k),
        "static_mean_db": float(Ls.mean()),
        "radar_median_seed_mean_db": float(np.median(Lr.mean(1))),
        "difference_db": float(Ls.mean() - float(np.median(Lr.mean(1)))),
        "ci95_whole_recording": [float(np.percentile(d, 2.5)),
                                 float(np.percentile(d, 97.5))],
        "n_recordings": int(len(groups)),
        "n_recordings_static_better": int(better),
        "static_below_every_seed": bool((Ls.mean() < Lr.mean(1)).all()),
    }


def main():
    ST = json.load(open(os.path.join(RES, "static_shortlist.json")))["scenarios"]
    Q = json.load(open(os.path.join(RES, "beam_power_quality.json")))["scenarios"]
    out = {"seed": SEED, "n_boot": NBOOT, "scenarios": {}}

    for sc in (31, 33):
        print(f"\n=== Scenario {sc} ===")
        y, seq, P, fin = load(sc)
        z = np.load(os.path.join(RES, f"beam_shortlists_sc{sc}.npz"))
        n, ntr, nte = len(y), int(z["n_train"]), int(z["n_test"])
        nval = max(1, ntr // 10)
        val = np.array(z["val_row_index"])
        fit = np.arange(0, ntr - nval)
        te = np.arange(n - nte, n)
        ff, ft = fit[fin[fit]], te[fin[te]]
        tmask = fin[te]

        S = ST[str(sc)]
        F = Q[str(sc)]["frontier"]

        # the greedy order, rebuilt on the regret with no log identity
        order = greedy_direct(P[ff], 16)
        check(order == [int(b) for b in S["order"]],
              f"Sc.{sc}: the greedy static order reproduces from the regret directly")

        # ---- 1. the SUPERSEDED six-length policy, recomputed for the record ----
        # Before 2026-10-14 the stored paired result was computed here, at the frontier
        # JSON's own selection, while the prose had moved to the expanded grid's cheapest
        # radar. These numbers are kept so the change is documented rather than asserted.
        old = F["selection"]["dedicated"]["selected"]
        sl_old = z[f"spatial|{old['setting']}|shortlist"]
        got = paired(P[ft], order, sl_old, tmask, int(old["k"]), seq[ft])
        st = S["paired_vs_selected_radar"]
        print(f"  superseded six-length policy: {old['setting']}, k={old['k']}  "
              f"diff {got['difference_db']:+.6f} "
              f"CI [{got['ci95_whole_recording'][0]:+.6f}, "
              f"{got['ci95_whole_recording'][1]:+.6f}] "
              f"{got['n_recordings_static_better']}/{got['n_recordings']}")
        check(int(st["k"]) != int(old["k"]),
              f"Sc.{sc}: the stored paired result no longer uses the six-length grid's "
              f"policy (stored k={st['k']}, six-length k={old['k']})")

        # ---- 2. the same statistic at the EXPANDED grid's cheapest radar ----
        fam = S["selection_dedicated"]["cheapest_by_family"]["radar"]
        setting = fam["policy"].replace("radar, ", "")
        kcur = int(fam["k"])
        check((setting, kcur) != (old["setting"], int(old["k"])),
              f"Sc.{sc}: the expanded grid's cheapest radar ({setting}, k={kcur}) "
              f"differs from the stored comparison policy "
              f"({old['setting']}, k={old['k']})")
        sl_cur = z[f"spatial|{setting}|shortlist"]
        cur = paired(P[ft], order, sl_cur, tmask, kcur, seq[ft])
        cur["radar_setting"] = setting
        # the STORED record must now be this one, value for value
        for key, tol in (("static_mean_db", 1e-12),
                         ("radar_median_seed_mean_db", 1e-12),
                         ("difference_db", 1e-12)):
            approx(cur[key], st[key], tol,
                   f"Sc.{sc}: the stored paired {key} is the corrected policy's")
        for i_ in (0, 1):
            approx(cur["ci95_whole_recording"][i_], st["ci95_whole_recording"][i_], 1e-9,
                   f"Sc.{sc}: the stored paired CI bound {i_} is the corrected policy's")
        check(cur["n_recordings_static_better"] == st["n_recordings_static_better"]
              and int(st["k"]) == kcur and st["radar_setting"] == setting,
              f"Sc.{sc}: the stored paired policy and per-recording count match the "
              f"selection's cheapest radar ({setting}, k={kcur})")
        print(f"  CORRECTED at {setting}, k={kcur}:")
        for key in ("static_mean_db", "radar_median_seed_mean_db", "difference_db",
                    "ci95_whole_recording", "n_recordings_static_better",
                    "static_below_every_seed"):
            print(f"    {key} = {cur[key]}")
        # the radar arm must agree with the grid's own stored test value
        approx(cur["radar_median_seed_mean_db"], fam["test"]["mean_regret_db"], 1e-9,
               f"Sc.{sc}: the corrected radar arm equals the selection grid's own "
               f"stored test mean regret")
        check(cur["difference_db"] < 0.0,
              f"Sc.{sc}: the static set still has the lower mean at the corrected k")

        # ---- 3. the crossover over ALL sixteen lengths ----
        sl_full = z["spatial|chirps=250|shortlist"]
        rows = []
        for k in KS_ALL:
            s_ = float(static_regret(P[ft], order, k).mean())
            r_ = float(np.median(radar_per_seed(P[ft], sl_full, tmask, k).mean(1)))
            rows.append({"k": k, "static_db": s_, "radar_db": r_,
                         "static_minus_radar_db": s_ - r_})
        xs = [r["k"] for r in rows if r["static_minus_radar_db"] <= 0]
        cross = min(xs) if xs else None
        print(f"  crossover over all sixteen lengths: k={cross}")
        for r in rows[:8]:
            print(f"    k={r['k']:2d}  static={r['static_db']:.6f}  "
                  f"radar={r['radar_db']:.6f}  diff={r['static_minus_radar_db']:+.6f}")
        # the six stored lengths must reproduce
        for m in S["matched_k"]:
            r = [x for x in rows if x["k"] == m["k"]][0]
            approx(r["static_db"], m["static"]["mean_regret_db"], 1e-9,
                   f"Sc.{sc}: stored matched static at k={m['k']} reproduces")
            approx(r["radar_db"], m["radar_full_frame"]["mean_regret_db"], 1e-9,
                   f"Sc.{sc}: stored matched radar at k={m['k']} reproduces")

        out["scenarios"][str(sc)] = {
            "stored_paired_policy": {"setting": old["setting"], "k": int(old["k"])},
            "stored_paired_reproduced": got,
            "corrected_paired": cur,
            "matched_k_all_lengths": rows,
            "k_crossover_mean_all_lengths": cross,
            "k_crossover_mean_stored": S["k_crossover_mean"],
        }

    p = os.path.join(HERE, "paired_and_crossover.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")
    print(f"\n{PASSED}/{PASSED + len(FAILED)} independent checks passed")
    for m in FAILED:
        print(f"  FAILED: {m}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
