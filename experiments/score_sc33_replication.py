#!/usr/bin/env python
"""Score Scenario 33 against the criteria fixed on 2026-10-02, before it was run.

The criteria, verbatim from `exp_sc31_resource_ablation.py`:
  R1  median top-1 at full resources exceeds the bound by more than 0.05
  R2  the 8 -> 64 chirp step has a bootstrap 95% CI on the median difference excluding zero
  R3  utilization against median gain over the bound is positive with p < 0.05

They are scored exactly as written. Two notes that are themselves part of the result:

R3 IS RETIRED, NOT SCORED. After the criteria were fixed, utilization was shown to be an affine
image of the gain: utilization = gain + (bound - shuffled), and the bracket is near-constant
(`audit_utilization_definitional.py` measures bracket s.d. / gain s.d. = 0.114 on Sc.31). R3 is
therefore satisfied automatically and tests nothing. Counting it as a pass would inflate the
verdict. Its honest replacement, reported alongside, is the bearing-mediation correlation, whose
two sides come from different tasks with different labels.

The 2026-10-02 Sc.33 run that failed used the DECIMATING readout and the pre-recipe-G optimizer.
Block-max resampling and recipe G were both shown to matter on Sc.31 (4 of 10 verdicts flipped;
collapse fell from ~1-in-10 to 0/15), so this is a re-test of the same hypothesis on a repaired
instrument, not a second bite at the same measurement. Say so when reporting it.
"""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
rng = np.random.default_rng(20261004)
FULL = "chirps=250"


def boot_diff(a, b, n=20000):
    a, b = np.asarray(a), np.asarray(b)
    d = np.array([np.median(rng.choice(a, a.size, True)) - np.median(rng.choice(b, b.size, True))
                  for _ in range(n)])
    pool = np.concatenate([a, b]); obs = np.median(a) - np.median(b); c = 0
    for _ in range(n):
        p = rng.permutation(pool)
        c += abs(np.median(p[:a.size]) - np.median(p[a.size:])) >= abs(obs) - 1e-12
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5), (c + 1) / (n + 1)


def main():
    sfx = "_max" if (len(sys.argv) < 2 or sys.argv[1] == "max") else ""
    p33 = os.path.join(RES, f"sc33_resource_ablation_G{sfx}.json")
    p31 = os.path.join(RES, f"sc31_resource_ablation_G{sfx}.json")
    if not os.path.exists(p33):
        print(f"not present yet: {p33}"); return
    d33, d31 = json.load(open(p33)), json.load(open(p31))
    S, b = d33["settings"], d33["bound_const"]
    print("=" * 80)
    print(f"SCENARIO 33 vs the criteria fixed 2026-10-02.  pool={d33.get('pool')}  "
          f"recipe={d33['recipe']}  seeds={d33['seeds']}")
    print(f"bound {b:.4f}   train {d33['n_train']}   test {d33['n_test']}")
    print(f"(Sc.31 for reference: bound {d31['bound_const']:.4f}, train {d31['n_train']})")
    print("=" * 80)
    print(f"\n  {'setting':<13}{'median':>9}{'vs bound':>10}{'sd':>8}{'top5':>8}{'coll':>8}")
    for k, v in S.items():
        print(f"  {k:<13}{v['top1_median']:>9.4f}{v['top1_median']-b:>+10.4f}"
              f"{v['top1_sd']:>8.4f}{v['top5_mean']:>8.4f}"
              f"{v['n_collapsed']:>6}/{v['n_seeds']}")

    out = {"scenario": "Scenario33", "pool": d33.get("pool"), "seeds": d33["seeds"],
           "bound_const": b, "criteria": {}}
    # ---- R1 ----
    m = S[FULL]["top1_median"]; g = m - b
    r1 = bool(g > 0.05)
    print(f"\nR1  median top-1 at full resources exceeds the bound by more than 0.05")
    print(f"    {FULL}: median {m:.4f}, bound {b:.4f}, gain {g:+.4f}")
    print(f"    -> {'PASS' if r1 else 'FAIL'}"
          f"{'' if r1 else f' (short of the threshold by {0.05-g:.4f})'}")
    out["criteria"]["R1"] = {"gain": float(g), "threshold": 0.05, "pass": r1}
    # ---- R2 ----
    o, lo, hi, pv = boot_diff(S["chirps=64"]["top1_seeds"], S["chirps=8"]["top1_seeds"])
    r2 = bool(lo > 0)
    print(f"\nR2  the 8 -> 64 chirp step has a bootstrap 95% CI excluding zero")
    print(f"    median difference {o:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  permutation p = {pv:.4f}")
    print(f"    -> {'PASS' if r2 else 'FAIL'}")
    out["criteria"]["R2"] = {"delta": float(o), "ci": [float(lo), float(hi)],
                             "p": float(pv), "pass": r2}
    # ---- R3 retired ----
    print(f"\nR3  RETIRED, NOT SCORED. Utilization is an affine image of the gain over the")
    print(f"    bound, so this criterion is satisfied automatically and tests nothing.")
    print(f"    See audit_utilization_definitional.py. Scoring it would inflate the verdict.")
    out["criteria"]["R3"] = {"retired": True,
                             "reason": "utilization = gain + near-constant bracket; "
                                       "see audit_utilization_definitional.py"}
    # ---- the qualitative dose-response, reported regardless ----
    ch = sorted([v for v in S.values() if v["sweep"] == "chirps"], key=lambda v: v["n_chirps"])
    med = [v["top1_median"] for v in ch]; nn = [v["n_chirps"] for v in ch]
    coll = [v["n_collapsed"] for v in ch]
    sp = stats.spearmanr(nn, med)
    o2, lo2, hi2, p2 = boot_diff(ch[-1]["top1_seeds"], ch[0]["top1_seeds"])
    print(f"\nDOSE-RESPONSE, reported whatever the criteria say")
    print(f"    medians {' -> '.join(f'{x:.4f}' for x in med)}")
    print(f"    Spearman(chirps, median) = {sp.statistic:+.4f}, p = {sp.pvalue:.4f}")
    print(f"    8 -> 250: {o2:+.4f}  CI [{lo2:+.4f}, {hi2:+.4f}]  p = {p2:.4f}")
    print(f"    settings clearing the bound: "
          f"{[v['n_chirps'] for v in ch if v['top1_median'] > b]}")
    print(f"    collapse count falls with resources: {coll}")
    out["dose_response"] = {"n_chirps": nn, "median": med, "collapsed": coll,
                            "spearman": [float(sp.statistic), float(sp.pvalue)],
                            "full_range": {"delta": float(o2), "ci": [float(lo2), float(hi2)],
                                           "p": float(p2)},
                            "clears_bound_at": [v["n_chirps"] for v in ch
                                                if v["top1_median"] > b]}
    npass = sum(1 for k in ("R1", "R2") if out["criteria"][k].get("pass"))
    verdict = ("REPLICATION" if npass == 2 else
               "PARTIAL REPLICATION" if npass == 1 else "NON-REPLICATION")
    out["verdict"] = verdict
    print(f"\n{'='*80}\nVERDICT: {verdict} ({npass}/2 scored criteria passed; R3 retired)")
    if verdict == "PARTIAL REPLICATION":
        print("Report it as a partial. The qualitative dose-response carries over; the")
        print("pre-registered magnitude threshold does not. Sc.33 has 43% less training data")
        print(f"({d33['n_train']} vs {d31['n_train']}) and a higher bound "
              f"({b:.4f} vs {d31['bound_const']:.4f}), so a smaller gain is expected, but the")
        print("threshold was fixed in advance and is not renegotiated after the fact.")
    print("=" * 80)
    q = os.path.join(RES, f"sc33_replication_score{sfx}.json")
    json.dump(out, open(q, "w"), indent=2)
    print(f"\nSaved: {q}")


if __name__ == "__main__":
    main()
