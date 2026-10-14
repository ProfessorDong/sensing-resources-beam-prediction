#!/usr/bin/env python
"""Is 'measurement utilization' on the beam task a relabelling of the gain over the bound?

Utilization is defined here as real_arm - shuffled_arm. The gain is real_arm - bound. So
    utilization = gain + (bound - shuffled)
and if the shuffled arm sits at a FIXED offset below the bound across settings, the bracket is
a constant, utilization is an affine image of the gain, and correlating the two measures
nothing. The earlier Paper B headline, Pearson r = +0.97 between utilization and the gain over
ten settings, was exactly that.

This script quantifies the bracket. A small spread in (bound - shuffled) relative to the spread
in gain means the correlation is definitional and must not be reported.

Usage: python audit_utilization_definitional.py [results/sc31_resource_ablation_G*.json ...]
"""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def check(path):
    d = json.load(open(path)); S, b = d["settings"], d["bound_const"]
    k = list(S)
    shuf = np.array([S[x]["shuffled_top1"] for x in k])
    real = np.array([S[x]["top1_mean"] for x in k])
    util = np.array([S[x]["utilization"] for x in k])
    gain = real - b
    off = b - shuf                                   # the bracket
    print(f"\n{'='*76}\n{os.path.basename(path)}   pool={d.get('pool','decimate')}   "
          f"seeds={d['seeds']}   bound={b:.4f}\n{'='*76}")
    print(f"{'setting':<13}{'real':>8}{'shuffled':>10}{'gain':>9}{'util':>9}{'bound-shuf':>12}")
    for i, x in enumerate(k):
        print(f"{x:<13}{real[i]:>8.4f}{shuf[i]:>10.4f}{gain[i]:>+9.4f}{util[i]:>+9.4f}"
              f"{off[i]:>+12.4f}")
    r = stats.pearsonr(util, gain)
    print(f"\n  spread of the bracket (bound - shuffled): mean {off.mean():+.4f}, "
          f"s.d. {off.std(ddof=1):.4f}, range {off.max()-off.min():.4f}")
    print(f"  spread of the gain:                       s.d. {gain.std(ddof=1):.4f}, "
          f"range {gain.max()-gain.min():.4f}")
    ratio = off.std(ddof=1) / gain.std(ddof=1)
    print(f"  bracket s.d. / gain s.d. = {ratio:.3f}")
    print(f"  utilization vs gain: Pearson r = {r.statistic:+.4f}  p = {r.pvalue:.2e}")
    # residual of utilization after removing the gain: what is left that is not the gain?
    resid = util - np.polyval(np.polyfit(gain, util, 1), gain)
    print(f"  s.d. of utilization after removing its linear dependence on the gain: "
          f"{resid.std(ddof=1):.4f}")
    verdict = ("DEFINITIONAL: the bracket is near-constant, so utilization is an affine image "
               "of\n      the gain. Do NOT report the correlation."
               if ratio < 0.25 else
               "The bracket varies enough that utilization carries information beyond the\n"
               "      gain. Report it only with this ratio stated.")
    print(f"  -> {verdict}")
    return {"file": os.path.basename(path), "pool": d.get("pool", "decimate"),
            "bracket_mean": float(off.mean()), "bracket_sd": float(off.std(ddof=1)),
            "gain_sd": float(gain.std(ddof=1)), "ratio": float(ratio),
            "pearson_r": float(r.statistic), "pearson_p": float(r.pvalue),
            "residual_sd": float(resid.std(ddof=1)),
            "definitional": bool(ratio < 0.25)}


def main():
    args = sys.argv[1:] or [os.path.join(RES, "sc31_resource_ablation_G.json")]
    out = [check(a if os.path.isabs(a) else os.path.join(HERE, a))
           for a in args if os.path.exists(a if os.path.isabs(a) else os.path.join(HERE, a))]
    p = os.path.join(RES, "utilization_definitional_audit.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
