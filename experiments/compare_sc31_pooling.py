#!/usr/bin/env python
"""Which Sc.31 verdicts depend on the readout's resampling operator?

The original readout DECIMATED both axes to 64 bins (`np.linspace(0,n-1,64).astype(int)`),
discarding every bin in between. That caps the Doppler axis at 64 distinct bins for chirps >= 64
and throws away half the range bins at 256 samples, so a setting could not benefit from
resolution it paid for. `--pool max` block-reduces instead and discards nothing. Below 64 bins
both modes replicate nearest-neighbour, so the two runs differ in exactly one factor.

This compares them setting by setting and, more importantly, re-tests every verdict that the
paper would state, so a conclusion that only holds under one operator is caught here rather
than by a reviewer.
"""
import itertools, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
rng = np.random.default_rng(20261004)

PAIRS = [("chirps=8", "chirps=64"), ("chirps=64", "chirps=125"),
         ("chirps=125", "chirps=250"), ("chirps=8", "chirps=250"),
         ("samples=64", "samples=128"), ("samples=128", "chirps=250"),
         ("rx=1", "rx=2"), ("rx=2", "chirps=250")]
ISO = [("chirps=125", "samples=128"), ("chirps=64", "samples=64")]


def perm(a, b, stat=np.median, n=20000):
    a, b = np.asarray(a), np.asarray(b)
    obs = stat(a) - stat(b); pool = np.concatenate([a, b]); c = 0
    for _ in range(n):
        p = rng.permutation(pool)
        c += abs(stat(p[:a.size]) - stat(p[a.size:])) >= abs(obs) - 1e-12
    bs = np.array([stat(rng.choice(a, a.size, True)) - stat(rng.choice(b, b.size, True))
                   for _ in range(n)])
    return obs, np.percentile(bs, 2.5), np.percentile(bs, 97.5), (c + 1) / (n + 1)


def load(sfx):
    p = os.path.join(RES, f"sc31_resource_ablation_G{sfx}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    D, M = load(""), load("_max")
    if M is None:
        print("pool=max results not present yet: results/sc31_resource_ablation_G_max.json")
        return
    assert abs(D["bound_const"] - M["bound_const"]) < 1e-12, "bounds differ; not comparable"
    b = D["bound_const"]
    print("=" * 84)
    print(f"READOUT RESAMPLING: decimate vs block-max.  bound {b:.4f} (identical, labels only)")
    print(f"seeds: decimate {D['seeds']}, max {M['seeds']}")
    print("=" * 84)
    print(f"\n{'setting':<13}{'decimate':>10}{'max':>9}{'delta':>9}{'p':>9}"
          f"{'coll d':>8}{'coll m':>8}")
    rows = []
    for k in D["settings"]:
        if k not in M["settings"]: continue
        a, c = M["settings"][k]["top1_seeds"], D["settings"][k]["top1_seeds"]
        o, lo, hi, p = perm(a, c)
        rows.append({"setting": k, "decimate": D["settings"][k]["top1_median"],
                     "max": M["settings"][k]["top1_median"], "delta": float(o),
                     "ci": [float(lo), float(hi)], "p": float(p),
                     "coll_decimate": D["settings"][k]["n_collapsed"],
                     "coll_max": M["settings"][k]["n_collapsed"]})
        print(f"{k:<13}{rows[-1]['decimate']:>10.4f}{rows[-1]['max']:>9.4f}"
              f"{o:>+9.4f}{p:>9.4f}"
              f"{rows[-1]['coll_decimate']:>6}/{D['seeds']}"
              f"{rows[-1]['coll_max']:>6}/{M['seeds']}")
    imp = [r for r in rows if r["delta"] > 0 and r["p"] < 0.05]
    print(f"\n  block-max beats decimate significantly in {len(imp)}/{len(rows)} settings; "
          f"mean change {np.mean([r['delta'] for r in rows]):+.4f}")

    print("\n" + "=" * 84)
    print("DO THE VERDICTS SURVIVE? Each comparison re-tested under both operators.")
    print("=" * 84)
    print(f"\n  {'comparison':<28}{'decimate':>22}{'max':>22}{'same?':>8}")
    verd = []
    for lo_k, hi_k in PAIRS + ISO:
        out = {}
        for tag, d in (("decimate", D), ("max", M)):
            if lo_k not in d["settings"] or hi_k not in d["settings"]:
                out[tag] = None; continue
            o, l, h, p = perm(d["settings"][hi_k]["top1_seeds"],
                              d["settings"][lo_k]["top1_seeds"])
            sign = "rise" if l > 0 else ("fall" if h < 0 else "flat")
            out[tag] = {"delta": float(o), "ci": [float(l), float(h)], "p": float(p),
                        "verdict": sign}
        same = (out["decimate"] and out["max"]
                and out["decimate"]["verdict"] == out["max"]["verdict"])
        lbl = f"{lo_k} -> {hi_k}"
        f = lambda z: f"{z['delta']:+.4f} {z['verdict']:<4} p={z['p']:.3f}" if z else "n/a"
        print(f"  {lbl:<28}{f(out['decimate']):>22}{f(out['max']):>22}"
              f"{('yes' if same else 'NO'):>8}")
        verd.append({"comparison": lbl, "iso_payload": (lo_k, hi_k) in ISO, **out,
                     "verdict_unchanged": bool(same)})
    flipped = [v for v in verd if not v["verdict_unchanged"]]
    print(f"\n  {len(verd)-len(flipped)}/{len(verd)} verdicts unchanged.")
    if flipped:
        print("  POOLING-DEPENDENT, must be reported as such or resolved:")
        for v in flipped:
            print(f"    {v['comparison']}: decimate {v['decimate']['verdict']}, "
                  f"max {v['max']['verdict']}")
    p = os.path.join(RES, "sc31_pooling_comparison.json")
    json.dump({"bound": b, "seeds": {"decimate": D["seeds"], "max": M["seeds"]},
               "per_setting": rows, "verdicts": verd,
               "n_verdicts_unchanged": len(verd) - len(flipped),
               "n_verdicts": len(verd)}, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
