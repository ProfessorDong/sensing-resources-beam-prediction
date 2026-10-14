#!/usr/bin/env python
"""Alignment-cost model for radar-aided beam selection, rebuilt after the 2026-10-06 audit.

WHAT THE PREVIOUS MODEL GOT WRONG
---------------------------------
It charged radar acquisition against the communication link's airtime. On DeepSense Testbed 5,
which collects Scenarios 31-34, unit 1 carries a 60 GHz receiver with a 16-element array and a
64-beam combining codebook, and SEPARATELY an FMCW radar at 76-81 GHz; unit 2 transmits with a
60 GHz quasi-omni antenna. The radar and the link are different radios in different bands, so
radar time is not communication airtime. It is sensor latency on a parallel chain.

It also concluded that reuse of an already collected frame "delivers no saving", which
contradicts its own equation: with the frame sunk, the expected overhead falls to
k*T_sec + (1-p_k)*N_B*T_sec, well below an exhaustive sweep.

And it compared only against an exhaustive sweep, the weakest possible policy, rather than the
deployable no-sensing policy, which is to probe the k most frequent beams in training.

WHAT THIS MODEL DOES
--------------------
Beam probing occupies the 60 GHz link, so it is charged as communication airtime:

    A(k, p_k) = k*T_sec + (1 - p_k) * (N_B - k) * T_sec        (retained measurements)
    A_restart(k, p_k) = k*T_sec + (1 - p_k) * N_B * T_sec      (full restart)

Radar acquisition and inference occupy the sensor chain, so they are charged as latency:

    L = N_c * T_chirp + C_proc

and the three deployment regimes differ only in which of these the decision must wait for:

    reuse       the frame is already available; the link pays A, the decision waits C_proc
    concurrent  radar runs in parallel; the link pays A, the decision waits max(0, L - slack)
    dedicated   the radar is run on demand before alignment; the decision waits L, then pays A

Nothing here is a hardware measurement. T_sec, T_chirp and C_proc are stated parameters, and the
outputs are break-even conditions in those parameters rather than verdicts about real hardware.

Output: results/alignment_cost_model.json
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
T_SEC = 5.0          # us per probed beam pair, a modelling assumption borrowed from
                     # Va et al., IEEE TVT 67(5), 2018, Sec. VI, not a mandated 802.11ad value
N_BEAM = 64
N_ADC = 256


def airtime(k, pk, restart):
    return k * T_SEC + (1 - pk) * (N_BEAM if restart else N_BEAM - k) * T_SEC


def main():
    out = {"T_sec_us": T_SEC, "n_beam": N_BEAM, "n_adc": N_ADC,
           "T_sec_provenance": "modelling assumption borrowed from Va et al., IEEE TVT 2018, "
                               "Sec. VI; not a mandated 802.11ad per-pair duration",
           "band_note": "radar 76-81 GHz, link 60 GHz, separate radios on DeepSense Testbed 5; "
                        "radar time is sensor latency, not communication airtime",
           "scenarios": {}}
    for sc in ("31", "33"):
        A = json.load(open(os.path.join(RES, f"sc{sc}_resource_ablation_G_max.json")))
        R = json.load(open(os.path.join(RES, f"sc{sc}_reference_predictors.json")))["predictors"]
        S = A["settings"]
        pk_prior = R["train-prior top-5"]["top5"]
        pk_gps = R["GPS reference (knn 20, val-selected)"]["top5"]
        pk_pers = R["persistence (temporally eligible)"]["top5"]
        rows = []
        for c in (8, 16, 32, 64, 125, 250):
            v = S[f"chirps={c}"]
            pk = v["top5_mean"]
            rows.append({"n_chirps": c, "top5": pk, "top1_median": v["top1_median"],
                         "airtime_retained_us": airtime(5, pk, False),
                         "airtime_restart_us": airtime(5, pk, True)})
        best = rows[-1]
        a_prior = airtime(5, pk_prior, False)
        a_exh = N_BEAM * T_SEC
        a_radar = best["airtime_retained_us"]
        # the budget a sensing-aided predictor must fit inside to beat each policy
        out["scenarios"][sc] = {
            "per_setting": rows,
            "p5_prior": pk_prior, "p5_gps": pk_gps, "p5_persistence": pk_pers,
            "airtime_exhaustive_us": a_exh,
            "airtime_prior_us": a_prior,
            "airtime_radar_full_us": a_radar,
            "airtime_gps_us": airtime(5, pk_gps, False),
            "airtime_persistence_us": airtime(5, pk_pers, False),
            # REUSE regime: frame already collected, only inference latency is incremental
            "reuse_budget_vs_prior_us": a_prior - a_radar,
            "reuse_budget_vs_exhaustive_us": a_exh - a_radar,
            # DEDICATED regime: the decision additionally waits for the acquisition
            "dedicated_breakeven_T_chirp_vs_prior_us":
                (a_prior - a_radar) / best["n_chirps"],
            "dedicated_breakeven_T_chirp_vs_exhaustive_us":
                (a_exh - a_radar) / best["n_chirps"],
        }
        # does any setting beat the prior under dedicated acquisition at a given T_chirp?
        grid = {}
        for tc in (2.56, 5.12, 10.24, 25.6, 51.2):
            per = []
            for r in rows:
                tot = r["n_chirps"] * tc + r["airtime_retained_us"]
                per.append({"n_chirps": r["n_chirps"], "total_us": tot,
                            "beats_prior": bool(tot < a_prior),
                            "beats_exhaustive": bool(tot < a_exh)})
            grid[f"{tc}"] = per
        out["scenarios"][sc]["dedicated_latency_plus_airtime"] = grid
        # Break-even as a function of PREDICTOR QUALITY, not only of frame length. At short
        # frames the binding constraint is accuracy and not airtime, which "dedicated sensing
        # never pays" hides: a perfect shortlist breaks even at a reachable chirp duration
        # there, while at the full frame the airtime term dominates whatever the accuracy.
        qual = []
        for r in rows:
            N = r["n_chirps"]
            q = {"n_chirps": N}
            for name, p5 in (("measured", r["top5"]), ("position", pk_gps), ("perfect", 1.0)):
                tc = (a_prior - airtime(5, p5, False)) / N
                q[f"breakeven_Tc_{name}_us"] = tc
                q[f"adc_{name}_msps"] = out["n_adc"] / tc if tc > 0 else float("inf")
            qual.append(q)
        out["scenarios"][sc]["dedicated_breakeven_by_predictor_quality"] = {
            "reference_strategy": "fixed training prior", "reference_cost_us": a_prior,
            "k": 5, "p5_position_reference": pk_gps, "rows": qual}
        print(f"=== Scenario {sc} ===")
        print(f"  no-sensing prior (k=5)      p5={pk_prior:.4f}  airtime {a_prior:7.2f} us")
        print(f"  radar, full frame (k=5)     p5={best['top5']:.4f}  airtime {a_radar:7.2f} us")
        print(f"  persistence (k=5)           p5={pk_pers:.4f}  airtime "
              f"{airtime(5,pk_pers,False):7.2f} us")
        print(f"  GPS reference (k=5)         p5={pk_gps:.4f}  airtime "
              f"{airtime(5,pk_gps,False):7.2f} us")
        print(f"  exhaustive sweep                      airtime {a_exh:7.2f} us")
        print(f"  --> REUSE: radar saves {a_prior-a_radar:.2f} us vs the prior and "
              f"{a_exh-a_radar:.2f} us vs exhaustive,")
        print(f"      provided incremental processing latency stays below that budget.")
        print(f"  --> DEDICATED: beating the prior needs T_chirp < "
              f"{(a_prior-a_radar)/best['n_chirps']:.3f} us at 250 chirps.")
        for tc in ("10.24", "25.6"):
            wins = [p["n_chirps"] for p in grid[tc] if p["beats_prior"]]
            winx = [p["n_chirps"] for p in grid[tc] if p["beats_exhaustive"]]
            print(f"      at T_chirp={tc} us: beats prior at {wins or 'no setting'}; "
                  f"beats exhaustive at {winx or 'no setting'}")
        print()
    p = os.path.join(RES, "alignment_cost_model.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
