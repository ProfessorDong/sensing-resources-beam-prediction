#!/usr/bin/env python
"""Does radar-aided beam prediction pay for its own airtime?

The beam-prediction literature reports accuracy and stops there. Accuracy is not the
quantity a link budget cares about: the radar frame costs airtime, and the thing it
replaces, exhaustive beam training, also costs airtime. This script does the accounting
in one unit, microseconds, using only measured numbers plus one sourced constant.

Sourced constant
  T_sec = 5.0 us per trained beam pair. Va, Choi, Shimizu, Bansal and Heath, "Inverse
  multipath fingerprinting for millimeter wave V2I beam alignment", IEEE Trans. Veh.
  Technol. 67(5):4042-4058, 2018, Sec. VI: the 802.11ad SSW frame is 26.8 us (4.3 us
  preamble + 22.5 us header and information) and with the standard's 32x spreading for
  quasi-omni patterns the per-pair sector training cost is 4.3 + 22.5/32 = 5.0 us.

Measured here
  top-1 and top-5 accuracy per chirp count, Sc.31, 15 seeds (sc31_resource_ablation_*.json)
  the no-information bound, i.e. the best constant predictor on the test split
  the radar frame period, from the dataset timestamps

Deliberately NOT assumed
  the chirp duration. DeepSense ships no radar configuration, so instead of inventing one
  the script solves for the BREAK-EVEN chirp duration and converts it to the ADC sample
  rate a 256-sample chirp would need to achieve it. That turns an assumption into a
  falsifiable statement about hardware.
"""
import csv, datetime as dt, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DS = os.path.join(HERE, "..", "datasets", "deepsense6g", "Scenario31")
T_SEC = 5.0          # us per trained beam pair, Va et al. 2018 Sec. VI
N_BEAM = 64          # unit1_beam codebook size on Sc.31
N_ADC = 256          # ADC samples per chirp in the DeepSense cube
ADC_MIN, ADC_MAX = 10.0, 25.0   # typical automotive FMCW front-end sample rates


def frame_period_us():
    rows = list(csv.DictReader(open(os.path.join(DS, "scenario31_dev.csv"))))
    t = []
    for r in rows:
        s = r["time_stamp"]
        hms, frac = s.split("-")
        h, m, sec = (int(x) for x in hms.split(":"))
        t.append((h * 3600 + m * 60 + sec) * 1e6 + int(frac))
    d = np.diff(np.array(t))
    d = d[(d > 0) & (d < 1e6)]                 # drop sequence boundaries
    return float(np.median(d)), len(rows), float(np.percentile(d, 5)), float(np.percentile(d, 95))


def main():
    pool = sys.argv[1] if len(sys.argv) > 1 else "decimate"
    sfx = "" if pool == "decimate" else f"_{pool}"
    path = os.path.join(RES, f"sc31_resource_ablation_G{sfx}.json")
    d = json.load(open(path))
    S, bound = d["settings"], d["bound_const"]
    Tf, nfr, lo, hi = frame_period_us()
    print(f"source: {os.path.basename(path)}   pool={d.get('pool','decimate')}   "
          f"seeds={d['seeds']}")
    print(f"radar frame period, measured from {nfr} timestamps: {Tf/1000:.1f} ms "
          f"(5-95 pct {lo/1000:.1f}-{hi/1000:.1f} ms)")
    print(f"exhaustive beam training over {N_BEAM} beams: "
          f"{N_BEAM*T_SEC:.0f} us at T_sec = {T_SEC} us/pair\n")

    ch = sorted([v for v in S.values() if v["sweep"] == "chirps"], key=lambda v: v["n_chirps"])
    print("BREAK-EVEN ANALYSIS. A predictor proposing the top-k beams trains k pairs instead")
    print("of 64, but when the best beam is NOT in the top k the link must still fall back to")
    print("the full sweep. Ignoring that fallback understates the chirp budget, so the honest")
    print("break-even solves the SAME equation the expected-overhead model uses:")
    print("    N*Tc + k*T_sec + (1-p_k)*N_BEAM*T_sec = N_BEAM*T_sec")
    print("    =>  Tc = T_sec * (N_BEAM*p_k - k) / N")
    print("The optimistic form, Tc = (N_BEAM-k)*T_sec/N, is reported alongside only to show how")
    print("much it flatters the result. The ADC column is the rate a 256-sample chirp needs.\n")
    print(f"{'chirps':>7}{'top1':>8}{'top5':>8}{'vs bound':>10}"
          f"{'Tc opt':>9}{'ADC opt':>9}{'Tc hon':>9}{'ADC hon':>9}{'reachable':>11}")
    rows = []
    for v in ch:
        N, p1, p5 = v["n_chirps"], v["top1_median"], v["top5_mean"]
        t_opt = (N_BEAM - 5) * T_SEC / N
        t_hon = T_SEC * (N_BEAM * p5 - 5) / N
        a_opt, a_hon = N_ADC / t_opt, N_ADC / t_hon
        reach = a_hon <= ADC_MAX
        rows.append({"n_chirps": N, "top1_median": p1, "top5_mean": p5,
                     "vs_bound": p1 - bound,
                     "breakeven_T_chirp_us_optimistic": t_opt,
                     "breakeven_adc_msps_optimistic": a_opt,
                     "breakeven_T_chirp_us": t_hon,
                     "breakeven_adc_msps": a_hon,
                     "reachable_by_automotive_adc": bool(reach)})
        print(f"{N:>7}{p1:>8.4f}{p5:>8.4f}{p1-bound:>+10.4f}"
              f"{t_opt:>9.2f}{a_opt:>9.1f}{t_hon:>9.2f}{a_hon:>9.1f}"
              f"{('YES' if reach else 'no'):>11}")
    und = [r['breakeven_adc_msps'] / r['breakeven_adc_msps_optimistic'] for r in rows]
    print(f"\n  Ignoring the fallback understates the required ADC rate by "
          f"{min(und):.1f}x to {max(und):.1f}x.")

    kopt = int(np.argmax([r["top1_median"] for r in rows]))
    ropt = rows[kopt]
    print(f"\nAutomotive FMCW front ends sample at roughly 10-25 Msps, so a 256-sample chirp")
    print(f"lasts about {N_ADC/25:.0f}-{N_ADC/10:.0f} us of active time, before any idle time.")
    reach = [r for r in rows if r["reachable_by_automotive_adc"]]
    print(f"At the accuracy optimum ({ropt['n_chirps']} chirps) break-even needs "
          f"{ropt['breakeven_adc_msps']:.0f} Msps, which is "
          f"{ropt['breakeven_adc_msps']/ADC_MAX:.1f}x to "
          f"{ropt['breakeven_adc_msps']/ADC_MIN:.1f}x that range.")
    if reach:
        r0 = reach[0]
        print(f"The only setting within reach is {r0['n_chirps']} chirps "
              f"({r0['breakeven_adc_msps']:.1f} Msps), and it sits {r0['vs_bound']:+.4f} over "
              f"the no-information bound.\n")
    else:
        print("No setting is within reach of an automotive front end.\n")

    print("EXPECTED-OVERHEAD MODEL, including the fallback. If the true best beam is not in")
    print("the proposed top-k the link must fall back to the exhaustive sweep, so")
    print("  E[overhead] = N*T_chirp + k*T_sec + (1 - p_k)*N_BEAM*T_sec.")
    print("Evaluated at two plausible chirp durations, k = 5:\n")
    best = {}
    for Tc in (25.6, 51.2):
        print(f"  T_chirp = {Tc} us  (ADC {N_ADC/Tc:.0f} Msps)")
        print(f"  {'chirps':>7}{'sensing':>10}{'train':>8}{'fallback':>10}{'E[total]':>11}"
              f"{'vs exhaustive':>15}")
        tab = []
        for r in rows:
            N, p5 = r["n_chirps"], r["top5_mean"]
            sens, tr = N * Tc, 5 * T_SEC
            fb = (1 - p5) * N_BEAM * T_SEC
            tot = sens + tr + fb
            tab.append((N, tot))
            print(f"  {N:>7}{sens:>10.0f}{tr:>8.0f}{fb:>10.0f}{tot:>11.0f}"
                  f"{tot - N_BEAM*T_SEC:>+14.0f} us")
        bn, bt = min(tab, key=lambda x: x[1])
        best[Tc] = {"best_n_chirps": bn, "best_total_us": bt,
                    "exhaustive_us": N_BEAM * T_SEC, "pays_for_itself": bool(bt < N_BEAM * T_SEC)}
        print(f"  -> cheapest sensing-aided option is {bn} chirps at {bt:.0f} us; exhaustive "
              f"is {N_BEAM*T_SEC:.0f} us")
        print(f"  -> dedicated radar {'PAYS' if bt < N_BEAM*T_SEC else 'DOES NOT PAY'} "
              f"for its own airtime\n")

    print("AMORTIZED CASE. In an ISAC system the radar frame is collected for the sensing")
    print("function regardless, so beam prediction pays only the INCREMENTAL airtime. Whether")
    print("any airtime can be handed back depends on where accuracy peaks:")
    acc = [r["top1_median"] for r in rows]
    k = int(np.argmax(acc))
    print(f"  top-1 median peaks at {rows[k]['n_chirps']} chirps ({acc[k]:.4f}); "
          f"the full frame is {rows[-1]['n_chirps']} chirps ({acc[-1]:.4f})")
    frac = 1 - rows[k]["n_chirps"] / rows[-1]["n_chirps"]
    print(f"  airtime recoverable at no measured cost in beam accuracy: "
          f"{frac*100:.0f}% of the radar frame")
    if frac <= 0:
        print("  -> accuracy peaks at the FULL frame, so there is NO saturation to exploit and")
        print("     no airtime can be handed back. The only surviving positive statement is")
        print("     that beam prediction is free as a byproduct of a frame collected in full;")
        print("     it justifies neither extra airtime nor a reduction.")
    else:
        print(f"  -> accuracy peaks before the full frame, so that fraction is free to reclaim.")
    out = {"source": os.path.basename(path), "pool": d.get("pool", "decimate"),
           "T_sec_us": T_SEC, "T_sec_source":
           "Va et al., IEEE TVT 67(5):4042-4058, 2018, Sec. VI (802.11ad SSW 26.8 us, 32x spreading)",
           "n_beam": N_BEAM, "n_adc_samples": N_ADC, "bound_const": bound,
           "frame_period_us": Tf, "frame_period_p5_p95_us": [lo, hi], "n_frames": nfr,
           "exhaustive_us": N_BEAM * T_SEC, "per_setting": rows,
           "expected_overhead": best,
           "saturation_n_chirps": rows[k]["n_chirps"],
           "full_frame_n_chirps": rows[-1]["n_chirps"],
           "recoverable_airtime_frac": 1 - rows[k]["n_chirps"] / rows[-1]["n_chirps"]}
    p = os.path.join(RES, f"isac_airtime_budget{sfx}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
