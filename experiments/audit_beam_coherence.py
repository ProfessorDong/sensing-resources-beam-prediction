#!/usr/bin/env python
"""How stale is a radar-aided beam prediction, in units of the beam coherence time?

Section VII previously said the latency term could not be priced, because it depends on a
vehicle speed and a beam coherence time "the dataset does not let us vary". That was too
pessimistic. Va, Choi and Heath (IEEE TVT 66(6), 2017) define the beam coherence time as the
average time over which a beam stays aligned, and for the line-of-sight case derive

    T_B = D * acos(theta^2 * log(zeta) + 1) / (v * sin(mu_r))                        (their 50)

with D the link distance, v the receiver speed, mu_r the angle between the direction of travel
and the beam, theta the beamwidth parameter of a von Mises pattern (k_r = 1/theta^2) and zeta
the tolerated fraction of peak receive power.

TWO ADAPTATIONS, both stated rather than hidden.

1. Va's receiver moves and carries the beam. Here unit 1 is STATIONARY and carries the beam
   while unit 2 moves. The derivation is purely geometric: the beam goes stale because the
   line of sight rotates. So the quantity that matters is the LOS angular rate, and
   D/(v sin mu_r) is simply the reciprocal of that rate.

2. We therefore do not estimate v and mu_r at all. The LOS bearing from unit 1 to unit 2 is
   recorded per frame, so d(bearing)/dt is MEASURED directly from consecutive frames. This
   removes two estimated quantities and the vehicle-heading assumption with them.

The tolerated pointing error acos(theta^2 log zeta + 1) is kept exactly as Va defines it.
theta comes from the deployed array: a uniform linear array of m elements at half-wavelength
spacing has a half-power beamwidth of about 0.886 * lambda / (m d) radians at broadside, and a
von Mises pattern matches a Gaussian of standard deviation theta whose half-power full width is
2 * sqrt(2 log 2) * theta, so theta = HPBW / 2.355.

Output: results/beam_coherence.json
"""
import csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DS = os.path.join(HERE, "..", "datasets", "deepsense6g")
ZETA = 0.5          # Va's own choice: the beam is "misaligned" at the half-power point


def hpbw_rad(m, d_over_lambda=0.5):
    """Half-power beamwidth of an m-element ULA at broadside, radians."""
    return 0.886 / (m * d_over_lambda)


def main():
    TB = json.load(open(os.path.join(RES, "testbed_facts.json")))
    m = TB["n_array_elements"]
    hp = hpbw_rad(m)
    theta = hp / (2 * np.sqrt(2 * np.log(2)))
    tol = float(np.arccos(theta ** 2 * np.log(ZETA) + 1))     # radians of pointing error
    out = {"zeta": ZETA, "n_elements": m,
           "hpbw_deg": float(np.degrees(hp)), "theta_deg": float(np.degrees(theta)),
           "tolerated_pointing_error_deg": float(np.degrees(tol)),
           "source": "Va, Choi and Heath, IEEE TVT 66(6):5014-5029, 2017, eq. (50)",
           "adaptation": "unit 1 is stationary and carries the beam, so D/(v sin mu_r) is the "
                         "reciprocal of the measured LOS angular rate; no heading is assumed",
           "scenarios": {}}
    print(f"array {m} elements -> HPBW {np.degrees(hp):.2f} deg, theta {np.degrees(theta):.2f} deg")
    print(f"tolerated pointing error at zeta={ZETA}: {np.degrees(tol):.3f} deg")
    print(f"(check: a Gaussian pattern gives 1.177*theta = {np.degrees(1.1774*theta):.3f} deg)\n")

    for sc in ("31", "33"):
        rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}",
                                                     f"scenario{sc}_dev.csv"))))
        lab = np.load(os.path.join(RES, f"sc{sc}_bearing_labels.npz"))
        brg = np.radians(lab["bearing_deg"].astype(float))
        rng_ = lab["range_m"].astype(float)
        seq = np.array([int(r["seq_index"]) for r in rows])
        idx = np.array([int(r["index"]) for r in rows])
        t = []
        for r in rows:
            hms, frac = r["time_stamp"].split("-")
            h, mi, s = (int(x) for x in hms.split(":"))
            t.append((h * 3600 + mi * 60 + s) + int(frac) / 1e6)
        t = np.array(t)
        # Only consecutive frames of the same recording, as in the persistence eligibility rule.
        dt = np.diff(t)
        ok = (seq[1:] == seq[:-1]) & (idx[1:] == idx[:-1] + 1) & (dt > 0) & (dt <= 0.2)
        dphi = np.abs(np.diff(brg))[ok]
        dt_ok = dt[ok]
        rate = dphi / dt_ok                                   # rad/s of LOS rotation
        pos = rate > 0
        rate, dt_pos = rate[pos], dt_ok[pos]
        tb = tol / rate                                       # seconds
        period = float(np.median(dt_ok))
        q = np.percentile(rate, [5, 50, 95])
        tq = np.percentile(tb, [5, 50, 95])
        out["scenarios"][sc] = {
            "n_eligible_pairs": int(ok.sum()), "n_rate_positive": int(len(rate)),
            "median_frame_period_s": period,
            "los_rate_deg_per_s": {"p5": float(np.degrees(q[0])),
                                   "median": float(np.degrees(q[1])),
                                   "p95": float(np.degrees(q[2]))},
            "beam_coherence_s": {"p5": float(tq[0]), "median": float(tq[1]),
                                 "p95": float(tq[2])},
            "beam_coherence_median_ms": float(1000 * np.median(tb)),
            "ratio_TB_over_frame_period_median": float(np.median(tb) / period),
            "frac_pairs_TB_below_frame_period": float(np.mean(tb < period)),
            # The line above compares each pair's ideal pointing time against the MEDIAN
            # period, not against that pair's own interval. The two differ, and a statistic
            # has to be named by the denominator it used.
            "frac_pairs_TB_below_own_interval": float(np.mean(tb < dt_pos)),
            "median_range_m": float(np.median(rng_)),
            # NOT A FRESHNESS VERDICT. A 3 dB pointing criterion is not top-1 label
            # persistence, the calculation starts from a correctly centred beam, and a
            # logging period does not bound decision age. This quantity is an ideal
            # line-of-sight pointing-time reference and nothing more; whether a prediction
            # survives being used later is measured directly in
            # audit_beam_power_regret.py, which evaluates saved shortlists at later frames.
            "scope": ("ideal LOS pointing time under a 3 dB criterion and an ideal 16-element "
                      "broadside pattern; not a measurement of prediction freshness"),
        }
        v = out["scenarios"][sc]
        print(f"=== Scenario {sc} ===")
        print(f"  eligible consecutive pairs      {v['n_eligible_pairs']}")
        print(f"  median frame period             {1000*period:.1f} ms")
        print(f"  LOS angular rate (deg/s)        p5 {v['los_rate_deg_per_s']['p5']:.2f}, "
              f"median {v['los_rate_deg_per_s']['median']:.2f}, "
              f"p95 {v['los_rate_deg_per_s']['p95']:.2f}")
        print(f"  beam coherence time             p5 {1000*v['beam_coherence_s']['p5']:.1f} ms, "
              f"median {v['beam_coherence_median_ms']:.1f} ms, "
              f"p95 {1000*v['beam_coherence_s']['p95']:.1f} ms")
        print(f"  T_B / frame period (median)     {v['ratio_TB_over_frame_period_median']:.2f}")
        print(f"  pairs with T_B < the median period   "
              f"{100*v['frac_pairs_TB_below_frame_period']:.2f}%")
        print(f"  pairs with T_B < their own interval  "
              f"{100*v['frac_pairs_TB_below_own_interval']:.2f}%")
        print(f"  --> {v['scope']}\n")
    p = os.path.join(RES, "beam_coherence.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
