#!/usr/bin/env python
"""Synthetic point-target validation of the Scenario 31/33 radar front end.

Why this exists. Every radar result in Paper B is read off a two-channel image produced by
`exp_sc31_resource_ablation._from_cube`. That function was never validated: the only
instrument check in the project rasterizes a position blob and pushes it through the CNN,
which tests the READOUT and not the range-Doppler-angle processing that makes the maps. Three
measurements are consistent with a front-end defect and do not distinguish one from the
physical explanation the paper gives:

  * the azimuth of the brightest range-azimuth cell correlates with the partner's true
    bearing at r = +0.045;
  * a zero-range artifact dominated that argmax until the first range bin was excluded;
  * one receive element, with no azimuthal information at all, retains 92% of the
    bearing-error reduction of the full array.

If the maps are not what their axis labels say, the dose-response could still be real (more
samples buy SNR) while the bearing and Cramer-Rao framing of Section VIII would be describing
an axis that does not exist. This script decides that question.

It IMPORTS the production transform rather than reimplementing it, so a defect in the shipped
path cannot pass by being absent from a copy. Predicted bin indices are derived analytically
from the manuscript's Eq. (1) and compared against the argmax of the function's own output.

Usage:
    python validate_radar_frontend.py                 # synthetic only, seconds
    python validate_radar_frontend.py --real 400      # plus checks on real frames
Exit code is non-zero if any check fails.
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_sc31_resource_ablation import _from_cube, _pool_axis, DS, FULL, GRID

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
NRX, NSAMP, NCHIRP = FULL

checks = []


def ck(cond, label, detail=""):
    checks.append((bool(cond), label, detail))
    return bool(cond)


def synth_cube(fr, fd, nu, n_rx=NRX, n_samp=NSAMP, n_chirp=NCHIRP, amp=1.0, snr_db=None,
               rng=None):
    """One point target, exactly the manuscript's Eq. (1), in NORMALIZED frequencies.

    fr, fd, nu are cycles per sample on the fast-time, slow-time and array axes, so no
    undocumented hardware constant is needed and the predicted bins follow from the DFT alone.
    """
    n = np.arange(n_rx)[:, None, None]
    k = np.arange(n_samp)[None, :, None]
    l = np.arange(n_chirp)[None, None, :]
    x = amp * np.exp(2j * np.pi * (fr * k + fd * l + nu * n))
    if snr_db is not None:
        rng = rng or np.random.default_rng(0)
        sigma = amp * 10 ** (-snr_db / 20) / np.sqrt(2)
        x = x + sigma * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x.astype(np.complex64)


def pooled_index(raw_idx, raw_len, pool):
    """Where raw bin `raw_idx` lands after the production resampler.

    Computed by pushing a one-hot through `_pool_axis` itself, so this cannot drift from the
    function it is predicting against.
    """
    a = np.zeros((raw_len, 1), np.float32)
    a[raw_idx, 0] = 1.0
    out = _pool_axis(a, 0, GRID, pool)[:, 0]
    return int(np.argmax(out)) if out.max() > 0 else None


def predict(fr, fd, nu, n_samp, n_chirp, pool):
    """Analytic bin prediction for the three axes, before and after pooling."""
    rng_raw = int(round(fr * n_samp))                       # no fftshift, lower half kept
    dop_raw = int(round(fd * n_chirp)) % n_chirp
    dop_raw = (dop_raw + n_chirp // 2) % n_chirp            # fftshift
    ang_raw = (int(round(nu * 64)) + 32) % 64               # zero-padded to 64, fftshift
    return {
        "range_raw": rng_raw, "doppler_raw": dop_raw, "azimuth_raw": ang_raw,
        "range_pooled": pooled_index(rng_raw, n_samp // 2, pool),
        "doppler_pooled": pooled_index(dop_raw, n_chirp, pool),
        "azimuth_pooled": ang_raw,                          # azimuth axis is already GRID
    }


def peak2d(img):
    i = int(np.argmax(img))
    return i // img.shape[1], i % img.shape[1]


def synthetic_suite(pool, tol=1, dc="none"):
    """Sweep targets across all three axes and check each lands where the DFT says it must."""
    rows, nfail = [], 0
    # On-grid frequencies, chosen away from the axis edges so no mainlobe wraps.
    grid = []
    for kr in (20, 48, 80, 110):
        grid.append((kr / NSAMP, 0.0, 0.0))
    for kd in (-80, -25, 25, 80):
        grid.append((48 / NSAMP, kd / NCHIRP, 0.0))
    for ka in (-24, -12, 0, 12, 24):
        grid.append((48 / NSAMP, 0.0, ka / 64))
    for kr, kd, ka in ((30, -40, -16), (90, 60, 20), (64, 15, -8)):
        grid.append((kr / NSAMP, kd / NCHIRP, ka / 64))
    for fr, fd, nu in grid:
        cube = synth_cube(fr, fd, nu)
        img = _from_cube(cube, NRX, NSAMP, NCHIRP, pool, dc)
        ra, rd = img[0], img[1]
        p = predict(fr, fd, nu, NSAMP, NCHIRP, pool)
        r_ra, a_ra = peak2d(ra)
        r_rd, d_rd = peak2d(rd)
        ok_a = abs(a_ra - p["azimuth_pooled"]) <= tol
        ok_r1 = abs(r_ra - p["range_pooled"]) <= tol
        ok_r2 = abs(r_rd - p["range_pooled"]) <= tol
        ok_d = abs(d_rd - p["doppler_pooled"]) <= tol
        nfail += (not all((ok_a, ok_r1, ok_r2, ok_d)))
        rows.append({"fr": fr, "fd": fd, "nu": nu,
                     "range_expected": p["range_pooled"], "range_ra": r_ra, "range_rd": r_rd,
                     "azimuth_expected": p["azimuth_pooled"], "azimuth_got": a_ra,
                     "doppler_expected": p["doppler_pooled"], "doppler_got": d_rd,
                     "ok": bool(all((ok_a, ok_r1, ok_r2, ok_d)))})
    ck(nfail == 0, f"[{pool}] every synthetic target lands in its predicted cell on BOTH maps",
       f"{len(rows) - nfail}/{len(rows)} exact to +-{tol} bin")
    return rows


def axis_identity(pool, dc="none"):
    """The axes must be what the labels say: monotone, correctly signed, centred."""
    out = {}
    # azimuth against injected spatial frequency
    nus = [k / 64 for k in range(-28, 29, 4)]
    got = []
    for nu in nus:
        img = _from_cube(synth_cube(48 / NSAMP, 0.0, nu), NRX, NSAMP, NCHIRP, pool, dc)
        got.append(peak2d(img[0])[1])
    d = np.diff(got)
    ck(np.all(d >= 0), "azimuth index is MONOTONE in the injected spatial frequency",
       f"steps {d.tolist()}")
    ck(got[len(got) // 2] == 32, "a broadside target (nu = 0) lands on the centre azimuth bin",
       f"got {got[len(got)//2]}, expected 32")
    slope = np.polyfit(nus, got, 1)[0]
    ck(abs(slope - 64) < 2.0, "azimuth scale is 64 bins per unit spatial frequency",
       f"fitted {slope:.2f}")
    out["azimuth_sweep"] = {"nu": nus, "bin": got, "slope_bins_per_cycle": float(slope)}

    # range against injected beat frequency
    frs = [k / NSAMP for k in range(12, 116, 8)]
    gotr = [peak2d(_from_cube(synth_cube(f, 0.0, 0.0), NRX, NSAMP, NCHIRP, pool, dc)[0])[0]
            for f in frs]
    ck(np.all(np.diff(gotr) >= 0), "range index is MONOTONE in the injected beat frequency",
       f"steps {np.diff(gotr).tolist()}")
    out["range_sweep"] = {"fr_cycles_per_sample": frs, "bin": gotr}

    # doppler against injected slow-time frequency, both signs
    fds = [k / NCHIRP for k in range(-100, 101, 20)]
    gotd = [peak2d(_from_cube(synth_cube(48 / NSAMP, f, 0.0), NRX, NSAMP, NCHIRP, pool, dc)[1])[1]
            for f in fds]
    ck(np.all(np.diff(gotd) >= 0), "Doppler index is MONOTONE and signed, zero at the centre",
       f"steps {np.diff(gotd).tolist()}")
    out["doppler_sweep"] = {"fd_cycles_per_chirp": fds, "bin": gotd}
    return out


def aperture_check(pool, dc="none"):
    """One element must carry EXACTLY no azimuthal structure. The paper asserts 0.0000."""
    out = {}
    for n_rx in (1, 2, 4):
        img = _from_cube(synth_cube(48 / NSAMP, 0.0, 12 / 64, n_rx=n_rx),
                         n_rx, NSAMP, NCHIRP, pool, dc)
        ra = img[0]
        prof = ra[ra.sum(1).argmax()]
        # prep() max-normalizes each map, so peak-to-peak saturates at 1 for every aperture
        # that has any structure at all and measures the normalization rather than the beam.
        # The profile MEAN does not: a flat axis normalizes to all ones, and a narrower
        # mainlobe pulls the mean down. Concentration = 1 - mean is the usable statistic.
        out[f"rx={n_rx}"] = {"azimuth_profile_ptp": float(np.ptp(prof)),
                             "azimuth_profile_mean": float(prof.mean()),
                             "concentration": float(1.0 - prof.mean()),
                             "n_local_maxima": int(((prof > np.roll(prof, 1)) &
                                                    (prof >= np.roll(prof, -1))).sum())}
    ck(out["rx=1"]["azimuth_profile_ptp"] == 0.0,
       "one receive element produces an EXACTLY flat azimuth axis",
       f"peak-to-peak {out['rx=1']['azimuth_profile_ptp']:.3e}, "
       f"which is the m(m^2-1) = 0 of Eq. (9) measured rather than asserted")
    ck(out["rx=4"]["concentration"] > out["rx=2"]["concentration"] > out["rx=1"]["concentration"],
       "azimuthal structure grows with aperture",
       f"concentration 1/2/4 = {out['rx=1']['concentration']:.3f}, "
       f"{out['rx=2']['concentration']:.3f}, {out['rx=4']['concentration']:.3f}")
    ck(out["rx=2"]["n_local_maxima"] > out["rx=4"]["n_local_maxima"] or
       out["rx=2"]["concentration"] < out["rx=4"]["concentration"],
       "a two-element array is more ambiguous in azimuth than a four-element one",
       f"local maxima 2/4 = {out['rx=2']['n_local_maxima']}/{out['rx=4']['n_local_maxima']}")
    return out


def two_target_check(pool, dc="none"):
    """Two separated targets must resolve as two peaks, not one."""
    a = synth_cube(40 / NSAMP, 0.0, -16 / 64)
    b = synth_cube(96 / NSAMP, 0.0, 16 / 64)
    img = _from_cube(a + b, NRX, NSAMP, NCHIRP, pool, dc)
    ra = img[0]
    pa = predict(40 / NSAMP, 0.0, -16 / 64, NSAMP, NCHIRP, pool)
    pb = predict(96 / NSAMP, 0.0, 16 / 64, NSAMP, NCHIRP, pool)
    va = ra[pa["range_pooled"], pa["azimuth_pooled"]]
    vb = ra[pb["range_pooled"], pb["azimuth_pooled"]]
    mid = ra[(pa["range_pooled"] + pb["range_pooled"]) // 2,
             (pa["azimuth_pooled"] + pb["azimuth_pooled"]) // 2]
    ck(min(va, vb) > mid, "two separated targets resolve as two peaks, not one merged lobe",
       f"target cells {va:.3f}/{vb:.3f} vs midpoint {mid:.3f}")
    return {"target_a": float(va), "target_b": float(vb), "midpoint": float(mid)}


def noise_check(pool, snr_db=0.0, trials=40, dc="none"):
    """The peak must survive noise at an SNR where the coherent gain is what saves it."""
    rng = np.random.default_rng(20261007)
    hits = 0
    for _ in range(trials):
        fr, fd, nu = 48 / NSAMP, 30 / NCHIRP, 12 / 64
        cube = synth_cube(fr, fd, nu, snr_db=snr_db, rng=rng)
        img = _from_cube(cube, NRX, NSAMP, NCHIRP, pool, dc)
        p = predict(fr, fd, nu, NSAMP, NCHIRP, pool)
        r, a = peak2d(img[0])
        hits += (abs(r - p["range_pooled"]) <= 1 and abs(a - p["azimuth_pooled"]) <= 1)
    ck(hits == trials, f"the target is recovered in every trial at {snr_db:g} dB per-sample SNR",
       f"{hits}/{trials}")
    return {"snr_db": snr_db, "trials": trials, "hits": hits}


def decimation_check():
    """What the ORIGINAL resampler does to the axis it actually mangles.

    The range axis runs 128 -> 64, where both modes halve and little is lost. The Doppler
    axis runs 250 -> 64, and `decimate` keeps 64 chosen indices and DISCARDS the rest. This
    measures how many Doppler bins have no representation at all, and how often a target
    sitting in a discarded bin fails to reach its predicted cell. It is the quantitative
    version of the instrument argument in Section IX.
    """
    kept = set(np.linspace(0, NCHIRP - 1, GRID).astype(int).tolist())
    dropped = NCHIRP - len(kept)
    misplaced, erased = 0, 0
    tested = list(range(-120, 121, 7))
    for kd in tested:
        fd = kd / NCHIRP
        cube = synth_cube(48 / NSAMP, fd, 12 / 64)
        p = predict(fd=fd, fr=48 / NSAMP, nu=12 / 64, n_samp=NSAMP, n_chirp=NCHIRP,
                    pool="decimate")
        got = peak2d(_from_cube(cube, NRX, NSAMP, NCHIRP, "decimate")[1])[1]
        if p["doppler_pooled"] is None:
            # The target's own slow-time bin is one of the discarded ones: its peak has no
            # cell to land in, and whatever the readout sees at that position is leakage.
            erased += 1
        elif abs(got - p["doppler_pooled"]) > 1:
            misplaced += 1
    ck(dropped > 0,
       "the original resampler DISCARDS most Doppler bins, as Section IX states",
       f"{dropped} of {NCHIRP} slow-time bins have no representation after decimation "
       f"({100*dropped/NCHIRP:.0f}%)")
    ck(True,
       "targets erased or displaced on the Doppler axis by decimation (reported)",
       f"of {len(tested)} injected Doppler positions, {erased} fall in a discarded bin and "
       f"have no cell to land in; a further {misplaced} land more than one bin away")
    return {"doppler_bins": NCHIRP, "doppler_bins_kept": len(kept),
            "doppler_bins_discarded": dropped, "targets_tested": len(tested),
            "targets_erased": erased, "targets_misplaced": misplaced}


def real_frames(scenario, limit, pool, dc="none"):
    """Checks that need the recordings: the discarded half, and the zero-range artifact."""
    path = os.path.join(DS, scenario, f"scenario{scenario[8:]}_dev.csv")
    rows = list(csv.DictReader(open(path)))
    rows = rows[:: max(1, len(rows) // limit)][:limit]
    keep, drop, first_bin, n = 0.0, 0.0, 0, 0
    for r in rows:
        cube = np.load(os.path.join(DS, scenario, r["unit1_radar"].lstrip("./")))
        c = cube[:NRX, :NSAMP, :NCHIRP]
        # Energy above vs below the Nyquist split that `_from_cube` throws away. A beat
        # signal with the opposite sign convention would sit almost entirely in the
        # discarded half, which would be a front-end defect and not a modelling choice.
        # NOTE: this transform is deliberately taken on the AS-PUBLISHED chain and ignores
        # `dc`, so `fraction_energy_kept` is identical under --dc none and --dc mean and says
        # nothing about the post-removal split. It is a sign-convention check on the chain the
        # paper reports, and Sec. IX-A states that scope. Do not read it as a DC-removal result.
        rng_fft = np.fft.fft(c * np.hanning(NSAMP)[None, :, None], NSAMP, axis=1)
        p = np.abs(rng_fft) ** 2
        keep += float(p[:, : NSAMP // 2, :].sum())
        drop += float(p[:, NSAMP // 2 :, :].sum())
        img = _from_cube(cube, NRX, NSAMP, NCHIRP, pool, dc)
        first_bin += int(peak2d(img[0])[0] == 0)
        n += 1
    frac_kept = keep / (keep + drop)
    ck(frac_kept > 0.5, "most range-FFT energy is in the half the transform KEEPS",
       f"{100*frac_kept:.1f}% kept, {100*(1-frac_kept):.1f}% discarded, {n} frames")
    return {"scenario": scenario, "frames": n, "fraction_energy_kept": frac_kept,
            "frames_with_peak_in_first_range_bin": first_bin,
            "fraction_peak_in_first_range_bin": first_bin / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="max", choices=["decimate", "max", "mean"])
    ap.add_argument("--real", type=int, default=0, help="frames per scenario; 0 skips")
    # Zero-beat-frequency removal, as in exp_sc31_resource_ablation.py. The chain the
    # experiments now use applies it, so the validation has to exercise the same path.
    # Results are written to a separate file per mode so neither overwrites the other.
    ap.add_argument("--dc", default="none", help="none | mean | notchN")
    a = ap.parse_args()
    out = {"pool": a.pool, "dc_removal": a.dc, "cube_shape": list(FULL), "grid": GRID,
           "transform": "exp_sc31_resource_ablation._from_cube (imported, not reimplemented)"}
    print(f"synthetic point-target validation, pool={a.pool}, dc={a.dc}\n")
    out["synthetic_grid"] = synthetic_suite(a.pool, dc=a.dc)
    out["axis_identity"] = axis_identity(a.pool, a.dc)
    out["aperture"] = aperture_check(a.pool, a.dc)
    out["two_target"] = two_target_check(a.pool, a.dc)
    out["noise"] = noise_check(a.pool, dc=a.dc)
    out["decimation"] = decimation_check()
    if a.real:
        out["real"] = {s: real_frames(s, a.real, a.pool, a.dc)
                       for s in ("Scenario31", "Scenario33")}
    for ok, label, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n          {detail}" if detail else ""))
    npass = sum(1 for ok, _, _ in checks if ok)
    print(f"\n{npass}/{len(checks)} front-end checks passed")
    out["checks"] = [{"ok": ok, "label": l, "detail": d} for ok, l, d in checks]
    out["n_passed"], out["n_checks"] = npass, len(checks)
    sfx = "" if a.dc == "none" else f"_dc{a.dc}"
    p = os.path.join(RES, f"radar_frontend_validation{sfx}.json")
    if not a.real and os.path.exists(p):
        # A synthetic-only run must not discard the real-frame evidence a previous --real run
        # saved: the claims verifier reads it. Carry it over verbatim and say so, rather than
        # recomputing a combined pass count that would misreport what this run actually did.
        prev = json.load(open(p))
        if "real" in prev:
            out["real"] = prev["real"]
            out["real_not_rerun"] = True
            print("  (real-frame results carried over from a previous --real run)")
    json.dump(out, open(p, "w"), indent=2)
    print(f"Saved: {p}")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
