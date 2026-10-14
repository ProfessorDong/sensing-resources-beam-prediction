#!/usr/bin/env python
"""Does removing the zero-beat-frequency term repair the range-azimuth map?

`validate_radar_frontend.py` established that the front end is correct and, in passing, that
the brightest cell of the Scenario 31 range-azimuth map lies in the FIRST RANGE BIN in 100.0%
of frames (Scenario 33: 4.3%). That bin is where transmit leakage and direct antenna coupling
land, because both arrive at essentially zero delay and so at zero beat frequency. The chain as
published applies no leakage removal, so on Scenario 31 the map's peak is not a scatterer and
its max-normalization is set by a quantity unrelated to the scene.

This script measures, per removal mode, on systematically sampled real frames:

  * where the range-azimuth argmax lands, raw and after pooling to the readout grid
  * how much of the range-transform energy sits in the first two bins
  * whether the peak's AZIMUTH then tracks unit 2's true bearing

The last is the one that matters. A repaired peak should carry bearing; the published peak
correlates with the true bearing at only +0.0454 (Sc.31) and +0.0216 (Sc.33).

Output: results/dc_removal_audit.json
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_sc31_resource_ablation import _from_cube, _pool_axis, DS, FULL, GRID

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODES = ["none", "mean", "notch1", "notch2", "notch3", "notch4"]


def ra_raw(cube, dc):
    """The range-azimuth magnitude map BEFORE pooling, so raw bin indices are visible."""
    c = cube[: FULL[0], : FULL[1], : FULL[2]]
    nr, ns, nc = c.shape
    w = np.hanning(ns)[None, :, None]
    if dc != "none":
        c = c - (c * w).sum(axis=1, keepdims=True) / w.sum()
    rng = np.fft.fft(c * w, ns, axis=1)
    e_all = float((np.abs(rng[:, : ns // 2, :]) ** 2).sum())
    e_lo = float((np.abs(rng[:, :2, :]) ** 2).sum())
    if dc.startswith("notch"):
        rng[:, : int(dc[5:]), :] = 0
    rd = np.fft.fftshift(np.fft.fft(rng * np.hanning(nc)[None, None, :], nc, axis=2), axes=2)
    rd = rd[:, : ns // 2, :]
    ang = np.fft.fftshift(np.fft.fft(rd, 64, axis=0), axes=0)
    mag = np.abs(ang).astype(np.float32)
    return mag.sum(axis=2).T, e_lo, e_all      # (range, azimuth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    a = ap.parse_args()
    out = {"n_frames_requested": a.frames, "modes": MODES,
           "note": "frames systematically strided across the whole recording, never head-truncated",
           "scenarios": {}}
    for sc in ("31", "33"):
        rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}",
                                                     f"scenario{sc}_dev.csv"))))
        lab = np.load(os.path.join(RES, f"sc{sc}_bearing_labels.npz"))
        brg = lab["bearing_deg"].astype(float)
        step = max(1, len(rows) // a.frames)
        sel = list(range(0, len(rows), step))[: a.frames]
        print(f"=== Scenario {sc}: {len(sel)} of {len(rows)} frames, stride {step} ===")
        acc = {m: {"rbin": [], "pbin": [], "az": [], "e_lo": [], "e_all": []} for m in MODES}
        for j, i in enumerate(sel):
            cube = np.load(os.path.join(DS, f"Scenario{sc}",
                                        rows[i]["unit1_radar"].lstrip("./")))
            for m in MODES:
                M, e_lo, e_all = ra_raw(cube, m)
                r, z = np.unravel_index(M.argmax(), M.shape)
                P = _pool_axis(_pool_axis(np.log1p(np.maximum(M, 0)), 0, GRID, "max"),
                               1, GRID, "max")
                pr, _ = np.unravel_index(P.argmax(), P.shape)
                acc[m]["rbin"].append(int(r)); acc[m]["pbin"].append(int(pr))
                acc[m]["az"].append(int(z))
                acc[m]["e_lo"].append(e_lo); acc[m]["e_all"].append(e_all)
            if j % 100 == 0:
                print(f"    {j}/{len(sel)}", flush=True)
        b = brg[sel]
        res = {}
        print(f"  {'mode':<9}{'argmax in':>11}{'argmax in':>11}{'E in':>9}"
              f"{'peak az vs bearing':>22}")
        print(f"  {'':<9}{'raw bin 0':>11}{'pooled b0':>11}{'bins 0-1':>9}"
              f"{'Pearson':>11}{'p':>11}")
        for m in MODES:
            d = acc[m]
            az = np.array(d["az"], float)
            r = stats.pearsonr(az, b)
            sp = stats.spearmanr(az, b)
            res[m] = {
                "frac_argmax_raw_bin0": float(np.mean(np.array(d["rbin"]) == 0)),
                "frac_argmax_raw_first2": float(np.mean(np.array(d["rbin"]) < 2)),
                "frac_argmax_pooled_bin0": float(np.mean(np.array(d["pbin"]) == 0)),
                "median_argmax_raw_bin": float(np.median(d["rbin"])),
                "energy_frac_first2_range_bins": float(np.sum(d["e_lo"]) / np.sum(d["e_all"])),
                "peak_azimuth_vs_bearing_pearson": float(r.statistic),
                "peak_azimuth_vs_bearing_p": float(r.pvalue),
                "peak_azimuth_vs_bearing_spearman": float(sp.statistic),
                "distinct_peak_azimuths": int(len(set(d["az"]))),
            }
            v = res[m]
            print(f"  {m:<9}{v['frac_argmax_raw_bin0']:>11.3f}"
                  f"{v['frac_argmax_pooled_bin0']:>11.3f}"
                  f"{v['energy_frac_first2_range_bins']:>9.3f}"
                  f"{v['peak_azimuth_vs_bearing_pearson']:>+11.4f}"
                  f"{v['peak_azimuth_vs_bearing_p']:>11.2e}"
                  f"   {v['distinct_peak_azimuths']:>3} distinct azimuths")
        out["scenarios"][sc] = {"n_frames": len(sel), "stride": step, "modes": res}
        print()
    p = os.path.join(RES, "dc_removal_audit.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
