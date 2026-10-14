"""Degeneracy screen for radar-aided beam prediction on DeepSense Scenarios 31 and 33.

WHY THIS SCENARIO PAIR
----------------------
Paper B needs a communication-side ISAC result. The Sc.36 attempt failed, and the literature
tells us why it was the wrong instance: Demirhan and Alkhateeb reach ~90% top-5 on DeepSense
Scenario 9, where unit 1 is STATIONARY and the codebook has 64 beams. Sc.36 has a mobile unit
1 and a 256-beam label pooled over four arrays with different boresights.

Scenarios 31 and 33 are the well-posed family and are already on disk:
  * `unit1_loc` is a SINGLE file for every frame, so unit 1 is stationary; unit 2 is mobile
  * `unit1_beam` is an explicit best-beam label over a 64-beam codebook
  * `unit1_radar` is a raw FMCW cube, (4 Rx, 256 ADC samples, 250 chirps), complex64
  * modal-beam share is 0.0669 on Sc.31, so there is real headroom above chance

THE SCREEN (this project's standard, applied before any design work)
  bound       best constant on the TEST split. Not the train-modal floor: that is the error
              that produced the retracted Sc.36 result.
  gps_oracle  PREMISE. unit 1 is fixed, so the bearing to unit 2 should determine the beam.
              If this does not clear the bound the task is not well posed and we stop.
  radar       is the task solvable from the sensing measurement at all?
  power       CIRCULARITY CHECK ONLY. `unit1_beam` is the argmax of `unit1_pwr_60ghz`
              (verified here), so power is never an input to a reported result.
  shuffled    radar from another frame. Separates per-frame evidence from marginal statistics.

Splits are contiguous temporal blocks with a gap, per scenario, never random: consecutive
frames are near-duplicates.

Output: results/sc31_beam_screen.json
"""
import os, sys, csv, json, collections, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_beam_prediction import train_readout

DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "deepsense6g")
GRID, N_AZ = 64, 64
GAP_FRAC, TRAIN_FRAC = 0.02, 0.70


def radar_to_ra(path, n_rx=None, n_samp=None, n_chirp=None):
    """Range-angle and range-Doppler maps from the raw cube, with optional resource limits.

    Angle FFT is applied coherently BEFORE magnitude and Doppler is integrated incoherently.
    Doing it the other way makes the static and dynamic parts cancel; this project has hit
    that defect twice.
    """
    cube = np.load(path)                                   # (4, 256, 250) complex64
    if n_rx:    cube = cube[:n_rx]
    if n_samp:  cube = cube[:, :n_samp]
    if n_chirp: cube = cube[:, :, :n_chirp]
    nr, ns, nc = cube.shape
    rng = np.fft.fft(cube * np.hanning(ns)[None, :, None], ns, axis=1)
    rd = np.fft.fftshift(np.fft.fft(rng * np.hanning(nc)[None, None, :], nc, axis=2), axes=2)
    rd = rd[:, : ns // 2, :]
    ang = np.fft.fftshift(np.fft.fft(rd, N_AZ, axis=0), axes=0)
    mag = np.abs(ang).astype(np.float32)
    ra = mag.sum(axis=2).T                                  # (range, azimuth)
    rdm = mag.sum(axis=0)                                   # (range, doppler)
    def prep(a):
        a = np.log1p(np.maximum(a, 0))
        i = np.linspace(0, a.shape[0] - 1, GRID).astype(int)
        j = np.linspace(0, a.shape[1] - 1, GRID).astype(int)
        a = a[np.ix_(i, j)]
        m = a.max()
        return (a / m if m > 0 else a).astype(np.float32)
    return np.stack([prep(ra), prep(rdm)], 0)


def sanitize(a, name):
    """DeepSense's mmWave power files contain occasional NaN, 79 of 7012 frames on Sc.31.

    The NaN sits at the PEAK beam (argmax matches the published label on those frames too),
    so it is a saturation artifact, not a dropout. One NaN row propagates through the readout
    and turns every weight to NaN, which is why the circular instrument check scored exactly
    0.0000 instead of ~1.0 and made the whole screen uninterpretable.

    Replace each NaN with that row's finite maximum, which preserves the argmax and therefore
    the label, then assert the array is finite. Nothing may reach a readout unchecked.
    """
    a = np.asarray(a, np.float32)
    bad = ~np.isfinite(a)
    n_bad_rows = int(bad.reshape(len(a), -1).any(1).sum())
    if bad.any():
        flat = a.reshape(len(a), -1)
        for i in np.where(bad.reshape(len(a), -1).any(1))[0]:
            row = flat[i]
            finite = row[np.isfinite(row)]
            row[~np.isfinite(row)] = finite.max() if finite.size else 0.0
        a = flat.reshape(a.shape)
    assert np.isfinite(a).all(), f"{name} still not finite"
    if n_bad_rows:
        print(f"    sanitized {name}: {n_bad_rows}/{len(a)} rows had non-finite values")
    return a, n_bad_rows


def load_scenario(sc, limit=None):
    dev = os.path.join(DS, sc, f"scenario{sc[8:]}_dev.csv")
    rows = list(csv.DictReader(open(dev)))
    if limit:
        rows = rows[:: max(1, len(rows) // limit)][:limit]
    beams = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1      # to 0-based
    u1 = np.loadtxt(os.path.join(DS, sc, rows[0]["unit1_loc"].lstrip("./")))
    gps, pwr, meas = [], [], []
    for k, r in enumerate(rows):
        p2 = np.loadtxt(os.path.join(DS, sc, r["unit2_loc"].lstrip("./")))
        dlat, dlon = p2[0] - u1[0], p2[1] - u1[1]
        north = dlat * 111320.0
        east = dlon * 111320.0 * np.cos(np.radians(u1[0]))
        gps.append([north, east, np.hypot(north, east), np.arctan2(east, north)])
        pwr.append(np.loadtxt(os.path.join(DS, sc, r["unit1_pwr_60ghz"].lstrip("./"))))
        meas.append(radar_to_ra(os.path.join(DS, sc, r["unit1_radar"].lstrip("./"))))
        if k % 500 == 0:
            print(f"    {sc} {k}/{len(rows)}", flush=True)
    gps, nb_g = sanitize(np.asarray(gps, np.float32), f"{sc} gps")
    pwr, nb_p = sanitize(np.asarray(pwr, np.float32), f"{sc} power")
    meas, nb_m = sanitize(np.stack(meas), f"{sc} radar")
    return gps, pwr, meas, beams, {"gps": nb_g, "power": nb_p, "radar": nb_m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="subsample frames for a fast screen")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {"config": vars(a)}

    for sc in ("Scenario31", "Scenario33"):
        print(f"\n=== {sc} ===")
        cache = os.path.join(RESULTS_DIR, f"cache_{sc}_{a.limit or 'full'}.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            gps, pwr, meas, beams = z["gps"], z["pwr"], z["meas"], z["beams"]
            nonfinite = dict(z["nonfinite"].item())
            print(f"  loaded cache {os.path.basename(cache)}")
        else:
            gps, pwr, meas, beams, nonfinite = load_scenario(sc, a.limit or None)
            np.savez_compressed(cache, gps=gps, pwr=pwr, meas=meas, beams=beams,
                                nonfinite=nonfinite)
        n = len(beams)
        circ = float((pwr.argmax(1) == beams).mean())
        ntr = int(TRAIN_FRAC * n); gap = int(GAP_FRAC * n)
        tr = slice(0, ntr); te = slice(ntr + gap, n)
        ytr, yte = beams[tr], beams[te]
        bmb, bn = collections.Counter(yte.tolist()).most_common(1)[0]
        bound = bn / len(yte)
        tmb = collections.Counter(ytr.tolist()).most_common(1)[0][0]
        floor = float((yte == tmb).mean())
        print(f"  frames {n}  train {ntr}  gap {gap}  test {len(yte)}")
        print(f"  label = argmax(power): match {circ:.4f}  (circular, power is never an input)")
        print(f"  train-modal floor {floor:.4f}   NO-INFORMATION BOUND {bound:.4f} (class {bmb})")

        rng = np.random.RandomState(0)
        perm = rng.permutation(len(yte))
        inputs = {"gps_oracle": (gps[tr], gps[te]),
                  "radar": (meas[tr], meas[te]),
                  "radar_shuffled": (meas[tr], meas[te][perm]),
                  "power_CIRCULAR": (pwr[tr], pwr[te])}
        rec = {"n": n, "n_train": int(ntr), "n_test": int(len(yte)),
               "nonfinite_rows_sanitized": nonfinite,
               "label_is_argmax_power": circ, "train_modal_floor": floor,
               "bound_const": bound, "bound_class": int(bmb), "results": {}}
        print(f"  {'input':<16}{'top1':>8}{'top5':>8}{'vs bound':>10}")
        for name, (Xtr, Xte) in inputs.items():
            m, _ = train_readout(torch.from_numpy(np.ascontiguousarray(Xtr)).float(),
                                 torch.from_numpy(ytr).long(),
                                 torch.from_numpy(np.ascontiguousarray(Xte)).float(),
                                 torch.from_numpy(yte).long(),
                                 64, 42, a.epochs, dev, False)
            rec["results"][name] = {**{k: float(v) for k, v in m.items()},
                                    "vs_bound": float(m["top1"] - bound)}
            print(f"  {name:<16}{m['top1']:>8.4f}{m.get('top5', float('nan')):>8.4f}"
                  f"{m['top1']-bound:>+10.4f}", flush=True)
        instr = rec["results"]["power_CIRCULAR"]["top1"]
        rec["instrument_valid"] = bool(instr > 0.80)
        print(f"  INSTRUMENT power_CIRCULAR {instr:.4f} -> "
              f"{'VALID' if rec['instrument_valid'] else 'INVALID, no row below may be read'}")
        g = rec["results"]["gps_oracle"]["vs_bound"]
        r = rec["results"]["radar"]["vs_bound"]
        rec["premise_holds"] = bool(g > 0.05)
        rec["radar_informative"] = bool(r > 0.05)
        print(f"  PREMISE  gps_oracle {g:+.4f} -> "
              f"{'beam IS determined by unit2 position' if g > 0.05 else 'PREMISE FAILS'}")
        print(f"  SENSING  radar      {r:+.4f} -> "
              f"{'radar predicts the beam' if r > 0.05 else 'radar does NOT predict the beam'}")
        out[sc] = rec

    ok = [s for s in ("Scenario31", "Scenario33")
          if out[s]["instrument_valid"] and out[s]["premise_holds"]
          and out[s]["radar_informative"]]
    out["usable_scenarios"] = ok
    print(f"\n{'='*70}\nUSABLE FOR PAPER B: {ok if ok else 'NONE'}")
    p = os.path.join(RESULTS_DIR, "sc31_beam_screen.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
