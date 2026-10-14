"""Paper B: does measurement utilization govern the marginal value of sensing resources?

THE CLAIM AND WHY THIS IS THE RIGHT EXPERIMENT
----------------------------------------------
Paper B's thesis is that utilization governs what additional sensing buys. Testing it needs a
communication-side quantity, which the reconstruction benchmarks do not have. DeepSense
Scenario 31 does: unit 1 is stationary, the codebook has 64 beams, `unit1_beam` is an explicit
best-beam label, and `unit1_radar` is a raw FMCW cube (4 Rx, 256 ADC samples, 250 chirps) whose
resources can be cut at will.

Three knobs, and one of them is the ISAC quantity:

  ADC samples   range resolution        bandwidth
  Rx antennas   angular resolution      aperture
  CHIRPS        Doppler and integration AIRTIME SPENT SENSING RATHER THAN COMMUNICATING

Chirps are the axis Xiong et al. formalize as a tradeoff, so the chirp sweep is the headline
and the other two are secondary.

Reported per setting, against the best constant on the TEST split, never the train-modal floor:
  top-1                 beam accuracy
  top-5                 the operational quantity: search 5 of 64 beams instead of all 64
  utilization           real minus shuffled radar, the part that depends on THIS frame

PROTOCOL
  * contiguous temporal split with a gap, never random
  * `ImgReadoutSpatial`, whose image path is validated on this scenario: a rasterized position
    image clears the bound by +0.4170 through it (`readout_spatial_audit.json`)
  * 3 seeds per setting for the real arm; the shuffled control is a control, so 1 seed
  * every radar cube is loaded ONCE and all settings are derived in that pass, because disk
    I/O dominates the FFT cost

REPLICATION CRITERION, fixed 2026-10-02 BEFORE running Scenario 33
-----------------------------------------------------------------
Scenario 31 gave: radar clears the no-information bound at full resources; the chirp sweep
rises over 8 -> 64 with a bootstrap CI on the median excluding zero; utilization correlates
with the median gain across settings. Scenario 33 is called a replication only if all three
hold there too:

  R1  median top-1 at full resources exceeds the bound by more than 0.05
  R2  the 8 -> 64 chirp step has a bootstrap 95% CI on the median difference excluding zero
  R3  utilization against median gain over the bound is positive with p < 0.05

Anything else is a non-replication and must be reported as one. Writing this down first is the
point: the Sc.33 radar number that currently exists (-0.0143) came from the POOLED readout,
before that architecture was found to under-read distributed inputs by up to 0.07, so there is
a live temptation to accept whichever answer the new run gives.

Output: results/<scenario>_resource_ablation.json
"""
import os, sys, csv, json, collections, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_beam_prediction import train_readout
import exp_beam_prediction as EBP
from audit_readout_spatial import ImgReadoutSpatial
from audit_collapse_fix import train as train_fixed
from screen_sc31_beam import sanitize, GRID

DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "deepsense6g")
TRAIN_FRAC, GAP_FRAC = 0.70, 0.02
FULL = (4, 256, 250)

# (n_rx, n_samp, n_chirp), label, which sweep it belongs to
SETTINGS = [((4, 256, c), f"chirps={c}", "chirps") for c in (8, 16, 32, 64, 125, 250)] + \
           [((4, s, 250), f"samples={s}", "samples") for s in (64, 128)] + \
           [((r, 256, 250), f"rx={r}", "antennas") for r in (1, 2)]


def build_cache(path, sc, limit=0, pool="decimate", dc="none"):
    rows = list(csv.DictReader(open(os.path.join(DS, sc, f"scenario{sc[8:]}_dev.csv"))))
    if limit:
        rows = rows[:: max(1, len(rows) // limit)][:limit]
    beams = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    keys = [k for k, _, _ in SETTINGS]
    acc = {k: np.zeros((len(rows), 2, GRID, GRID), np.float32) for k in keys}
    t0 = time.time()
    for i, r in enumerate(rows):
        p = os.path.join(DS, sc, r["unit1_radar"].lstrip("./"))
        cube = np.load(p)                                  # loaded ONCE per frame
        for (nrx, ns, nc) in keys:
            acc[(nrx, ns, nc)][i] = _from_cube(cube, nrx, ns, nc, pool, dc)
        if i % 500 == 0:
            el = time.time() - t0
            print(f"    {i}/{len(rows)}  {el/60:.1f} min  eta {el/max(i,1)*(len(rows)-i)/60:.1f} min",
                  flush=True)
    out = {"beams": beams}
    for k in keys:
        a, nb = sanitize(acc[k], f"{k}")
        out["|".join(map(str, k))] = a
    np.savez_compressed(path, **out)
    return out


def _pool_axis(a, axis, G, how):
    """Resample one axis to length G.

    `decimate` is the original operator: pick G evenly spaced indices. When the axis is
    longer than G that DISCARDS the bins in between, so a setting cannot benefit from
    resolution it paid for, and a finer bin spacing is aliased rather than integrated.
    `max`/`mean` block-reduce instead, so no bin is thrown away. Below length G there is
    nothing to recover and every mode replicates nearest-neighbour.
    """
    n = a.shape[axis]
    if n == G:
        return a
    if n < G or how == "decimate":
        i = np.linspace(0, n - 1, G).astype(int)
        return np.take(a, i, axis=axis)
    starts = np.linspace(0, n, G + 1).astype(int)[:-1]
    red = np.maximum if how == "max" else np.add
    out = red.reduceat(a, starts, axis=axis)
    if how == "mean":
        cnt = np.diff(np.append(starts, n)).astype(np.float32)
        out = out / np.expand_dims(cnt, tuple(i for i in range(a.ndim) if i != axis % a.ndim))
    return out


def _from_cube(cube, n_rx, n_samp, n_chirp, pool="decimate", dc="none"):
    """radar_to_ra's transform, applied to an already-loaded cube.

    `dc` controls removal of the zero-beat-frequency term. Transmit leakage and direct
    antenna coupling arrive at essentially zero delay, so they appear at zero beat
    frequency and land in range bin 0, where they dominate the map and set its
    normalization. `none` is the original chain and reproduces every earlier result.

      mean    subtract the WINDOW-WEIGHTED mean along fast time. Bin 0 of the windowed
              transform is sum_k w[k] x[k], so subtracting m = sum_k w[k] x[k] / sum_k w[k]
              from every sample annihilates it EXACTLY. Subtracting the unweighted mean
              would not, because the window is not constant.
      notchN  mean removal, then zero range bins 0..N-1. Mean removal nulls bin 0 but a
              large DC term also leaks into its neighbours through the window sidelobes,
              and leakage at a short but nonzero delay occupies more than one bin.
    """
    c = cube[:n_rx, :n_samp, :n_chirp]
    nr, ns, nc = c.shape
    w = np.hanning(ns)[None, :, None]
    if dc != "none":
        c = c - (c * w).sum(axis=1, keepdims=True) / w.sum()
    rng = np.fft.fft(c * w, ns, axis=1)
    if dc.startswith("notch"):
        rng[:, : int(dc[5:]), :] = 0
    rd = np.fft.fftshift(np.fft.fft(rng * np.hanning(nc)[None, None, :], nc, axis=2), axes=2)
    rd = rd[:, : ns // 2, :]
    ang = np.fft.fftshift(np.fft.fft(rd, 64, axis=0), axes=0)
    mag = np.abs(ang).astype(np.float32)
    ra, rdm = mag.sum(axis=2).T, mag.sum(axis=0)
    def prep(a):
        a = np.log1p(np.maximum(a, 0))
        a = _pool_axis(_pool_axis(a, 0, GRID, pool), 1, GRID, pool)
        m = a.max()
        return (a / m if m > 0 else a).astype(np.float32)
    return np.stack([prep(ra), prep(rdm)], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="Scenario31")
    # Arm G of collapse_fix_audit.json: lower LR plus balanced-accuracy checkpoint selection.
    # The accuracy-based selector prefers collapsed epochs when the no-information bound is
    # high; balanced accuracy cannot be won by a constant predictor. Class balancing is NOT
    # used: it destroys accuracy on both scenarios.
    ap.add_argument("--recipe", default="G", choices=["baseline", "G"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    # The readout resamples both maps to GRID. `decimate` (the original) discards bins
    # above GRID, which caps the Doppler axis at 64 distinct bins for chirps >= 64 and
    # throws away half the range bins at 256 samples. `max` block-reduces instead. Every
    # pre-2026-10-04 number is pool=decimate; the flag keeps that path reproducible.
    ap.add_argument("--pool", default="decimate", choices=["decimate", "max", "mean"])
    # Zero-beat-frequency removal. The chain as published applies none, and the brightest
    # cell of the range-azimuth map then sits in the first range bin in 100.0% of Sc.31
    # frames: transmit leakage, not a scatterer. `none` keeps that path reproducible.
    ap.add_argument("--dc", default="none",
                    help="none | mean | notchN (mean removal, then blank N range bins)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    tag = a.scenario[8:]
    psfx = "" if a.pool == "decimate" else f"_{a.pool}"
    psfx += "" if a.dc == "none" else f"_dc{a.dc}"
    cache = os.path.join(RESULTS_DIR,
                         f"cache_sc{tag}_ablation{psfx}{'_smoke' if a.limit else ''}.npz")
    if os.path.exists(cache):
        z = np.load(cache); data = {k: z[k] for k in z.files}
        print(f"loaded {os.path.basename(cache)}")
    else:
        print("building the ablation cache (one pass over the cubes)")
        data = build_cache(cache, a.scenario, a.limit, a.pool, a.dc)

    beams = data["beams"]
    n = len(beams); ntr = int(TRAIN_FRAC * n); gap = int(GAP_FRAC * n)
    tr, te = slice(0, ntr), slice(ntr + gap, n)
    ytr, yte = beams[tr], beams[te]
    bmb, bn = collections.Counter(yte.tolist()).most_common(1)[0]
    bound = bn / len(yte)
    floor = float((yte == collections.Counter(ytr.tolist()).most_common(1)[0][0]).mean())
    print(f"\ntrain {ntr}  test {len(yte)}   NO-INFORMATION BOUND {bound:.4f} (class {bmb})"
          f"   train-modal floor {floor:.4f}")
    yt = torch.from_numpy(ytr).long(); ye = torch.from_numpy(yte).long()
    perm = np.random.RandomState(0).permutation(len(yte))

    out = {"scenario": a.scenario, "recipe": a.recipe, "lr": a.lr, "pool": a.pool,
           "dc_removal": a.dc, "bound_const": bound, "n_train": int(ntr), "n_test": int(len(yte)),
           "readout": "ImgReadoutSpatial (image path validated at +0.4170 on this scenario)",
           "seeds": a.seeds, "epochs": a.epochs, "settings": {}}
    orig = EBP.ImgReadout; EBP.ImgReadout = ImgReadoutSpatial
    try:
        print(f"\n  {'setting':<14}{'sweep':<10}{'top1':>8}{'sd':>7}{'top5':>8}"
              f"{'vs bound':>10}{'util':>8}")
        for key, label, sweep in SETTINGS:
            X = data["|".join(map(str, key))]
            Xtr = torch.from_numpy(np.ascontiguousarray(X[tr])).float()
            Xte = torch.from_numpy(np.ascontiguousarray(X[te])).float()
            t1, t5 = [], []
            if a.recipe == "G":
                for s in range(a.seeds):
                    o1, o5 = train_fixed(Xtr, yt, Xte, ye, 64, 42 + s, dev, lr=a.lr,
                                         epochs=a.epochs, balanced=False, select="bal")
                    t1.append(o1); t5.append(o5)
                shuf, _ = train_fixed(Xtr, yt, Xte[perm], ye, 64, 42, dev, lr=a.lr,
                                      epochs=a.epochs, balanced=False, select="bal")
                ms = {"top1": shuf}
            else:
                for s in range(a.seeds):
                    m, _ = train_readout(Xtr, yt, Xte, ye, 64, 42 + s, a.epochs, dev, False)
                    t1.append(m["top1"]); t5.append(m["top5"])
                ms, _ = train_readout(Xtr, yt, Xte[perm], ye, 64, 42, a.epochs, dev, False)
            # A run that lands on the train-modal floor or the bound has predicted a
            # constant class. Roughly 1 in 10 runs does, in every setting
            # (`sc31_rx_audit.json`), so the median and a collapse count are reported
            # alongside the mean rather than letting a bimodal distribution hide inside one.
            arr = np.asarray(t1)
            coll = [float(x) for x in arr
                    if abs(x - floor) < 0.005 or abs(x - bound) < 0.005]
            keep = arr[[not (abs(x - floor) < 0.005 or abs(x - bound) < 0.005) for x in arr]]
            rec = {"n_rx": key[0], "n_samples": key[1], "n_chirps": key[2], "sweep": sweep,
                   "top1_mean": float(np.mean(t1)),
                   "top1_median": float(np.median(t1)),
                   "top1_sd": float(np.std(t1, ddof=1)) if len(t1) > 1 else 0.0,
                   "top1_min": float(arr.min()), "top1_max": float(arr.max()),
                   "n_collapsed": len(coll), "n_seeds": len(arr),
                   "top1_mean_excluding_collapsed": float(keep.mean()) if len(keep) else None,
                   "top1_seeds": [float(x) for x in t1],
                   "top5_mean": float(np.mean(t5)),
                   "vs_bound": float(np.mean(t1) - bound),
                   "shuffled_top1": float(ms["top1"]),
                   "utilization": float(np.mean(t1) - ms["top1"])}
            out["settings"][label] = rec
            print(f"  {label:<14}{sweep:<10}{rec['top1_mean']:>8.4f}{rec['top1_sd']:>7.4f}"
                  f"{rec['top5_mean']:>8.4f}{rec['vs_bound']:>+10.4f}"
                  f"{rec['utilization']:>+8.4f}  med {rec['top1_median']:.4f}"
                  f"  collapsed {rec['n_collapsed']}/{rec['n_seeds']}", flush=True)
    finally:
        EBP.ImgReadout = orig

    ch = [(k, v) for k, v in out["settings"].items() if v["sweep"] == "chirps"]
    ch.sort(key=lambda kv: kv[1]["n_chirps"])
    acc = [v["top1_mean"] for _, v in ch]
    med = [v["top1_median"] for _, v in ch]
    utl = [v["utilization"] for _, v in ch]
    npts = [v["n_chirps"] for _, v in ch]
    out["chirp_trend"] = {
        "n_chirps": npts, "top1": acc, "top1_median": med, "utilization": utl,
        "monotone_median": bool(all(x <= y + 1e-9 for x, y in zip(med, med[1:]))),
        "monotone_top1": bool(all(x <= y + 1e-9 for x, y in zip(acc, acc[1:]))),
        "slope_top1_per_log2_chirp": float(np.polyfit(np.log2(npts), acc, 1)[0]),
        "slope_util_per_log2_chirp": float(np.polyfit(np.log2(npts), utl, 1)[0]),
        "clears_bound_at": [c for c, v in zip(npts, [x - bound for x in acc]) if v > 0.05]}
    print(f"\nCHIRP SWEEP (sensing airtime): top-1 "
          f"{' -> '.join(f'{x:.4f}' for x in acc)}")
    print(f"  slope {out['chirp_trend']['slope_top1_per_log2_chirp']:+.4f} top-1 per doubling; "
          f"utilization slope {out['chirp_trend']['slope_util_per_log2_chirp']:+.4f}")
    print(f"  clears the bound at chirps = {out['chirp_trend']['clears_bound_at']}")
    p = os.path.join(RESULTS_DIR,
                     f"sc{tag}_resource_ablation_{a.recipe}{psfx}"
                     f"{'_smoke' if a.limit else ''}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
