"""
DeepSense 6G Scenario 36 loader — the ISAC-native, non-degenerate benchmark.

WHY THIS EXISTS
---------------
Sc.42/43/44 (indoor ISAC) is degenerate: test scenes are 91.35% identical by
pairwise IoU, and a constant predictor beats every method in the paper on IoU,
NLL and Brier (results/trivial_baselines.json). It also ships only pre-processed
magnitude range-Doppler maps, so the manuscript's Eq. (2) declared N_nu Doppler
bins that the processing never produced (R3 #3, R4 #1).

Sc.36 fixes all of that:
  * 24,799 frames (vs 2,479 indoor training frames)
  * unit1 is MOVING (23,430 distinct GPS fixes) -> genuine scene diversity
  * radar is RAW COMPLEX with a chirp axis -> a real range-Doppler-angle chain
  * `pwr1..4` are 64-element beam-power vectors  ->  h_beam, explicitly
  * `unit1_overall-beam` is a ground-truth best-beam label -> a COMMUNICATION-side
    metric, which is what editor points 1 & 9 and R1 #2 demanded and the paper
    could not supply.

VERIFIED FACTS (2026-09-21, checked against the data on disk)
--------------------------------------------------------------
* All 24,799 CSV rows resolve for every modality, once paths are based at
  'scenario36/'. The CSV stores 'unit1/radar1/...' but extraction writes
  'scenario36/unit1/radar1/...'.
* radar .mat -> key 'data', complex64 (4, 256, 128) = (Rx, ADC samples, chirps).
  Four boards -> 16 Rx total when stacked.
* lidar CSV columns: TIMESTAMP, RANGE_mm, SIGNAL, NEAR_IR, REFLECTIVITY,
  X_mm, Y_mm, Z_mm. ~14,470 valid returns per frame (RANGE>0).
* LiDAR extent: +-32 m captures 95.0% of points, +-48 m 98.5%. Z p1/p99 = -2.5/6.8 m.
* pwr vectors: argmax == the CSV's `best-beam` column, exactly. Confirmed on
  all four arrays. `unit1_overall-beam` spans 1..255 across the 4x64 codebook.
* scenario36.mat carries `unit1_pwr1..4` as (24799, 64) matrices — far faster
  than reading 24,800 text files. This loader uses the .mat when present.

NOT VERIFIED — do not fabricate
-------------------------------
No radar parameter file ships with Sc.36. Range/Doppler axes are therefore
reported in BINS, not metres or m/s. The manuscript quotes TI AWR2243 at
76-81 GHz with 4 GHz bandwidth for Sc.31/33; DO NOT assume that carries over to
Sc.36 without confirmation from DeepSense. Physical scaling is a TODO.

Usage
-----
    python preprocess_deepsense_sc36.py --selftest
    python preprocess_deepsense_sc36.py --build --limit 500 --stride 10

    from preprocess_deepsense_sc36 import load_sc36
    scenes, meas, hbeam, beamlabel, meta = load_sc36(max_frames=2000, stride=10)

Then ALWAYS run experiments/verify_new_dataset.py before training.
"""
import os, sys, csv, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "datasets", "deepsense6g", "Scenario36")
CSV = os.path.join(SC_DIR, "scenario36.csv")
MAT = os.path.join(SC_DIR, "scenario36.mat")
BASE = os.path.join(SC_DIR, "scenario36")          # <- the path base the CSV omits

GRID = 64
EXTENT = 32.0            # +- metres in X and Y (95.0% of LiDAR points)
Z_BAND = (-2.5, 7.0)     # p1..p99 of the height distribution
N_AZ_BINS = 64
N_RADAR_BOARDS = 4
N_RX_PER_BOARD = 4


def _resize2d(a, H, W):
    h, w = a.shape
    yi = (np.arange(H + 1) * h / H).astype(int)
    xi = (np.arange(W + 1) * w / W).astype(int)
    out = np.empty((H, W), np.float32)
    for i in range(H):
        ys, ye = yi[i], max(yi[i + 1], yi[i] + 1)
        row = a[ys:ye]
        for j in range(W):
            xs, xe = xi[j], max(xi[j + 1], xi[j] + 1)
            out[i, j] = row[:, xs:xe].mean()
    return out


# =============================================================================
# radar: raw complex ADC -> range-Doppler-angle -> 4-channel polar measurement
# =============================================================================
def radar_to_measurement(boards, grid=GRID, n_az=N_AZ_BINS):
    """
    boards: list of 4 arrays, each complex (4 Rx, 256 ADC, 128 chirps).
    Returns [4, grid, grid] polar measurement:
      ch0 range-azimuth all-Doppler   ch1 range-azimuth static
      ch2 range-azimuth dynamic       ch3 range-Doppler (angle-collapsed)
    Angle FFT is applied coherently BEFORE magnitude; Doppler integration is
    incoherent. Doing it the other way makes static and dynamic identical.
    Kept in the SENSOR's polar domain; the BEV target is Cartesian, so the
    network must learn the coordinate transform.
    """
    cube = np.concatenate([np.asarray(b) for b in boards], axis=0)    # (16, 256, 128)
    n_rx, n_s, n_c = cube.shape

    wr = np.hanning(n_s)[None, :, None]
    wd = np.hanning(n_c)[None, None, :]
    rng = np.fft.fft(cube * wr, n_s, axis=1)                          # range along ADC axis
    rd = np.fft.fftshift(np.fft.fft(rng * wd, n_c, axis=2), axes=2)   # (16, n_s, n_c)
    rd = rd[:, : n_s // 2, :].astype(np.complex64)                    # positive ranges

    # Coherent angle FFT FIRST, then magnitude, then INCOHERENT integration.
    # Summing complex bins before taking |.| cancels: sum over all Doppler bins is
    # ~0, which makes static == -dynamic and the two channels numerically identical.
    ang = np.fft.fftshift(np.fft.fft(rd, n_az, axis=0), axes=0)       # (n_az, n_range, n_c)
    mag = np.abs(ang).astype(np.float32)
    del ang, rd, rng

    zero = mag.shape[2] // 2
    ra_all = mag.sum(axis=2).T                                        # (n_range, n_az)
    ra_st = mag[:, :, zero - 2:zero + 3].sum(axis=2).T
    ra_dy = ra_all - ra_st
    rdm = mag.sum(axis=0)                                             # (n_range, n_c)

    def prep(a):
        a = np.log1p(np.maximum(a, 0).astype(np.float32))
        a = _resize2d(a, grid, grid)
        mx = a.max()
        return a / mx if mx > 0 else a

    return np.stack([prep(ra_all), prep(ra_st), prep(ra_dy), prep(rdm)], 0).astype(np.float32)


# =============================================================================
# LiDAR -> Cartesian BEV occupancy (INDEPENDENT ground truth)
# =============================================================================
def lidar_to_bev(path, grid=GRID, extent=EXTENT, z_band=Z_BAND):
    # pandas is ~4.3x faster than genfromtxt here and this is the pipeline bottleneck
    try:
        import pandas as pd
        d = pd.read_csv(path, skiprows=1, header=None).values
    except Exception:
        d = np.genfromtxt(path, delimiter=",", skip_header=1)
    if d.ndim != 2 or d.shape[0] == 0:
        return None, None
    valid = d[:, 1] > 0
    x, y, z = d[valid, 5] / 1000., d[valid, 6] / 1000., d[valid, 7] / 1000.
    m = (np.abs(x) < extent) & (np.abs(y) < extent) & (z > z_band[0]) & (z < z_band[1])
    x, y, z = x[m], y[m], z[m]
    occ = np.zeros((grid, grid), np.float32)
    hgt = np.zeros((grid, grid), np.float32)
    if len(x):
        gx = ((x + extent) / (2 * extent) * grid).astype(int).clip(0, grid - 1)
        gy = ((y + extent) / (2 * extent) * grid).astype(int).clip(0, grid - 1)
        occ[gy, gx] = 1.0
        np.maximum.at(hgt, (gy, gx), (z - z_band[0]) / (z_band[1] - z_band[0]))
    return occ, np.clip(hgt, 0, 1)


# =============================================================================
def _load_beam_tables():
    """Prefer scenario36.mat (24799x64 per array) over 24,800 text files."""
    try:
        import scipy.io
        m = scipy.io.loadmat(MAT)
        pw = np.stack([np.asarray(m[f"unit1_pwr{i}"], np.float32) for i in (1, 2, 3, 4)], 1)
        ob = np.asarray(m["unit1_overall-beam"], np.float32).ravel()
        return pw, ob                                  # (N,4,64), (N,)
    except Exception:
        return None, None


def load_sc36(max_frames=None, stride=1, grid=GRID, extent=EXTENT, verbose=True):
    """
    Returns (scenes, meas, hbeam, beamlabel, meta).
      scenes    [N,3,G,G]  [lidar occupancy, height proxy, zeros(reserved)]
      meas      [N,4,G,G]  polar radar (RA all/static/dynamic + range-Doppler)
      hbeam     [N,256]    4 arrays x 64 beams, normalized per frame -> h_beam
      beamlabel [N]        unit1_overall-beam, the COMMUNICATION-side label
    """
    import scipy.io
    rows = list(csv.DictReader(open(CSV)))
    pw_tab, ob_tab = _load_beam_tables()

    scenes, meas, hbeams, labels, frames = [], [], [], [], []
    sel = range(0, len(rows), stride)
    for n_done, i in enumerate(sel):
        if max_frames and len(scenes) >= max_frames:
            break
        r = rows[i]
        try:
            occ, hgt = lidar_to_bev(os.path.join(BASE, r["unit1_lidar1"].lstrip("./")),
                                    grid, extent)
            if occ is None or occ.sum() < 5:
                continue
            boards = []
            for b in range(1, N_RADAR_BOARDS + 1):
                mm = scipy.io.loadmat(os.path.join(BASE, r[f"unit1_radar{b}"].lstrip("./")))
                boards.append(mm["data"])
            if pw_tab is not None:
                hb = pw_tab[i].reshape(-1)
                lb = ob_tab[i]
            else:
                hb = np.concatenate([np.loadtxt(os.path.join(BASE, r[f"unit1_pwr{b}"].lstrip("./")))
                                     for b in range(1, 5)])
                lb = float(r["unit1_overall-beam"] or -1)
        except Exception as e:
            if verbose and len(frames) < 5:
                print(f"    row {i}: skipped ({type(e).__name__}: {e})")
            continue

        mx = hb.max()
        hbeams.append((hb / mx if mx > 0 else hb).astype(np.float32))
        labels.append(lb)
        scenes.append(np.stack([occ, hgt, np.zeros_like(occ)], 0))
        meas.append(radar_to_measurement(boards, grid))
        frames.append({"row": int(i), "seq_index": r.get("seq_index"),
                       "unit2_lat": r.get("unit2_gps1_lat"), "unit2_lon": r.get("unit2_gps1_lon")})
        if verbose and len(scenes) % 50 == 0:
            print(f"    {len(scenes)} frames ...")

    meta = {"frames": frames, "grid": grid, "extent_m": extent, "z_band_m": list(Z_BAND),
            "stride": stride, "n_rows_total": len(rows),
            "channels": {"scene": ["lidar_occupancy", "height_proxy", "reserved_zero"],
                         "meas": ["RA_all", "RA_static", "RA_dynamic", "range_doppler"],
                         "hbeam": "4 arrays x 64 beams, per-frame max-normalized",
                         "beamlabel": "unit1_overall-beam (1..255)"},
            "measurement_domain": "polar (range x azimuth) bins; PHYSICAL SCALING UNKNOWN",
            "caveat": "No radar parameter file ships with Sc.36. Axes are bins, not metres."}
    return (np.asarray(scenes, np.float32), np.asarray(meas, np.float32),
            np.asarray(hbeams, np.float32), np.asarray(labels, np.float32), meta)


# =============================================================================
def selftest():
    import scipy.io
    print("DeepSense Sc.36 loader self-test\n" + "=" * 64)
    ok = True
    rows = list(csv.DictReader(open(CSV)))
    print(f"  csv rows: {len(rows)}")

    miss = 0
    for r in rows[::500]:
        for c in ("unit1_radar1", "unit1_radar2", "unit1_radar3", "unit1_radar4",
                  "unit1_lidar1", "unit1_pwr1"):
            if not os.path.exists(os.path.join(BASE, r[c].lstrip("./"))):
                miss += 1
    print(f"  path resolution (every 500th row, 6 modalities): {miss} missing "
          f"{'OK' if miss == 0 else '*** FAIL ***'}")
    ok &= miss == 0

    r = rows[5000]
    mm = scipy.io.loadmat(os.path.join(BASE, r["unit1_radar1"].lstrip("./")))
    d = mm["data"]
    shp_ok = d.shape == (4, 256, 128) and np.iscomplexobj(d)
    print(f"  radar .mat: shape={d.shape} dtype={d.dtype} {'OK' if shp_ok else '*** FAIL ***'}")
    ok &= shp_ok

    # is axis 1 raw ADC (flat) or already range-transformed (peaked)?
    prof = np.abs(d[0, :, 0]); prof = prof / (prof.sum() + 1e-12)
    ent = -(prof * np.log(prof + 1e-12)).sum() / np.log(len(prof))
    print(f"  axis-1 normalized entropy = {ent:.3f}  -> "
          f"{'looks like RAW ADC (range FFT needed, as implemented)' if ent > 0.8 else 'looks ALREADY range-transformed — CHECK'}")

    pw_tab, ob_tab = _load_beam_tables()
    if pw_tab is not None:
        i = 5000
        agree = all(int(pw_tab[i, b].argmax()) == int(r[f"unit1_pwr{b+1}_best-beam"])
                    for b in range(4))
        print(f"  .mat pwr tables {pw_tab.shape}, argmax == csv best-beam: "
              f"{'OK' if agree else '*** FAIL ***'}")
        ok &= agree
    else:
        print("  .mat beam tables unavailable — will fall back to text files")

    occ, hgt = lidar_to_bev(os.path.join(BASE, r["unit1_lidar1"].lstrip("./")))
    print(f"  lidar BEV: fill={100*occ.mean():.2f}%  height in [{hgt.min():.2f},{hgt.max():.2f}]")
    ok &= occ.sum() > 5

    boards = [scipy.io.loadmat(os.path.join(BASE, r[f"unit1_radar{b}"].lstrip("./")))["data"]
              for b in range(1, 5)]
    m = radar_to_measurement(boards)
    print(f"  measurement: {m.shape}  per-channel max={[round(float(m[c].max()),3) for c in range(4)]}")
    distinct = len({m[c].tobytes() for c in range(4)}) == 4
    print(f"  4 channels distinct (RADIal's were identical copies): "
          f"{'OK' if distinct else '*** FAIL ***'}")
    ok &= distinct

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest or not a.build:
        if not selftest():
            sys.exit(1)
    if a.build:
        print(f"\nbuilding (limit {a.limit}, stride {a.stride}) ...")
        sc, me, hb, lb, meta = load_sc36(max_frames=a.limit, stride=a.stride)
        print(f"  scenes {sc.shape}  meas {me.shape}  hbeam {hb.shape}  labels {lb.shape}")
        print(f"  occupancy fill: {100*(sc[:,0]>0.5).mean():.2f}%   "
              f"beam label range: {lb.min():.0f}..{lb.max():.0f}")
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "results", "sc36_sample.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, scenes=sc, meas=me, hbeam=hb, beamlabel=lb)
        json.dump(meta, open(out.replace(".npz", "_meta.json"), "w"), indent=2)
        print(f"  saved: {out}")
        print("\nNEXT: run experiments/verify_new_dataset.py on this before training.")


if __name__ == "__main__":
    main()
