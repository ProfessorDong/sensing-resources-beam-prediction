"""
RADIal loader using the OFFICIAL annotations and LiDAR — replaces the circular one.

WHY THIS EXISTS
---------------
`train_all_real.preprocess_radial` built both the measurement and the "ground
truth" from the same `rd_mag` array: the target was a threshold of the input
(0 violations in 500 frames, see results/dataset_audit.json), all four
measurement channels were identical copies, and a one-line threshold beat the
diffusion model by 10.7 IoU points. That benchmark cannot test reconstruction.

This loader instead uses:
  * ground truth  -> the `scala` LiDAR point cloud (an INDEPENDENT sensor),
                     optionally cross-checked against the official object
                     annotations in labels_CVPR.csv (laser_X_m / laser_Y_m).
  * measurement   -> the raw ADC cube processed to a range-azimuth / Doppler
                     representation in the SENSOR's native polar domain.

The target is Cartesian BEV; the measurement stays polar. The network must
therefore learn the coordinate transform, which is the actual inverse problem
Eq. (6)-(7) describes — and which a 7x7 convolutional surrogate provably cannot
represent (see results/physics_loss_audit.json, T5).

VERIFIED FACTS (2026-09-21, all checked against the data on disk)
-----------------------------------------------------------------
* Label `index` indexes SyncReader's synchronised samples, NOT raw radar frames.
  Per-sequence sync counts vs max label index: 965/961, 386/374, 472/467,
  96/77, 477/474 — all consistent.
* scala columns: 0=X(forward), 1=Y, 2=Z, 4=range, 5=elevation, 6=azimuth.
  Verified by atan2(col1,col0) == col6.
* EXTRINSIC: the label frame is (forward, lateral) = (col0, -col1). Tested all
  four sign/axis conventions over 60 annotated objects: this one gives median
  closest-LiDAR-point distance 0.05 m with 100% of objects inside 2 m; the
  alternatives give 1.46 / 8.10 / 11.92 m. Do not change it without re-running
  that test (`--selftest`).
* Object extent: laser_X_m in [-35.2, 30.8], laser_Y_m in [4.1, 89.9]. A grid of
  X=+-32, Y=[0,96] captures 99.9% of annotated objects; X=+-36,Y=[0,96] captures 100%.
* Radar (miscellaneous/RADAR_CONFIG.m + tools/SignalProcessing/rpl.py):
  512 samples/chirp, 256 chirps, 16 Rx (4 chips x 4), 12 Tx, 76 GHz,
  chirpSlope 14.677048e12 Hz/s -> range resolution ~0.20 m, max range ~102 m,
  max unambiguous velocity ~17.2 m/s.
  Per-chip ADC layout (rpl.py:158): reshape(I+jQ, (512, 4, 256), order='F').transpose(0,2,1)

Usage
-----
    python preprocess_radial_official.py --selftest        # verify extrinsic + mapping
    python preprocess_radial_official.py --build --limit 200

    from preprocess_radial_official import load_radial_official
    scenes, meas, meta = load_radial_official(max_frames=500)

Then ALWAYS run experiments/verify_new_dataset.py on the result before training.
"""
import os, sys, csv, glob, json, types, argparse, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "radial")
LABELS = os.path.join(DS_ROOT, "labels_CVPR.csv")
EXTRACTED = os.path.join(DS_ROOT, "extracted")
DBREADER = os.path.join(DS_ROOT, "tools", "DBReader")

# ---- radar constants, from miscellaneous/RADAR_CONFIG.m ----------------------
N_SAMPLE_PER_CHIRP = 512
N_CHIRPS = 256
N_RX_PER_CHIP = 4
N_CHIPS = 4
ADC_RATE = 10e6
CHIRP_SLOPE = 14.677048e12
CHIRP_RAMP_END = 108.4e-6
CHIRP_IDLE = 6.0e-6
START_FREQ = 76e9
ADC_START = 4.0e-6
C = 3e8

CHIRP_RAMP_TIME = N_SAMPLE_PER_CHIRP / ADC_RATE            # 51.2 us
CHIRP_BW = CHIRP_SLOPE * CHIRP_RAMP_TIME                   # ~751 MHz
MAX_RANGE = C * ADC_RATE * CHIRP_RAMP_TIME / (2 * CHIRP_BW)  # ~102 m
RANGE_RES = MAX_RANGE / N_SAMPLE_PER_CHIRP                 # ~0.20 m
F_C = START_FREQ + (ADC_START + CHIRP_RAMP_TIME / 2) * CHIRP_SLOPE
LAMBDA = C / F_C
CHIRP_INTERVAL = CHIRP_RAMP_END + CHIRP_IDLE
MAX_VELOCITY = LAMBDA / (2 * CHIRP_INTERVAL)               # ~17.2 m/s

# ---- BEV grid ---------------------------------------------------------------
GRID = 64
X_EXTENT = 32.0        # lateral, +-
Y_MIN, Y_MAX = 0.0, 96.0
Z_BAND = (-3.0, 6.0)   # height band around the sensor, metres
N_AZ_BINS = 64         # azimuth bins for the polar measurement


def _install_cantools_stub():
    """DBReader imports cantools only for the CAN bus, which we never read."""
    if "cantools" in sys.modules:
        return
    m = types.ModuleType("cantools")
    m.database = types.ModuleType("cantools.database")
    m.database.load_file = lambda *a, **k: None
    sys.modules["cantools"] = m
    sys.modules["cantools.database"] = m.database


def get_syncreader(seq_dir, tolerance=20000):
    _install_cantools_stub()
    if DBREADER not in sys.path:
        sys.path.insert(0, DBREADER)
    from DBReader.DBReader import SyncReader
    return SyncReader(seq_dir, tolerance=tolerance, silent=True)


# =============================================================================
# radar: ADC -> range-Doppler-angle -> 4-channel polar measurement
# =============================================================================
def adc_to_rd_cube(adc_per_chip):
    """4 x [1048576] int16 -> complex range-Doppler cube [512 range, 256 dop, 16 rx]."""
    frames = []
    for adc in adc_per_chip:
        adc = np.asarray(adc, dtype=np.int16)
        cx = adc[0::2].astype(np.float32) + 1j * adc[1::2].astype(np.float32)
        f = cx.reshape(N_SAMPLE_PER_CHIRP, N_RX_PER_CHIP, N_CHIRPS, order="F")
        frames.append(f.transpose(0, 2, 1))               # (512, 256, 4)
    cube = np.concatenate(frames, axis=2)                  # (512, 256, 16)

    wr = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_SAMPLE_PER_CHIRP) / (N_SAMPLE_PER_CHIRP - 1))
    wd = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_CHIRPS) / (N_CHIRPS - 1))
    cube = cube * wr[:, None, None]
    rng = np.fft.fft(cube, N_SAMPLE_PER_CHIRP, axis=0)
    rng = rng * wd[None, :, None]
    rd = np.fft.fftshift(np.fft.fft(rng, N_CHIRPS, axis=1), axes=1)
    return rd                                              # (512, 256, 16) complex


def rd_cube_to_measurement(rd, grid=GRID, n_az=N_AZ_BINS):
    """
    Polar 4-channel measurement, kept in the SENSOR domain (range x azimuth):
      ch0 range-azimuth, all Doppler   ch1 range-azimuth, static (|v| ~ 0)
      ch2 range-azimuth, dynamic       ch3 range-Doppler (angle-collapsed)
    Angle FFT is coherent and applied BEFORE magnitude; Doppler integration is
    incoherent. The reverse order makes static and dynamic identical.
    Each log-compressed and normalized to [0,1].
    """
    n_dop = rd.shape[1]
    rd = rd.astype(np.complex64)

    # Coherent angle FFT FIRST, then magnitude, then INCOHERENT Doppler integration.
    # Summing complex Doppler bins before |.| cancels (sum over all bins ~ 0), which
    # makes the static and dynamic channels numerically identical -- the same defect
    # as RADIal's original four-identical-channel measurement tensor.
    ang = np.fft.fftshift(np.fft.fft(rd, n_az, axis=2), axes=2)   # (512, 256, n_az)
    mag = np.abs(ang).astype(np.float32)
    del ang

    zero = n_dop // 2
    ra_all = mag.sum(axis=1)                                      # (512, n_az)
    ra_st = mag[:, zero - 2:zero + 3, :].sum(axis=1)
    ra_dy = ra_all - ra_st
    rdm = mag.sum(axis=2)                                         # (512, 256)
    del mag

    def prep(a, H, W):
        a = np.log1p(np.maximum(a, 0).astype(np.float32))
        a = _resize2d(a, H, W)
        mx = a.max()
        return (a / mx) if mx > 0 else a

    return np.stack([prep(ra_all, grid, grid), prep(ra_st, grid, grid),
                     prep(ra_dy, grid, grid), prep(rdm, grid, grid)], 0).astype(np.float32)


def _resize2d(a, H, W):
    """Area-average resize without torch."""
    h, w = a.shape
    yi = (np.arange(H + 1) * h / H).astype(int)
    xi = (np.arange(W + 1) * w / W).astype(int)
    out = np.empty((H, W), dtype=np.float32)
    for i in range(H):
        ys, ye = yi[i], max(yi[i + 1], yi[i] + 1)
        row = a[ys:ye]
        for j in range(W):
            xs, xe = xi[j], max(xi[j + 1], xi[j] + 1)
            out[i, j] = row[:, xs:xe].mean()
    return out


# =============================================================================
# LiDAR -> Cartesian BEV occupancy (the INDEPENDENT ground truth)
# =============================================================================
def scala_to_bev(scala, grid=GRID):
    """
    scala: (N, 11). col0=X(forward), col1=Y, col2=Z.
    EXTRINSIC (verified, see module docstring): (forward, lateral) = (col0, -col1).
    Returns (occupancy, reflectivity-proxy) each [grid, grid], occupancy binary.
    """
    s = np.asarray(scala, dtype=np.float32)
    fwd, lat, z = s[:, 0], -s[:, 1], s[:, 2]
    m = (z > Z_BAND[0]) & (z < Z_BAND[1]) & (fwd >= Y_MIN) & (fwd < Y_MAX) & (np.abs(lat) < X_EXTENT)
    fwd, lat, z = fwd[m], lat[m], z[m]

    occ = np.zeros((grid, grid), np.float32)
    hgt = np.zeros((grid, grid), np.float32)
    if len(fwd):
        gy = ((fwd - Y_MIN) / (Y_MAX - Y_MIN) * grid).astype(int).clip(0, grid - 1)
        gx = ((lat + X_EXTENT) / (2 * X_EXTENT) * grid).astype(int).clip(0, grid - 1)
        occ[gy, gx] = 1.0
        np.maximum.at(hgt, (gy, gx), (z - Z_BAND[0]) / (Z_BAND[1] - Z_BAND[0]))
    return occ, np.clip(hgt, 0, 1)


def objects_to_bev(objs, grid=GRID, radius_m=2.0):
    """Official annotations -> BEV disc mask. objs: list of (lat, fwd)."""
    o = np.zeros((grid, grid), np.float32)
    cy = (Y_MAX - Y_MIN) / grid
    cx = 2 * X_EXTENT / grid
    ry, rx = max(1, int(radius_m / cy)), max(1, int(radius_m / cx))
    for lat, fwd in objs:
        if not (Y_MIN <= fwd < Y_MAX and abs(lat) < X_EXTENT):
            continue
        gy = int((fwd - Y_MIN) / (Y_MAX - Y_MIN) * grid)
        gx = int((lat + X_EXTENT) / (2 * X_EXTENT) * grid)
        y0, y1 = max(0, gy - ry), min(grid, gy + ry + 1)
        x0, x1 = max(0, gx - rx), min(grid, gx + rx + 1)
        o[y0:y1, x0:x1] = 1.0
    return o


# =============================================================================
# main loader
# =============================================================================
def load_radial_official(sequences=None, max_frames=None, grid=GRID,
                         annotation_tiers=("strong", "weak"), labelled_only=True,
                         verbose=True):
    """
    Returns (scenes, meas, meta).
      scenes [N,3,G,G] = [LiDAR occupancy, height proxy, annotated-object mask]
      meas   [N,4,G,G] = polar radar (range-azimuth x3 + range-Doppler)
      meta   dict with per-frame sequence/index and the config used
    """
    rows = list(csv.DictReader(open(LABELS)))
    seq_dirs = sorted(glob.glob(os.path.join(EXTRACTED, "RECORD*")))
    if sequences:
        seq_dirs = [d for d in seq_dirs if os.path.basename(d) in set(sequences)]
    by_seq = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["Annotation"] in annotation_tiers:
            by_seq[r["dataset"]][int(r["index"])].append(
                (float(r["laser_X_m"]), float(r["laser_Y_m"])))

    scenes, meas, frames = [], [], []
    for d in seq_dirs:
        name = os.path.basename(d)
        db = get_syncreader(d)
        idxs = sorted(by_seq[name]) if labelled_only else list(range(len(db)))
        idxs = [i for i in idxs if i < len(db)]
        if verbose:
            print(f"  {name}: {len(db)} sync samples, {len(idxs)} usable")
        for i in idxs:
            if max_frames and len(scenes) >= max_frames:
                break
            try:
                s = db.GetSensorData(i)
                adc = [np.asarray(s[f"radar_ch{c}"]["data"]) for c in range(N_CHIPS)]
                sc = np.asarray(s["scala"]["data"])
            except Exception as e:
                if verbose:
                    print(f"    idx {i}: skipped ({e})")
                continue
            occ, hgt = scala_to_bev(sc, grid)
            if occ.sum() < 5:
                continue
            obj = objects_to_bev(by_seq[name].get(i, []), grid)
            scenes.append(np.stack([occ, hgt, obj], 0))
            meas.append(rd_cube_to_measurement(adc_to_rd_cube(adc), grid))
            frames.append({"sequence": name, "index": i,
                           "n_objects": len(by_seq[name].get(i, []))})
        if max_frames and len(scenes) >= max_frames:
            break

    meta = {"frames": frames, "grid": grid,
            "x_extent_m": X_EXTENT, "y_range_m": [Y_MIN, Y_MAX], "z_band_m": list(Z_BAND),
            "annotation_tiers": list(annotation_tiers),
            "radar": {"range_res_m": RANGE_RES, "max_range_m": MAX_RANGE,
                      "max_velocity_mps": MAX_VELOCITY, "lambda_m": LAMBDA,
                      "f_c_Hz": F_C, "bandwidth_Hz": CHIRP_BW},
            "measurement_domain": "polar (range x azimuth), target is Cartesian BEV",
            "channels": {"scene": ["lidar_occupancy", "height_proxy", "annotated_objects"],
                         "meas": ["RA_all", "RA_static", "RA_dynamic", "range_doppler"]}}
    return (np.asarray(scenes, np.float32), np.asarray(meas, np.float32), meta)


# =============================================================================
def selftest():
    """Re-verify the extrinsic and the index mapping. Cheap; run after any change."""
    print("RADIal loader self-test\n" + "=" * 62)
    print(f"radar: range res {RANGE_RES:.3f} m, max range {MAX_RANGE:.1f} m, "
          f"max |v| {MAX_VELOCITY:.1f} m/s, lambda {1000*LAMBDA:.2f} mm, B {CHIRP_BW/1e6:.0f} MHz")
    rows = list(csv.DictReader(open(LABELS)))
    lab = collections.defaultdict(list)
    for r in rows:
        lab[r["dataset"]].append(r)
    ok = True
    for d in sorted(glob.glob(os.path.join(EXTRACTED, "RECORD*"))):
        n = os.path.basename(d)
        db = get_syncreader(d)
        mx = max(int(r["index"]) for r in lab[n])
        good = mx < len(db)
        ok &= good
        print(f"  {n:<34} sync={len(db):>5} max_label_idx={mx:>5} {'OK' if good else '*** FAIL ***'}")

    n = "RECORD@2020-11-21_11.54.31"
    db = get_syncreader(os.path.join(EXTRACTED, n))
    sel = [r for r in lab[n] if int(r["index"]) < len(db)][:40]
    conv = {"(c0,-c1)  <-- expected": lambda s: (s[:, 0], -s[:, 1]),
            "(c0, c1)": lambda s: (s[:, 0], s[:, 1]),
            "(c1, c0)": lambda s: (s[:, 1], s[:, 0]),
            "(c1,-c0)": lambda s: (s[:, 1], -s[:, 0])}
    cache = {}
    print("\n  extrinsic check (median closest LiDAR point to an annotated object):")
    best = (1e9, None)
    for name, f in conv.items():
        ds = []
        for r in sel:
            i = int(r["index"])
            if i not in cache:
                cache[i] = np.asarray(db.GetSensorData(i)["scala"]["data"])
            fwd, lat = f(cache[i])
            lx, ly = float(r["laser_X_m"]), float(r["laser_Y_m"])
            ds.append(np.sqrt((fwd - ly) ** 2 + (lat - lx) ** 2).min())
        med = float(np.median(ds))
        print(f"    {name:<24} {med:7.3f} m   frac<2m={np.mean(np.array(ds) < 2):.2f}")
        if med < best[0]:
            best = (med, name)
    good = best[1].startswith("(c0,-c1)")
    ok &= good
    print(f"  -> best: {best[1]} @ {best[0]:.3f} m  {'OK' if good else '*** EXTRINSIC MISMATCH ***'}")

    i0 = int(sel[0]["index"])
    s0 = db.GetSensorData(i0)
    adc = [np.asarray(s0[f"radar_ch{c}"]["data"]) for c in range(N_CHIPS)]
    mm = rd_cube_to_measurement(adc_to_rd_cube(adc))
    distinct = len({mm[c].tobytes() for c in range(4)}) == 4
    cors = [float(np.corrcoef(mm[i].ravel(), mm[j].ravel())[0, 1])
            for i in range(4) for j in range(i + 1, 4)]
    print(f"\n  measurement {mm.shape}; 4 channels distinct: "
          f"{'OK' if distinct else '*** FAIL: duplicated channels ***'}")
    print(f"    max pairwise corr = {max(cors):.4f}")
    ok &= distinct
    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest or not a.build:
        if not selftest():
            sys.exit(1)
    if a.build:
        print(f"\nbuilding (limit {a.limit}) ...")
        sc, me, meta = load_radial_official(max_frames=a.limit)
        print(f"  scenes {sc.shape}  meas {me.shape}")
        print(f"  occupancy fill: {100*(sc[:,0]>0.5).mean():.2f}%   "
              f"annotated-object fill: {100*(sc[:,2]>0.5).mean():.2f}%")
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "results", "radial_official_sample.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, scenes=sc, meas=me)
        json.dump(meta, open(out.replace(".npz", "_meta.json"), "w"), indent=2)
        print(f"  saved: {out}")
        print("\nNEXT: run experiments/verify_new_dataset.py on this before training.")


if __name__ == "__main__":
    main()
