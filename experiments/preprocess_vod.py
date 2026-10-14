"""
View-of-Delft loader — the sparsest, most genuinely ill-posed benchmark in this project.

WHY VoD MATTERS
---------------
Its radar is a SPARSE POINT CLOUD, not a dense tensor: ~318 points per frame
against ~177,600 LiDAR points, a ratio of 1:559. Reconstructing a dense BEV from
~318 radar returns is an actually ill-posed inverse problem -- which is what this
project's paper claims to solve and what none of its previous benchmarks tested
(indoor ISAC was solved by a constant, RADIal's target was a threshold of its own
input).

It also ships a built-in MEASUREMENT-SPARSITY SWEEP: radar/, radar_3frames/ and
radar_5frames/ give ~318 / ~928 / ~1567 points. Utilization theory predicts that
measurement utilization should rise with density; VoD can test that directly, and
no other dataset here can.

VERIFIED FACTS (2026-09-25, all checked against the data on disk)
-----------------------------------------------------------------
* Formats (confirmed against docs/SENSORS_AND_DATA.md):
    radar .bin  N x 7  float32  [x, y, z, RCS, v_r, v_r_compensated, time]
    lidar .bin  N x 4  float32  [x, y, z, reflectance]
* Calibration is CONSTANT across all frames for both sensors (checked on 10).
* RADAR AND LIDAR ARE IN DIFFERENT FRAMES. radar/calib and lidar/calib carry
  different Tr_velo_to_cam. radar->lidar = inv(Tr_lidar_to_cam) @ Tr_radar_to_cam,
  which is a near-identity rotation plus translation (2.514, 0.061, -1.153) m --
  the radar sits 2.5 m forward of and 1.15 m below the roof LiDAR.
* LABELS ARE IN CAMERA COORDINATES; cam->lidar = inv(Tr_lidar_to_cam). Validated
  by nearest-LiDAR-point distance to annotated object centers: median 0.229 m,
  100% within 2 m. Alternatives for comparison: no transform 18.62 m (0% within
  2 m), swapped x/z 3.64 m (10%). Do not change it without rerunning --selftest.
* label z is the box BOTTOM: Car labels average -1.59 m in the lidar frame, i.e.
  +0.08 m above the ground plane. Consistent, as it should be.
* lidar frame: x forward, y lateral, z up. Ground plane at z = -1.67 m
  (sd 0.07 across frames), estimated as the mode of z inside the forward grid.
* Radar covers the FORWARD HALF-PLANE only (x >= 0, azimuth +-90 deg, to ~85 m),
  so the BEV grid is forward-only. Including the rear would ask the model to
  hallucinate regions the radar can never observe and would dilute utilization.
* Annotations cover only objects within 50 m AND inside the camera's +-32 deg
  horizontal FoV, so the object mask is a NARROW WEDGE. The dense target therefore
  comes from the LiDAR cloud, not from labels; labels are a cross-check and an
  optional channel.
* Official splits (lidar/ImageSets): train 5139, val 1296, test 2247 (test is the
  unlabeled benchmark holdout). Use these -- do not invent a split.

GROUND EXCLUSION -- the fix for the Sc.36 metric problem
--------------------------------------------------------
On DeepSense Sc.36, aggregate BEV IoU turned out to be dominated by near-field
ground return: a constant predictor scored 0.8895 within 8 m and 0.0000 beyond
24 m (results/sc36_range_stratified.json). Here the road surface is excluded at
source: only points above ground+0.5 m enter the target, which retains 43% of
forward points. Still report range-stratified metrics as well.

Usage
-----
    python preprocess_vod.py --selftest
    python preprocess_vod.py --build --split val --limit 200 --radar-frames 1

    from preprocess_vod import load_vod
    scenes, meas, meta = load_vod(split="train", n_radar_frames=1)

ALWAYS run experiments/verify_new_dataset.py on the result before training.
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "datasets", "view_of_delft", "view_of_delft_PUBLIC")
LIDAR = os.path.join(ROOT, "lidar", "training")
IMAGESETS = os.path.join(ROOT, "lidar", "ImageSets")
RADAR_DIR = {1: "radar", 3: "radar_3frames", 5: "radar_5frames"}

GRID = 64
X_RANGE = (0.0, 64.0)      # forward, metres (radar sees the forward half-plane)
Y_RANGE = (-32.0, 32.0)    # lateral, metres
GROUND_MARGIN = 0.5        # keep z > ground + this; removes the road surface
LABEL_CLASSES = ("Car", "Pedestrian", "Cyclist", "rider", "bicycle", "motor",
                 "truck", "moped_scooter", "other_vehicle", "other_ride")


# --------------------------------------------------------------------- calib
def _read_calib(path):
    d = {}
    for ln in open(path):
        if ":" in ln:
            k, v = ln.split(":", 1)
            v = v.strip()
            if v:
                d[k] = np.array([float(x) for x in v.split()])
    return d


def transforms():
    """Returns (radar->lidar, camera->lidar) 4x4. Calib is frame-invariant."""
    fid = sorted(os.listdir(os.path.join(LIDAR, "calib")))[0][:-4]
    Tl = np.vstack([_read_calib(os.path.join(LIDAR, "calib", f"{fid}.txt"))
                    ["Tr_velo_to_cam"].reshape(3, 4), [0, 0, 0, 1]])
    Tr = np.vstack([_read_calib(os.path.join(ROOT, "radar", "training", "calib", f"{fid}.txt"))
                    ["Tr_velo_to_cam"].reshape(3, 4), [0, 0, 0, 1]])
    C2L = np.linalg.inv(Tl)
    return C2L @ Tr, C2L


def split_ids(split):
    p = os.path.join(IMAGESETS, f"{split}.txt")
    if not os.path.exists(p):
        raise SystemExit(f"no such split: {p}. Available: "
                         f"{[os.path.basename(x)[:-4] for x in glob.glob(IMAGESETS+'/*.txt')]}")
    return [ln.strip() for ln in open(p) if ln.strip()]


# --------------------------------------------------------------- rasterizing
def _grid_index(x, y, grid=GRID):
    gx = ((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * grid).astype(int)
    gy = ((y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * grid).astype(int)
    ok = (gx >= 0) & (gx < grid) & (gy >= 0) & (gy < grid)
    return gx[ok], gy[ok], ok


def ground_z(pts):
    """Mode of z inside the forward grid. Per frame -- the road can slope."""
    m = ((pts[:, 0] > X_RANGE[0]) & (pts[:, 0] < X_RANGE[1]) &
         (pts[:, 1] > Y_RANGE[0]) & (pts[:, 1] < Y_RANGE[1]))
    z = pts[m, 2]
    if len(z) < 100:
        return -1.67
    h, e = np.histogram(z, bins=120, range=(-5, 3))
    return float(e[h.argmax()])


def lidar_to_bev(L, grid=GRID, margin=GROUND_MARGIN):
    """L: [N,4] in the lidar frame -> (occupancy, height, reflectance), ground removed."""
    gz = ground_z(L)
    keep = L[:, 2] > gz + margin
    P = L[keep]
    occ = np.zeros((grid, grid), np.float32)
    hgt = np.zeros((grid, grid), np.float32)
    ref = np.zeros((grid, grid), np.float32)
    if len(P):
        gx, gy, ok = _grid_index(P[:, 0], P[:, 1], grid)
        z = P[ok, 2] - gz
        r = P[ok, 3] / 255.0
        occ[gx, gy] = 1.0
        np.maximum.at(hgt, (gx, gy), np.clip(z / 5.0, 0, 1))
        np.maximum.at(ref, (gx, gy), np.clip(r, 0, 1))
    return occ, hgt, ref, gz


def radar_to_bev(R, R2L, grid=GRID):
    """R: [N,7] in the radar frame -> 4-channel BEV measurement in the lidar frame.

    ch0 log point density   ch1 RCS (max, normalized)
    ch2 v_r (signed)        ch3 v_r_compensated (signed)
    Velocities are mapped to [0,1] with 0.5 = stationary, so sign is preserved.
    """
    xyz = np.c_[R[:, :3], np.ones(len(R))] @ R2L.T        # radar -> lidar
    cnt = np.zeros((grid, grid), np.float32)
    rcs = np.zeros((grid, grid), np.float32)
    vr = np.full((grid, grid), 0.5, np.float32)
    vrc = np.full((grid, grid), 0.5, np.float32)
    if len(R):
        gx, gy, ok = _grid_index(xyz[:, 0], xyz[:, 1], grid)
        np.add.at(cnt, (gx, gy), 1.0)
        np.maximum.at(rcs, (gx, gy), np.clip((R[ok, 3] + 45.0) / 80.0, 0, 1))
        vr[gx, gy] = np.clip(R[ok, 4] / 14.0 + 0.5, 0, 1)
        vrc[gx, gy] = np.clip(R[ok, 5] / 14.0 + 0.5, 0, 1)
    cnt = np.log1p(cnt)
    if cnt.max() > 0:
        cnt = cnt / cnt.max()
    return np.stack([cnt, rcs, vr, vrc], 0).astype(np.float32)


def labels_to_bev(path, C2L, grid=GRID, radius_m=1.5):
    """Annotated-object mask. NARROW WEDGE ONLY: labels cover <=50 m and the
    camera's +-32 deg FoV, so this is not a dense target -- it is a cross-check."""
    o = np.zeros((grid, grid), np.float32)
    n = 0
    if not os.path.exists(path):
        return o, 0
    cw = (X_RANGE[1] - X_RANGE[0]) / grid
    ch = (Y_RANGE[1] - Y_RANGE[0]) / grid
    rx, ry = max(1, int(radius_m / cw)), max(1, int(radius_m / ch))
    for ln in open(path):
        p = ln.split()
        if len(p) < 14 or p[0] not in LABEL_CLASSES:
            continue
        lid = C2L @ np.array([float(p[11]), float(p[12]), float(p[13]), 1.0])
        gx = int((lid[0] - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * grid)
        gy = int((lid[1] - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * grid)
        if not (0 <= gx < grid and 0 <= gy < grid):
            continue
        o[max(0, gx-rx):min(grid, gx+rx+1), max(0, gy-ry):min(grid, gy+ry+1)] = 1.0
        n += 1
    return o, n


# ------------------------------------------------------------------- loader
def load_vod(split="train", n_radar_frames=1, limit=None, stride=1,
             grid=GRID, verbose=True):
    """
    Returns (scenes, meas, meta).
      scenes [N,3,G,G] = [lidar occupancy (ground removed), height, reflectance]
      meas   [N,4,G,G] = [radar density, RCS, v_r, v_r_compensated]
    """
    R2L, C2L = transforms()
    rdir = os.path.join(ROOT, RADAR_DIR[n_radar_frames], "training", "velodyne")
    ids = split_ids(split)[::stride]
    if limit:
        ids = ids[:limit]
    scenes, meas, frames = [], [], []
    for k, fid in enumerate(ids):
        lp = os.path.join(LIDAR, "velodyne", f"{fid}.bin")
        rp = os.path.join(rdir, f"{fid}.bin")
        if not (os.path.exists(lp) and os.path.exists(rp)):
            continue
        L = np.fromfile(lp, dtype=np.float32).reshape(-1, 4)
        R = np.fromfile(rp, dtype=np.float32).reshape(-1, 7)
        occ, hgt, ref, gz = lidar_to_bev(L, grid)
        if occ.sum() < 5:
            continue
        obj, nobj = labels_to_bev(os.path.join(LIDAR, "label_2", f"{fid}.txt"), C2L, grid)
        scenes.append(np.stack([occ, hgt, ref], 0))
        meas.append(radar_to_bev(R, R2L, grid))
        frames.append({"id": fid, "n_radar_pts": int(len(R)), "n_lidar_pts": int(len(L)),
                       "ground_z": gz, "n_labels_in_grid": nobj})
        if verbose and (k + 1) % 500 == 0:
            print(f"    {len(scenes)}/{k+1} frames ...", flush=True)
    meta = {"split": split, "n_radar_frames": n_radar_frames, "grid": grid,
            "x_range_m": list(X_RANGE), "y_range_m": list(Y_RANGE),
            "ground_margin_m": GROUND_MARGIN, "n": len(scenes), "frames": frames,
            "channels": {"scene": ["lidar_occupancy_ground_removed", "height", "reflectance"],
                         "meas": ["radar_density_log", "RCS", "v_r", "v_r_compensated"]},
            "mean_radar_pts": float(np.mean([f["n_radar_pts"] for f in frames])) if frames else 0,
            "mean_lidar_pts": float(np.mean([f["n_lidar_pts"] for f in frames])) if frames else 0,
            "note": "forward half-plane only; radar cannot observe the rear. "
                    "Labels cover <=50 m and +-32 deg camera FoV, so the dense "
                    "target comes from LiDAR, not labels."}
    return (np.asarray(scenes, np.float32), np.asarray(meas, np.float32), meta)


# ----------------------------------------------------------------- selftest
def selftest():
    print("View-of-Delft loader self-test\n" + "=" * 70)
    ok = True
    R2L, C2L = transforms()
    print(f"  radar->lidar translation: {np.round(R2L[:3,3],3)} m  "
          f"rotation near-identity: {np.allclose(R2L[:3,:3], np.eye(3), atol=0.05)}")

    for s in ("train", "val", "test", "full"):
        try:
            print(f"  split {s:<6} {len(split_ids(s)):>5} frames")
        except SystemExit:
            print(f"  split {s:<6} MISSING"); ok = False

    ids = split_ids("val")[:40]
    # 1. label transform, against alternatives
    alts = {"cam->lidar (expected)": lambda c: (C2L @ c)[:3],
            "no transform": lambda c: c[:3],
            "swap x/z": lambda c: np.array([c[2], c[0], c[1]])}
    print("\n  label-frame check (median nearest LiDAR point to an annotated center):")
    best = (1e9, None)
    for name, f in alts.items():
        ds = []
        for fid in ids[:12]:
            lp = os.path.join(LIDAR, "velodyne", f"{fid}.bin")
            lab = os.path.join(LIDAR, "label_2", f"{fid}.txt")
            if not os.path.exists(lab):
                continue
            L = np.fromfile(lp, dtype=np.float32).reshape(-1, 4)
            for ln in open(lab):
                p = ln.split()
                if p[0] not in ("Car", "Pedestrian", "Cyclist"):
                    continue
                c = np.array([float(p[11]), float(p[12]), float(p[13]), 1.0])
                ds.append(np.linalg.norm(L[:, :3] - f(c), axis=1).min())
        med = float(np.median(ds)) if ds else np.nan
        print(f"    {name:<24} {med:7.3f} m  frac<2m={np.mean(np.array(ds)<2):.2f}  n={len(ds)}")
        if med < best[0]:
            best = (med, name)
    good = best[1].startswith("cam->lidar")
    ok &= good
    print(f"    -> best: {best[1]} @ {best[0]:.3f} m  "
          f"{'OK' if good else '*** FRAME MISMATCH ***'}")

    # 2. ground estimate stability
    gz = []
    for fid in ids[:20]:
        L = np.fromfile(os.path.join(LIDAR, "velodyne", f"{fid}.bin"),
                        dtype=np.float32).reshape(-1, 4)
        gz.append(ground_z(L))
    print(f"\n  ground plane: mean {np.mean(gz):+.2f} m  sd {np.std(gz):.3f} m  "
          f"{'OK (stable)' if np.std(gz) < 0.3 else '*** UNSTABLE ***'}")
    ok &= np.std(gz) < 0.3

    # 3. sparsity sweep + channel distinctness
    print("\n  measurement sparsity sweep and channel distinctness:")
    for nf in (1, 3, 5):
        sc, me, meta = load_vod("val", n_radar_frames=nf, limit=12, verbose=False)
        distinct = all(len({me[i, c].tobytes() for c in range(4)}) == 4 for i in range(len(me)))
        print(f"    {nf}-frame radar: {meta['mean_radar_pts']:7.1f} pts  "
              f"meas fill={100*(me[:,0]>0).mean():5.2f}%  "
              f"scene fill={100*(sc[:,0]>0.5).mean():5.2f}%  "
              f"4 channels distinct: {'OK' if distinct else '*** FAIL ***'}")
        ok &= distinct
    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--split", default="val")
    ap.add_argument("--radar-frames", type=int, default=1, choices=(1, 3, 5))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest or not a.build:
        if not selftest():
            sys.exit(1)
    if a.build:
        print(f"\nbuilding split={a.split} radar_frames={a.radar_frames} ...")
        sc, me, meta = load_vod(a.split, a.radar_frames,
                               limit=a.limit or None, stride=a.stride)
        print(f"  scenes {sc.shape}  meas {me.shape}")
        print(f"  radar {meta['mean_radar_pts']:.0f} pts/frame vs lidar "
              f"{meta['mean_lidar_pts']:.0f}  (1:{meta['mean_lidar_pts']/max(meta['mean_radar_pts'],1):.0f})")
        print(f"  scene occupancy fill {100*(sc[:,0]>0.5).mean():.2f}%   "
              f"measurement fill {100*(me[:,0]>0).mean():.2f}%")
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                                    f"vod_{a.split}_r{a.radar_frames}.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, scenes=sc, meas=me)
        json.dump(meta, open(out.replace(".npz", "_meta.json"), "w"), indent=2)
        print(f"  saved: {out}")
        print("\nNEXT: run experiments/verify_new_dataset.py before training.")


if __name__ == "__main__":
    main()
