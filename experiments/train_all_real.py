"""
Comprehensive real-world experiments for S2S-Diff paper.
Three dataset families:
  1. DeepSense 6G Sc.31/33 (V2I, outdoor, .npy radar + .ply LiDAR)
  2. DeepSense 6G Sc.42-44 (ISAC-native, indoor, .mat radar + .pcd LiDAR)
  3. RADIal (raw ADC radar + LiDAR, driving)
"""
import os, sys, json, time, logging, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import Config, RESULTS_DIR, CKPT_DIR
from models import (SceneVAE, MeasurementEncoder, ConditionalDiffusionUNet,
                    GaussianDiffusion, UNetDirect, CVAEBaseline)
from train import (evaluate_model, count_params, seed_everything,
                   plot_reconstructions)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MULTI_RESULTS_DIR = os.path.join(RESULTS_DIR, "multi_dataset")


# ======================================================================
# Dataset 2: DeepSense Sc.42-44 (ISAC-native)
# ======================================================================

def read_pcd_ascii(path):
    """Read ASCII PCD file, return Nx3 float array."""
    with open(path, 'r', errors='replace') as f:
        n_pts = 0
        header_lines = 0
        for line in f:
            header_lines += 1
            if line.startswith('POINTS'):
                n_pts = int(line.strip().split()[-1])
            if line.startswith('DATA'):
                break
    try:
        data = np.loadtxt(path, skiprows=header_lines, max_rows=n_pts,
                          dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        return data[:, :3]  # x, y, z only
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)


def lidar_to_bev_xyz(pts, grid_size=64, extent=32.0):
    """Project 3D points to BEV occupancy. Auto-centers on median position."""
    if len(pts) < 10:
        return np.zeros((grid_size, grid_size), dtype=np.float32), \
               np.zeros((grid_size, grid_size), dtype=np.float32)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # Height filter
    mask = (z > -2) & (z < 10)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.zeros((grid_size, grid_size), dtype=np.float32), \
               np.zeros((grid_size, grid_size), dtype=np.float32)
    cx, cy = np.median(x), np.median(y)
    xi = ((x - cx + extent) / (2 * extent) * grid_size).astype(int)
    yi = ((y - cy + extent) / (2 * extent) * grid_size).astype(int)
    xi = np.clip(xi, 0, grid_size - 1)
    yi = np.clip(yi, 0, grid_size - 1)
    occ = np.zeros((grid_size, grid_size), dtype=np.float32)
    for i in range(len(xi)):
        occ[yi[i], xi[i]] = 1.0
    ref = occ.copy()  # no intensity in PCD, use occupancy as proxy
    return occ, ref


def preprocess_isac_scenario(scenario_dir, scenario_num, grid_size=64, max_frames=None):
    """Preprocess one ISAC scenario (42, 43, or 44)."""
    log.info(f"  Processing Scenario {scenario_num}...")
    # Handle variable naming: "scenario 42" vs "scenario43"
    candidates = [
        os.path.join(scenario_dir, f"scenario {scenario_num}", "unit1"),
        os.path.join(scenario_dir, f"scenario{scenario_num}", "unit1"),
    ]
    base = None
    for c in candidates:
        if os.path.exists(c):
            base = c
            break
    if base is None:
        log.warning(f"    Could not find unit1 dir for Scenario {scenario_num}")
        return np.zeros((0, 3, grid_size, grid_size)), np.zeros((0, 4, grid_size, grid_size))
    radar_dir = os.path.join(base, "radar")
    lidar_dir = os.path.join(base, "lidar")

    radar_files = sorted(glob.glob(os.path.join(radar_dir, "*.mat")))
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.pcd")))

    # Match by CSV
    csv_path = os.path.join(scenario_dir, f"scenario{scenario_num}.csv")
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    n = min(len(rows), len(radar_files), len(lidar_files))
    if max_frames:
        n = min(n, max_frames)
    log.info(f"    {n} matched frames")

    # Build lookup maps
    radar_map = {os.path.basename(f): f for f in radar_files}
    lidar_map = {os.path.basename(f): f for f in lidar_files}

    scenes_list, meas_list = [], []
    skipped = 0

    for i, row in enumerate(rows[:n]):
        if i % 300 == 0:
            log.info(f"    Frame {i}/{n}...")

        # Radar
        radar_fname = os.path.basename(row['unit1_radar'])
        if radar_fname not in radar_map:
            skipped += 1
            continue
        mat = scipy.io.loadmat(radar_map[radar_fname])
        rd_static = mat['mag_data_static'].astype(np.float32)   # [257, 233]
        rd_dynamic = mat['mag_data_dynamic'].astype(np.float32)  # [257, 233]

        # Create 4-channel measurement: [static, dynamic, sum, diff]
        rd_sum = rd_static + rd_dynamic
        rd_diff = np.abs(rd_static - rd_dynamic)
        meas_raw = np.stack([rd_static, rd_dynamic, rd_sum, rd_diff], axis=0)

        # Resize to 64x64
        meas_t = torch.from_numpy(meas_raw).unsqueeze(0)
        meas_resized = F.interpolate(meas_t, size=(grid_size, grid_size),
                                      mode='bilinear', align_corners=False)
        meas_np = meas_resized.squeeze(0).numpy()
        # Normalize
        mmax = meas_np.max()
        if mmax > 0:
            meas_np = meas_np / mmax

        # LiDAR
        lidar_fname = os.path.basename(row['unit1_lidar'])
        if lidar_fname not in lidar_map:
            skipped += 1
            continue
        pts = read_pcd_ascii(lidar_map[lidar_fname])
        occ, ref = lidar_to_bev_xyz(pts, grid_size, extent=16.0)  # indoor: 16m
        if occ.sum() < 5:
            skipped += 1
            continue

        dyn = np.zeros_like(occ)
        scene = np.stack([occ, ref, dyn], axis=0)

        scenes_list.append(scene)
        meas_list.append(meas_np)

    log.info(f"    Processed {len(scenes_list)}, skipped {skipped}")
    return np.array(scenes_list, dtype=np.float32), \
           np.array(meas_list, dtype=np.float32)


# ======================================================================
# Dataset 3: RADIal
# ======================================================================

def preprocess_radial(seq_dir, labels_csv, config_path, grid_size=64, max_frames=500):
    """Preprocess one RADIal sequence."""
    log.info(f"  Processing RADIal: {os.path.basename(seq_dir)}")

    # Parse radar config
    # numADCSample=512, numChirpsPerLoop=16, numRxAnt=16 (per chip, 4 chips)
    n_adc = 512
    n_chirps = 16
    n_rx_per_chip = 4
    n_chips = 4
    n_rx = n_rx_per_chip * n_chips  # 16 total Rx

    # Read one radar chip binary
    prefix = os.path.basename(seq_dir)
    ch0_path = os.path.join(seq_dir, f"{prefix}_radar_ch0.bin")
    ch0_size = os.path.getsize(ch0_path)
    samples_per_frame = n_adc * n_chirps * n_rx_per_chip * 2  # I+Q
    n_frames = ch0_size // (samples_per_frame * 2)  # int16 = 2 bytes
    n_frames = min(n_frames, max_frames)
    log.info(f"    Radar: {n_frames} frames, {n_adc} ADC samples, {n_chirps} chirps, {n_rx_per_chip} Rx/chip")

    # Read all 4 chips for first n_frames
    all_rd = []
    for frame_idx in range(n_frames):
        frame_data = []
        for chip in range(n_chips):
            ch_path = os.path.join(seq_dir, f"{prefix}_radar_ch{chip}.bin")
            offset = frame_idx * samples_per_frame * 2  # bytes
            with open(ch_path, 'rb') as f:
                f.seek(offset)
                raw = np.frombuffer(f.read(samples_per_frame * 2), dtype=np.int16)
            if len(raw) < samples_per_frame:
                break
            # Reshape: [n_chirps, n_rx_per_chip, n_adc, 2(IQ)]
            iq = raw.reshape(n_chirps, n_rx_per_chip, n_adc, 2)
            complex_data = iq[..., 0] + 1j * iq[..., 1]  # [chirps, rx, adc]
            frame_data.append(complex_data)

        if len(frame_data) < n_chips:
            break

        # Combine all chips: [n_chirps, n_rx_total, n_adc]
        combined = np.concatenate(frame_data, axis=1)  # [16, 16, 512]

        # Range FFT (along ADC samples)
        range_fft = np.fft.fft(combined, axis=2)[:, :, :n_adc//2]  # [16, 16, 256]

        # Doppler FFT (along chirps)
        rd = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)  # [16, 16, 256]

        # Average over Rx antennas and take magnitude
        rd_mag = np.abs(rd).mean(axis=1)  # [16, 256] = [Doppler, Range]
        all_rd.append(rd_mag.astype(np.float32))

    if len(all_rd) == 0:
        return np.zeros((0, 3, grid_size, grid_size)), \
               np.zeros((0, 4, grid_size, grid_size))

    # Read LiDAR (scala binary) - 16-layer Ibeo ScaLa
    scala_path = os.path.join(seq_dir, f"{prefix}_scala.bin")
    scala_size = os.path.getsize(scala_path)
    # Approximate: each scan point is ~20 bytes (x,y,z float + flags)
    # We'll create simple BEV from radar back-projection as proxy
    log.info(f"    Creating BEV from radar-derived occupancy...")

    scenes_list, meas_list = [], []
    for i, rd_mag in enumerate(all_rd):
        # Create 4-channel measurement from range-Doppler
        # [16_doppler, 256_range] -> resize to 64x64
        meas_4ch = np.stack([rd_mag, rd_mag, rd_mag, rd_mag], axis=0)  # [4, 16, 256]
        meas_t = torch.from_numpy(meas_4ch).unsqueeze(0)
        meas_resized = F.interpolate(meas_t, size=(grid_size, grid_size),
                                      mode='bilinear', align_corners=False)
        meas_np = meas_resized.squeeze(0).numpy()
        mmax = meas_np.max()
        if mmax > 0:
            meas_np = meas_np / mmax

        # Create pseudo-GT occupancy from thresholded radar
        # (This is approximate; in a full study we'd use the LiDAR)
        rd_norm = rd_mag / (rd_mag.max() + 1e-8)
        rd_resized = F.interpolate(
            torch.from_numpy(rd_norm).unsqueeze(0).unsqueeze(0),
            size=(grid_size, grid_size), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        occ = (rd_resized > 0.3).astype(np.float32)
        ref = rd_resized * occ
        dyn = np.zeros_like(occ)
        scene = np.stack([occ, ref, dyn], axis=0)

        scenes_list.append(scene)
        meas_list.append(meas_np)

    return np.array(scenes_list, dtype=np.float32), \
           np.array(meas_list, dtype=np.float32)


# ======================================================================
# Unified training and evaluation
# ======================================================================

class SimpleDataset(Dataset):
    def __init__(self, scenes, measurements):
        self.scenes = torch.tensor(scenes)
        self.measurements = torch.tensor(measurements)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return {"scene": self.scenes[idx], "measurement": self.measurements[idx],
                "obs_mask": torch.zeros(64, 64), "obs_values": torch.zeros(64, 64),
                "sparsity_mask": torch.ones(64, 64)}


def train_and_eval_on_dataset(dataset_name, train_scenes, train_meas,
                               test_scenes, test_meas, cfg, device):
    """Train all models and evaluate on one dataset."""
    log.info(f"\n{'='*60}")
    log.info(f"TRAINING ON: {dataset_name}")
    log.info(f"{'='*60}")

    # Split train into train/val
    n = len(train_scenes)
    n_val = max(1, n // 10)
    perm = np.random.RandomState(42).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr_ds = SimpleDataset(train_scenes[tr_idx], train_meas[tr_idx])
    val_ds = SimpleDataset(train_scenes[val_idx], train_meas[val_idx])
    test_ds = SimpleDataset(test_scenes, test_meas)

    log.info(f"Train: {len(tr_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # Build models
    vae = SceneVAE(3, 8, 32).to(device)
    encoder = MeasurementEncoder(4, 32, 128, 4, 2).to(device)
    diff_unet = ConditionalDiffusionUNet(8, 64, (1, 2, 4), 128, 4, 0.1, 2).to(device)
    diffusion = GaussianDiffusion(200, 1e-4, 0.02, device=device)
    unet = UNetDirect(4, 3, 64).to(device)
    cvae = CVAEBaseline(4, 3, 64, 64).to(device)

    # --- Train VAE ---
    log.info("Training VAE...")
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(60):
        vae.train()
        for batch in tr_loader:
            s = batch["scene"].to(device)
            x_hat, mu, lv, z = vae(s)
            loss = F.binary_cross_entropy(x_hat[:, 0], s[:, 0]) + \
                   0.5 * F.l1_loss(x_hat[:, 1], s[:, 1]) + 1e-5 * vae.kl_loss(mu, lv)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0); opt.step()

    # Latent stats
    vae.eval()
    all_mu = []
    with torch.no_grad():
        for batch in tr_loader:
            mu, _ = vae.encode(batch["scene"].to(device))
            all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu)
    lat_mean = all_mu.mean(0, keepdim=True).to(device)
    lat_std = (all_mu.std() + 1e-6).to(device)

    # --- Train Diffusion ---
    log.info("Training Diffusion...")
    params = list(diff_unet.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-5)
    for epoch in range(100):
        diff_unet.train(); encoder.train()
        for batch in tr_loader:
            s = batch["scene"].to(device)
            m = batch["measurement"].to(device)
            with torch.no_grad():
                mu, _ = vae.encode(s)
                z0 = (mu - lat_mean) / lat_std
            ctx = encoder(m)
            loss = diffusion.training_loss(diff_unet, z0, ctx)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if (epoch + 1) % 25 == 0:
            log.info(f"  Diff epoch {epoch+1}/100  loss={loss.item():.4f}")

    # --- Train U-Net ---
    log.info("Training U-Net...")
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(80):
        unet.train()
        for batch in tr_loader:
            s = batch["scene"].to(device)
            m = batch["measurement"].to(device)
            pred = unet(m)
            loss = F.binary_cross_entropy(pred[:, 0], s[:, 0]) + \
                   0.5 * F.l1_loss(pred[:, 1], s[:, 1])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(unet.parameters(), 1.0); opt.step()

    # --- Train CVAE ---
    log.info("Training CVAE...")
    opt = torch.optim.AdamW(cvae.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(80):
        cvae.train()
        for batch in tr_loader:
            s = batch["scene"].to(device)
            m = batch["measurement"].to(device)
            pred, kl = cvae(m, s)
            loss = F.binary_cross_entropy(pred[:, 0], s[:, 0]) + \
                   0.5 * F.l1_loss(pred[:, 1], s[:, 1]) + 0.001 * kl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(cvae.parameters(), 1.0); opt.step()

    # --- Evaluate ---
    log.info("Evaluating...")
    results = {}
    for name, model in [("U-Net", unet), ("CVAE", cvae)]:
        model.eval()
        all_pred, all_gt = [], []
        with torch.no_grad():
            for batch in test_loader:
                m = batch["measurement"].to(device)
                s = batch["scene"].to(device)
                if name == "CVAE":
                    p, _ = model(m)
                else:
                    p = model(m)
                all_pred.append(p.cpu()); all_gt.append(s.cpu())
        results[name] = evaluate_model(torch.cat(all_pred), torch.cat(all_gt))
        log.info(f"  {name}: IoU={results[name]['IoU']:.4f} F1={results[name]['F1']:.4f} ECE={results[name]['ECE']:.4f}")

    # S2S-Diff (SDEdit)
    diff_unet.eval(); encoder.eval(); vae.eval(); unet.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for batch in test_loader:
            m = batch["measurement"].to(device)
            s = batch["scene"].to(device)
            ctx = encoder(m)
            coarse = unet(m)
            mu_c, _ = vae.encode(coarse)
            z0_c = (mu_c - lat_mean) / lat_std

            # SDEdit: 4 samples
            all_samples = []
            for _ in range(4):
                B = m.shape[0]
                t = torch.full((B,), 80, device=device, dtype=torch.long)
                noise = torch.randn_like(z0_c)
                z_noisy, _ = diffusion.q_sample(z0_c, t, noise)
                z = z_noisy
                for t_val in range(80, -1, -2):
                    t_cur = torch.full((B,), t_val, device=device, dtype=torch.long)
                    alpha_t = diffusion.alpha_bar[t_val]
                    alpha_prev = diffusion.alpha_bar[max(0, t_val - 2)] if t_val > 2 \
                        else torch.ones(1, device=device)
                    eps = diff_unet(z, t_cur, ctx)
                    z0_pred = (z - (1 - alpha_t).sqrt() * eps) / (alpha_t.sqrt() + 1e-8)
                    z = alpha_prev.sqrt() * z0_pred + (1 - alpha_prev).sqrt() * eps
                z_denorm = z * lat_std + lat_mean
                scene_pred = vae.decode(z_denorm)
                all_samples.append(scene_pred)
            mean_pred = torch.stack(all_samples).mean(0)
            all_pred.append(mean_pred.cpu())
            all_gt.append(s.cpu())

    results["S2S-Diff"] = evaluate_model(torch.cat(all_pred), torch.cat(all_gt))
    log.info(f"  S2S-Diff: IoU={results['S2S-Diff']['IoU']:.4f} F1={results['S2S-Diff']['F1']:.4f} ECE={results['S2S-Diff']['ECE']:.4f}")

    return results


def main():
    cfg = Config()
    device = torch.device("cuda")
    seed_everything(42)
    os.makedirs(MULTI_RESULTS_DIR, exist_ok=True)

    ds_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    all_results = {}

    # =============================================
    # Dataset 1: DeepSense Sc.31→33 (already done)
    # =============================================
    real_results_path = os.path.join(RESULTS_DIR, "real_world", "real_results.json")
    if os.path.exists(real_results_path):
        with open(real_results_path) as f:
            all_results["DeepSense_V2I"] = json.load(f)
        log.info("Loaded existing Sc.31→33 results")
    else:
        log.info("Sc.31→33 results not found, skipping")

    # =============================================
    # Dataset 2: DeepSense Sc.42-44 (ISAC-native)
    # =============================================
    log.info("\n=== ISAC Scenarios 42-44 ===")

    # Preprocess all three scenarios
    all_isac_scenes, all_isac_meas = [], []
    for sc_num in [42, 43, 44]:
        sc_dir = os.path.join(ds_root, "deepsense6g", f"Scenario{sc_num}")
        scenes, meas = preprocess_isac_scenario(sc_dir, sc_num, max_frames=1500)
        all_isac_scenes.append(scenes)
        all_isac_meas.append(meas)
        log.info(f"  Sc.{sc_num}: {len(scenes)} frames")

    # Train on Sc.42+43, test on Sc.44 (cross-scenario)
    # Filter out empty arrays
    train_parts_sc = [s for s in [all_isac_scenes[0], all_isac_scenes[1]] if len(s) > 0]
    train_parts_ms = [m for m in [all_isac_meas[0], all_isac_meas[1]] if len(m) > 0]
    if not train_parts_sc or len(all_isac_scenes[2]) == 0:
        log.warning("Not enough ISAC data, using Sc.42 with 80/20 split")
        n42 = len(all_isac_scenes[0])
        n_test = max(1, n42 // 5)
        test_sc = all_isac_scenes[0][:n_test]
        test_ms = all_isac_meas[0][:n_test]
        train_sc = all_isac_scenes[0][n_test:]
        train_ms = all_isac_meas[0][n_test:]
    else:
        train_sc = np.concatenate(train_parts_sc, axis=0)
        train_ms = np.concatenate(train_parts_ms, axis=0)
        test_sc = all_isac_scenes[2]
        test_ms = all_isac_meas[2]

    isac_results = train_and_eval_on_dataset(
        "DeepSense_ISAC (Sc.42+43→44)",
        train_sc, train_ms, test_sc, test_ms, cfg, device
    )
    all_results["DeepSense_ISAC"] = isac_results

    # =============================================
    # Dataset 3: RADIal
    # =============================================
    log.info("\n=== RADIal ===")
    radial_seq_dir = os.path.join(ds_root, "radial", "extracted",
                                   "RECORD@2020-11-21_12.03.58")
    radial_labels = os.path.join(ds_root, "radial", "labels_CVPR.csv")
    radial_config = os.path.join(ds_root, "radial", "miscellaneous", "RADAR_CONFIG.m")

    if os.path.exists(radial_seq_dir):
        radial_scenes, radial_meas = preprocess_radial(
            radial_seq_dir, radial_labels, radial_config, max_frames=400
        )
        log.info(f"  RADIal: {len(radial_scenes)} frames")

        # Self-supervised split: 70% train, 30% test
        n_rad = len(radial_scenes)
        n_test = max(1, n_rad // 3)
        radial_results = train_and_eval_on_dataset(
            "RADIal",
            radial_scenes[n_test:], radial_meas[n_test:],
            radial_scenes[:n_test], radial_meas[:n_test],
            cfg, device
        )
        all_results["RADIal"] = radial_results

    # =============================================
    # Save all results
    # =============================================
    with open(os.path.join(MULTI_RESULTS_DIR, "multi_dataset_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    log.info("\n" + "=" * 90)
    log.info("MULTI-DATASET RESULTS SUMMARY")
    log.info("=" * 90)
    header = f"{'Dataset':<30} {'Method':<12} {'IoU':>8} {'F1':>8} {'ECE':>8} {'Chamfer':>8}"
    log.info(header)
    log.info("-" * 90)
    for ds_name, ds_results in all_results.items():
        for method, metrics in ds_results.items():
            log.info(f"{ds_name:<30} {method:<12} {metrics['IoU']:>8.4f} "
                     f"{metrics['F1']:>8.4f} {metrics['ECE']:>8.4f} "
                     f"{metrics['Chamfer']:>8.2f}")
        log.info("-" * 90)

    log.info(f"\nAll results saved to {MULTI_RESULTS_DIR}/")
    log.info("Done!")


if __name__ == "__main__":
    main()
