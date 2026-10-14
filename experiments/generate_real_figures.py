#!/usr/bin/env python3
"""
Generate publication-quality figures for the S2S-Diff paper using real DeepSense 6G data.

Figure 1: Real-world ISAC data visualization (2x5 grid)
  - Row 1 & 2: Different frames from Scenario 33 (test data)
  - Columns: Camera | Radar RD | LiDAR BEV | S2S-Diff Reconstruction | Uncertainty

Figure 2: Posterior diversity visualization
  - For 1 test frame: 4 posterior samples + mean + uncertainty
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# -- Project imports --
sys.path.insert(0, os.path.dirname(__file__))
from models import (SceneVAE, MeasurementEncoder, ConditionalDiffusionUNet,
                    GaussianDiffusion, UNetDirect)
from preprocess_deepsense import read_ply_ascii, lidar_to_bev, radar_to_range_doppler

# -- Paths --
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SCENARIO33_DIR = os.path.join(PROJECT_ROOT, "datasets", "deepsense6g", "Scenario33", "unit1")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "real")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- Matplotlib global settings for publication quality --
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.02,
    "text.usetex": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})


# ===========================================================================
# Data loading helpers
# ===========================================================================

def load_camera_image(frame_id):
    """Load camera JPG for a given frame index."""
    path = os.path.join(SCENARIO33_DIR, "camera_data", f"image_{frame_id}.jpg")
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path))


def load_radar_rd(frame_id):
    """Load radar data and compute range-Doppler magnitude (mean over antennas)."""
    path = os.path.join(SCENARIO33_DIR, "radar_data", f"radar_data_{frame_id}.npy")
    if not os.path.exists(path):
        return None
    radar_raw = np.load(path)  # [4, 256, 250] complex
    # FFT along axis=2 (chirps/Doppler dimension)
    rd = np.fft.fftshift(np.fft.fft(radar_raw, axis=2), axes=2)
    rd_mag = np.abs(rd)  # [4, 256, 250]
    # Mean magnitude over antennas for visualization
    rd_mean = rd_mag.mean(axis=0)  # [256, 250]
    return rd_mean


def normalize_radar_for_display(rd_mag):
    """
    Normalize radar range-Doppler for display.
    Removes zero-Doppler DC line by replacing with noise-matched fill,
    then normalizes with dB-scale contrast showing full signal detail.
    Returns an array in [0, 1].
    """
    rd = rd_mag.copy()
    n_range, n_doppler = rd.shape

    # --- 1. Remove zero-Doppler DC line ---
    # Detect columns with anomalously high energy relative to background
    col_means = np.array([rd[:, c].mean() for c in range(n_doppler)])
    bg_median = np.median(col_means)
    bg_mad = np.median(np.abs(col_means - bg_median))  # median absolute deviation

    # Mark any column > 5 MAD above median as DC-contaminated
    threshold = bg_median + 8 * max(bg_mad, 1e-10)
    dc_mask = col_means > threshold

    # Get background statistics from clean columns
    clean_cols = np.where(~dc_mask)[0]
    if len(clean_cols) > 20:
        bg_data = rd[:, clean_cols]
        bg_mean_per_row = bg_data.mean(axis=1)
        bg_std_per_row = bg_data.std(axis=1)
    else:
        bg_mean_per_row = np.median(rd, axis=1)
        bg_std_per_row = np.std(rd, axis=1) * 0.3

    # Replace contaminated columns with noise matching background
    rng_fill = np.random.RandomState(42)
    dc_cols = np.where(dc_mask)[0]
    for c in dc_cols:
        rd[:, c] = np.abs(bg_mean_per_row + bg_std_per_row * rng_fill.randn(n_range) * 0.8)

    # --- 2. Convert to dB ---
    rd_db = 20 * np.log10(rd + 1e-10)

    # --- 3. Percentile-based contrast ---
    p_low = np.percentile(rd_db, 5)
    p_high = np.percentile(rd_db, 99.95)
    rd_norm = (rd_db - p_low) / (p_high - p_low + 1e-10)
    rd_norm = np.clip(rd_norm, 0, 1)

    return rd_norm


def load_lidar_bev(frame_id, grid_size=64):
    """Load LiDAR PLY and project to BEV occupancy grid."""
    path = os.path.join(SCENARIO33_DIR, "lidar_data", f"lidar_data_{frame_id}.ply")
    if not os.path.exists(path):
        return None, None
    pts = read_ply_ascii(path)
    if len(pts) < 10:
        return None, None

    # Determine spatial extent from the point cloud itself
    x_med = np.median(pts[:, 0])
    y_med = np.median(pts[:, 1])
    half = grid_size * 0.5
    x_range = (x_med - half, x_med + half)
    y_range = (y_med - half, y_med + half)

    occ, ref = lidar_to_bev(pts, grid_size, x_range, y_range, z_range=(-2, 10))
    return occ, ref


def get_available_frame_ids(n_desired=6, min_lidar_fill=0.03):
    """
    Get frame IDs that have all three modalities and sufficient LiDAR content.
    Filters for frames with enough occupied BEV cells for interesting visualization.
    """
    camera_files = sorted(glob.glob(os.path.join(SCENARIO33_DIR, "camera_data", "image_*.jpg")))
    all_ids = []
    for f in camera_files:
        fid = int(os.path.basename(f).replace("image_", "").replace(".jpg", ""))
        radar_path = os.path.join(SCENARIO33_DIR, "radar_data", f"radar_data_{fid}.npy")
        lidar_path = os.path.join(SCENARIO33_DIR, "lidar_data", f"lidar_data_{fid}.ply")
        if os.path.exists(radar_path) and os.path.exists(lidar_path):
            all_ids.append(fid)

    if len(all_ids) == 0:
        raise RuntimeError("No frames found with all three modalities!")

    # Filter for frames with interesting LiDAR content
    # Sample a subset to avoid loading all files
    rng = np.random.RandomState(42)
    candidate_ids = rng.choice(all_ids, size=min(200, len(all_ids)), replace=False)
    good_ids = []
    for fid in sorted(candidate_ids):
        occ, _ = load_lidar_bev(fid)
        if occ is not None:
            fill = occ.mean()
            if fill >= min_lidar_fill:
                good_ids.append((fid, fill))

    if len(good_ids) < n_desired:
        # Fall back to evenly spaced frames
        n = len(all_ids)
        step = max(1, n // (n_desired + 1))
        return [all_ids[min(i * step, n - 1)] for i in range(1, n_desired + 1)]

    # Sort by fill rate and pick diverse ones (spread through the sorted list)
    good_ids.sort(key=lambda x: x[1], reverse=True)
    # Take from top half (good content) with some spread
    top_half = good_ids[:max(n_desired * 3, len(good_ids) // 2)]
    # Sort by frame ID to get temporal diversity
    top_half.sort(key=lambda x: x[0])
    step = max(1, len(top_half) // n_desired)
    selected = [top_half[min(i * step, len(top_half) - 1)][0]
                for i in range(n_desired)]

    return selected[:n_desired]


# ===========================================================================
# Model loading and inference
# ===========================================================================

def load_models(device):
    """Load all trained models."""
    vae = SceneVAE(3, 8, 32).to(device)
    encoder = MeasurementEncoder(4, 32, 128, 4, 2).to(device)
    diff_unet = ConditionalDiffusionUNet(8, 64, (1, 2, 4), 128, 4, 0.1, 2).to(device)
    unet = UNetDirect(4, 3, 64).to(device)
    diffusion = GaussianDiffusion(200, 1e-4, 0.02, device=device)

    vae.load_state_dict(torch.load(os.path.join(CKPT_DIR, "vae_real.pt"),
                                    map_location=device, weights_only=True))
    encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "enc_real.pt"),
                                        map_location=device, weights_only=True))
    diff_unet.load_state_dict(torch.load(os.path.join(CKPT_DIR, "diff_real.pt"),
                                          map_location=device, weights_only=True))
    unet.load_state_dict(torch.load(os.path.join(CKPT_DIR, "unet_real.pt"),
                                     map_location=device, weights_only=True))

    vae.eval()
    encoder.eval()
    diff_unet.eval()
    unet.eval()

    return vae, encoder, diff_unet, unet, diffusion


def compute_latent_stats(vae, measurements_batch, device):
    """Load or compute latent normalization statistics."""
    latent_stats_path = os.path.join(os.path.dirname(__file__), "checkpoints", "latent_stats.pt")
    if os.path.exists(latent_stats_path):
        stats = torch.load(latent_stats_path, map_location=device, weights_only=True)
        if isinstance(stats, dict):
            return stats.get("mean", torch.zeros(1, 8, 16, 16, device=device)), \
                   stats.get("std", torch.ones(1, device=device))
    return torch.zeros(1, 8, 16, 16, device=device), torch.tensor(1.0, device=device)


def prepare_measurement_tensor(frame_id, grid_size=64):
    """Prepare a measurement tensor from raw radar data, same as preprocessing."""
    radar_path = os.path.join(SCENARIO33_DIR, "radar_data", f"radar_data_{frame_id}.npy")
    radar_raw = np.load(radar_path)  # [4, 256, 250] complex
    rd = np.fft.fftshift(np.fft.fft(radar_raw, axis=2), axes=2)
    rd_mag = np.abs(rd).astype(np.float32)  # [4, 256, 250]

    # Resize to grid size
    t = torch.from_numpy(rd_mag).unsqueeze(0)  # [1, 4, 256, 250]
    t = F.interpolate(t, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0).numpy()  # [4, 64, 64]

    # Normalize
    rd_max = t.max()
    if rd_max > 0:
        t = t / rd_max

    return torch.from_numpy(t).float()


@torch.no_grad()
def s2s_diff_inference(diff_unet, encoder, vae, diffusion, measurements,
                       unet_coarse, latent_mean, latent_std,
                       n_samples=8, n_steps=50, t_start=80, device="cuda"):
    """S2S-Diff inference with SDEdit refinement."""
    B = measurements.shape[0]
    context = encoder(measurements)

    # Coarse estimate from U-Net
    coarse_scene = unet_coarse(measurements)
    mu_coarse, _ = vae.encode(coarse_scene)
    z0_coarse = (mu_coarse - latent_mean) / latent_std

    all_scenes = []
    for s in range(n_samples):
        t = torch.full((B,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(z0_coarse)
        z, _ = diffusion.q_sample(z0_coarse, t, noise)

        # Reverse diffusion
        step_size = max(1, t_start // n_steps)
        timesteps = list(range(t_start, -1, -step_size))
        if 0 not in timesteps:
            timesteps.append(0)

        for i, t_val in enumerate(timesteps[:-1]):
            t_cur = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_next = timesteps[i + 1]

            alpha_t = diffusion.alpha_bar[t_val]
            alpha_prev = diffusion.alpha_bar[t_next] if t_next > 0 \
                else torch.ones(1, device=device)

            eps_pred = diff_unet(z, t_cur, context)
            z0_pred = (z - (1 - alpha_t).sqrt() * eps_pred) / (alpha_t.sqrt() + 1e-8)

            eta = 0.3 if n_samples > 1 else 0.0
            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t + 1e-8)).sqrt() * \
                    (1 - alpha_t / (alpha_prev + 1e-8)).sqrt()
            dir_z = (1 - alpha_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
            rand_noise = torch.randn_like(z) if i + 1 < len(timesteps) - 1 \
                else torch.zeros_like(z)
            z = alpha_prev.sqrt() * z0_pred + dir_z + sigma * rand_noise

        z_denorm = z * latent_std + latent_mean
        scene = vae.decode(z_denorm)
        all_scenes.append(scene)

    samples = torch.stack(all_scenes, dim=0)  # [n_samples, B, 3, 64, 64]
    mean_scene = samples.mean(dim=0)
    uncertainty = samples.var(dim=0) if n_samples > 1 else torch.zeros_like(mean_scene)

    return mean_scene, uncertainty, samples


# ===========================================================================
# Figure 1: Real-world ISAC data visualization
# ===========================================================================

def generate_figure1(device):
    """Generate Figure 1: 2-row real-world ISAC data visualization."""
    print("Generating Figure 1: Real-world ISAC data visualization...")

    # Load models
    vae, encoder, diff_unet, unet, diffusion = load_models(device)
    latent_mean, latent_std = compute_latent_stats(vae, None, device)

    # Select frames with good LiDAR content
    frame_ids = get_available_frame_ids(n_desired=6, min_lidar_fill=0.03)
    selected_frames = [frame_ids[1], frame_ids[4]]
    print(f"  Selected frame IDs: {selected_frames}")

    # Prepare measurements for model inference
    meas_tensors = []
    for fid in selected_frames:
        mt = prepare_measurement_tensor(fid)
        meas_tensors.append(mt)
    meas_batch = torch.stack(meas_tensors, dim=0).to(device)  # [2, 4, 64, 64]

    # Run S2S-Diff inference
    mean_scene, uncertainty, samples = s2s_diff_inference(
        diff_unet, encoder, vae, diffusion, meas_batch, unet,
        latent_mean, latent_std, n_samples=8, n_steps=50, device=device
    )

    # Compute shared uncertainty vmax for consistent colorbar across rows
    unc_arrays = []
    for row_idx in range(2):
        unc_arrays.append(uncertainty[row_idx, 0].cpu().numpy())
    unc_vmax = max(a.max() for a in unc_arrays)
    unc_vmax = max(unc_vmax, 1e-6)

    # -- Create figure --
    # IEEE two-column: full width ~7.16 inches
    fig_width = 7.16
    fig_height = 3.2
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use gridspec for precise control
    gs = gridspec.GridSpec(2, 5, figure=fig,
                           wspace=0.04, hspace=0.15,
                           left=0.05, right=0.93, top=0.88, bottom=0.02)

    col_titles = [
        "Camera",
        "Radar R-D",
        "LiDAR BEV",
        "S2S-Diff recon.",
        "Uncertainty"
    ]

    for row_idx, fid in enumerate(selected_frames):
        # Col 0: Camera image
        ax0 = fig.add_subplot(gs[row_idx, 0])
        cam = load_camera_image(fid)
        if cam is not None:
            ax0.imshow(cam)
        ax0.axis("off")

        # Col 1: Radar range-Doppler with percentile contrast
        ax1 = fig.add_subplot(gs[row_idx, 1])
        rd = load_radar_rd(fid)
        if rd is not None:
            rd_vis = normalize_radar_for_display(rd)
            ax1.imshow(rd_vis, cmap="magma", aspect="auto", vmin=0, vmax=1)
        ax1.axis("off")

        # Col 2: LiDAR BEV occupancy
        ax2 = fig.add_subplot(gs[row_idx, 2])
        occ, ref = load_lidar_bev(fid)
        if occ is not None:
            ax2.imshow(occ, cmap="gray_r", vmin=0, vmax=1,
                       origin="lower", aspect="equal")
        ax2.axis("off")

        # Col 3: S2S-Diff reconstruction (occupancy channel)
        ax3 = fig.add_subplot(gs[row_idx, 3])
        recon = mean_scene[row_idx, 0].cpu().numpy()
        ax3.imshow(recon, cmap="gray_r", vmin=0, vmax=1,
                   origin="lower", aspect="equal")
        ax3.axis("off")

        # Col 4: Uncertainty map
        ax4 = fig.add_subplot(gs[row_idx, 4])
        unc = unc_arrays[row_idx]
        im_unc = ax4.imshow(unc, cmap="inferno", vmin=0, vmax=unc_vmax,
                            origin="lower", aspect="equal")
        ax4.axis("off")

        # Row labels
        row_label = f"Frame {row_idx + 1}"
        ax0.text(-0.08, 0.5, row_label, transform=ax0.transAxes,
                 fontsize=9, rotation=90, va="center", ha="right",
                 fontweight="bold")

        # Subplot labels
        axes_row = [ax0, ax1, ax2, ax3, ax4]
        for col_idx, ax in enumerate(axes_row):
            label_char = chr(ord('a') + row_idx * 5 + col_idx)
            ax.text(0.03, 0.94, f"({label_char})", transform=ax.transAxes,
                    fontsize=8, fontweight="bold", color="white",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="black",
                              alpha=0.7, edgecolor="none"))

            # Column titles (only on first row)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=8, fontweight="bold",
                             pad=3)

    # Add a thin colorbar for the uncertainty column
    cbar_ax = fig.add_axes([0.94, 0.08, 0.012, 0.74])
    cbar = fig.colorbar(im_unc, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6.5, width=0.4, length=2)
    cbar.outline.set_linewidth(0.4)

    save_path = os.path.join(RESULTS_DIR, "real_data_visualization.pdf")
    fig.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"  Saved Figure 1 to: {save_path}")
    return save_path


# ===========================================================================
# Figure 2: Posterior diversity visualization
# ===========================================================================

def generate_figure2(device):
    """Generate Figure 2: Posterior diversity - multiple samples from one input."""
    print("Generating Figure 2: Posterior diversity visualization...")

    # Load models
    vae, encoder, diff_unet, unet, diffusion = load_models(device)
    latent_mean, latent_std = compute_latent_stats(vae, None, device)

    # Pick one frame with good content
    frame_ids = get_available_frame_ids(n_desired=4, min_lidar_fill=0.03)
    fid = frame_ids[1]
    print(f"  Using frame ID: {fid}")

    # Prepare measurement
    meas_tensor = prepare_measurement_tensor(fid).unsqueeze(0).to(device)  # [1, 4, 64, 64]

    # Generate multiple posterior samples
    mean_scene, uncertainty, samples = s2s_diff_inference(
        diff_unet, encoder, vae, diffusion, meas_tensor, unet,
        latent_mean, latent_std, n_samples=8, n_steps=50, device=device
    )

    # Select 4 diverse samples
    sample_scenes = samples[:4, 0].cpu()  # [4, 3, 64, 64]
    mean_occ = mean_scene[0, 0].cpu().numpy()
    unc_occ = uncertainty[0, 0].cpu().numpy()

    # Load raw data for context
    occ_gt, _ = load_lidar_bev(fid)
    rd = load_radar_rd(fid)

    # -- Create figure --
    fig_width = 7.16
    fig_height = 3.8

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           wspace=0.05, hspace=0.22,
                           left=0.02, right=0.92, top=0.92, bottom=0.02)

    panels = {}

    # Row 1: Input radar | GT LiDAR BEV | Sample 1 | Sample 2
    ax_radar = fig.add_subplot(gs[0, 0])
    if rd is not None:
        rd_vis = normalize_radar_for_display(rd)
        ax_radar.imshow(rd_vis, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax_radar.set_title("Input radar", fontsize=9, fontweight="bold", pad=3)
    ax_radar.axis("off")
    panels["a"] = ax_radar

    ax_gt = fig.add_subplot(gs[0, 1])
    if occ_gt is not None:
        ax_gt.imshow(occ_gt, cmap="gray_r", vmin=0, vmax=1,
                     origin="lower", aspect="equal")
    ax_gt.set_title("LiDAR ground truth", fontsize=9, fontweight="bold", pad=3)
    ax_gt.axis("off")
    panels["b"] = ax_gt

    for s_idx, col_idx in enumerate([2, 3]):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.imshow(sample_scenes[s_idx, 0].numpy(), cmap="gray_r", vmin=0, vmax=1,
                  origin="lower", aspect="equal")
        ax.set_title(f"Sample {s_idx + 1}", fontsize=9, fontweight="bold", pad=3)
        ax.axis("off")
        panels[chr(ord('c') + s_idx)] = ax

    # Row 2: Sample 3 | Sample 4 | Posterior mean | Uncertainty
    for s_idx, col_idx in enumerate([0, 1]):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.imshow(sample_scenes[s_idx + 2, 0].numpy(), cmap="gray_r", vmin=0, vmax=1,
                  origin="lower", aspect="equal")
        ax.set_title(f"Sample {s_idx + 3}", fontsize=9, fontweight="bold", pad=3)
        ax.axis("off")
        panels[chr(ord('e') + s_idx)] = ax

    ax_mean = fig.add_subplot(gs[1, 2])
    ax_mean.imshow(mean_occ, cmap="gray_r", vmin=0, vmax=1,
                   origin="lower", aspect="equal")
    ax_mean.set_title("Posterior mean", fontsize=9, fontweight="bold", pad=3)
    ax_mean.axis("off")
    panels["g"] = ax_mean

    ax_unc = fig.add_subplot(gs[1, 3])
    unc_vmax = max(unc_occ.max(), 1e-6)
    im_unc = ax_unc.imshow(unc_occ, cmap="inferno", vmin=0, vmax=unc_vmax,
                           origin="lower", aspect="equal")
    ax_unc.set_title("Uncertainty", fontsize=9, fontweight="bold", pad=3)
    ax_unc.axis("off")
    panels["h"] = ax_unc

    # Add colorbar for uncertainty
    cbar_ax = fig.add_axes([0.935, 0.06, 0.015, 0.38])
    cbar = fig.colorbar(im_unc, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7, width=0.4, length=2)
    cbar.set_label("Variance", fontsize=8, labelpad=2)
    cbar.outline.set_linewidth(0.4)

    # Add subplot labels
    for label_char, ax in panels.items():
        ax.text(0.03, 0.94, f"({label_char})", transform=ax.transAxes,
                fontsize=8, fontweight="bold", color="white",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="black",
                          alpha=0.7, edgecolor="none"))

    save_path = os.path.join(RESULTS_DIR, "posterior_diversity.pdf")
    fig.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"  Saved Figure 2 to: {save_path}")
    return save_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Verify data exists
    assert os.path.isdir(SCENARIO33_DIR), f"Scenario 33 data not found at {SCENARIO33_DIR}"
    assert os.path.exists(os.path.join(CKPT_DIR, "vae_real.pt")), "Missing vae_real.pt checkpoint"

    path1 = generate_figure1(device)
    path2 = generate_figure2(device)

    print("\nDone! Figures saved to:")
    print(f"  {path1}")
    print(f"  {path2}")


if __name__ == "__main__":
    main()
