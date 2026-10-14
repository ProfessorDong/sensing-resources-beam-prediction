"""
Regenerate results/isac_reconstructions.pdf with larger fonts.
3 frames x 5 columns: Radar (input), LiDAR GT, U-Net recon, S2S-Diff recon, Uncertainty.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, SimpleForwardModel,
    UNetDirect, SimpleDS, _try_load,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISAC_CKPT = os.path.join(CKPT_DIR, "v2", "DeepSense_ISAC__Sc_42_43_44_")
OUT_PATH = os.path.join(ROOT, "results", "isac_reconstructions.pdf")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    # Load Sc.44 test data
    print("Loading ISAC Sc.44 test data...")
    from train_all_real import preprocess_isac_scenario
    sc_dir = os.path.join(ROOT, "datasets", "deepsense6g", "Scenario44")
    scenes, meas = preprocess_isac_scenario(sc_dir, 44, max_frames=1500)
    print(f"  loaded {len(scenes)} frames")

    # Pick 3 test frames spread out
    frame_idxs = [50, 500, 1000]
    frames_to_show = [(scenes[i], meas[i]) for i in frame_idxs]

    # Models
    diff_model = PixelDiffusionUNet().to(device)
    fwd_model = SimpleForwardModel(scene_ch=3, meas_ch=4).to(device)
    diffusion = DiffusionV2(n_steps=300, device=device)
    unet = UNetDirect(in_ch=4, out_ch=3, base_ch=96).to(device)
    _try_load(diff_model, os.path.join(ISAC_CKPT, "diff_v2.pt"), "diff_v2")
    _try_load(fwd_model, os.path.join(ISAC_CKPT, "fwd_v2.pt"), "fwd_v2")
    _try_load(unet, os.path.join(ISAC_CKPT, "unet_v2.pt"), "unet_v2")
    diff_model.eval(); unet.eval()

    # Generate predictions
    print("Generating predictions...")
    radar_imgs, lidar_imgs, unet_imgs, s2s_imgs, unc_imgs = [], [], [], [], []
    with torch.no_grad():
        for scene_np, meas_np in frames_to_show:
            scene_t = torch.from_numpy(scene_np).unsqueeze(0).float()
            meas_t = torch.from_numpy(meas_np).unsqueeze(0).float().to(device)
            # U-Net
            unet_out = torch.sigmoid(unet(meas_t))[0, 0].cpu().numpy()
            # S2S-Diff (S=4 samples, average)
            samples = []
            for _ in range(4):
                s = diffusion.ddim_sample(diff_model, meas_t, n_steps=50, eta=0.3)
                samples.append(s)
            stack = torch.stack(samples)
            s2s_mean = stack.mean(0)[0, 0].cpu().numpy()
            s2s_var = stack.var(0)[0, 0].cpu().numpy()
            # Radar input: combined magnitude (channel 2 is the "sum" channel)
            radar_imgs.append(meas_np[2])
            # LiDAR GT: occupancy channel
            lidar_imgs.append(scene_np[0])
            unet_imgs.append(unet_out)
            s2s_imgs.append(s2s_mean)
            unc_imgs.append(s2s_var)

    # Plot
    print("Plotting with enlarged fonts...")
    n_rows, n_cols = 3, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 6.5),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.10})

    # Title font sizes (LARGER than original)
    TITLE_FS = 14
    ROW_FS = 13
    CBAR_FS = 12
    CBAR_LABEL_FS = 13

    col_titles = ["Radar (input)", "LiDAR GT", "U-Net recon.", "S2S-Diff recon.", "Uncertainty"]
    row_labels = [f"Frame {i+1}" for i in range(n_rows)]

    # Shared uncertainty vmax
    unc_vmax = max(u.max() for u in unc_imgs)
    unc_vmax = max(unc_vmax, 1e-6)

    im_unc = None
    for r in range(n_rows):
        # Radar input: log scale + magma
        rd = radar_imgs[r]
        rd_db = 20 * np.log10(np.abs(rd) + 1e-10)
        rd_norm = (rd_db - rd_db.min()) / (rd_db.max() - rd_db.min() + 1e-8)
        axes[r, 0].imshow(rd_norm, cmap="magma", aspect="auto", vmin=0, vmax=1)
        # LiDAR GT
        axes[r, 1].imshow(lidar_imgs[r], cmap="gray_r", vmin=0, vmax=1)
        # U-Net
        axes[r, 2].imshow(unet_imgs[r], cmap="gray_r", vmin=0, vmax=1)
        # S2S-Diff
        axes[r, 3].imshow(s2s_imgs[r], cmap="gray_r", vmin=0, vmax=1)
        # Uncertainty
        im_unc = axes[r, 4].imshow(unc_imgs[r], cmap="inferno", vmin=0, vmax=unc_vmax)

        for c in range(n_cols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_linewidth(0.6)

        # Top-row column titles
        if r == 0:
            for c in range(n_cols):
                axes[0, c].set_title(col_titles[c], fontsize=TITLE_FS, fontweight="bold", pad=6)

        # Left-side row labels
        axes[r, 0].set_ylabel(row_labels[r], fontsize=ROW_FS, fontweight="bold",
                              rotation=90, labelpad=8)

    # Colorbar for uncertainty (right side)
    fig.subplots_adjust(left=0.06, right=0.91, top=0.95, bottom=0.03)
    cbar_ax = fig.add_axes([0.925, 0.05, 0.015, 0.88])
    cbar = fig.colorbar(im_unc, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=CBAR_FS, width=0.6, length=3)
    cbar.set_label("Variance", fontsize=CBAR_LABEL_FS, labelpad=4)
    cbar.outline.set_linewidth(0.6)

    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
