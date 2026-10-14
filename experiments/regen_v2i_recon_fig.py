"""
Regenerate results/real_data_visualization.pdf for V2I (Scenario 33) with
larger fonts and aspect ratio sized for ~0.9 textwidth two-column figure.
3 frames x 5 columns: Camera, Radar R-D, LiDAR BEV, S2S-Diff recon., Uncertainty.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, _try_load,
)
from generate_real_figures import (
    load_camera_image, load_radar_rd, load_lidar_bev,
    normalize_radar_for_display, get_available_frame_ids,
    prepare_measurement_tensor,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2I_CKPT = os.path.join(CKPT_DIR, "v2", "DeepSense_V2I__Sc_31_33_")
OUT_PATH = os.path.join(ROOT, "results", "real_data_visualization.pdf")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    # Pick 2 frames with good LiDAR coverage (matches original layout)
    print("Selecting V2I frames...")
    frame_ids = get_available_frame_ids(n_desired=6, min_lidar_fill=0.03)
    selected_frames = [frame_ids[1], frame_ids[4]]
    print(f"  selected: {selected_frames}")

    # Load V2I models
    diff_model = PixelDiffusionUNet().to(device)
    diffusion = DiffusionV2(n_steps=300, device=device)
    _try_load(diff_model, os.path.join(V2I_CKPT, "diff_v2.pt"), "diff_v2")
    diff_model.eval()

    # Inference per frame
    print("Generating S2S-Diff predictions...")
    meas_tensors = []
    for fid in selected_frames:
        mt = prepare_measurement_tensor(fid)
        meas_tensors.append(mt)
    meas_batch = torch.stack(meas_tensors, dim=0).to(device)  # [2,4,64,64]

    # Generate S=8 samples for posterior mean + variance
    samples = []
    for _ in range(8):
        s = diffusion.ddim_sample(diff_model, meas_batch, n_steps=50, eta=0.3)
        samples.append(s)
    stack = torch.stack(samples)  # [S, 2, 3, 64, 64]
    mean_scene = stack.mean(0)  # [2, 3, 64, 64]
    uncertainty = stack.var(0)  # [2, 3, 64, 64]

    # Shared colorbar vmax
    unc_arrays = [uncertainty[i, 0].cpu().numpy() for i in range(2)]
    unc_vmax = max(a.max() for a in unc_arrays)
    unc_vmax = max(unc_vmax, 1e-6)

    # ---- Plot ----
    print("Plotting with enlarged fonts...")
    # IEEE two-column page: full text width ~ 7.16 in. Figure used at 0.9 textwidth.
    # Make the source figure wider so fonts remain crisp after the LaTeX scaling.
    fig_width = 11.0
    fig_height = 4.5
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = gridspec.GridSpec(2, 5, figure=fig,
                           wspace=0.05, hspace=0.12,
                           left=0.06, right=0.93, top=0.88, bottom=0.04)

    # LARGER font sizes (was 6.5-9 in original)
    TITLE_FS = 17
    ROW_FS = 16
    LABEL_FS = 14
    CBAR_FS = 14
    CBAR_LABEL_FS = 15

    col_titles = ["Camera", "Radar R-D", "LiDAR BEV", "S2S-Diff recon.", "Uncertainty"]

    im_unc = None
    for row_idx, fid in enumerate(selected_frames):
        # Camera
        ax0 = fig.add_subplot(gs[row_idx, 0])
        cam = load_camera_image(fid)
        if cam is not None:
            ax0.imshow(cam)
        ax0.axis("off")

        # Radar R-D
        ax1 = fig.add_subplot(gs[row_idx, 1])
        rd = load_radar_rd(fid)
        if rd is not None:
            rd_vis = normalize_radar_for_display(rd)
            ax1.imshow(rd_vis, cmap="magma", aspect="auto", vmin=0, vmax=1)
        ax1.axis("off")

        # LiDAR BEV
        ax2 = fig.add_subplot(gs[row_idx, 2])
        occ, _ = load_lidar_bev(fid)
        if occ is not None:
            ax2.imshow(occ, cmap="gray_r", vmin=0, vmax=1,
                       origin="lower", aspect="equal")
        ax2.axis("off")

        # S2S-Diff reconstruction (occupancy channel)
        ax3 = fig.add_subplot(gs[row_idx, 3])
        recon = mean_scene[row_idx, 0].cpu().numpy()
        ax3.imshow(recon, cmap="gray_r", vmin=0, vmax=1,
                   origin="lower", aspect="equal")
        ax3.axis("off")

        # Uncertainty
        ax4 = fig.add_subplot(gs[row_idx, 4])
        im_unc = ax4.imshow(unc_arrays[row_idx], cmap="inferno",
                            vmin=0, vmax=unc_vmax, origin="lower", aspect="equal")
        ax4.axis("off")

        # Row label
        row_label = f"Frame {row_idx + 1}"
        ax0.text(-0.10, 0.5, row_label, transform=ax0.transAxes,
                 fontsize=ROW_FS, rotation=90, va="center", ha="right",
                 fontweight="bold")

        # Subplot labels (a)-(j)
        axes_row = [ax0, ax1, ax2, ax3, ax4]
        for col_idx, ax in enumerate(axes_row):
            label_char = chr(ord('a') + row_idx * 5 + col_idx)
            ax.text(0.03, 0.94, f"({label_char})", transform=ax.transAxes,
                    fontsize=LABEL_FS, fontweight="bold", color="white",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="black",
                              alpha=0.7, edgecolor="none"))
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=TITLE_FS,
                             fontweight="bold", pad=6)

    # Colorbar
    cbar_ax = fig.add_axes([0.94, 0.06, 0.013, 0.80])
    cbar = fig.colorbar(im_unc, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=CBAR_FS, width=0.7, length=4)
    cbar.set_label("Variance", fontsize=CBAR_LABEL_FS, labelpad=4)
    cbar.outline.set_linewidth(0.7)

    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
