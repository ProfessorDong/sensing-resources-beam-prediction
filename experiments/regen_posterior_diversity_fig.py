"""
Generate posterior sample diversity figure for one ISAC test frame.

Layout: 2 rows x 6 columns.
  Row 1 (raw):    LiDAR GT, Posterior mean, Sample 1, Sample 2, Sample 3, Sample 4
  Row 2 (resid):  Radar input, Per-cell std, |S1-mean|, |S2-mean|, |S3-mean|, |S4-mean|

The bottom row is the load-bearing one: residuals show that disagreement is
spatially coherent (not pixel noise) and that each sample contributes a
distinct plausible deviation from the posterior mean.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CKPT_DIR
from s2s_diff_v2 import PixelDiffusionUNet, DiffusionV2, _try_load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISAC_CKPT = os.path.join(CKPT_DIR, "v2", "DeepSense_ISAC__Sc_42_43_44_")
OUT_PATH = os.path.join(ROOT, "results", "posterior_diversity.pdf")


def pairwise_diversity(samples):
    n = len(samples)
    pairs = [np.abs(samples[i] - samples[j]).mean()
             for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(pairs)) if pairs else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    print("Loading ISAC Sc.44 test data...")
    from train_all_real import preprocess_isac_scenario
    sc_dir = os.path.join(ROOT, "datasets", "deepsense6g", "Scenario44")
    scenes, meas = preprocess_isac_scenario(sc_dir, 44, max_frames=1500)
    print(f"  loaded {len(scenes)} frames")

    diff_model = PixelDiffusionUNet().to(device)
    diffusion = DiffusionV2(n_steps=300, device=device)
    _try_load(diff_model, os.path.join(ISAC_CKPT, "diff_v2.pt"), "diff_v2")
    diff_model.eval()

    print("Scoring candidate frames by posterior diversity...")
    candidate_idxs = list(range(50, len(scenes), 50))
    best_idx, best_score = None, -1.0
    with torch.no_grad():
        for ci in candidate_idxs:
            occ_frac = float(scenes[ci][0].mean())
            if not (0.04 <= occ_frac <= 0.30):
                continue
            meas_t = torch.from_numpy(meas[ci]).unsqueeze(0).float().to(device)
            quick = []
            for k in range(4):
                torch.manual_seed(1000 + ci + k * 7)
                s = diffusion.ddim_sample(diff_model, meas_t, n_steps=30, eta=0.3)
                quick.append(s[0, 0].cpu().numpy())
            div = pairwise_diversity(quick)
            if div > best_score:
                best_score = div
                best_idx = ci
    print(f"  selected frame {best_idx}: pairwise diversity = {best_score:.4f}, "
          f"occupancy = {scenes[best_idx][0].mean():.3f}")

    scene_np = scenes[best_idx]
    meas_np = meas[best_idx]

    print("Generating S=4 posterior samples...")
    n_samples = 4
    meas_t = torch.from_numpy(meas_np).unsqueeze(0).float().to(device)
    samples = []
    with torch.no_grad():
        for k in range(n_samples):
            torch.manual_seed(2024 + k * 17)
            s = diffusion.ddim_sample(diff_model, meas_t, n_steps=50, eta=0.3)
            samples.append(s[0, 0].cpu().numpy())
    samples_arr = np.stack(samples)
    sample_std = samples_arr.std(0)
    sample_mean = samples_arr.mean(0)
    residuals = [samples_arr[i] - sample_mean for i in range(n_samples)]

    print(f"  std max:             {sample_std.max():.4f}")
    print(f"  std mean:            {sample_std.mean():.4f}")
    print(f"  pairwise diversity:  {pairwise_diversity(samples_arr):.4f}")
    print(f"  fraction(std>0.05):  {(sample_std > 0.05).mean():.3f}")
    print(f"  fraction(std>0.10):  {(sample_std > 0.10).mean():.3f}")

    # ---- Plot: 2 rows x 6 cols ----
    print("Plotting...")
    fig, axes = plt.subplots(2, 6, figsize=(11, 2.3),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.40})

    TITLE_FS = 11
    CBAR_FS = 9
    CBAR_LBL_FS = 10

    # Row 1: raw scenes
    axes[0, 0].imshow(scene_np[0], cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("LiDAR GT", fontsize=TITLE_FS, fontweight="bold")

    axes[0, 1].imshow(sample_mean, cmap="gray_r", vmin=0, vmax=1)
    axes[0, 1].set_title("Posterior mean", fontsize=TITLE_FS, fontweight="bold")

    for i in range(n_samples):
        axes[0, i + 2].imshow(samples_arr[i], cmap="gray_r", vmin=0, vmax=1)
        axes[0, i + 2].set_title(f"Sample {i+1}", fontsize=TITLE_FS)

    # Row 2: radar input + std + 4 residuals
    rd = meas_np[2]  # combined radar magnitude
    rd_db = 20 * np.log10(np.abs(rd) + 1e-10)
    rd_norm = (rd_db - rd_db.min()) / (rd_db.max() - rd_db.min() + 1e-8)
    axes[1, 0].imshow(rd_norm, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axes[1, 0].set_title("Radar (input)", fontsize=TITLE_FS, fontweight="bold")

    std_vmax = max(float(sample_std.max()), 1e-6)
    im_std = axes[1, 1].imshow(sample_std, cmap="inferno", vmin=0, vmax=std_vmax)
    axes[1, 1].set_title("Per-cell std.", fontsize=TITLE_FS, fontweight="bold")

    # Use 99th-percentile of |residual| so medium variations register at print
    # size instead of being washed out by a few near-saturated cells.
    res_vmax = max(float(np.percentile(np.abs(residuals), 99.0)), 1e-6)
    im_res = None
    for i in range(n_samples):
        im_res = axes[1, i + 2].imshow(residuals[i], cmap="RdBu_r",
                                       vmin=-res_vmax, vmax=res_vmax)
        axes[1, i + 2].set_title(f"$S_{{{i+1}}} - \\bar{{S}}$", fontsize=TITLE_FS)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

    # Colorbars
    fig.subplots_adjust(left=0.01, right=0.92, top=0.92, bottom=0.04,
                        wspace=0.06, hspace=0.40)
    # Std colorbar (over panel [1,1])
    cbar_std_ax = fig.add_axes([0.927, 0.55, 0.008, 0.35])
    cbar_std = fig.colorbar(im_std, cax=cbar_std_ax)
    cbar_std.ax.tick_params(labelsize=CBAR_FS, width=0.5, length=2)
    cbar_std.set_label("std", fontsize=CBAR_LBL_FS, labelpad=2)
    cbar_std.outline.set_linewidth(0.5)
    # Residual colorbar (bottom row)
    cbar_res_ax = fig.add_axes([0.927, 0.07, 0.008, 0.35])
    cbar_res = fig.colorbar(im_res, cax=cbar_res_ax)
    cbar_res.ax.tick_params(labelsize=CBAR_FS, width=0.5, length=2)
    cbar_res.set_label("residual", fontsize=CBAR_LBL_FS, labelpad=2)
    cbar_res.outline.set_linewidth(0.5)

    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", pad_inches=0.04)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
