"""
S2S-Diff v4: Reviewer-rigor experiments for paper revision.
Reuses v2 checkpoints + v3 pix2pix/CVAE-balanced checkpoints.

Adds (inference-only on existing checkpoints):
  (1) Reliability diagrams       — bin predicted prob vs empirical freq, all methods
  (2) DPS comparison             — DDIM with measurement gradient at each step
  (3) Multi-seed S2S-Diff (5)    — variance over DDIM sampling seeds
  (4) Posterior coverage         — fraction of GT cells inside [mean ± q_alpha * std]
  (5) Sample count S ablation    — S in {1, 2, 4, 8, 16}
  (6) SNR robustness sweep       — Gaussian noise at SNR in {30, 20, 10, 5, 0, -5} dB

Per-dataset incremental save to experiments/results/v4/v4_results.json.
"""
import os, sys, json, time, logging, math, gc, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, SimpleForwardModel,
    UNetDirect, SimpleDS, _try_load, _clear_gpu,
)
from s2s_diff_v3 import (
    Pix2PixGenerator,
    compute_iou, compute_f1, compute_nll, compute_brier, compute_ece,
    compute_chamfer, evaluate_full,
    infer_unet, infer_cvae, infer_pix2pix, infer_diffusion,
)
from models import CVAEBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

V2_CKPT = os.path.join(CKPT_DIR, "v2")
V4_RESULTS = os.path.join(RESULTS_DIR, "v4")
os.makedirs(V4_RESULTS, exist_ok=True)
SAVE_PATH = os.path.join(V4_RESULTS, "v4_results.json")


# =====================================================================
# Helpers: deterministic seeding, save partial results
# =====================================================================

def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_partial(all_results):
    with open(SAVE_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"  [partial save -> {SAVE_PATH}]")


# =====================================================================
# (1) Reliability diagram data
# =====================================================================

def reliability_bins(pred, gt, n_bins=15, channel=0):
    """
    Bin predicted occupancy probabilities into n_bins; for each bin record
    the mean predicted prob and the empirical positive frequency.
    Returns dict suitable for JSON.
    """
    p = pred[:, channel].flatten().numpy()
    y = (gt[:, channel].flatten() > 0.5).numpy().astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred_means, emp_freqs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        counts.append(n)
        if n > 0:
            pred_means.append(float(p[mask].mean()))
            emp_freqs.append(float(y[mask].mean()))
        else:
            pred_means.append(float(centers[i]))
            emp_freqs.append(0.0)
    return {
        "bin_centers": centers.tolist(),
        "predicted_means": pred_means,
        "empirical_freqs": emp_freqs,
        "bin_counts": counts,
    }


# =====================================================================
# (2) Diffusion Posterior Sampling (DPS) inference
# =====================================================================

def infer_dps(diff_model, fwd_model, diffusion, test_loader,
              n_steps=50, n_samples=4, eta=0.3, zeta=0.1, device="cuda"):
    """
    DDIM with measurement-gradient guidance at each reverse step.
    Returns (pred, gt, latency_ms).
    """
    diff_model.eval()
    fwd_model.eval()
    all_pred, all_gt = [], []
    n_total = 0
    t_start = time.time()

    for scene, meas in test_loader:
        meas = meas.to(device)
        B = meas.shape[0]

        samples = []
        for _ in range(n_samples):
            x = torch.randn(B, 3, 64, 64, device=device)
            step_ratio = diffusion.n_steps // n_steps
            timesteps = list(range(0, diffusion.n_steps, step_ratio))[:n_steps]
            timesteps = list(reversed(timesteps))

            for i, t_val in enumerate(timesteps):
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                with torch.no_grad():
                    eps_pred = diff_model(x, t, meas)
                    x0_pred = diffusion.predict_x0(x, t, eps_pred)
                    x0_pred = x0_pred.clamp(0, 1)

                # DPS gradient: backprop measurement-consistency loss through fwd model
                x0_grad = x0_pred.detach().requires_grad_(True)
                with torch.enable_grad():
                    meas_pred = fwd_model(x0_grad)
                    loss = ((meas_pred - meas) ** 2).sum()
                    grad = torch.autograd.grad(loss, x0_grad)[0]
                # Normalize gradient by its norm to keep step size scale-invariant
                grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-6)
                grad = grad / grad_norm.view(-1, 1, 1, 1)
                x0_pred = (x0_pred - zeta * grad).clamp(0, 1).detach()

                if i + 1 < len(timesteps):
                    t_next = timesteps[i + 1]
                    alpha_t = diffusion.alpha_bar[t_val]
                    alpha_next = diffusion.alpha_bar[t_next]
                    sigma = eta * ((1 - alpha_next) / (1 - alpha_t + 1e-8)).sqrt() * \
                            (1 - alpha_t / (alpha_next + 1e-8)).sqrt()
                    dir_x = (1 - alpha_next - sigma**2).clamp(min=0).sqrt() * eps_pred
                    noise = torch.randn_like(x) * sigma
                    x = alpha_next.sqrt() * x0_pred + dir_x + noise
                else:
                    x = x0_pred
            samples.append(x.clamp(0, 1).cpu())

        pred = torch.stack(samples).mean(0)
        all_pred.append(pred)
        all_gt.append(scene)
        n_total += scene.shape[0]

    elapsed = time.time() - t_start
    latency_ms = elapsed / max(n_total, 1) * 1000
    return torch.cat(all_pred), torch.cat(all_gt), latency_ms


# =====================================================================
# (3) Multi-seed S2S-Diff inference
# =====================================================================

def multi_seed_eval(diff_model, diffusion, test_loader,
                    seeds=(42, 7, 123, 0, 999), n_steps=50, n_samples=4, device="cuda"):
    metrics_per_seed = []
    for seed in seeds:
        _seed_all(seed)
        pred, gt = infer_diffusion(diff_model, diffusion, test_loader,
                                   n_steps=n_steps, n_samples=n_samples,
                                   eta=0.3, device=device)
        m = evaluate_full(pred, gt)
        m["seed"] = int(seed)
        metrics_per_seed.append(m)
        log.info(f"    seed={seed}: IoU={m['IoU']:.4f}  ECE={m['ECE']:.4f}  "
                 f"NLL={m['NLL']:.4f}  Brier={m['Brier']:.4f}")

    summary = {}
    for k in ["IoU", "F1", "NLL", "Brier", "ECE", "Chamfer", "RMSE"]:
        vals = np.array([m[k] for m in metrics_per_seed])
        summary[f"{k}_mean"] = float(vals.mean())
        summary[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return {"seeds": list(seeds), "metrics_per_seed": metrics_per_seed, "summary": summary}


# =====================================================================
# (4) Posterior coverage at multiple alpha levels
# =====================================================================

def posterior_coverage_eval(diff_model, diffusion, test_loader,
                            n_post_samples=16, n_steps=50, eta=0.3,
                            alphas=(0.10, 0.05, 0.01), device="cuda"):
    """
    Generate n_post_samples DDIM samples per measurement; compute per-cell
    mean/std on occupancy channel; check whether GT lies within
    [mean - q_a*std, mean + q_a*std] for each alpha.
    """
    diff_model.eval()
    per_alpha_hits = {a: 0.0 for a in alphas}
    total_cells = 0
    # Also accumulate sample-mean predictions for an aggregate IoU/ECE summary
    all_pred_mean = []
    all_gt = []

    z_quantiles = {a: float(torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - a/2.0)))
                   for a in alphas}

    with torch.no_grad():
        for scene, meas in test_loader:
            meas = meas.to(device)
            B = meas.shape[0]
            samples = []
            for _ in range(n_post_samples):
                s = diffusion.ddim_sample(diff_model, meas, n_steps=n_steps, eta=eta)
                samples.append(s.cpu())
            # samples: list of [B,3,64,64]
            stack = torch.stack(samples)  # [S, B, 3, 64, 64]
            mean = stack.mean(0)
            std = stack.std(0) + 1e-6  # bias = 0 (population std)
            gt_occ = scene[:, 0]  # occupancy channel as continuous in [0,1]
            mean_occ = mean[:, 0]
            std_occ = std[:, 0]
            for a in alphas:
                q = z_quantiles[a]
                lo = mean_occ - q * std_occ
                hi = mean_occ + q * std_occ
                hit = ((gt_occ >= lo) & (gt_occ <= hi)).float().sum().item()
                per_alpha_hits[a] += hit
            total_cells += int(gt_occ.numel())
            all_pred_mean.append(mean)
            all_gt.append(scene)

    coverage = {f"alpha={a:.2f}": per_alpha_hits[a] / max(total_cells, 1)
                for a in alphas}
    # also nominal coverage (1 - alpha) for context
    coverage["nominal"] = {f"alpha={a:.2f}": 1.0 - a for a in alphas}
    return coverage


# =====================================================================
# (5) Sample count S ablation
# =====================================================================

def sample_count_ablation(diff_model, diffusion, test_loader,
                          counts=(1, 2, 4, 8, 16), n_steps=50, eta=0.3, device="cuda"):
    results = {}
    for S in counts:
        _seed_all(42)  # consistent seed across S to isolate the S effect
        pred, gt = infer_diffusion(diff_model, diffusion, test_loader,
                                   n_steps=n_steps, n_samples=S,
                                   eta=eta, device=device)
        m = evaluate_full(pred, gt)
        results[str(S)] = m
        log.info(f"    S={S:2d}: IoU={m['IoU']:.4f}  ECE={m['ECE']:.4f}  "
                 f"NLL={m['NLL']:.4f}  Brier={m['Brier']:.4f}")
    return results


# =====================================================================
# (6) SNR robustness sweep
# =====================================================================

def snr_sweep_eval(diff_model, diffusion, unet, cvae, pix2pix,
                   test_loader, snr_db_list=(30, 20, 10, 5, 0, -5),
                   n_steps=50, n_samples=4, eta=0.3, device="cuda"):
    """For each SNR, perturb measurements with Gaussian noise and eval all 4 methods."""
    results = {}
    for snr in snr_db_list:
        _seed_all(42)  # deterministic noise per SNR
        all_pred = {"S2S-Diff": [], "U-Net": [], "CVAE": [], "pix2pix": []}
        all_gt = []
        with torch.no_grad():
            for scene, meas in test_loader:
                meas = meas.to(device)
                # Per-sample signal RMS, additive Gaussian noise
                sig_rms = meas.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt()
                noise_std = sig_rms * (10.0 ** (-snr / 20.0))
                noise = torch.randn_like(meas) * noise_std.view(-1, 1, 1, 1)
                meas_n = (meas + noise).clamp(min=0.0)

                # S2S-Diff
                samples = []
                for _ in range(n_samples):
                    s = diffusion.ddim_sample(diff_model, meas_n, n_steps=n_steps, eta=eta)
                    samples.append(s)
                p_d = torch.stack(samples).mean(0).cpu()
                all_pred["S2S-Diff"].append(p_d)

                # U-Net
                p_u = torch.sigmoid(unet(meas_n)).cpu()
                all_pred["U-Net"].append(p_u)

                # CVAE (prior sampling, average)
                cv_samples = []
                for _ in range(n_samples):
                    p_c, _ = cvae(meas_n)
                    cv_samples.append(p_c)
                all_pred["CVAE"].append(torch.stack(cv_samples).mean(0).cpu())

                # pix2pix
                all_pred["pix2pix"].append(pix2pix(meas_n).cpu())
                all_gt.append(scene)

        gt_cat = torch.cat(all_gt)
        snr_metrics = {}
        for m, plist in all_pred.items():
            pcat = torch.cat(plist)
            snr_metrics[m] = evaluate_full(pcat, gt_cat)
        results[str(snr)] = snr_metrics
        line = "  ".join([f"{m}:IoU={snr_metrics[m]['IoU']:.3f}/ECE={snr_metrics[m]['ECE']:.3f}"
                          for m in ["S2S-Diff","U-Net","CVAE","pix2pix"]])
        log.info(f"    SNR={snr:3d}dB  {line}")
    return results


# =====================================================================
# Per-dataset orchestrator
# =====================================================================

def run_v4_on_dataset(name, test_scenes, test_meas, device="cuda"):
    safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
    ckpt_dir = os.path.join(V2_CKPT, safe_name)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"Checkpoint dir not found: {ckpt_dir}\n"
            f"  Available v2 dirs: {sorted(os.listdir(V2_CKPT))}\n"
            f"  Did the dataset name string match v3's safe_name?")
    # Verify required checkpoints actually exist; bail loudly if not.
    for fn in ("diff_v2.pt", "fwd_v2.pt", "unet_v2.pt", "cvae_balanced.pt", "pix2pix.pt"):
        p = os.path.join(ckpt_dir, fn)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing checkpoint: {p}")
    log.info(f"\n{'='*70}\nDATASET: {name}\n{'='*70}")
    log.info(f"  Test set: {len(test_scenes)} frames")
    log.info(f"  Checkpoint dir: {ckpt_dir}")

    test_ds = SimpleDS(test_scenes, test_meas)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # Models
    diff_model = PixelDiffusionUNet().to(device)
    fwd_model = SimpleForwardModel(scene_ch=3, meas_ch=4).to(device)
    diffusion = DiffusionV2(n_steps=300, device=device)
    unet = UNetDirect(in_ch=4, out_ch=3, base_ch=96).to(device)
    cvae = CVAEBaseline(meas_ch=4, scene_ch=3, latent_dim=64, base_ch=64).to(device)
    pix2pix = Pix2PixGenerator(in_ch=4, out_ch=3, base_ch=64).to(device)

    _try_load(diff_model, os.path.join(ckpt_dir, "diff_v2.pt"), "diff_v2")
    _try_load(fwd_model, os.path.join(ckpt_dir, "fwd_v2.pt"), "fwd_v2")
    _try_load(unet, os.path.join(ckpt_dir, "unet_v2.pt"), "unet_v2")
    _try_load(cvae, os.path.join(ckpt_dir, "cvae_balanced.pt"), "cvae_balanced")
    _try_load(pix2pix, os.path.join(ckpt_dir, "pix2pix.pt"), "pix2pix")

    res = {}

    # ---------- (1) Reliability diagrams ----------
    log.info("(1) Reliability diagram bins (15 bins) ...")
    _seed_all(42)
    pred_d, gt = infer_diffusion(diff_model, diffusion, test_loader,
                                 n_steps=50, n_samples=4, eta=0.3, device=device)
    pred_u, _ = infer_unet(unet, test_loader, device=device)
    pred_c, _ = infer_cvae(cvae, test_loader, n_samples=4, device=device)
    pred_p, _ = infer_pix2pix(pix2pix, test_loader, device=device)
    res["reliability"] = {
        "S2S-Diff": reliability_bins(pred_d, gt, n_bins=15),
        "U-Net":    reliability_bins(pred_u, gt, n_bins=15),
        "CVAE":     reliability_bins(pred_c, gt, n_bins=15),
        "pix2pix":  reliability_bins(pred_p, gt, n_bins=15),
    }
    log.info("  reliability bins computed for 4 methods")

    # ---------- (2) DPS comparison ----------
    log.info("(2) DPS (inference-time physics) ...")
    _seed_all(42)
    pred_dps, gt2, lat_dps = infer_dps(diff_model, fwd_model, diffusion, test_loader,
                                       n_steps=50, n_samples=4, eta=0.3, zeta=0.1, device=device)
    m_dps = evaluate_full(pred_dps, gt2)
    m_dps["Latency_ms"] = lat_dps
    res["dps"] = m_dps
    log.info(f"  DPS: IoU={m_dps['IoU']:.4f}  F1={m_dps['F1']:.4f}  "
             f"NLL={m_dps['NLL']:.4f}  Brier={m_dps['Brier']:.4f}  "
             f"ECE={m_dps['ECE']:.4f}  Latency={lat_dps:.1f}ms")

    # ---------- (3) Multi-seed S2S-Diff ----------
    log.info("(3) Multi-seed S2S-Diff (5 seeds) ...")
    res["multi_seed_s2s"] = multi_seed_eval(diff_model, diffusion, test_loader,
                                             seeds=(42, 7, 123, 0, 999), device=device)

    # ---------- (4) Posterior coverage ----------
    log.info("(4) Posterior coverage (S=16, alphas in {0.10, 0.05, 0.01}) ...")
    _seed_all(42)
    res["coverage"] = posterior_coverage_eval(diff_model, diffusion, test_loader,
                                              n_post_samples=16, n_steps=50, eta=0.3,
                                              alphas=(0.10, 0.05, 0.01), device=device)
    log.info(f"  coverage: {res['coverage']}")

    # ---------- (5) Sample count S ablation ----------
    log.info("(5) Sample count S ablation, S in {1,2,4,8,16} ...")
    res["sample_count_ablation"] = sample_count_ablation(diff_model, diffusion, test_loader,
                                                         counts=(1, 2, 4, 8, 16), device=device)

    # ---------- (6) SNR sweep ----------
    log.info("(6) SNR sweep, dB in {30,20,10,5,0,-5} ...")
    res["snr_sweep"] = snr_sweep_eval(diff_model, diffusion, unet, cvae, pix2pix,
                                      test_loader, snr_db_list=(30, 20, 10, 5, 0, -5),
                                      device=device)

    # cleanup
    del diff_model, fwd_model, unet, cvae, pix2pix
    _clear_gpu()
    gc.collect()
    return res


# =====================================================================
# Main
# =====================================================================

def main():
    device = torch.device("cuda")
    _seed_all(42)
    all_results = {}

    ds_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

    # ------------------------- ISAC -------------------------
    log.info("Loading ISAC data ...")
    from train_all_real import preprocess_isac_scenario, preprocess_radial
    isac_scenes, isac_meas = [], []
    for sc in [42, 43, 44]:
        sc_dir = os.path.join(ds_root, "deepsense6g", f"Scenario{sc}")
        s, m = preprocess_isac_scenario(sc_dir, sc, max_frames=1500)
        isac_scenes.append(s); isac_meas.append(m)
    if len(isac_scenes[2]) > 0:
        try:
            all_results["DeepSense_ISAC"] = run_v4_on_dataset(
                "DeepSense ISAC (Sc.42+43→44)",  # unicode → = '→' to match v3 safe_name
                isac_scenes[2], isac_meas[2], device=device)
            _save_partial(all_results)
        except Exception as e:
            log.exception(f"ISAC failed: {e}")
    _clear_gpu(); gc.collect()

    # ------------------------- V2I -------------------------
    log.info("\nLoading V2I data ...")
    try:
        from preprocess_deepsense import preprocess_scenario as preprocess_v2i
        v2i_test_dir = os.path.join(ds_root, "deepsense6g", "Scenario33")
        if os.path.exists(v2i_test_dir):
            ts_scenes, ts_meas, _ = preprocess_v2i(v2i_test_dir)
            if len(ts_scenes) > 0:
                all_results["DeepSense_V2I"] = run_v4_on_dataset(
                    "DeepSense V2I (Sc.31→33)",  # → = '→'
                    ts_scenes, ts_meas, device=device)
                _save_partial(all_results)
    except Exception as e:
        log.exception(f"V2I failed: {e}")
    _clear_gpu(); gc.collect()

    # ------------------------- RADIal -------------------------
    log.info("\nLoading RADIal data ...")
    try:
        radial_base = os.path.join(ds_root, "radial")
        seq_dirs = sorted(glob.glob(os.path.join(radial_base, "extracted", "RECORD*")))
        labels_csv = os.path.join(radial_base, "labels_CVPR.csv")
        config_path = os.path.join(radial_base, "miscellaneous", "RADAR_CONFIG.m")
        if seq_dirs:
            scenes, meas = preprocess_radial(seq_dirs[0], labels_csv, config_path,
                                              max_frames=500)
            if len(scenes) > 50:
                n_train = int(0.8 * len(scenes))
                all_results["RADIal"] = run_v4_on_dataset(
                    "RADIal", scenes[n_train:], meas[n_train:], device=device)
                _save_partial(all_results)
    except Exception as e:
        log.exception(f"RADIal failed: {e}")

    log.info("\n" + "=" * 70)
    log.info("v4 EXPERIMENTS COMPLETE")
    log.info("=" * 70)
    _save_partial(all_results)
    log.info(f"Results: {SAVE_PATH}")


if __name__ == "__main__":
    main()
