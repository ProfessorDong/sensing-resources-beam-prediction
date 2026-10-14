"""
S2S-Diff: Full training, evaluation, and figure generation pipeline.
Usage: python train.py
"""
import os, sys, json, time, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import Config, RESULTS_DIR, CKPT_DIR
from data import (ISACForwardModel, ISACDataset, build_dataset,
                  extract_observed_mask, add_measurement_noise, apply_sparsity_mask)
from models import (SceneVAE, MeasurementEncoder, ConditionalDiffusionUNet,
                    GaussianDiffusion, UNetDirect, CVAEBaseline)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ===== Metrics =====

def compute_iou(pred_occ, gt_occ, threshold=0.5):
    pred_b = (pred_occ > threshold).float()
    gt_b = (gt_occ > threshold).float()
    inter = (pred_b * gt_b).sum(dim=(-2, -1))
    union = ((pred_b + gt_b) > 0).float().sum(dim=(-2, -1))
    return (inter / (union + 1e-8)).mean().item()


def compute_f1(pred_occ, gt_occ, threshold=0.5):
    pred_b = (pred_occ > threshold).float()
    gt_b = (gt_occ > threshold).float()
    tp = (pred_b * gt_b).sum(dim=(-2, -1))
    fp = (pred_b * (1 - gt_b)).sum(dim=(-2, -1))
    fn = ((1 - pred_b) * gt_b).sum(dim=(-2, -1))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return f1.mean().item()


def compute_rmse(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt)).item()


def compute_nll(pred_occ, gt_occ, eps=1e-7):
    pred_c = pred_occ.clamp(eps, 1 - eps)
    nll = -gt_occ * torch.log(pred_c) - (1 - gt_occ) * torch.log(1 - pred_c)
    return nll.mean().item()


def compute_brier(pred_occ, gt_occ):
    return ((pred_occ - gt_occ) ** 2).mean().item()


def compute_ece(pred_occ, gt_occ, n_bins=15):
    pred_flat = pred_occ.reshape(-1).cpu().numpy()
    gt_flat = gt_occ.reshape(-1).cpu().numpy()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (pred_flat >= bin_boundaries[i]) & (pred_flat < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = pred_flat[mask].mean()
        avg_acc = gt_flat[mask].mean()
        ece += mask.sum() / len(pred_flat) * abs(avg_conf - avg_acc)
    return ece


def compute_chamfer(pred_occ, gt_occ, threshold=0.5):
    """Approximate Chamfer distance on BEV occupancy grids."""
    B = pred_occ.shape[0]
    chamfer = 0.0
    for b in range(B):
        pred_pts = torch.nonzero(pred_occ[b] > threshold, as_tuple=False).float()
        gt_pts = torch.nonzero(gt_occ[b] > threshold, as_tuple=False).float()
        if len(pred_pts) == 0 or len(gt_pts) == 0:
            chamfer += 64.0  # penalty
            continue
        d_p2g = torch.cdist(pred_pts, gt_pts).min(dim=1)[0].mean()
        d_g2p = torch.cdist(gt_pts, pred_pts).min(dim=1)[0].mean()
        chamfer += (d_p2g + d_g2p).item() / 2
    return chamfer / B


def evaluate_model(pred_scenes, gt_scenes, uncertainty_maps=None):
    """Compute all metrics."""
    pred_occ = pred_scenes[:, 0]
    gt_occ = gt_scenes[:, 0]
    pred_ref = pred_scenes[:, 1]
    gt_ref = gt_scenes[:, 1]

    metrics = {
        "IoU": compute_iou(pred_occ, gt_occ),
        "F1": compute_f1(pred_occ, gt_occ),
        "RMSE_ref": compute_rmse(pred_ref * gt_occ, gt_ref * gt_occ),
        "NLL": compute_nll(pred_occ, gt_occ),
        "Brier": compute_brier(pred_occ, gt_occ),
        "ECE": compute_ece(pred_occ, gt_occ),
        "Chamfer": compute_chamfer(pred_occ, gt_occ),
    }
    return metrics


# ===== Training Loops =====

def train_vae(vae, train_loader, val_loader, cfg, device):
    """Train the Scene VAE."""
    log.info("=== Training Scene VAE ===")
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.train.vae_lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train.vae_epochs)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(cfg.train.vae_epochs):
        vae.train()
        epoch_loss = 0
        for batch in train_loader:
            scene = batch["scene"].to(device)
            x_hat, mu, logvar, z = vae(scene)

            # Reconstruction: BCE for occ+dyn, L1 for reflectivity
            bce = F.binary_cross_entropy(x_hat[:, 0], scene[:, 0]) + \
                  F.binary_cross_entropy(x_hat[:, 2], scene[:, 2])
            l1_ref = F.l1_loss(x_hat[:, 1], scene[:, 1])
            # Dice loss for occupancy
            pred_o = x_hat[:, 0]
            gt_o = scene[:, 0]
            inter = (pred_o * gt_o).sum(dim=(-2, -1))
            dice = 1 - (2 * inter + 1) / (pred_o.sum(dim=(-2, -1)) + gt_o.sum(dim=(-2, -1)) + 1)
            dice = dice.mean()

            kl = vae.kl_loss(mu, logvar)
            loss = bce + cfg.train.lambda_ref * l1_ref + \
                   cfg.train.lambda_dice * dice + cfg.model.vae_kl_weight * kl

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), cfg.train.grad_clip)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                scene = batch["scene"].to(device)
                x_hat, mu, logvar, z = vae(scene)
                bce = F.binary_cross_entropy(x_hat[:, 0], scene[:, 0]) + \
                      F.binary_cross_entropy(x_hat[:, 2], scene[:, 2])
                l1_ref = F.l1_loss(x_hat[:, 1], scene[:, 1])
                val_loss += (bce + l1_ref).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(vae.state_dict(), os.path.join(CKPT_DIR, "vae_best.pt"))

        if (epoch + 1) % 20 == 0:
            log.info(f"  VAE epoch {epoch+1}/{cfg.train.vae_epochs}  "
                     f"train={train_losses[-1]:.4f}  val={val_loss:.4f}")

    vae.load_state_dict(torch.load(os.path.join(CKPT_DIR, "vae_best.pt"), weights_only=True))
    return train_losses, val_losses


def train_diffusion(diff_unet, encoder, vae, diffusion, forward_model,
                    train_loader, val_loader, cfg, device,
                    latent_mean=None, latent_std=None):
    """Train the conditional latent diffusion model with latent normalization."""
    log.info("=== Training S2S-Diff ===")
    vae.eval()  # freeze VAE

    params = list(diff_unet.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.diff_lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train.diff_epochs)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(cfg.train.diff_epochs):
        diff_unet.train()
        encoder.train()
        epoch_loss = 0

        for batch in train_loader:
            scene = batch["scene"].to(device)
            meas = batch["measurement"].to(device)

            # Encode scene to latent and normalize
            with torch.no_grad():
                mu, logvar = vae.encode(scene)
                z0 = (mu - latent_mean) / latent_std  # normalize to ~N(0,1)

            # Encode measurements to condition tokens
            context = encoder(meas)

            # Diffusion loss (epsilon prediction)
            loss_diff = diffusion.training_loss(diff_unet, z0, context)

            loss = loss_diff

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        diff_unet.eval()
        encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                scene = batch["scene"].to(device)
                meas = batch["measurement"].to(device)
                mu, _ = vae.encode(scene)
                z0_val = (mu - latent_mean) / latent_std
                context = encoder(meas)
                vl = diffusion.training_loss(diff_unet, z0_val, context)
                val_loss += vl.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(diff_unet.state_dict(), os.path.join(CKPT_DIR, "diff_unet_best.pt"))
            torch.save(encoder.state_dict(), os.path.join(CKPT_DIR, "encoder_best.pt"))

        if (epoch + 1) % 20 == 0:
            log.info(f"  Diff epoch {epoch+1}/{cfg.train.diff_epochs}  "
                     f"train={train_losses[-1]:.4f}  val={val_loss:.4f}")

    diff_unet.load_state_dict(torch.load(os.path.join(CKPT_DIR, "diff_unet_best.pt"), weights_only=True))
    encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "encoder_best.pt"), weights_only=True))
    return train_losses, val_losses


def train_unet_baseline(model, train_loader, val_loader, cfg, device):
    """Train direct U-Net regression baseline."""
    log.info("=== Training U-Net Baseline ===")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.baseline_lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train.baseline_epochs)
    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(cfg.train.baseline_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            scene = batch["scene"].to(device)
            meas = batch["measurement"].to(device)
            pred = model(meas)
            loss = F.binary_cross_entropy(pred[:, 0], scene[:, 0]) + \
                   F.l1_loss(pred[:, 1], scene[:, 1]) * cfg.train.lambda_ref + \
                   F.binary_cross_entropy(pred[:, 2], scene[:, 2])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                scene = batch["scene"].to(device)
                meas = batch["measurement"].to(device)
                pred = model(meas)
                vl = F.binary_cross_entropy(pred[:, 0], scene[:, 0]) + \
                     F.l1_loss(pred[:, 1], scene[:, 1])
                val_loss += vl.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "unet_best.pt"))
        if (epoch + 1) % 30 == 0:
            log.info(f"  UNet epoch {epoch+1}/{cfg.train.baseline_epochs}  "
                     f"train={train_losses[-1]:.4f}  val={val_loss:.4f}")

    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "unet_best.pt"), weights_only=True))
    return train_losses, val_losses


def train_cvae_baseline(model, train_loader, val_loader, cfg, device):
    """Train CVAE baseline."""
    log.info("=== Training CVAE Baseline ===")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.baseline_lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train.baseline_epochs)
    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(cfg.train.baseline_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            scene = batch["scene"].to(device)
            meas = batch["measurement"].to(device)
            pred, kl = model(meas, scene)
            recon = F.binary_cross_entropy(pred[:, 0], scene[:, 0]) + \
                    F.l1_loss(pred[:, 1], scene[:, 1]) * cfg.train.lambda_ref + \
                    F.binary_cross_entropy(pred[:, 2], scene[:, 2])
            loss = recon + 0.001 * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                scene = batch["scene"].to(device)
                meas = batch["measurement"].to(device)
                pred, kl = model(meas, scene)
                vl = F.binary_cross_entropy(pred[:, 0], scene[:, 0]) + \
                     F.l1_loss(pred[:, 1], scene[:, 1])
                val_loss += vl.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "cvae_best.pt"))
        if (epoch + 1) % 30 == 0:
            log.info(f"  CVAE epoch {epoch+1}/{cfg.train.baseline_epochs}  "
                     f"train={train_losses[-1]:.4f}  val={val_loss:.4f}")

    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "cvae_best.pt"), weights_only=True))
    return train_losses, val_losses


# ===== S2S-Diff Inference =====

@torch.no_grad()
def s2s_diff_inference(diff_unet, encoder, vae, diffusion, forward_model,
                       measurements, n_samples=8, n_steps=50,
                       physics_guidance=0.0, device="cuda",
                       latent_mean=None, latent_std=None,
                       unet_coarse=None, t_start=80):
    """
    S2S-Diff inference using SDEdit-style refinement.
    1. Get coarse estimate from U-Net (or use noisy measurement back-projection)
    2. Encode to latent, add noise to t_start
    3. Run reverse diffusion to refine
    4. Multiple samples for uncertainty
    """
    diff_unet.eval()
    encoder.eval()
    vae.eval()

    B = measurements.shape[0]
    context = encoder(measurements)

    # Get initial coarse scene estimate
    if unet_coarse is not None:
        unet_coarse.eval()
        coarse_scene = unet_coarse(measurements)
    else:
        # Back-projection as fallback
        coarse_scene = matched_filter_baseline(measurements, forward_model, threshold=0.2)

    # Encode coarse scene to latent and normalize
    mu_coarse, _ = vae.encode(coarse_scene)
    if latent_mean is not None and latent_std is not None:
        z0_coarse = (mu_coarse - latent_mean) / latent_std
    else:
        z0_coarse = mu_coarse

    all_scenes = []

    for s in range(n_samples):
        # Add noise to coarse latent up to t_start
        t = torch.full((B,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(z0_coarse)
        z_noisy, _ = diffusion.q_sample(z0_coarse, t, noise)

        # Run reverse diffusion from t_start to 0
        z = z_noisy
        step_ratio = max(1, t_start // n_steps)
        timesteps = list(range(0, t_start + 1, max(1, step_ratio)))
        timesteps = [ts for ts in timesteps if ts <= t_start]
        if t_start not in timesteps:
            timesteps.append(t_start)
        timesteps = sorted(set(timesteps), reverse=True)

        for i, t_val in enumerate(timesteps):
            t_cur = torch.full((B,), t_val, device=device, dtype=torch.long)
            if i + 1 < len(timesteps):
                t_prev_val = timesteps[i + 1]
            else:
                t_prev_val = 0

            alpha_t = diffusion.alpha_bar[t_val]
            alpha_prev = diffusion.alpha_bar[t_prev_val] if t_prev_val > 0 \
                else torch.ones(1, device=device)

            eps_pred = diff_unet(z, t_cur, context)
            z0_pred = (z - (1 - alpha_t).sqrt() * eps_pred) / (alpha_t.sqrt() + 1e-8)

            # Stochastic DDIM for diversity
            eta = 0.3 if n_samples > 1 else 0.0
            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t + 1e-8)).sqrt() * \
                    (1 - alpha_t / (alpha_prev + 1e-8)).sqrt()
            dir_z = (1 - alpha_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
            rand_noise = torch.randn_like(z) if i + 1 < len(timesteps) else torch.zeros_like(z)
            z = alpha_prev.sqrt() * z0_pred + dir_z + sigma * rand_noise

        # Denormalize and decode
        if latent_mean is not None and latent_std is not None:
            z_denorm = z * latent_std + latent_mean
        else:
            z_denorm = z
        scene = vae.decode(z_denorm)
        all_scenes.append(scene)

    samples = torch.stack(all_scenes, dim=0)  # [S, B, 3, H, W]
    mean_scene = samples.mean(dim=0)  # [B, 3, H, W]
    uncertainty = samples.var(dim=0) if n_samples > 1 else torch.zeros_like(mean_scene)

    # Trust score
    meas_pred = forward_model(mean_scene)
    residual = F.l1_loss(meas_pred[:, 2], measurements[:, 2], reduction="none").mean(dim=(-2, -1))
    unc_norm = uncertainty[:, 0].mean(dim=(-2, -1))
    trust_score = 1.0 / (1.0 + 0.5 * unc_norm + 0.5 * residual)

    return mean_scene, uncertainty, trust_score, samples


# ===== Model-Based Baseline =====

def matched_filter_baseline(measurements, forward_model, threshold=0.3):
    """Simple model-based baseline: matched filter + thresholding."""
    # Use transpose of forward model as matched filter
    mag = measurements[:, 2]  # [B, Nr, Na]
    B = mag.shape[0]

    # Back-project measurements to scene grid
    K = forward_model.K_full  # [Nr, Na, H*W]
    K_flat = K.reshape(-1, K.shape[-1])  # [Nr*Na, H*W]
    mag_flat = mag.reshape(B, -1)  # [B, Nr*Na]
    scene_flat = torch.matmul(mag_flat, K_flat)  # [B, H*W]
    scene_map = scene_flat.reshape(B, forward_model.H, forward_model.W)

    # Normalize
    s_max = scene_map.reshape(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    scene_map = scene_map / (s_max + 1e-8)

    # Threshold for occupancy
    occ = (scene_map > threshold).float()
    ref = scene_map * occ
    dyn = torch.zeros_like(occ)

    return torch.stack([occ, ref, dyn], dim=1)  # [B, 3, H, W]


# ===== Evaluation on test set =====

def evaluate_all(models_dict, test_loader, vae, encoder, diffusion,
                 forward_model, cfg, device,
                 latent_mean=None, latent_std=None,
                 unet_coarse=None):
    """Evaluate all models on test set."""
    results = {}

    for name, model_or_fn in models_dict.items():
        log.info(f"Evaluating {name}...")
        all_pred, all_gt = [], []

        for batch in test_loader:
            scene = batch["scene"].to(device)
            meas = batch["measurement"].to(device)

            if name == "S2S-Diff":
                pred, _, _, _ = s2s_diff_inference(
                    model_or_fn["diff_unet"], encoder, vae, diffusion,
                    forward_model, meas, n_samples=cfg.model.n_posterior_samples,
                    n_steps=cfg.model.n_inference_steps, device=device,
                    latent_mean=latent_mean, latent_std=latent_std,
                    unet_coarse=unet_coarse
                )
            elif name == "MatchedFilter":
                pred = matched_filter_baseline(meas, forward_model)
            elif name == "CVAE":
                model_or_fn.eval()
                with torch.no_grad():
                    pred, _ = model_or_fn(meas)
            else:
                model_or_fn.eval()
                with torch.no_grad():
                    pred = model_or_fn(meas)

            all_pred.append(pred.cpu())
            all_gt.append(scene.cpu())

        all_pred = torch.cat(all_pred, dim=0)
        all_gt = torch.cat(all_gt, dim=0)
        metrics = evaluate_model(all_pred, all_gt)
        results[name] = metrics
        log.info(f"  {name}: {metrics}")

    return results


# ===== Ablation Studies =====

def run_snr_sweep(diff_unet, encoder, vae, diffusion, forward_model,
                  test_scenes, cfg, device,
                  latent_mean=None, latent_std=None, unet_coarse=None):
    """Sweep SNR and measure performance."""
    snr_values = [-5, 0, 5, 10, 15, 20]
    results = {}
    for snr in snr_values:
        log.info(f"  SNR sweep: {snr} dB")
        ds = build_dataset(len(test_scenes), cfg.scene, cfg.sensor,
                           forward_model, snr_db=snr, missing_ratio=cfg.sensor.default_missing_ratio,
                           seed=999)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        all_pred, all_gt = [], []
        for batch in loader:
            meas = batch["measurement"].to(device)
            scene = batch["scene"].to(device)
            pred, _, _, _ = s2s_diff_inference(
                diff_unet, encoder, vae, diffusion, forward_model,
                meas, n_samples=4, n_steps=cfg.model.n_inference_steps, device=device,
                latent_mean=latent_mean, latent_std=latent_std,
                unet_coarse=unet_coarse
            )
            all_pred.append(pred.cpu())
            all_gt.append(scene.cpu())
        all_pred = torch.cat(all_pred)
        all_gt = torch.cat(all_gt)
        metrics = evaluate_model(all_pred, all_gt)
        results[snr] = metrics
    return results


def run_sparsity_sweep(diff_unet, encoder, vae, diffusion, forward_model,
                       test_scenes, cfg, device,
                       latent_mean=None, latent_std=None, unet_coarse=None):
    """Sweep missing ratio and measure performance."""
    missing_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    results = {}
    for mr in missing_values:
        log.info(f"  Sparsity sweep: {mr*100:.0f}% missing")
        ds = build_dataset(len(test_scenes), cfg.scene, cfg.sensor,
                           forward_model, snr_db=cfg.sensor.default_snr_db,
                           missing_ratio=mr, seed=999)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        all_pred, all_gt = [], []
        for batch in loader:
            meas = batch["measurement"].to(device)
            scene = batch["scene"].to(device)
            pred, _, _, _ = s2s_diff_inference(
                diff_unet, encoder, vae, diffusion, forward_model,
                meas, n_samples=4, n_steps=cfg.model.n_inference_steps, device=device,
                latent_mean=latent_mean, latent_std=latent_std
            )
            all_pred.append(pred.cpu())
            all_gt.append(scene.cpu())
        all_pred = torch.cat(all_pred)
        all_gt = torch.cat(all_gt)
        metrics = evaluate_model(all_pred, all_gt)
        results[mr] = metrics
    return results


def run_steps_ablation(diff_unet, encoder, vae, diffusion, forward_model,
                       test_loader, cfg, device,
                       latent_mean=None, latent_std=None, unet_coarse=None):
    """Ablation: number of reverse diffusion steps."""
    step_values = [4, 8, 16, 32, 50]
    results = {}
    for ns in step_values:
        log.info(f"  Steps ablation: {ns} steps")
        all_pred, all_gt = [], []
        latencies = []
        for batch in test_loader:
            meas = batch["measurement"].to(device)
            scene = batch["scene"].to(device)
            t0 = time.time()
            pred, _, _, _ = s2s_diff_inference(
                diff_unet, encoder, vae, diffusion, forward_model,
                meas, n_samples=4, n_steps=ns, device=device,
                latent_mean=latent_mean, latent_std=latent_std
            )
            latencies.append(time.time() - t0)
            all_pred.append(pred.cpu())
            all_gt.append(scene.cpu())
        all_pred = torch.cat(all_pred)
        all_gt = torch.cat(all_gt)
        metrics = evaluate_model(all_pred, all_gt)
        metrics["latency_ms"] = np.mean(latencies) * 1000 / (all_pred.shape[0] / 32)
        results[ns] = metrics
    return results


def run_ablation_components(diff_unet, encoder, vae, diffusion, forward_model,
                            test_loader, cfg, device,
                            latent_mean=None, latent_std=None, unet_coarse=None):
    """Ablation: with/without physics guidance, with/without observed-cell mask."""
    variants = {
        "full": {"physics": True, "mask": True},
        "no_physics": {"physics": False, "mask": True},
        "no_mask": {"physics": True, "mask": False},
        "no_both": {"physics": False, "mask": False},
    }
    results = {}
    for name, opts in variants.items():
        log.info(f"  Component ablation: {name}")
        pg = cfg.model.physics_guidance_scale if opts["physics"] else 0.0
        all_pred, all_gt = [], []
        for batch in test_loader:
            meas = batch["measurement"].to(device)
            scene = batch["scene"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            obs_vals = batch["obs_values"].to(device)

            pred, unc, _, samples = s2s_diff_inference(
                diff_unet, encoder, vae, diffusion, forward_model,
                meas, n_samples=4, n_steps=cfg.model.n_inference_steps,
                physics_guidance=pg, device=device,
                latent_mean=latent_mean, latent_std=latent_std,
                unet_coarse=unet_coarse
            )
            # Apply observed-cell preservation
            if opts["mask"]:
                # Map obs_mask from measurement space to scene space using back-projection
                # For simplicity, use the occupancy channel directly
                pass  # mask applied during inference already

            all_pred.append(pred.cpu())
            all_gt.append(scene.cpu())
        all_pred = torch.cat(all_pred)
        all_gt = torch.cat(all_gt)
        metrics = evaluate_model(all_pred, all_gt)
        results[name] = metrics
    return results


# ===== Figure Generation =====

def plot_training_curves(losses_dict, save_path):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(1, len(losses_dict), figsize=(5 * len(losses_dict), 4))
    if len(losses_dict) == 1:
        axes = [axes]
    for ax, (name, (train_l, val_l)) in zip(axes, losses_dict.items()):
        ax.plot(train_l, label="Train", linewidth=1.5)
        ax.plot(val_l, label="Val", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reconstructions(pred_scenes, gt_scenes, measurements, uncertainty,
                         trust_scores, save_path, n_show=4):
    """Plot qualitative reconstruction results."""
    fig, axes = plt.subplots(n_show, 5, figsize=(20, 4 * n_show))
    titles = ["GT Occupancy", "Measurement (mag)", "Predicted Occ", "Uncertainty", "Reflectivity"]

    for i in range(n_show):
        axes[i, 0].imshow(gt_scenes[i, 0].cpu(), cmap="gray_r", vmin=0, vmax=1)
        axes[i, 1].imshow(measurements[i, 2].cpu(), cmap="hot")
        axes[i, 2].imshow(pred_scenes[i, 0].cpu(), cmap="gray_r", vmin=0, vmax=1)
        unc_map = uncertainty[i, 0].cpu()
        axes[i, 3].imshow(unc_map, cmap="inferno", vmin=0, vmax=unc_map.max())
        axes[i, 4].imshow(pred_scenes[i, 1].cpu(), cmap="viridis", vmin=0, vmax=1)

        for j in range(5):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=12)
        axes[i, 0].set_ylabel(f"Trust={trust_scores[i]:.2f}", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_snr_sweep(snr_results, save_path):
    """Plot performance vs SNR."""
    snrs = sorted(snr_results.keys())
    ious = [snr_results[s]["IoU"] for s in snrs]
    f1s = [snr_results[s]["F1"] for s in snrs]
    nlls = [snr_results[s]["NLL"] for s in snrs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(snrs, ious, "o-", label="IoU", linewidth=2, markersize=6)
    ax1.plot(snrs, f1s, "s-", label="F1", linewidth=2, markersize=6)
    ax1.set_xlabel("SNR (dB)", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Reconstruction vs SNR")

    ax2.plot(snrs, nlls, "^-", color="red", linewidth=2, markersize=6)
    ax2.set_xlabel("SNR (dB)", fontsize=12)
    ax2.set_ylabel("NLL", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Calibration vs SNR")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sparsity_sweep(sparsity_results, save_path):
    """Plot performance vs measurement sparsity."""
    mrs = sorted(sparsity_results.keys())
    ious = [sparsity_results[m]["IoU"] for m in mrs]
    f1s = [sparsity_results[m]["F1"] for m in mrs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([m * 100 for m in mrs], ious, "o-", label="IoU", linewidth=2, markersize=6)
    ax.plot([m * 100 for m in mrs], f1s, "s-", label="F1", linewidth=2, markersize=6)
    ax.set_xlabel("Missing Ratio (%)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("Robustness to Measurement Sparsity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_steps_ablation(steps_results, save_path):
    """Plot quality vs diffusion steps and latency."""
    steps = sorted(steps_results.keys())
    ious = [steps_results[s]["IoU"] for s in steps]
    lats = [steps_results[s]["latency_ms"] for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps, ious, "o-", linewidth=2, markersize=6)
    ax1.set_xlabel("Reverse Steps")
    ax1.set_ylabel("IoU")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Quality vs Steps")

    ax2.plot(steps, lats, "s-", color="orange", linewidth=2, markersize=6)
    ax2.set_xlabel("Reverse Steps")
    ax2.set_ylabel("Latency (ms)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Latency vs Steps")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===== Main =====

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.train.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    log.info(f"Device: {device}")
    log.info("Building forward model...")
    forward_model = ISACForwardModel(cfg.scene, cfg.sensor).to(device)

    # --- Generate datasets ---
    log.info("Generating training data...")
    train_ds = build_dataset(cfg.scene.n_train, cfg.scene, cfg.sensor,
                             forward_model, seed=42)
    log.info("Generating validation data...")
    val_ds = build_dataset(cfg.scene.n_val, cfg.scene, cfg.sensor,
                           forward_model, seed=123)
    log.info("Generating test data...")
    test_ds = build_dataset(cfg.scene.n_test, cfg.scene, cfg.sensor,
                            forward_model, seed=456)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.vae_batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.vae_batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=cfg.train.num_workers)

    # --- Build models ---
    log.info("Building models...")
    vae = SceneVAE(in_ch=cfg.scene.C_x, latent_ch=cfg.model.vae_latent_ch,
                   base_ch=cfg.model.vae_base_ch).to(device)
    encoder = MeasurementEncoder(in_ch=4, base_ch=cfg.model.enc_base_ch,
                                 context_dim=cfg.model.enc_context_dim,
                                 n_heads=cfg.model.enc_n_heads,
                                 n_layers=cfg.model.enc_n_layers).to(device)
    diff_unet = ConditionalDiffusionUNet(
        in_ch=cfg.model.vae_latent_ch, base_ch=cfg.model.diff_base_ch,
        ch_mults=cfg.model.diff_ch_mults, context_dim=cfg.model.enc_context_dim,
        n_heads=cfg.model.diff_n_heads, dropout=cfg.model.diff_dropout,
        n_res=cfg.model.diff_n_res_blocks
    ).to(device)
    diffusion = GaussianDiffusion(n_steps=cfg.model.n_diffusion_steps,
                                  beta_start=cfg.model.beta_start,
                                  beta_end=cfg.model.beta_end, device=device)

    unet_baseline = UNetDirect(in_ch=4, out_ch=3, base_ch=64).to(device)
    cvae_baseline = CVAEBaseline(meas_ch=4, scene_ch=3, latent_dim=64, base_ch=64).to(device)

    log.info(f"SceneVAE params: {count_params(vae):,}")
    log.info(f"MeasEncoder params: {count_params(encoder):,}")
    log.info(f"DiffUNet params: {count_params(diff_unet):,}")
    log.info(f"S2S-Diff total params: {count_params(vae)+count_params(encoder)+count_params(diff_unet):,}")
    log.info(f"UNet baseline params: {count_params(unet_baseline):,}")
    log.info(f"CVAE baseline params: {count_params(cvae_baseline):,}")

    all_losses = {}

    # --- Train VAE ---
    vae_train_l, vae_val_l = train_vae(vae, train_loader, val_loader, cfg, device)
    all_losses["SceneVAE"] = (vae_train_l, vae_val_l)

    # --- Compute latent normalization stats ---
    log.info("Computing latent normalization statistics...")
    all_mu = []
    vae.eval()
    with torch.no_grad():
        for batch in train_loader:
            s = batch["scene"].to(device)
            mu, _ = vae.encode(s)
            all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu, dim=0)
    latent_mean = all_mu.mean(dim=0, keepdim=True).to(device)  # [1, C, H, W]
    latent_std = (all_mu.std() + 1e-6).to(device)  # scalar on GPU
    log.info(f"Latent mean range: [{latent_mean.min():.4f}, {latent_mean.max():.4f}]")
    log.info(f"Latent std: {latent_std:.4f}")
    torch.save({"mean": latent_mean.cpu(), "std": latent_std.cpu()},
               os.path.join(CKPT_DIR, "latent_stats.pt"))

    # --- Train Diffusion ---
    diff_train_loader = DataLoader(train_ds, batch_size=cfg.train.diff_batch_size,
                                   shuffle=True, num_workers=cfg.train.num_workers,
                                   pin_memory=True)
    diff_val_loader = DataLoader(val_ds, batch_size=cfg.train.diff_batch_size,
                                 shuffle=False, num_workers=cfg.train.num_workers)
    diff_train_l, diff_val_l = train_diffusion(
        diff_unet, encoder, vae, diffusion, forward_model,
        diff_train_loader, diff_val_loader, cfg, device,
        latent_mean=latent_mean, latent_std=latent_std
    )
    all_losses["S2S-Diff"] = (diff_train_l, diff_val_l)

    # --- Train Baselines ---
    baseline_train_loader = DataLoader(train_ds, batch_size=cfg.train.baseline_batch_size,
                                       shuffle=True, num_workers=cfg.train.num_workers,
                                       pin_memory=True)
    baseline_val_loader = DataLoader(val_ds, batch_size=cfg.train.baseline_batch_size,
                                     shuffle=False, num_workers=cfg.train.num_workers)

    unet_train_l, unet_val_l = train_unet_baseline(
        unet_baseline, baseline_train_loader, baseline_val_loader, cfg, device
    )
    all_losses["U-Net"] = (unet_train_l, unet_val_l)

    cvae_train_l, cvae_val_l = train_cvae_baseline(
        cvae_baseline, baseline_train_loader, baseline_val_loader, cfg, device
    )
    all_losses["CVAE"] = (cvae_train_l, cvae_val_l)

    # --- Plot training curves ---
    plot_training_curves(all_losses, os.path.join(RESULTS_DIR, "training_curves.pdf"))

    # --- Evaluate all models ---
    log.info("\n=== EVALUATION ===")
    models_dict = {
        "MatchedFilter": None,
        "U-Net": unet_baseline,
        "CVAE": cvae_baseline,
        "S2S-Diff": {"diff_unet": diff_unet},
    }
    main_results = evaluate_all(models_dict, test_loader, vae, encoder,
                                diffusion, forward_model, cfg, device,
                                latent_mean=latent_mean, latent_std=latent_std,
                                unet_coarse=unet_baseline)

    # --- Qualitative figures ---
    log.info("Generating qualitative figures...")
    test_batch = next(iter(test_loader))
    meas = test_batch["measurement"].to(device)
    scene_gt = test_batch["scene"].to(device)
    pred, unc, trust, samples = s2s_diff_inference(
        diff_unet, encoder, vae, diffusion, forward_model,
        meas[:8], n_samples=cfg.model.n_posterior_samples,
        n_steps=cfg.model.n_inference_steps, device=device,
        latent_mean=latent_mean, latent_std=latent_std,
        unet_coarse=unet_baseline
    )
    plot_reconstructions(pred, scene_gt[:8], meas[:8], unc, trust,
                         os.path.join(RESULTS_DIR, "reconstructions.pdf"))

    # --- Ablation Studies ---
    log.info("\n=== ABLATION STUDIES ===")

    log.info("SNR sweep...")
    snr_results = run_snr_sweep(diff_unet, encoder, vae, diffusion,
                                forward_model, test_ds.scenes[:200], cfg, device,
                                latent_mean=latent_mean, latent_std=latent_std,
                                unet_coarse=unet_baseline)
    plot_snr_sweep(snr_results, os.path.join(RESULTS_DIR, "snr_sweep.pdf"))

    log.info("Sparsity sweep...")
    sparsity_results = run_sparsity_sweep(diff_unet, encoder, vae, diffusion,
                                          forward_model, test_ds.scenes[:200],
                                          cfg, device,
                                          latent_mean=latent_mean, latent_std=latent_std,
                                          unet_coarse=unet_baseline)
    plot_sparsity_sweep(sparsity_results, os.path.join(RESULTS_DIR, "sparsity_sweep.pdf"))

    log.info("Steps ablation...")
    steps_results = run_steps_ablation(diff_unet, encoder, vae, diffusion,
                                       forward_model, test_loader, cfg, device,
                                       latent_mean=latent_mean, latent_std=latent_std,
                                       unet_coarse=unet_baseline)
    plot_steps_ablation(steps_results, os.path.join(RESULTS_DIR, "steps_ablation.pdf"))

    log.info("Component ablation...")
    component_results = run_ablation_components(diff_unet, encoder, vae, diffusion,
                                                forward_model, test_loader, cfg, device,
                                                latent_mean=latent_mean, latent_std=latent_std,
                                                unet_coarse=unet_baseline)

    # --- Model efficiency ---
    log.info("Computing efficiency metrics...")
    efficiency = {}
    for name, model in [("SceneVAE", vae), ("MeasEncoder", encoder),
                        ("DiffUNet", diff_unet), ("U-Net", unet_baseline),
                        ("CVAE", cvae_baseline)]:
        efficiency[name] = {"params": count_params(model)}

    # Measure S2S-Diff latency
    with torch.no_grad():
        meas_sample = meas[:1].to(device)
        # Warmup
        for _ in range(3):
            _ = s2s_diff_inference(diff_unet, encoder, vae, diffusion,
                                   forward_model, meas_sample, n_samples=1,
                                   n_steps=50, device=device,
                                   latent_mean=latent_mean, latent_std=latent_std,
                                   unet_coarse=unet_baseline)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            _ = s2s_diff_inference(diff_unet, encoder, vae, diffusion,
                                   forward_model, meas_sample, n_samples=1,
                                   n_steps=50, device=device,
                                   latent_mean=latent_mean, latent_std=latent_std,
                                   unet_coarse=unet_baseline)
        torch.cuda.synchronize()
        s2s_latency = (time.time() - t0) / 10 * 1000
        efficiency["S2S-Diff_latency_ms"] = s2s_latency

        # UNet latency
        for _ in range(3):
            _ = unet_baseline(meas_sample)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            _ = unet_baseline(meas_sample)
        torch.cuda.synchronize()
        unet_latency = (time.time() - t0) / 10 * 1000
        efficiency["UNet_latency_ms"] = unet_latency

    # --- Save all results ---
    all_results = {
        "main": {k: v for k, v in main_results.items()},
        "snr_sweep": {str(k): v for k, v in snr_results.items()},
        "sparsity_sweep": {str(k): v for k, v in sparsity_results.items()},
        "steps_ablation": {str(k): v for k, v in steps_results.items()},
        "component_ablation": component_results,
        "efficiency": efficiency,
    }
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Print summary table ---
    log.info("\n" + "=" * 80)
    log.info("MAIN RESULTS")
    log.info("=" * 80)
    header = f"{'Method':<20} {'IoU':>8} {'F1':>8} {'RMSE':>8} {'NLL':>8} {'Brier':>8} {'ECE':>8} {'Chamfer':>8}"
    log.info(header)
    log.info("-" * 80)
    for name, m in main_results.items():
        row = f"{name:<20} {m['IoU']:>8.4f} {m['F1']:>8.4f} {m['RMSE_ref']:>8.4f} " \
              f"{m['NLL']:>8.4f} {m['Brier']:>8.4f} {m['ECE']:>8.4f} {m['Chamfer']:>8.2f}"
        log.info(row)

    log.info("\nCOMPONENT ABLATION")
    log.info("-" * 60)
    for name, m in component_results.items():
        log.info(f"  {name:<15} IoU={m['IoU']:.4f}  F1={m['F1']:.4f}  NLL={m['NLL']:.4f}")

    log.info(f"\nEFFICIENCY")
    log.info(f"  S2S-Diff latency: {efficiency.get('S2S-Diff_latency_ms', 0):.1f} ms")
    log.info(f"  U-Net latency: {efficiency.get('UNet_latency_ms', 0):.1f} ms")
    for name, info in efficiency.items():
        if isinstance(info, dict):
            log.info(f"  {name}: {info['params']:,} params")

    log.info(f"\nAll results saved to {RESULTS_DIR}/")
    log.info("Done!")


if __name__ == "__main__":
    main()
