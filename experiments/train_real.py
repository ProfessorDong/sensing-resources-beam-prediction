"""
S2S-Diff training and evaluation on real DeepSense 6G data.
Train on Scenario 31, evaluate on Scenario 33 (cross-scenario generalization).
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

sys.path.insert(0, os.path.dirname(__file__))
from config import Config, RESULTS_DIR, CKPT_DIR
from models import (SceneVAE, MeasurementEncoder, ConditionalDiffusionUNet,
                    GaussianDiffusion, UNetDirect, CVAEBaseline)
from train import (compute_iou, compute_f1, compute_rmse, compute_nll,
                   compute_brier, compute_ece, compute_chamfer,
                   evaluate_model, count_params, seed_everything,
                   plot_reconstructions, matched_filter_baseline)
from preprocess_deepsense import build_deepsense_datasets, DeepSenseDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

REAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "real_world")
REAL_CKPT_DIR = os.path.join(CKPT_DIR, "real")


def train_vae_real(vae, train_loader, val_loader, epochs=80, lr=5e-4, device="cuda"):
    """Train Scene VAE on real LiDAR-derived BEV scenes."""
    log.info("=== Training Scene VAE (real data) ===")
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")

    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0
        for batch in train_loader:
            scene = batch["scene"].to(device)
            x_hat, mu, logvar, z = vae(scene)
            bce = F.binary_cross_entropy(x_hat[:, 0], scene[:, 0]) + \
                  F.binary_cross_entropy(x_hat[:, 2].clamp(1e-7, 1-1e-7),
                                         scene[:, 2].clamp(0, 1))
            l1_ref = F.l1_loss(x_hat[:, 1], scene[:, 1])
            kl = vae.kl_loss(mu, logvar)
            loss = bce + 0.5 * l1_ref + 1e-5 * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()

        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                scene = batch["scene"].to(device)
                x_hat, mu, logvar, z = vae(scene)
                vl = F.binary_cross_entropy(x_hat[:, 0], scene[:, 0]) + \
                     F.l1_loss(x_hat[:, 1], scene[:, 1])
                val_loss += vl.item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(vae.state_dict(), os.path.join(REAL_CKPT_DIR, "vae_real.pt"))

        if (epoch + 1) % 20 == 0:
            log.info(f"  VAE epoch {epoch+1}/{epochs}  "
                     f"train={epoch_loss/len(train_loader):.4f}  val={val_loss:.4f}")

    vae.load_state_dict(torch.load(os.path.join(REAL_CKPT_DIR, "vae_real.pt"),
                                    weights_only=True))


def train_diffusion_real(diff_unet, encoder, vae, diffusion,
                         train_loader, val_loader, latent_mean, latent_std,
                         epochs=150, lr=2e-4, device="cuda"):
    """Train conditional diffusion on real data."""
    log.info("=== Training S2S-Diff (real data) ===")
    vae.eval()
    params = list(diff_unet.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")

    for epoch in range(epochs):
        diff_unet.train()
        encoder.train()
        epoch_loss = 0

        for batch in train_loader:
            scene = batch["scene"].to(device)
            meas = batch["measurement"].to(device)

            with torch.no_grad():
                mu, _ = vae.encode(scene)
                z0 = (mu - latent_mean) / latent_std

            context = encoder(meas)
            loss = diffusion.training_loss(diff_unet, z0, context)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()

        diff_unet.eval()
        encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                scene = batch["scene"].to(device)
                meas = batch["measurement"].to(device)
                mu, _ = vae.encode(scene)
                z0 = (mu - latent_mean) / latent_std
                context = encoder(meas)
                vl = diffusion.training_loss(diff_unet, z0, context)
                val_loss += vl.item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(diff_unet.state_dict(),
                       os.path.join(REAL_CKPT_DIR, "diff_real.pt"))
            torch.save(encoder.state_dict(),
                       os.path.join(REAL_CKPT_DIR, "enc_real.pt"))

        if (epoch + 1) % 20 == 0:
            log.info(f"  Diff epoch {epoch+1}/{epochs}  "
                     f"train={epoch_loss/len(train_loader):.4f}  val={val_loss:.4f}")

    diff_unet.load_state_dict(torch.load(os.path.join(REAL_CKPT_DIR, "diff_real.pt"),
                                          weights_only=True))
    encoder.load_state_dict(torch.load(os.path.join(REAL_CKPT_DIR, "enc_real.pt"),
                                        weights_only=True))


def train_unet_real(model, train_loader, val_loader, epochs=120, lr=1e-3, device="cuda"):
    """Train U-Net baseline on real data."""
    log.info("=== Training U-Net (real data) ===")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            scene = batch["scene"].to(device)
            meas = batch["measurement"].to(device)
            pred = model(meas)
            loss = F.binary_cross_entropy(pred[:, 0], scene[:, 0]) + \
                   0.5 * F.l1_loss(pred[:, 1], scene[:, 1])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()

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
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(REAL_CKPT_DIR, "unet_real.pt"))

        if (epoch + 1) % 30 == 0:
            log.info(f"  UNet epoch {epoch+1}/{epochs}  "
                     f"train={epoch_loss/len(train_loader):.4f}  val={val_loss:.4f}")

    model.load_state_dict(torch.load(os.path.join(REAL_CKPT_DIR, "unet_real.pt"),
                                      weights_only=True))


@torch.no_grad()
def s2s_diff_inference_real(diff_unet, encoder, vae, diffusion,
                            measurements, unet_coarse,
                            latent_mean, latent_std,
                            n_samples=8, n_steps=50, t_start=80,
                            device="cuda"):
    """S2S-Diff inference with SDEdit refinement on real data."""
    diff_unet.eval(); encoder.eval(); vae.eval(); unet_coarse.eval()

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

    samples = torch.stack(all_scenes, dim=0)
    mean_scene = samples.mean(dim=0)
    uncertainty = samples.var(dim=0) if n_samples > 1 else torch.zeros_like(mean_scene)

    unc_norm = uncertainty[:, 0].mean(dim=(-2, -1))
    trust_score = 1.0 / (1.0 + 0.5 * unc_norm)

    return mean_scene, uncertainty, trust_score, samples


def evaluate_real(models_dict, test_loader, vae, encoder, diffusion,
                  latent_mean, latent_std, unet_coarse, device):
    """Evaluate all models on real test data."""
    results = {}
    for name, model in models_dict.items():
        log.info(f"Evaluating {name} on real data...")
        all_pred, all_gt = [], []

        for batch in test_loader:
            scene = batch["scene"].to(device)
            meas = batch["measurement"].to(device)

            if name == "S2S-Diff":
                pred, _, _, _ = s2s_diff_inference_real(
                    model["diff_unet"], encoder, vae, diffusion,
                    meas, unet_coarse, latent_mean, latent_std,
                    n_samples=8, n_steps=50, device=device
                )
            elif name == "U-Net":
                model.eval()
                with torch.no_grad():
                    pred = model(meas)
            elif name == "CVAE":
                model.eval()
                with torch.no_grad():
                    pred, _ = model(meas)
            else:
                continue

            all_pred.append(pred.cpu())
            all_gt.append(scene.cpu())

        all_pred = torch.cat(all_pred, dim=0)
        all_gt = torch.cat(all_gt, dim=0)
        metrics = evaluate_model(all_pred, all_gt)
        results[name] = metrics
        log.info(f"  {name}: {metrics}")

    return results


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.train.seed)
    os.makedirs(REAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(REAL_CKPT_DIR, exist_ok=True)

    log.info(f"Device: {device}")

    # === Preprocess DeepSense data ===
    data_root = os.path.join(os.path.dirname(__file__), "..", "datasets", "deepsense6g")
    preprocessed_dir = os.path.join(os.path.dirname(__file__), "preprocessed")

    if os.path.exists(os.path.join(preprocessed_dir, "deepsense_train.pt")):
        log.info("Loading preprocessed data...")
        tr = torch.load(os.path.join(preprocessed_dir, "deepsense_train.pt"), weights_only=True)
        vl = torch.load(os.path.join(preprocessed_dir, "deepsense_val.pt"), weights_only=True)
        ts = torch.load(os.path.join(preprocessed_dir, "deepsense_test.pt"), weights_only=True)
        train_ds = DeepSenseDataset(tr["scenes"], tr["meas"], tr["beams"])
        val_ds = DeepSenseDataset(vl["scenes"], vl["meas"], vl["beams"])
        test_ds = DeepSenseDataset(ts["scenes"], ts["meas"], ts["beams"])
    else:
        log.info("Preprocessing DeepSense data (first run)...")
        train_ds, val_ds, test_ds = build_deepsense_datasets(data_root)

    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # === Build models ===
    vae = SceneVAE(3, 8, 32).to(device)
    encoder = MeasurementEncoder(4, 32, 128, 4, 2).to(device)
    diff_unet = ConditionalDiffusionUNet(8, 64, (1, 2, 4), 128, 4, 0.1, 2).to(device)
    diffusion = GaussianDiffusion(200, 1e-4, 0.02, device=device)
    unet_baseline = UNetDirect(4, 3, 64).to(device)
    cvae_baseline = CVAEBaseline(4, 3, 64, 64).to(device)

    log.info(f"S2S-Diff params: {count_params(vae)+count_params(encoder)+count_params(diff_unet):,}")

    # === Train VAE ===
    train_vae_real(vae, train_loader, val_loader, epochs=80, device=device)

    # === Compute latent stats ===
    log.info("Computing latent normalization...")
    vae.eval()
    all_mu = []
    with torch.no_grad():
        for batch in train_loader:
            mu, _ = vae.encode(batch["scene"].to(device))
            all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu)
    latent_mean = all_mu.mean(dim=0, keepdim=True).to(device)
    latent_std = (all_mu.std() + 1e-6).to(device)
    log.info(f"Latent mean range: [{latent_mean.min():.4f}, {latent_mean.max():.4f}], "
             f"std: {latent_std:.4f}")

    # === Train Diffusion ===
    diff_train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                                   num_workers=4, pin_memory=True)
    diff_val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    train_diffusion_real(diff_unet, encoder, vae, diffusion,
                         diff_train_loader, diff_val_loader,
                         latent_mean, latent_std, epochs=150, device=device)

    # === Train U-Net baseline ===
    train_unet_real(unet_baseline, train_loader, val_loader, epochs=120, device=device)

    # === Train CVAE baseline ===
    log.info("=== Training CVAE (real data) ===")
    cvae_opt = torch.optim.AdamW(cvae_baseline.parameters(), lr=1e-3, weight_decay=1e-5)
    cvae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(cvae_opt, 120)
    best_cv = float("inf")
    for epoch in range(120):
        cvae_baseline.train()
        el = 0
        for batch in train_loader:
            scene = batch["scene"].to(device)
            meas = batch["measurement"].to(device)
            pred, kl = cvae_baseline(meas, scene)
            loss = F.binary_cross_entropy(pred[:, 0], scene[:, 0]) + \
                   0.5 * F.l1_loss(pred[:, 1], scene[:, 1]) + 0.001 * kl
            cvae_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(cvae_baseline.parameters(), 1.0)
            cvae_opt.step()
            el += loss.item()
        cvae_sched.step()
        cvae_baseline.eval()
        vl_c = 0
        with torch.no_grad():
            for batch in val_loader:
                p, k = cvae_baseline(batch["measurement"].to(device), batch["scene"].to(device))
                vl_c += F.binary_cross_entropy(p[:, 0], batch["scene"][:, 0].to(device)).item()
        vl_c /= max(1, len(val_loader))
        if vl_c < best_cv:
            best_cv = vl_c
            torch.save(cvae_baseline.state_dict(), os.path.join(REAL_CKPT_DIR, "cvae_real.pt"))
        if (epoch + 1) % 30 == 0:
            log.info(f"  CVAE epoch {epoch+1}/120  train={el/len(train_loader):.4f}  val={vl_c:.4f}")
    cvae_baseline.load_state_dict(torch.load(os.path.join(REAL_CKPT_DIR, "cvae_real.pt"),
                                              weights_only=True))

    # === Evaluate all ===
    log.info("\n=== REAL-WORLD EVALUATION (Train: Sc.31, Test: Sc.33) ===")
    models_dict = {
        "U-Net": unet_baseline,
        "CVAE": cvae_baseline,
        "S2S-Diff": {"diff_unet": diff_unet},
    }
    real_results = evaluate_real(models_dict, test_loader, vae, encoder, diffusion,
                                 latent_mean, latent_std, unet_baseline, device)

    # === Qualitative figures ===
    log.info("Generating real-world figures...")
    test_batch = next(iter(test_loader))
    meas = test_batch["measurement"].to(device)
    scene_gt = test_batch["scene"].to(device)
    n_show = min(4, meas.shape[0])
    pred, unc, trust, samples = s2s_diff_inference_real(
        diff_unet, encoder, vae, diffusion, meas[:n_show],
        unet_baseline, latent_mean, latent_std,
        n_samples=8, n_steps=50, device=device
    )
    plot_reconstructions(pred, scene_gt[:n_show], meas[:n_show], unc, trust,
                         os.path.join(REAL_RESULTS_DIR, "real_reconstructions.pdf"),
                         n_show=n_show)

    # === Save results ===
    with open(os.path.join(REAL_RESULTS_DIR, "real_results.json"), "w") as f:
        json.dump(real_results, f, indent=2)

    # === Print summary ===
    log.info("\n" + "=" * 80)
    log.info("REAL-WORLD RESULTS (Train: Scenario 31 → Test: Scenario 33)")
    log.info("=" * 80)
    header = f"{'Method':<15} {'IoU':>8} {'F1':>8} {'RMSE':>8} {'NLL':>8} {'Brier':>8} {'ECE':>8} {'Chamfer':>8}"
    log.info(header)
    log.info("-" * 80)
    for name, m in real_results.items():
        row = f"{name:<15} {m['IoU']:>8.4f} {m['F1']:>8.4f} {m['RMSE_ref']:>8.4f} " \
              f"{m['NLL']:>8.4f} {m['Brier']:>8.4f} {m['ECE']:>8.4f} {m['Chamfer']:>8.2f}"
        log.info(row)

    log.info(f"\nResults saved to {REAL_RESULTS_DIR}/")
    log.info("Done!")


if __name__ == "__main__":
    main()
