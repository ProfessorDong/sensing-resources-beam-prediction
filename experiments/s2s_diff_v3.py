"""
S2S-Diff v3: Enhanced experiments for paper revision.
Adds:
  1. pix2pix (cGAN) baseline
  2. Class-balanced CVAE (fixes RADIal collapse)
  3. Brier score metric
  4. DDIM steps ablation (10/25/50/100)
  5. Comprehensive metric reporting (IoU, F1, NLL, Brier, ECE, Chamfer, RMSE)
Reuses existing S2S-Diff and U-Net checkpoints from v2.
"""
import os, sys, json, time, logging, math, gc, glob, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))
from config import Config, RESULTS_DIR, CKPT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

V2_CKPT = os.path.join(CKPT_DIR, "v2")
V3_RESULTS = os.path.join(RESULTS_DIR, "v3")

# Import v2 components
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, SimpleForwardModel,
    UNetDirect, SimpleDS, _try_load, _clear_gpu,
    timestep_embedding, GroupNorm32, norm, ResBlock, SelfAttention,
    Downsample, Upsample
)
from models import CVAEBaseline


# =====================================================================
# Metrics (enhanced with Brier score)
# =====================================================================

def compute_iou(pred, gt, thr=0.5):
    p = (pred > thr).float()
    g = (gt > thr).float()
    inter = (p * g).sum(dim=(-2, -1))
    union = ((p + g) > 0).float().sum(dim=(-2, -1))
    return (inter / (union + 1e-8)).mean().item()

def compute_f1(pred, gt, thr=0.5):
    p = (pred > thr).float()
    g = (gt > thr).float()
    tp = (p * g).sum(dim=(-2, -1))
    fp = (p * (1 - g)).sum(dim=(-2, -1))
    fn = ((1 - p) * g).sum(dim=(-2, -1))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return (2 * prec * rec / (prec + rec + 1e-8)).mean().item()

def compute_nll(pred, gt, eps=1e-7):
    p = pred.clamp(eps, 1 - eps)
    return (-gt * torch.log(p) - (1 - gt) * torch.log(1 - p)).mean().item()

def compute_brier(pred, gt):
    """Brier score: mean squared error between predicted probabilities and binary labels."""
    return ((pred - gt) ** 2).mean().item()

def compute_ece(pred, gt, n_bins=15):
    pf = pred.reshape(-1).cpu().numpy()
    gf = gt.reshape(-1).cpu().numpy()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (pf >= bins[i]) & (pf < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(pf) * abs(pf[mask].mean() - gf[mask].mean())
    return ece

def compute_chamfer(pred, gt, thr=0.5):
    B = pred.shape[0]
    cd = 0.0
    for b in range(B):
        pp = torch.nonzero(pred[b] > thr, as_tuple=False).float()
        gp = torch.nonzero(gt[b] > thr, as_tuple=False).float()
        if len(pp) == 0 or len(gp) == 0:
            cd += 64.0
            continue
        d1 = torch.cdist(pp, gp).min(1)[0].mean()
        d2 = torch.cdist(gp, pp).min(1)[0].mean()
        cd += (d1 + d2).item() / 2
    return cd / B

def evaluate_full(pred, gt):
    """Compute all metrics on occupancy channel."""
    o_p, o_g = pred[:, 0], gt[:, 0]
    metrics = {
        "IoU": compute_iou(o_p, o_g),
        "F1": compute_f1(o_p, o_g),
        "NLL": compute_nll(o_p, o_g),
        "Brier": compute_brier(o_p, o_g),
        "ECE": compute_ece(o_p, o_g),
        "Chamfer": compute_chamfer(o_p, o_g),
    }
    # RMSE on reflectivity (occupied cells only)
    occ_mask = gt[:, 0] > 0.5
    if occ_mask.any():
        metrics["RMSE"] = torch.sqrt(
            F.mse_loss(pred[:, 1][occ_mask], gt[:, 1][occ_mask])
        ).item()
    else:
        metrics["RMSE"] = 0.0
    return metrics


# =====================================================================
# pix2pix (cGAN) baseline
# =====================================================================

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for pix2pix."""
    def __init__(self, in_ch=7, base_ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = base_ch
        for i in range(1, n_layers):
            out_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, True),
            ]
            ch = out_ch
        layers += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Pix2PixGenerator(nn.Module):
    """U-Net generator for pix2pix (measurement→scene)."""
    def __init__(self, in_ch=4, out_ch=3, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 4, 2, 1))  # 32
        self.enc2 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),
                                  nn.InstanceNorm2d(base_ch*2))  # 16
        self.enc3 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),
                                  nn.InstanceNorm2d(base_ch*4))  # 8
        self.enc4 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1),
                                  nn.InstanceNorm2d(base_ch*8))  # 4

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.LeakyReLU(0.2, True),
                                        nn.Conv2d(base_ch*8, base_ch*8, 4, 2, 1))  # 2

        # Decoder with skip connections
        self.dec5 = nn.Sequential(nn.ReLU(True),
                                  nn.ConvTranspose2d(base_ch*8, base_ch*8, 4, 2, 1),
                                  nn.InstanceNorm2d(base_ch*8), nn.Dropout(0.5))  # 4
        self.dec4 = nn.Sequential(nn.ReLU(True),
                                  nn.ConvTranspose2d(base_ch*16, base_ch*4, 4, 2, 1),
                                  nn.InstanceNorm2d(base_ch*4))  # 8
        self.dec3 = nn.Sequential(nn.ReLU(True),
                                  nn.ConvTranspose2d(base_ch*8, base_ch*2, 4, 2, 1),
                                  nn.InstanceNorm2d(base_ch*2))  # 16
        self.dec2 = nn.Sequential(nn.ReLU(True),
                                  nn.ConvTranspose2d(base_ch*4, base_ch, 4, 2, 1),
                                  nn.InstanceNorm2d(base_ch))  # 32
        self.dec1 = nn.Sequential(nn.ReLU(True),
                                  nn.ConvTranspose2d(base_ch*2, out_ch, 4, 2, 1),
                                  nn.Sigmoid())  # 64

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d5 = self.dec5(b)
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        return d1


def train_pix2pix(gen, disc, train_loader, val_loader, epochs=150,
                  lr_g=2e-4, lr_d=2e-4, lambda_l1=100.0, device="cuda",
                  save_path="pix2pix.pt"):
    """Train pix2pix cGAN baseline."""
    log.info("=== Training pix2pix (cGAN) baseline ===")
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    best_val = float("inf")

    for epoch in range(epochs):
        gen.train(); disc.train()
        for scene, meas in train_loader:
            scene, meas = scene.to(device), meas.to(device)
            B = scene.shape[0]
            fake = gen(meas)

            # --- Discriminator ---
            real_pair = torch.cat([meas, scene], 1)
            fake_pair = torch.cat([meas, fake.detach()], 1)
            pred_real = disc(real_pair)
            pred_fake = disc(fake_pair)
            loss_d = 0.5 * (bce(pred_real, torch.ones_like(pred_real)) +
                            bce(pred_fake, torch.zeros_like(pred_fake)))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # --- Generator ---
            fake_pair = torch.cat([meas, fake], 1)
            pred_fake = disc(fake_pair)
            loss_g_gan = bce(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = F.l1_loss(fake, scene) * lambda_l1
            # Weighted BCE for occupancy
            loss_g_bce = F.binary_cross_entropy(fake[:, 0], scene[:, 0])
            loss_g = loss_g_gan + loss_g_l1 + loss_g_bce
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        # Validation
        gen.eval()
        vl = 0
        with torch.no_grad():
            for scene, meas in val_loader:
                scene, meas = scene.to(device), meas.to(device)
                pred = gen(meas)
                vl += F.binary_cross_entropy(pred[:, 0], scene[:, 0]).item()
        vl /= max(1, len(val_loader))
        if vl < best_val:
            best_val = vl
            torch.save(gen.state_dict(), save_path)
        if (epoch + 1) % 30 == 0:
            log.info(f"  pix2pix epoch {epoch+1}/{epochs}  val={vl:.4f}")

    gen.load_state_dict(torch.load(save_path, weights_only=True))


# =====================================================================
# Class-balanced CVAE training
# =====================================================================

def train_cvae_balanced(model, train_loader, val_loader, epochs=150,
                        lr=1e-3, device="cuda", save_path="cvae_balanced.pt"):
    """Train CVAE with class-balanced loss to prevent collapse on imbalanced data."""
    log.info("=== Training CVAE (class-balanced) ===")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")

    # Compute class weight from training data
    occ_sum, total = 0.0, 0.0
    for scene, meas in train_loader:
        occ_sum += scene[:, 0].sum().item()
        total += scene[:, 0].numel()
    occ_ratio = occ_sum / total
    # pos_weight: ratio of negatives to positives
    pos_weight = (1 - occ_ratio) / (occ_ratio + 1e-8)
    pos_weight = min(pos_weight, 50.0)  # cap at 50
    pw_tensor = torch.tensor([pos_weight], device=device)
    log.info(f"  CVAE occ_ratio={occ_ratio:.4f}  pos_weight={pos_weight:.1f}")

    for epoch in range(epochs):
        model.train()
        for scene, meas in train_loader:
            scene, meas = scene.to(device), meas.to(device)
            pred, kl = model(meas, scene)
            # Class-balanced BCE
            loss_bce = F.binary_cross_entropy_with_logits(
                torch.logit(pred[:, 0].clamp(1e-6, 1-1e-6)),
                scene[:, 0],
                pos_weight=pw_tensor
            )
            loss = loss_bce + 0.5 * F.l1_loss(pred[:, 1], scene[:, 1]) + 0.001 * kl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for scene, meas in val_loader:
                scene, meas = scene.to(device), meas.to(device)
                pred, kl = model(meas, scene)
                vl += F.binary_cross_entropy(pred[:, 0], scene[:, 0]).item()
        vl /= max(1, len(val_loader))
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_path)
        if (epoch + 1) % 30 == 0:
            log.info(f"  CVAE-bal epoch {epoch+1}/{epochs}  val={vl:.4f}")

    model.load_state_dict(torch.load(save_path, weights_only=True))


# =====================================================================
# DDIM Steps Ablation
# =====================================================================

def ddim_steps_ablation(model, diffusion, test_loader, device="cuda",
                        step_counts=(10, 25, 50, 100)):
    """Evaluate S2S-Diff at different DDIM step counts."""
    log.info("=== DDIM Steps Ablation ===")
    model.eval()
    results = {}

    for n_steps in step_counts:
        all_pred, all_gt = [], []
        t_start = time.time()
        n_samples_total = 0

        with torch.no_grad():
            for scene, meas in test_loader:
                meas = meas.to(device)
                samples = []
                for _ in range(4):  # S=4
                    s = diffusion.ddim_sample(model, meas, n_steps=n_steps, eta=0.3)
                    samples.append(s)
                pred = torch.stack(samples).mean(0)
                all_pred.append(pred.cpu())
                all_gt.append(scene)
                n_samples_total += scene.shape[0]

        elapsed = time.time() - t_start
        latency_ms = elapsed / n_samples_total * 1000

        all_pred = torch.cat(all_pred)
        all_gt = torch.cat(all_gt)
        metrics = evaluate_full(all_pred, all_gt)
        metrics["Latency_ms"] = latency_ms

        results[n_steps] = metrics
        log.info(f"  Steps={n_steps:3d}: IoU={metrics['IoU']:.4f}  "
                 f"F1={metrics['F1']:.4f}  ECE={metrics['ECE']:.4f}  "
                 f"Latency={latency_ms:.1f}ms")

    return results


# =====================================================================
# Inference helpers
# =====================================================================

def infer_diffusion(model, diffusion, test_loader, n_steps=50, n_samples=4,
                    eta=0.3, device="cuda"):
    """Run DDIM inference and return predictions."""
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for scene, meas in test_loader:
            meas = meas.to(device)
            samples = []
            for _ in range(n_samples):
                s = diffusion.ddim_sample(model, meas, n_steps=n_steps, eta=eta)
                samples.append(s)
            pred = torch.stack(samples).mean(0)
            all_pred.append(pred.cpu())
            all_gt.append(scene)
    return torch.cat(all_pred), torch.cat(all_gt)


def infer_unet(model, test_loader, device="cuda"):
    """Run U-Net inference and return predictions."""
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for scene, meas in test_loader:
            pred = torch.sigmoid(model(meas.to(device)))
            all_pred.append(pred.cpu())
            all_gt.append(scene)
    return torch.cat(all_pred), torch.cat(all_gt)


def infer_cvae(model, test_loader, n_samples=4, device="cuda"):
    """Run CVAE inference (prior sampling) and return predictions."""
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for scene, meas in test_loader:
            meas = meas.to(device)
            samples = []
            for _ in range(n_samples):
                pred, _ = model(meas)
                samples.append(pred)
            pred = torch.stack(samples).mean(0)
            all_pred.append(pred.cpu())
            all_gt.append(scene)
    return torch.cat(all_pred), torch.cat(all_gt)


def infer_pix2pix(model, test_loader, device="cuda"):
    """Run pix2pix inference and return predictions."""
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for scene, meas in test_loader:
            pred = model(meas.to(device))
            all_pred.append(pred.cpu())
            all_gt.append(scene)
    return torch.cat(all_pred), torch.cat(all_gt)


# =====================================================================
# Main experiment runner
# =====================================================================

def run_on_dataset(name, train_scenes, train_meas, test_scenes, test_meas,
                   diff_epochs=150, device="cuda"):
    """Full experiment on one dataset."""
    safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
    ckpt_dir = os.path.join(V2_CKPT, safe_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log.info(f"\n{'='*60}\nDATASET: {name}\n{'='*60}")

    # Build data loaders
    n = len(train_scenes)
    nv = max(1, n // 10)
    perm = np.random.RandomState(42).permutation(n)
    tr_ds = SimpleDS(train_scenes[perm[nv:]], train_meas[perm[nv:]])
    val_ds = SimpleDS(train_scenes[perm[:nv]], train_meas[perm[:nv]])
    test_ds = SimpleDS(test_scenes, test_meas)

    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    log.info(f"Train: {len(tr_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    diffusion = DiffusionV2(n_steps=300, device=device)
    results = {}

    # --- 1. Load existing S2S-Diff v2 (full) ---
    model_full = PixelDiffusionUNet(base_ch=96).to(device)
    fwd_full = SimpleForwardModel().to(device)
    diff_loaded = _try_load(model_full, os.path.join(ckpt_dir, "diff_v2.pt"), "diff_v2")
    fwd_loaded = _try_load(fwd_full, os.path.join(ckpt_dir, "fwd_v2.pt"), "fwd_v2")
    if not (diff_loaded and fwd_loaded):
        log.warning(f"  S2S-Diff checkpoints not found in {ckpt_dir}, skipping")
        return {}

    log.info("Evaluating S2S-Diff v2 (full)...")
    pred, gt = infer_diffusion(model_full, diffusion, test_loader, device=device)
    results["S2S-Diff"] = evaluate_full(pred, gt)
    log.info(f"  S2S-Diff: IoU={results['S2S-Diff']['IoU']:.4f}  "
             f"F1={results['S2S-Diff']['F1']:.4f}  ECE={results['S2S-Diff']['ECE']:.4f}  "
             f"NLL={results['S2S-Diff']['NLL']:.4f}  Brier={results['S2S-Diff']['Brier']:.4f}")

    # --- 2. Load existing ablations ---
    for abl_name, abl_file in [("no_physics", "abl_nophys.pt"),
                                ("no_obsweight", "abl_noobs.pt")]:
        model_abl = PixelDiffusionUNet(base_ch=96).to(device)
        if _try_load(model_abl, os.path.join(ckpt_dir, abl_file), abl_file):
            pred, gt = infer_diffusion(model_abl, diffusion, test_loader, device=device)
            results[f"S2S-Diff_{abl_name}"] = evaluate_full(pred, gt)
            log.info(f"  S2S-Diff_{abl_name}: IoU={results[f'S2S-Diff_{abl_name}']['IoU']:.4f}")
        del model_abl; _clear_gpu()

    # --- 3. Load existing U-Net ---
    unet = UNetDirect(base_ch=96).to(device)
    if _try_load(unet, os.path.join(ckpt_dir, "unet_v2.pt"), "unet_v2"):
        pred, gt = infer_unet(unet, test_loader, device=device)
        results["U-Net"] = evaluate_full(pred, gt)
        log.info(f"  U-Net: IoU={results['U-Net']['IoU']:.4f}  "
                 f"F1={results['U-Net']['F1']:.4f}  ECE={results['U-Net']['ECE']:.4f}  "
                 f"NLL={results['U-Net']['NLL']:.4f}  Brier={results['U-Net']['Brier']:.4f}")
    del unet; _clear_gpu()

    # --- 4. Train CVAE (class-balanced) ---
    cvae = CVAEBaseline(meas_ch=4, scene_ch=3, latent_dim=64, base_ch=64).to(device)
    cvae_path = os.path.join(ckpt_dir, "cvae_balanced.pt")
    if not _try_load(cvae, cvae_path, "cvae_balanced"):
        train_cvae_balanced(cvae, tr_loader, val_loader, epochs=150,
                           device=device, save_path=cvae_path)
    pred, gt = infer_cvae(cvae, test_loader, device=device)
    results["CVAE"] = evaluate_full(pred, gt)
    log.info(f"  CVAE (balanced): IoU={results['CVAE']['IoU']:.4f}  "
             f"F1={results['CVAE']['F1']:.4f}  ECE={results['CVAE']['ECE']:.4f}  "
             f"NLL={results['CVAE']['NLL']:.4f}  Brier={results['CVAE']['Brier']:.4f}")
    del cvae; _clear_gpu()

    # --- 5. Train pix2pix (cGAN) ---
    gen = Pix2PixGenerator(in_ch=4, out_ch=3, base_ch=64).to(device)
    disc = PatchDiscriminator(in_ch=7, base_ch=64).to(device)
    log.info(f"  pix2pix params: G={sum(p.numel() for p in gen.parameters()):,}, "
             f"D={sum(p.numel() for p in disc.parameters()):,}")
    pix2pix_path = os.path.join(ckpt_dir, "pix2pix.pt")
    if not _try_load(gen, pix2pix_path, "pix2pix"):
        train_pix2pix(gen, disc, tr_loader, val_loader, epochs=150,
                     device=device, save_path=pix2pix_path)
    pred, gt = infer_pix2pix(gen, test_loader, device=device)
    results["pix2pix"] = evaluate_full(pred, gt)
    log.info(f"  pix2pix: IoU={results['pix2pix']['IoU']:.4f}  "
             f"F1={results['pix2pix']['F1']:.4f}  ECE={results['pix2pix']['ECE']:.4f}  "
             f"NLL={results['pix2pix']['NLL']:.4f}  Brier={results['pix2pix']['Brier']:.4f}")
    del gen, disc; _clear_gpu()

    # --- 6. DDIM Steps Ablation ---
    log.info("Running DDIM steps ablation...")
    ddim_abl = ddim_steps_ablation(model_full, diffusion, test_loader,
                                    device=device, step_counts=(10, 25, 50, 100))
    results["ddim_ablation"] = ddim_abl

    del model_full, fwd_full; _clear_gpu()
    return results


def main():
    device = torch.device("cuda")
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    os.makedirs(V3_RESULTS, exist_ok=True)

    all_results = {}
    ds_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

    # === 1. DeepSense ISAC (Sc.42+43 → 44) ===
    log.info("Loading ISAC data...")
    from train_all_real import preprocess_isac_scenario, preprocess_radial
    isac_scenes, isac_meas = [], []
    for sc in [42, 43, 44]:
        sc_dir = os.path.join(ds_root, "deepsense6g", f"Scenario{sc}")
        s, m = preprocess_isac_scenario(sc_dir, sc, max_frames=1500)
        isac_scenes.append(s); isac_meas.append(m)

    train_parts = [s for s in isac_scenes[:2] if len(s) > 0]
    train_meas_parts = [m for m in isac_meas[:2] if len(m) > 0]
    if train_parts and len(isac_scenes[2]) > 0:
        all_results["DeepSense_ISAC"] = run_on_dataset(
            "DeepSense ISAC (Sc.42+43→44)",
            np.concatenate(train_parts), np.concatenate(train_meas_parts),
            isac_scenes[2], isac_meas[2],
            diff_epochs=150, device=device)
    _clear_gpu(); gc.collect()

    # === 2. DeepSense V2I (Sc.31→33) ===
    log.info("\nLoading V2I data...")
    from preprocess_deepsense import preprocess_scenario as preprocess_v2i
    v2i_train_dir = os.path.join(ds_root, "deepsense6g", "Scenario31")
    v2i_test_dir = os.path.join(ds_root, "deepsense6g", "Scenario33")
    if os.path.exists(v2i_train_dir) and os.path.exists(v2i_test_dir):
        tr_scenes, tr_meas, _ = preprocess_v2i(v2i_train_dir)
        ts_scenes, ts_meas, _ = preprocess_v2i(v2i_test_dir)
        if len(tr_scenes) > 0 and len(ts_scenes) > 0:
            all_results["DeepSense_V2I"] = run_on_dataset(
                "DeepSense V2I (Sc.31→33)",
                tr_scenes, tr_meas, ts_scenes, ts_meas,
                diff_epochs=150, device=device)
    _clear_gpu(); gc.collect()

    # === 3. RADIal ===
    log.info("\nLoading RADIal data...")
    radial_base = os.path.join(ds_root, "radial")
    seq_dirs = sorted(glob.glob(os.path.join(radial_base, "extracted", "RECORD*")))
    labels_csv = os.path.join(radial_base, "labels_CVPR.csv")
    config_path = os.path.join(radial_base, "miscellaneous", "RADAR_CONFIG.m")
    if seq_dirs:
        scenes, meas = preprocess_radial(seq_dirs[0], labels_csv, config_path,
                                          max_frames=500)
        if len(scenes) > 50:
            n_train = int(0.8 * len(scenes))
            all_results["RADIal"] = run_on_dataset(
                "RADIal", scenes[:n_train], meas[:n_train],
                scenes[n_train:], meas[n_train:],
                diff_epochs=150, device=device)

    # === Print comprehensive summary ===
    log.info("\n" + "=" * 100)
    log.info("S2S-DIFF V3 COMPREHENSIVE RESULTS")
    log.info("=" * 100)

    for ds_name, ds_res in all_results.items():
        log.info(f"\n--- {ds_name} ---")
        log.info(f"  {'Method':<25s} {'IoU':>6s} {'F1':>6s} {'NLL':>7s} {'Brier':>7s} "
                 f"{'ECE':>7s} {'Cham.':>7s} {'RMSE':>7s}")
        log.info(f"  {'-'*75}")
        for method, m in ds_res.items():
            if method == "ddim_ablation":
                continue
            log.info(f"  {method:<25s} {m['IoU']:.4f} {m['F1']:.4f} {m['NLL']:.4f} "
                     f"{m['Brier']:.4f} {m['ECE']:.4f} {m['Chamfer']:.2f}  "
                     f"{m.get('RMSE', 0):.4f}")

        if "ddim_ablation" in ds_res:
            log.info(f"\n  DDIM Steps Ablation:")
            log.info(f"  {'Steps':>6s} {'IoU':>6s} {'F1':>6s} {'ECE':>7s} {'Latency':>8s}")
            for steps, m in ds_res["ddim_ablation"].items():
                log.info(f"  {steps:>6d} {m['IoU']:.4f} {m['F1']:.4f} {m['ECE']:.4f} "
                         f"{m['Latency_ms']:.1f}ms")

    # Save results
    save_path = os.path.join(V3_RESULTS, "v3_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved to {save_path}")
    log.info("Done!")


if __name__ == "__main__":
    main()
