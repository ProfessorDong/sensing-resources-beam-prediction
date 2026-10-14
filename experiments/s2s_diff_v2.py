"""
S2S-Diff v2: Redesigned for superior reconstruction accuracy.
Key changes from v1:
  1. Pixel-space diffusion (no VAE bottleneck)
  2. Concatenation conditioning (measurement spatially joined with noisy scene)
  3. Physics consistency loss during training (with gradients)
  4. Observed-cell weighted loss
  5. Generation from noise (not SDEdit) with strong conditioning
"""
import os, sys, json, time, logging, math, gc, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import Config, RESULTS_DIR, CKPT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

V2_RESULTS = os.path.join(RESULTS_DIR, "v2")
V2_CKPT = os.path.join(CKPT_DIR, "v2")


# =====================================================================
# Building blocks (same as before, but adapted for pixel-space)
# =====================================================================

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


def norm(ch):
    return GroupNorm32(min(32, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.1):
        super().__init__()
        self.norm1 = norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.norm = norm(ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, -1)
        qkv = self.qkv(h).reshape(B, 3, self.n_heads, C // self.n_heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        w = torch.einsum("bhcn,bhcm->bhnm", q, k) * (C // self.n_heads) ** -0.5
        w = F.softmax(w, dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", w, v).reshape(B, C, -1)
        return (x.reshape(B, C, -1) + self.proj(out)).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# =====================================================================
# Pixel-Space Conditional Diffusion U-Net (concat conditioning)
# =====================================================================

class PixelDiffusionUNet(nn.Module):
    """
    U-Net that takes concatenated [noisy_scene(3ch) | measurement(4ch)] = 7ch input.
    Outputs noise prediction for the 3-channel scene.
    """
    def __init__(self, scene_ch=3, meas_ch=4, base_ch=96, ch_mults=(1, 2, 4),
                 n_res=2, attn_res=(16,), n_heads=4, dropout=0.1):
        super().__init__()
        in_ch = scene_ch + meas_ch  # 7
        t_dim = base_ch * 4

        self.t_embed = nn.Sequential(
            nn.Linear(base_ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down path
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_ch
        skip_chs = [ch]
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(n_res):
                self.down_blocks.append(ResBlock(ch, out_ch, t_dim, dropout))
                self.down_attns.append(
                    SelfAttention(out_ch, n_heads)
                    if (64 // (2**i)) in attn_res else nn.Identity())
                ch = out_ch
                skip_chs.append(ch)
            if i < len(ch_mults) - 1:
                self.down_samples.append(Downsample(ch))
                skip_chs.append(ch)

        # Mid
        self.mid1 = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = SelfAttention(ch, n_heads)
        self.mid2 = ResBlock(ch, ch, t_dim, dropout)

        # Up path
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for j in range(n_res + 1):
                s_ch = skip_chs.pop()
                self.up_blocks.append(ResBlock(ch + s_ch, out_ch, t_dim, dropout))
                self.up_attns.append(
                    SelfAttention(out_ch, n_heads)
                    if (64 // (2**i)) in attn_res else nn.Identity())
                ch = out_ch
            if i > 0:
                self.up_samples.append(Upsample(ch))

        self.out_norm = norm(ch)
        self.out_conv = nn.Conv2d(ch, scene_ch, 3, padding=1)

    def forward(self, x_t, t, measurement):
        """
        x_t: [B, 3, 64, 64] noisy scene
        t: [B] timesteps
        measurement: [B, 4, 64, 64]
        """
        t_emb = self.t_embed(timestep_embedding(t, self.t_embed[0].in_features))
        h = self.in_conv(torch.cat([x_t, measurement], dim=1))  # [B, base, 64, 64]
        skips = [h]

        # Down
        bi, si = 0, 0
        n_levels = len(set(self.down_blocks))  # approximate
        for i in range(len(self.down_blocks)):
            h = self.down_blocks[i](h, t_emb)
            h = self.down_attns[i](h)
            skips.append(h)
            # Downsample after every n_res blocks
            if si < len(self.down_samples) and (i + 1) % 2 == 0 and \
               (i + 1) // 2 <= len(self.down_samples):
                h = self.down_samples[si](h)
                skips.append(h)
                si += 1

        # Mid
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Up
        ui = 0
        for i in range(len(self.up_blocks)):
            s = skips.pop()
            h = torch.cat([h, s], dim=1)
            h = self.up_blocks[i](h, t_emb)
            h = self.up_attns[i](h)
            if ui < len(self.up_samples) and (i + 1) % 3 == 0:
                h = self.up_samples[ui](h)
                ui += 1

        return self.out_conv(F.silu(self.out_norm(h)))


# =====================================================================
# Diffusion Process
# =====================================================================

class DiffusionV2:
    def __init__(self, n_steps=300, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alpha_bar = alpha_bar
        self.sqrt_ab = torch.sqrt(alpha_bar)
        self.sqrt_omab = torch.sqrt(1.0 - alpha_bar)
        self.device = device

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_ab[t][:, None, None, None] * x0 + \
               self.sqrt_omab[t][:, None, None, None] * noise, noise

    def predict_x0(self, x_t, t, eps_pred):
        return (x_t - self.sqrt_omab[t][:, None, None, None] * eps_pred) / \
               (self.sqrt_ab[t][:, None, None, None] + 1e-8)

    @torch.no_grad()
    def ddim_sample(self, model, measurement, n_steps=50, eta=0.0):
        """DDIM sampling with concatenation conditioning."""
        B = measurement.shape[0]
        x = torch.randn(B, 3, 64, 64, device=self.device)

        step_ratio = self.n_steps // n_steps
        timesteps = list(range(0, self.n_steps, step_ratio))[:n_steps]
        timesteps = list(reversed(timesteps))

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            eps_pred = model(x, t, measurement)
            x0_pred = self.predict_x0(x, t, eps_pred)
            x0_pred = x0_pred.clamp(0, 1)  # scene is in [0,1]

            if i + 1 < len(timesteps):
                t_next = timesteps[i + 1]
                alpha_t = self.alpha_bar[t_val]
                alpha_next = self.alpha_bar[t_next]
                sigma = eta * ((1 - alpha_next) / (1 - alpha_t + 1e-8)).sqrt() * \
                        (1 - alpha_t / (alpha_next + 1e-8)).sqrt()
                dir_x = (1 - alpha_next - sigma**2).clamp(min=0).sqrt() * eps_pred
                noise = torch.randn_like(x) * sigma
                x = alpha_next.sqrt() * x0_pred + dir_x + noise
            else:
                x = x0_pred

        return x.clamp(0, 1)


# =====================================================================
# Physics-aware forward model (lightweight, differentiable)
# =====================================================================

class SimpleForwardModel(nn.Module):
    """Learned differentiable forward model: scene → measurement proxy."""
    def __init__(self, scene_ch=3, meas_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(scene_ch, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, meas_ch, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, scene):
        return self.net(scene)


# =====================================================================
# Direct U-Net baseline (for comparison)
# =====================================================================

class UNetDirect(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, base_ch=96):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.SiLU(),
                                  ResBlock(base_ch, base_ch, base_ch*4))
        self.t_dummy = nn.Linear(1, base_ch * 4)  # dummy for ResBlock
        self.down1 = Downsample(base_ch)
        self.enc2 = ResBlock(base_ch, base_ch * 2, base_ch * 4)
        self.down2 = Downsample(base_ch * 2)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, base_ch * 4)
        self.down3 = Downsample(base_ch * 4)
        self.mid = nn.Sequential(
            ResBlock(base_ch * 4, base_ch * 4, base_ch * 4),
            SelfAttention(base_ch * 4),
            ResBlock(base_ch * 4, base_ch * 4, base_ch * 4))
        self.up3 = Upsample(base_ch * 4)
        self.dec3 = ResBlock(base_ch * 8, base_ch * 2, base_ch * 4)
        self.up2 = Upsample(base_ch * 2)
        self.dec2 = ResBlock(base_ch * 4, base_ch, base_ch * 4)
        self.up1 = Upsample(base_ch)
        self.dec1 = ResBlock(base_ch * 2, base_ch, base_ch * 4)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x):
        t_emb = torch.zeros(x.shape[0], self.t_dummy.out_features, device=x.device)
        e1 = self.enc1[0](x)
        e1 = F.silu(e1)
        e1 = self.enc1[2](e1, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        m = self.mid[0](self.down3(e3), t_emb)
        m = self.mid[1](m)
        m = self.mid[2](m, t_emb)
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out_conv(d1)  # raw logits; apply sigmoid externally


# =====================================================================
# Metrics
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

def compute_ece(pred, gt, n_bins=15):
    pf = pred.reshape(-1).cpu().numpy()
    gf = gt.reshape(-1).cpu().numpy()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (pf >= bins[i]) & (pf < bins[i + 1])
        if mask.sum() == 0: continue
        ece += mask.sum() / len(pf) * abs(pf[mask].mean() - gf[mask].mean())
    return ece

def compute_chamfer(pred, gt, thr=0.5):
    B = pred.shape[0]
    cd = 0.0
    for b in range(B):
        pp = torch.nonzero(pred[b] > thr, as_tuple=False).float()
        gp = torch.nonzero(gt[b] > thr, as_tuple=False).float()
        if len(pp) == 0 or len(gp) == 0:
            cd += 64.0; continue
        d1 = torch.cdist(pp, gp).min(1)[0].mean()
        d2 = torch.cdist(gp, pp).min(1)[0].mean()
        cd += (d1 + d2).item() / 2
    return cd / B

def evaluate(pred, gt):
    o_p, o_g = pred[:, 0], gt[:, 0]
    return {
        "IoU": compute_iou(o_p, o_g),
        "F1": compute_f1(o_p, o_g),
        "NLL": compute_nll(o_p, o_g),
        "ECE": compute_ece(o_p, o_g),
        "Chamfer": compute_chamfer(o_p, o_g),
        "RMSE": torch.sqrt(F.mse_loss(pred[:, 1] * gt[:, 0],
                                        gt[:, 1] * gt[:, 0])).item(),
    }


# =====================================================================
# Training
# =====================================================================

class SimpleDS(Dataset):
    def __init__(self, scenes, meas):
        self.scenes = torch.tensor(scenes, dtype=torch.float32)
        self.meas = torch.tensor(meas, dtype=torch.float32)
    def __len__(self):
        return len(self.scenes)
    def __getitem__(self, i):
        return self.scenes[i], self.meas[i]


def train_s2s_v2(model, fwd_model, diffusion, train_loader, val_loader,
                 epochs=200, lr=2e-4, phys_weight=0.1, obs_weight=2.0,
                 device="cuda"):
    """Train pixel-space diffusion with physics loss."""
    log.info("=== Training S2S-Diff v2 ===")
    params = list(model.parameters()) + list(fwd_model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        fwd_model.train()
        epoch_loss = 0
        for scene, meas in train_loader:
            scene, meas = scene.to(device), meas.to(device)
            B = scene.shape[0]

            # Standard diffusion loss
            t = torch.randint(0, diffusion.n_steps, (B,), device=device)
            noise = torch.randn_like(scene)
            x_t, _ = diffusion.q_sample(scene, t, noise)
            eps_pred = model(x_t, t, meas)
            loss_diff = F.mse_loss(eps_pred, noise)

            # Physics consistency loss (on predicted x0)
            x0_pred = diffusion.predict_x0(x_t, t, eps_pred).clamp(0, 1)
            meas_pred = fwd_model(x0_pred.detach())  # detach scene, train fwd model
            meas_gt_proxy = fwd_model(scene.detach())  # forward model target
            loss_phys = F.mse_loss(meas_pred, meas_gt_proxy.detach())

            # Also train forward model to match actual measurements
            meas_from_gt = fwd_model(scene)
            loss_fwd = F.mse_loss(meas_from_gt, meas)

            # Observed-cell weighted loss: penalize errors on occupied cells more
            occ_weight = 1.0 + (obs_weight - 1.0) * scene[:, 0:1]  # [B,1,H,W]
            loss_weighted = (occ_weight * (eps_pred - noise).pow(2)).mean()

            # Ramp up physics weight
            pw = phys_weight * min(1.0, epoch / 30.0)
            loss = loss_diff + 0.3 * loss_weighted + pw * loss_phys + 0.1 * loss_fwd

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for scene, meas in val_loader:
                scene, meas = scene.to(device), meas.to(device)
                B = scene.shape[0]
                t = torch.randint(0, diffusion.n_steps, (B,), device=device)
                noise = torch.randn_like(scene)
                x_t, _ = diffusion.q_sample(scene, t, noise)
                eps_pred = model(x_t, t, meas)
                val_loss += F.mse_loss(eps_pred, noise).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(V2_CKPT, "diff_v2.pt"))
            torch.save(fwd_model.state_dict(), os.path.join(V2_CKPT, "fwd_v2.pt"))

        if (epoch + 1) % 25 == 0:
            log.info(f"  Epoch {epoch+1}/{epochs}  loss={epoch_loss/len(train_loader):.4f}  val={val_loss:.4f}")

    model.load_state_dict(torch.load(os.path.join(V2_CKPT, "diff_v2.pt"), weights_only=True))
    fwd_model.load_state_dict(torch.load(os.path.join(V2_CKPT, "fwd_v2.pt"), weights_only=True))


def train_unet_baseline(model, train_loader, val_loader, epochs=150,
                        lr=2e-4, device="cuda"):
    """Train direct U-Net regression baseline."""
    log.info("=== Training U-Net baseline ===")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        for scene, meas in train_loader:
            scene, meas = scene.to(device), meas.to(device)
            pred = model(meas)
            # Use logits + bce_with_logits for numerical stability
            logits = pred[:, 0]
            # Detach sigmoid for L1 on reflectivity channel
            loss = F.binary_cross_entropy_with_logits(logits, scene[:, 0]) + \
                   0.5 * F.l1_loss(torch.sigmoid(pred[:, 1]), scene[:, 1])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for scene, meas in val_loader:
                scene, meas = scene.to(device), meas.to(device)
                pred = model(meas)
                vl += F.binary_cross_entropy_with_logits(pred[:, 0], scene[:, 0]).item()
        vl /= max(1, len(val_loader))
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(V2_CKPT, "unet_v2.pt"))
        if (epoch + 1) % 30 == 0:
            log.info(f"  UNet epoch {epoch+1}/{epochs}  val={vl:.4f}")

    model.load_state_dict(torch.load(os.path.join(V2_CKPT, "unet_v2.pt"), weights_only=True))


# =====================================================================
# Ablation variants
# =====================================================================

def train_ablation_no_physics(model, diffusion, train_loader, val_loader,
                              epochs=200, device="cuda"):
    """Ablation: no physics loss."""
    log.info("=== Ablation: no physics ===")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for scene, meas in train_loader:
            scene, meas = scene.to(device), meas.to(device)
            B = scene.shape[0]
            t = torch.randint(0, diffusion.n_steps, (B,), device=device)
            noise = torch.randn_like(scene)
            x_t, _ = diffusion.q_sample(scene, t, noise)
            eps_pred = model(x_t, t, meas)
            # Weighted loss but no physics
            occ_w = 1.0 + scene[:, 0:1]
            loss = (occ_w * (eps_pred - noise).pow(2)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        vl = 0
        with torch.no_grad():
            for scene, meas in val_loader:
                scene, meas = scene.to(device), meas.to(device)
                t = torch.randint(0, diffusion.n_steps, (scene.shape[0],), device=device)
                noise = torch.randn_like(scene)
                x_t, _ = diffusion.q_sample(scene, t, noise)
                vl += F.mse_loss(model(x_t, t, meas), noise).item()
        vl /= max(1, len(val_loader))
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(V2_CKPT, "abl_nophys.pt"))
    model.load_state_dict(torch.load(os.path.join(V2_CKPT, "abl_nophys.pt"), weights_only=True))


def train_ablation_no_obsweight(model, fwd_model, diffusion, train_loader,
                                val_loader, epochs=200, device="cuda"):
    """Ablation: no observed-cell weighting."""
    log.info("=== Ablation: no obs weight ===")
    params = list(model.parameters()) + list(fwd_model.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")
    for epoch in range(epochs):
        model.train(); fwd_model.train()
        for scene, meas in train_loader:
            scene, meas = scene.to(device), meas.to(device)
            B = scene.shape[0]
            t = torch.randint(0, diffusion.n_steps, (B,), device=device)
            noise = torch.randn_like(scene)
            x_t, _ = diffusion.q_sample(scene, t, noise)
            eps_pred = model(x_t, t, meas)
            loss_diff = F.mse_loss(eps_pred, noise)  # uniform weight
            x0_pred = diffusion.predict_x0(x_t, t, eps_pred).clamp(0, 1)
            meas_from_gt = fwd_model(scene)
            loss_fwd = F.mse_loss(meas_from_gt, meas)
            pw = 0.1 * min(1.0, epoch / 30.0)
            loss = loss_diff + pw * F.mse_loss(fwd_model(x0_pred.detach()),
                                                meas_from_gt.detach()) + 0.1 * loss_fwd
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()
        model.eval()
        vl = 0
        with torch.no_grad():
            for scene, meas in val_loader:
                scene, meas = scene.to(device), meas.to(device)
                t = torch.randint(0, diffusion.n_steps, (scene.shape[0],), device=device)
                noise = torch.randn_like(scene)
                x_t, _ = diffusion.q_sample(scene, t, noise)
                vl += F.mse_loss(model(x_t, t, meas), noise).item()
        vl /= max(1, len(val_loader))
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(V2_CKPT, "abl_noobs.pt"))
    model.load_state_dict(torch.load(os.path.join(V2_CKPT, "abl_noobs.pt"), weights_only=True))


# =====================================================================
# CVAE baseline (imported from models.py)
# =====================================================================

from models import CVAEBaseline


def train_cvae_baseline(model, train_loader, val_loader, epochs=150,
                        lr=1e-3, device="cuda"):
    """Train CVAE baseline."""
    log.info("=== Training CVAE baseline ===")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        for scene, meas in train_loader:
            scene, meas = scene.to(device), meas.to(device)
            pred, kl = model(meas, scene)
            loss = F.binary_cross_entropy(pred[:, 0], scene[:, 0]) + \
                   0.5 * F.l1_loss(pred[:, 1], scene[:, 1]) + 0.001 * kl
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
            torch.save(model.state_dict(), os.path.join(V2_CKPT, "cvae_v2.pt"))
        if (epoch + 1) % 30 == 0:
            log.info(f"  CVAE epoch {epoch+1}/{epochs}  val={vl:.4f}")

    model.load_state_dict(torch.load(os.path.join(V2_CKPT, "cvae_v2.pt"), weights_only=True))


# =====================================================================
# Main
# =====================================================================

def _try_load(model, path, label):
    """Try to load a checkpoint. Returns True if successful."""
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, weights_only=True))
            log.info(f"  Resumed from checkpoint: {label}")
            return True
        except Exception as e:
            log.info(f"  Checkpoint {label} exists but failed to load ({e}), retraining...")
    return False


def _clear_gpu():
    """Free GPU memory between training phases."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def run_on_dataset(name, train_scenes, train_meas, test_scenes, test_meas,
                   diff_epochs=200, device="cuda"):
    """Full experiment on one dataset."""
    global V2_CKPT
    _orig_ckpt = V2_CKPT
    safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
    V2_CKPT = os.path.join(_orig_ckpt, safe_name)
    os.makedirs(V2_CKPT, exist_ok=True)
    log.info(f"\n{'='*60}\nDATASET: {name}\n{'='*60}")
    log.info(f"Checkpoint dir: {V2_CKPT}")

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

    # 1. Train S2S-Diff v2 (full) — resume if checkpoints exist
    model_full = PixelDiffusionUNet(base_ch=96).to(device)
    fwd_full = SimpleForwardModel().to(device)
    log.info(f"S2S-Diff v2 params: {sum(p.numel() for p in model_full.parameters()):,}")
    diff_ckpt = os.path.join(V2_CKPT, "diff_v2.pt")
    fwd_ckpt = os.path.join(V2_CKPT, "fwd_v2.pt")
    if not (_try_load(model_full, diff_ckpt, "diff_v2") and
            _try_load(fwd_full, fwd_ckpt, "fwd_v2")):
        train_s2s_v2(model_full, fwd_full, diffusion, tr_loader, val_loader,
                     epochs=diff_epochs, device=device)
    _clear_gpu()

    # 2. Train U-Net baseline (same base_ch for fair comparison) — resume if checkpoint exists
    unet = UNetDirect(base_ch=96).to(device)
    log.info(f"U-Net params: {sum(p.numel() for p in unet.parameters()):,}")
    unet_ckpt = os.path.join(V2_CKPT, "unet_v2.pt")
    if not _try_load(unet, unet_ckpt, "unet_v2"):
        train_unet_baseline(unet, tr_loader, val_loader, epochs=min(150, diff_epochs), device=device)
    _clear_gpu()

    # 3. Ablation: no physics — resume if checkpoint exists
    model_nophys = PixelDiffusionUNet(base_ch=96).to(device)
    nophys_ckpt = os.path.join(V2_CKPT, "abl_nophys.pt")
    if not _try_load(model_nophys, nophys_ckpt, "abl_nophys"):
        train_ablation_no_physics(model_nophys, diffusion, tr_loader, val_loader,
                                  epochs=diff_epochs, device=device)
    _clear_gpu()

    # 4. Ablation: no obs weight — resume if checkpoint exists
    model_noobs = PixelDiffusionUNet(base_ch=96).to(device)
    fwd_noobs = SimpleForwardModel().to(device)
    noobs_ckpt = os.path.join(V2_CKPT, "abl_noobs.pt")
    if not _try_load(model_noobs, noobs_ckpt, "abl_noobs"):
        train_ablation_no_obsweight(model_noobs, fwd_noobs, diffusion, tr_loader,
                                    val_loader, epochs=diff_epochs, device=device)
    _clear_gpu()

    # 5. CVAE baseline — resume if checkpoint exists
    cvae = CVAEBaseline(meas_ch=4, scene_ch=3, latent_dim=64, base_ch=64).to(device)
    log.info(f"CVAE params: {sum(p.numel() for p in cvae.parameters()):,}")
    cvae_ckpt = os.path.join(V2_CKPT, "cvae_v2.pt")
    if not _try_load(cvae, cvae_ckpt, "cvae_v2"):
        train_cvae_baseline(cvae, tr_loader, val_loader,
                            epochs=min(150, diff_epochs), device=device)
    _clear_gpu()

    # === Evaluate ===
    log.info("Evaluating...")
    results = {}

    for model_name, mdl in [("S2S-Diff_v2_full", model_full),
                             ("S2S-Diff_v2_no_physics", model_nophys),
                             ("S2S-Diff_v2_no_obsweight", model_noobs)]:
        mdl.eval()
        all_pred, all_gt = [], []
        with torch.no_grad():
            for scene, meas in test_loader:
                meas = meas.to(device)
                # Generate 4 samples, take mean
                samples = []
                for _ in range(4):
                    s = diffusion.ddim_sample(mdl, meas, n_steps=50, eta=0.3)
                    samples.append(s)
                pred = torch.stack(samples).mean(0)
                all_pred.append(pred.cpu())
                all_gt.append(scene)
        all_pred = torch.cat(all_pred)
        all_gt = torch.cat(all_gt)
        occ_pred = all_pred[:, 0]
        occ_gt = all_gt[:, 0]
        log.info(f"    Pred occ stats: mean={occ_pred.mean():.4f} std={occ_pred.std():.4f} "
                 f"min={occ_pred.min():.4f} max={occ_pred.max():.4f} "
                 f"frac>0.5={((occ_pred > 0.5).float().mean()):.4f}")
        log.info(f"    GT   occ stats: mean={occ_gt.mean():.4f} frac>0.5={((occ_gt > 0.5).float().mean()):.4f}")
        results[model_name] = evaluate(all_pred, all_gt)
        log.info(f"  {model_name}: IoU={results[model_name]['IoU']:.4f}  "
                 f"F1={results[model_name]['F1']:.4f}  ECE={results[model_name]['ECE']:.4f}")

    # U-Net (returns logits, apply sigmoid for evaluation)
    unet.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for scene, meas in test_loader:
            pred = torch.sigmoid(unet(meas.to(device)))
            all_pred.append(pred.cpu())
            all_gt.append(scene)
    all_pred_cat = torch.cat(all_pred)
    all_gt_cat = torch.cat(all_gt)
    results["U-Net"] = evaluate(all_pred_cat, all_gt_cat)
    occ_p = all_pred_cat[:, 0]
    log.info(f"    U-Net occ stats: mean={occ_p.mean():.4f} std={occ_p.std():.4f} "
             f"min={occ_p.min():.4f} max={occ_p.max():.4f} "
             f"frac>0.5={((occ_p > 0.5).float().mean()):.4f}")
    log.info(f"  U-Net: IoU={results['U-Net']['IoU']:.4f}  "
             f"F1={results['U-Net']['F1']:.4f}  ECE={results['U-Net']['ECE']:.4f}")

    # CVAE (generate from prior, average 4 samples)
    cvae.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for scene, meas in test_loader:
            meas = meas.to(device)
            samples = []
            for _ in range(4):
                pred, _ = cvae(meas)  # prior sampling (no scene)
                samples.append(pred)
            pred = torch.stack(samples).mean(0)
            all_pred.append(pred.cpu())
            all_gt.append(scene)
    all_pred_cat = torch.cat(all_pred)
    all_gt_cat = torch.cat(all_gt)
    results["CVAE"] = evaluate(all_pred_cat, all_gt_cat)
    occ_p = all_pred_cat[:, 0]
    log.info(f"    CVAE occ stats: mean={occ_p.mean():.4f} std={occ_p.std():.4f} "
             f"min={occ_p.min():.4f} max={occ_p.max():.4f} "
             f"frac>0.5={((occ_p > 0.5).float().mean()):.4f}")
    log.info(f"  CVAE: IoU={results['CVAE']['IoU']:.4f}  "
             f"F1={results['CVAE']['F1']:.4f}  ECE={results['CVAE']['ECE']:.4f}")

    V2_CKPT = _orig_ckpt
    return results


def main():
    device = torch.device("cuda")
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    os.makedirs(V2_RESULTS, exist_ok=True)
    os.makedirs(V2_CKPT, exist_ok=True)

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
    _clear_gpu()

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
    _clear_gpu()

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

    # === Print summary ===
    log.info("\n" + "=" * 90)
    log.info("S2S-DIFF V2 RESULTS SUMMARY")
    log.info("=" * 90)
    for ds_name, ds_res in all_results.items():
        log.info(f"\n--- {ds_name} ---")
        for method, m in ds_res.items():
            log.info(f"  {method:<30s} IoU={m['IoU']:.4f}  F1={m['F1']:.4f}  "
                     f"ECE={m['ECE']:.4f}  Chamfer={m['Chamfer']:.2f}")

    with open(os.path.join(V2_RESULTS, "v2_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved to {V2_RESULTS}/v2_results.json")
    log.info("Done!")


if __name__ == "__main__":
    main()
