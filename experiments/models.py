"""
Neural network models for S2S-Diff.
- SceneVAE: latent scene autoencoder with KL regularization
- MeasurementEncoder: complex CNN + transformer for wireless conditioning
- ConditionalDiffusionUNet: latent diffusion backbone with cross-attention
- Baselines: UNetDirect, CVAEBaseline
- GaussianDiffusion: DDPM/DDIM process with physics-guided sampling
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


def norm(ch):
    return GroupNorm32(min(32, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim=None, dropout=0.0):
        super().__init__()
        self.norm1 = norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        if t_dim is not None:
            self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        else:
            self.t_proj = None

    def forward(self, x, t_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if t_emb is not None and self.t_proj is not None:
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
        scale = (C // self.n_heads) ** -0.5
        w = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        w = F.softmax(w, dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", w, v).reshape(B, C, -1)
        return (x.reshape(B, C, -1) + self.proj(out)).reshape(B, C, H, W)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads=4, dim_head=32):
        super().__init__()
        inner_dim = n_heads * dim_head
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_c = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        """x: [B, N, D], context: [B, M, C]"""
        x_n = self.norm_q(x)
        c_n = self.norm_c(context)
        q = self.to_q(x_n).reshape(x.shape[0], -1, self.n_heads, self.dim_head).permute(0, 2, 1, 3)
        k = self.to_k(c_n).reshape(x.shape[0], -1, self.n_heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(c_n).reshape(x.shape[0], -1, self.n_heads, self.dim_head).permute(0, 2, 1, 3)
        w = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        w = F.softmax(w, dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", w, v)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], -1, self.n_heads * self.dim_head)
        return x + self.to_out(out)


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
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Scene VAE
# ---------------------------------------------------------------------------

class SceneVAE(nn.Module):
    """Convolutional VAE for scene tensor compression (64x64 -> 16x16)."""

    def __init__(self, in_ch=3, latent_ch=8, base_ch=32):
        super().__init__()
        # Encoder: 64->32->16
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.SiLU(),
            ResBlock(base_ch, base_ch),
            Downsample(base_ch),          # 32x32
            ResBlock(base_ch, base_ch * 2),
            Downsample(base_ch * 2),      # 16x16
            ResBlock(base_ch * 2, base_ch * 2),
            norm(base_ch * 2), nn.SiLU(),
        )
        self.enc_mu = nn.Conv2d(base_ch * 2, latent_ch, 1)
        self.enc_logvar = nn.Conv2d(base_ch * 2, latent_ch, 1)

        # Decoder: 16->32->64
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, base_ch * 2, 3, padding=1), nn.SiLU(),
            ResBlock(base_ch * 2, base_ch * 2),
            Upsample(base_ch * 2),        # 32x32
            ResBlock(base_ch * 2, base_ch),
            Upsample(base_ch),            # 64x64
            ResBlock(base_ch, base_ch),
            norm(base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        x_hat = self.dec(z)
        # Sigmoid for occupancy & dynamic (ch 0, 2); clamp reflectivity (ch 1)
        out = torch.sigmoid(x_hat)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ---------------------------------------------------------------------------
# Wireless Measurement Encoder
# ---------------------------------------------------------------------------

class MeasurementEncoder(nn.Module):
    """Encode sparse ISAC measurements into condition tokens for cross-attention."""

    def __init__(self, in_ch=4, base_ch=32, context_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        # CNN stem: 64x64 -> 8x8
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1), nn.SiLU(),      # 32
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.SiLU(),  # 16
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1), nn.SiLU(),  # 8
        )
        feat_dim = base_ch * 4  # 128
        self.proj = nn.Linear(feat_dim, context_dim)

        # Transformer layers for long-range fusion
        layer = nn.TransformerEncoderLayer(
            d_model=context_dim, nhead=n_heads, dim_feedforward=context_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, measurement: torch.Tensor) -> torch.Tensor:
        """
        Args:
            measurement: [B, 4, Nr, Na]
        Returns:
            tokens: [B, N_tokens, context_dim]
        """
        h = self.stem(measurement)  # [B, C, 8, 8]
        B, C, H, W = h.shape
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # [B, 64, C]
        h = self.proj(h)  # [B, 64, context_dim]
        h = self.transformer(h)
        return h


# ---------------------------------------------------------------------------
# Conditional Latent Diffusion U-Net
# ---------------------------------------------------------------------------

class ConditionalDiffusionUNet(nn.Module):
    """
    U-Net for latent diffusion with cross-attention conditioning
    on wireless measurement tokens.
    Input: noisy latent z_t [B, latent_ch, 16, 16]
    Output: predicted noise epsilon [B, latent_ch, 16, 16]
    """

    def __init__(self, in_ch=8, base_ch=64, ch_mults=(1, 2, 4),
                 context_dim=128, n_heads=4, dropout=0.1, n_res=2):
        super().__init__()
        t_dim = base_ch * 4
        self.t_embed = nn.Sequential(
            nn.Linear(base_ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
        )
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_cross = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_ch
        self.skip_channels = [ch]
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(n_res):
                self.down_blocks.append(ResBlock(ch, out_ch, t_dim, dropout))
                # Self-attention + cross-attention at lower resolutions
                self.down_attns.append(SelfAttention(out_ch, n_heads))
                self.down_cross.append(CrossAttention(out_ch, context_dim, n_heads))
                ch = out_ch
                self.skip_channels.append(ch)
            if i < len(ch_mults) - 1:
                self.down_samples.append(Downsample(ch))
                self.skip_channels.append(ch)

        # Middle
        self.mid_block1 = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = SelfAttention(ch, n_heads)
        self.mid_cross = CrossAttention(ch, context_dim, n_heads)
        self.mid_block2 = ResBlock(ch, ch, t_dim, dropout)

        # Up blocks
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_cross = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for j in range(n_res + 1):
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, t_dim, dropout))
                self.up_attns.append(SelfAttention(out_ch, n_heads))
                self.up_cross.append(CrossAttention(out_ch, context_dim, n_heads))
                ch = out_ch
            if i > 0:
                self.up_samples.append(Upsample(ch))

        self.out_norm = norm(ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, z_t, t, context):
        """
        Args:
            z_t: [B, C, H, W] noisy latent
            t: [B] integer timesteps
            context: [B, N, D] condition tokens
        Returns:
            eps_pred: [B, C, H, W] predicted noise
        """
        t_emb = self.t_embed(timestep_embedding(t, self.t_embed[0].in_features))

        h = self.in_conv(z_t)
        skips = [h]

        # Down
        d_idx = 0
        s_idx = 0
        for i in range(len(self.down_blocks)):
            h = self.down_blocks[i](h, t_emb)
            B, C, H_, W_ = h.shape
            h_flat = h.reshape(B, C, -1).permute(0, 2, 1)
            h_flat = self.down_cross[i](h_flat, context)
            h = h_flat.permute(0, 2, 1).reshape(B, C, H_, W_)
            h = self.down_attns[i](h)
            skips.append(h)
            # Downsample after each resolution level
            n_res = len(self.down_blocks) // len(self.down_samples) if len(self.down_samples) > 0 else 999
            # Check if we should downsample
            if s_idx < len(self.down_samples):
                # Count how many blocks per level
                blocks_per_level = 2  # n_res
                if (i + 1) % blocks_per_level == 0 and (i + 1) // blocks_per_level <= len(self.down_samples):
                    h = self.down_samples[s_idx](h)
                    skips.append(h)
                    s_idx += 1

        # Mid
        h = self.mid_block1(h, t_emb)
        B, C, H_, W_ = h.shape
        h_flat = h.reshape(B, C, -1).permute(0, 2, 1)
        h_flat = self.mid_cross(h_flat, context)
        h = h_flat.permute(0, 2, 1).reshape(B, C, H_, W_)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Up
        u_idx = 0
        us_idx = 0
        for i in range(len(self.up_blocks)):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = self.up_blocks[i](h, t_emb)
            B, C, H_, W_ = h.shape
            h_flat = h.reshape(B, C, -1).permute(0, 2, 1)
            h_flat = self.up_cross[i](h_flat, context)
            h = h_flat.permute(0, 2, 1).reshape(B, C, H_, W_)
            h = self.up_attns[i](h)
            # Upsample
            blocks_per_level = 3  # n_res + 1
            if us_idx < len(self.up_samples):
                if (i + 1) % blocks_per_level == 0:
                    h = self.up_samples[us_idx](h)
                    us_idx += 1

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ---------------------------------------------------------------------------
# Gaussian Diffusion (DDPM / DDIM)
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """DDPM forward process and DDIM reverse sampling."""

    def __init__(self, n_steps=200, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.device = device

    def q_sample(self, z0, t, noise=None):
        """Forward diffusion: add noise to z0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_ab * z0 + sqrt_omab * noise, noise

    def training_loss(self, model, z0, context):
        """Compute epsilon-prediction loss."""
        B = z0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t, _ = self.q_sample(z0, t, noise)
        eps_pred = model(z_t, t, context)
        return F.mse_loss(eps_pred, noise)

    def predict_z0(self, z_t, t, eps_pred):
        """Predict clean latent from noisy latent and predicted noise."""
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return (z_t - sqrt_omab * eps_pred) / (sqrt_ab + 1e-8)

    @torch.no_grad()
    def ddim_sample(self, model, shape, context, n_steps=50,
                    eta=0.0, guidance_fn=None, guidance_scale=0.0):
        """
        DDIM sampling with optional physics guidance.

        Args:
            guidance_fn: callable(z0_pred) -> gradient for physics consistency
            guidance_scale: strength of physics guidance
        """
        device = self.device
        B = shape[0]
        z = torch.randn(shape, device=device)

        # Subsequence of timesteps for DDIM
        step_ratio = self.n_steps // n_steps
        timesteps = list(range(0, self.n_steps, step_ratio))[:n_steps]
        timesteps = list(reversed(timesteps))

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), timesteps[i + 1] if i + 1 < len(timesteps) else 0,
                                device=device, dtype=torch.long)

            alpha_t = self.alpha_bar[t][:, None, None, None]
            alpha_prev = self.alpha_bar[t_prev][:, None, None, None] if i + 1 < len(timesteps) \
                else torch.ones_like(alpha_t)

            eps_pred = model(z, t, context)
            z0_pred = (z - torch.sqrt(1 - alpha_t) * eps_pred) / (torch.sqrt(alpha_t) + 1e-8)

            # Physics guidance (DPS-style)
            if guidance_fn is not None and guidance_scale > 0:
                grad = guidance_fn(z0_pred)
                z0_pred = z0_pred - guidance_scale * grad

            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t + 1e-8)) * \
                    torch.sqrt(1 - alpha_t / (alpha_prev + 1e-8))
            dir_z = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0)) * eps_pred
            noise = torch.randn_like(z) if i + 1 < len(timesteps) else torch.zeros_like(z)
            z = torch.sqrt(alpha_prev) * z0_pred + dir_z + sigma * noise

        return z

    @torch.no_grad()
    def ddim_sample_with_grad(self, model, vae, forward_model, measurements,
                              shape, context, n_steps=50, guidance_scale=0.5):
        """
        DDIM sampling with DPS physics-guided correction.
        Computes gradient through the VAE decoder and forward model.
        """
        device = self.device
        B = shape[0]
        z = torch.randn(shape, device=device)

        step_ratio = self.n_steps // n_steps
        timesteps = list(range(0, self.n_steps, step_ratio))[:n_steps]
        timesteps = list(reversed(timesteps))

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), timesteps[i + 1] if i + 1 < len(timesteps) else 0,
                                device=device, dtype=torch.long)

            alpha_t = self.alpha_bar[t][:, None, None, None]
            alpha_prev = self.alpha_bar[t_prev][:, None, None, None] if i + 1 < len(timesteps) \
                else torch.ones_like(alpha_t)

            eps_pred = model(z, t, context)
            z0_pred = (z - torch.sqrt(1 - alpha_t) * eps_pred) / (torch.sqrt(alpha_t) + 1e-8)

            # DPS: physics guidance via differentiable forward model
            if guidance_scale > 0 and forward_model is not None:
                z0_g = z0_pred.detach().requires_grad_(True)
                scene_pred = vae.decode(z0_g)
                meas_pred = forward_model(scene_pred)
                # L1 consistency loss on magnitude
                phys_loss = F.l1_loss(meas_pred[:, 2], measurements[:, 2])
                phys_loss.backward()
                grad = z0_g.grad
                z0_pred = z0_pred - guidance_scale * grad

            dir_z = torch.sqrt(torch.clamp(1 - alpha_prev, min=0)) * eps_pred
            z = torch.sqrt(alpha_prev) * z0_pred + dir_z * 0  # simplified DDIM step
            # More precise DDIM:
            sigma = 0.0
            dir_z = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0)) * eps_pred
            noise = torch.zeros_like(z)
            z = torch.sqrt(alpha_prev) * z0_pred + dir_z + sigma * noise

        return z


# ---------------------------------------------------------------------------
# Baseline: Direct U-Net Regression
# ---------------------------------------------------------------------------

class UNetDirect(nn.Module):
    """Direct regression baseline: measurements -> scene."""

    def __init__(self, in_ch=4, out_ch=3, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.SiLU(),
                                  ResBlock(base_ch, base_ch))
        self.down1 = Downsample(base_ch)  # 32
        self.enc2 = nn.Sequential(ResBlock(base_ch, base_ch * 2))
        self.down2 = Downsample(base_ch * 2)  # 16
        self.enc3 = nn.Sequential(ResBlock(base_ch * 2, base_ch * 4))
        self.down3 = Downsample(base_ch * 4)  # 8

        # Bottleneck
        self.mid = nn.Sequential(ResBlock(base_ch * 4, base_ch * 4),
                                 SelfAttention(base_ch * 4),
                                 ResBlock(base_ch * 4, base_ch * 4))

        # Decoder
        self.up3 = Upsample(base_ch * 4)  # 16
        self.dec3 = ResBlock(base_ch * 8, base_ch * 2)
        self.up2 = Upsample(base_ch * 2)  # 32
        self.dec2 = ResBlock(base_ch * 4, base_ch)
        self.up1 = Upsample(base_ch)  # 64
        self.dec1 = ResBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        m = self.mid(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


# ---------------------------------------------------------------------------
# Baseline: CVAE
# ---------------------------------------------------------------------------

class CVAEBaseline(nn.Module):
    """Conditional VAE baseline: measurements + scene -> latent -> scene."""

    def __init__(self, meas_ch=4, scene_ch=3, latent_dim=64, base_ch=64):
        super().__init__()
        self.latent_dim = latent_dim

        # Measurement encoder
        self.meas_enc = nn.Sequential(
            nn.Conv2d(meas_ch, base_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        meas_feat = base_ch * 2

        # Scene encoder (for training, encodes GT scene)
        self.scene_enc = nn.Sequential(
            nn.Conv2d(scene_ch, base_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        scene_feat = base_ch * 2

        self.fc_mu = nn.Linear(meas_feat + scene_feat, latent_dim)
        self.fc_logvar = nn.Linear(meas_feat + scene_feat, latent_dim)

        # Prior (conditioned on measurements only)
        self.prior_mu = nn.Linear(meas_feat, latent_dim)
        self.prior_logvar = nn.Linear(meas_feat, latent_dim)

        # Decoder
        dec_in = latent_dim + meas_feat
        self.dec_fc = nn.Linear(dec_in, base_ch * 2 * 16 * 16)
        self.decoder = nn.Sequential(
            ResBlock(base_ch * 2, base_ch * 2),
            Upsample(base_ch * 2),
            ResBlock(base_ch * 2, base_ch),
            Upsample(base_ch),
            ResBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, scene_ch, 3, padding=1),
        )

    def encode(self, measurement, scene=None):
        m_feat = self.meas_enc(measurement)
        if scene is not None:
            s_feat = self.scene_enc(scene)
            combined = torch.cat([m_feat, s_feat], dim=1)
            return self.fc_mu(combined), self.fc_logvar(combined), m_feat
        else:
            return self.prior_mu(m_feat), self.prior_logvar(m_feat), m_feat

    def decode(self, z, m_feat):
        h = self.dec_fc(torch.cat([z, m_feat], dim=1))
        h = h.reshape(-1, 128, 16, 16)
        return torch.sigmoid(self.decoder(h))

    def forward(self, measurement, scene=None):
        if scene is not None:
            mu, logvar, m_feat = self.encode(measurement, scene)
        else:
            mu, logvar, m_feat = self.encode(measurement)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_hat = self.decode(z, m_feat)

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return x_hat, kl
