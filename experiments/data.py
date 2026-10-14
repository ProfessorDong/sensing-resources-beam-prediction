"""
Synthetic ISAC scene generation and differentiable forward model.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from config import SceneConfig, SensorConfig


class ISACForwardModel(nn.Module):
    """Differentiable ISAC forward model: BEV scene -> range-angle measurements."""

    def __init__(self, scene_cfg: SceneConfig, sensor_cfg: SensorConfig):
        super().__init__()
        H, W = scene_cfg.H, scene_cfg.W
        Nr, Na = sensor_cfg.n_range, sensor_cfg.n_angle
        sx, sy = sensor_cfg.sensor_pos

        # Physical coordinates for each scene cell (meters)
        scene_extent = H * scene_cfg.cell_size_m
        x_phys = torch.linspace(
            -scene_extent / 2, scene_extent / 2, W
        )  # left-right
        y_phys = torch.linspace(0, scene_extent, H)  # near-far from sensor
        yy, xx = torch.meshgrid(y_phys, x_phys, indexing="ij")  # [H, W]

        # Sensor physical position
        sx_phys = (sx - W / 2) * scene_cfg.cell_size_m
        sy_phys = sy * scene_cfg.cell_size_m

        dx = xx - sx_phys  # [H, W]
        dy = yy - sy_phys  # [H, W]
        cell_ranges = torch.sqrt(dx**2 + dy**2)  # [H, W]
        cell_angles = torch.atan2(dx, dy)  # [H, W], angle from broadside

        # Measurement grid
        range_bins = torch.linspace(0, sensor_cfg.max_range_m, Nr)
        fov_rad = sensor_cfg.fov_deg * np.pi / 180
        angle_bins = torch.linspace(-fov_rad / 2, fov_rad / 2, Na)

        # Wavelength and resolution
        wavelength = 3e8 / (sensor_cfg.carrier_freq_ghz * 1e9)
        range_res = 3e8 / (2 * sensor_cfg.bandwidth_mhz * 1e6)
        # Rayleigh angular resolution
        d_ant = wavelength / 2
        array_length = sensor_cfg.n_antennas * d_ant
        angle_res = wavelength / array_length

        # Build sensing kernel K: [Nr*Na, H*W]
        # K[m, g] = exp(-(r_m - d_g)^2 / (2*sigma_r^2)) * exp(-(theta_m - phi_g)^2 / (2*sigma_theta^2))
        sigma_r = range_res / 2.0
        sigma_theta = angle_res / 2.0

        # For phase: round-trip phase 4*pi*d/lambda
        k_wave = 2 * np.pi / wavelength

        # Flatten cell positions
        flat_ranges = cell_ranges.reshape(-1)  # [H*W]
        flat_angles = cell_angles.reshape(-1)  # [H*W]

        # Field of view mask
        fov_mask = (flat_angles.abs() <= fov_rad / 2).float()  # [H*W]

        # Build kernel: vectorized over bins and cells
        # range_bins: [Nr], flat_ranges: [H*W]
        range_diff = range_bins.unsqueeze(1) - flat_ranges.unsqueeze(0)  # [Nr, H*W]
        K_range = torch.exp(-range_diff**2 / (2 * sigma_r**2))  # [Nr, H*W]

        angle_diff = angle_bins.unsqueeze(1) - flat_angles.unsqueeze(0)  # [Na, H*W]
        K_angle = torch.exp(-angle_diff**2 / (2 * sigma_theta**2))  # [Na, H*W]

        # Full 2D kernel: K[r, a, g] = K_range[r, g] * K_angle[a, g]
        # Reshape: K_range [Nr, 1, H*W], K_angle [1, Na, H*W]
        K_full = K_range.unsqueeze(1) * K_angle.unsqueeze(0)  # [Nr, Na, H*W]
        K_full = K_full * fov_mask.unsqueeze(0).unsqueeze(0)  # apply FoV mask

        # Path loss: 1/d^2 (avoid div by zero)
        path_loss = 1.0 / (flat_ranges**2 + 1e-6)  # [H*W]
        K_full = K_full * path_loss.unsqueeze(0).unsqueeze(0)

        # Normalize kernel
        K_max = K_full.max()
        if K_max > 0:
            K_full = K_full / K_max

        # Phase matrix for complex measurements
        phase_matrix = 2 * k_wave * flat_ranges  # round-trip phase [H*W]

        self.register_buffer("K_full", K_full)  # [Nr, Na, H*W]
        self.register_buffer("phase_matrix", phase_matrix)  # [H*W]
        self.register_buffer("cell_ranges", cell_ranges)  # [H, W]
        self.register_buffer("cell_angles", cell_angles)  # [H, W]

        self.Nr = Nr
        self.Na = Na
        self.H = H
        self.W = W

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        Map scene tensor to complex range-angle measurements.
        Args:
            scene: [B, 3, H, W] (occupancy, reflectivity, dynamic)
        Returns:
            measurements: [B, 4, Nr, Na] (real, imag, magnitude, phase)
        """
        B = scene.shape[0]
        occ = scene[:, 0]  # [B, H, W]
        ref = scene[:, 1]  # [B, H, W]

        # Effective scattering
        eff = (occ * ref).reshape(B, -1)  # [B, H*W]

        # Complex amplitude per cell: eff * exp(-j * phase)
        phase = self.phase_matrix.unsqueeze(0).expand(B, -1)  # [B, H*W]
        eff_real = eff * torch.cos(phase)  # [B, H*W]
        eff_imag = eff * (-torch.sin(phase))  # [B, H*W]

        # Apply sensing kernel: y[r, a] = sum_g K[r, a, g] * eff_complex[g]
        # K_full: [Nr, Na, H*W], eff: [B, H*W]
        K_flat = self.K_full.reshape(self.Nr * self.Na, -1)  # [Nr*Na, H*W]
        y_real = torch.matmul(K_flat, eff_real.T).T.reshape(B, self.Nr, self.Na)
        y_imag = torch.matmul(K_flat, eff_imag.T).T.reshape(B, self.Nr, self.Na)

        y_mag = torch.sqrt(y_real**2 + y_imag**2 + 1e-8)
        y_phase = torch.atan2(y_imag, y_real)

        return torch.stack([y_real, y_imag, y_mag, y_phase], dim=1)  # [B, 4, Nr, Na]


def add_measurement_noise(measurements: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add complex Gaussian noise to measurements at given SNR."""
    # measurements: [B, 4, Nr, Na] with channels [real, imag, mag, phase]
    real = measurements[:, 0]
    imag = measurements[:, 1]
    signal_power = (real**2 + imag**2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = torch.sqrt(noise_power / 2)

    noise_r = torch.randn_like(real) * noise_std
    noise_i = torch.randn_like(imag) * noise_std

    noisy_real = real + noise_r
    noisy_imag = imag + noise_i
    noisy_mag = torch.sqrt(noisy_real**2 + noisy_imag**2 + 1e-8)
    noisy_phase = torch.atan2(noisy_imag, noisy_real)

    return torch.stack([noisy_real, noisy_imag, noisy_mag, noisy_phase], dim=1)


def apply_sparsity_mask(
    measurements: torch.Tensor, missing_ratio: float
) -> tuple:
    """Randomly mask measurement bins to simulate sparse observations."""
    B, C, Nr, Na = measurements.shape
    mask = (torch.rand(B, 1, Nr, Na, device=measurements.device) > missing_ratio).float()
    masked = measurements * mask
    return masked, mask[:, 0]  # return [B, Nr, Na] mask


def generate_scene(H: int = 64, W: int = 64, rng: np.random.RandomState = None):
    """Generate a random urban BEV scene tensor [3, H, W]."""
    if rng is None:
        rng = np.random.RandomState()

    occ = np.zeros((H, W), dtype=np.float32)
    ref = np.zeros((H, W), dtype=np.float32)
    dyn = np.zeros((H, W), dtype=np.float32)

    # --- Buildings (large rectangles) ---
    n_buildings = rng.randint(3, 8)
    for _ in range(n_buildings):
        bw = rng.randint(4, 13)
        bh = rng.randint(4, 13)
        bx = rng.randint(2, max(3, W - bw - 2))
        by = rng.randint(2, max(3, H - bh - 2))
        if occ[by : by + bh, bx : bx + bw].sum() < 0.3 * bw * bh:
            occ[by : by + bh, bx : bx + bw] = 1.0
            r_val = rng.uniform(0.5, 1.0)
            ref[by : by + bh, bx : bx + bw] = r_val

    # --- Walls (thin lines) ---
    n_walls = rng.randint(0, 4)
    for _ in range(n_walls):
        if rng.rand() > 0.5:
            wy = rng.randint(5, H - 5)
            wx1 = rng.randint(0, W // 2)
            wx2 = rng.randint(wx1 + 5, min(wx1 + 25, W))
            occ[wy : wy + 1, wx1:wx2] = 1.0
            ref[wy : wy + 1, wx1:wx2] = rng.uniform(0.3, 0.7)
        else:
            wx = rng.randint(5, W - 5)
            wy1 = rng.randint(0, H // 2)
            wy2 = rng.randint(wy1 + 5, min(wy1 + 25, H))
            occ[wy1:wy2, wx : wx + 1] = 1.0
            ref[wy1:wy2, wx : wx + 1] = rng.uniform(0.3, 0.7)

    # --- L-shaped structures ---
    n_L = rng.randint(0, 3)
    for _ in range(n_L):
        lx = rng.randint(3, W - 15)
        ly = rng.randint(3, H - 15)
        lw = rng.randint(3, 8)
        lh = rng.randint(6, 12)
        r_val = rng.uniform(0.4, 0.9)
        # Vertical part
        if occ[ly : ly + lh, lx : lx + 2].sum() < 0.3 * lh * 2:
            occ[ly : ly + lh, lx : lx + 2] = 1.0
            ref[ly : ly + lh, lx : lx + 2] = r_val
            # Horizontal part
            occ[ly : ly + 2, lx : lx + lw] = 1.0
            ref[ly : ly + 2, lx : lx + lw] = r_val

    # --- Vehicles (small, dynamic) ---
    n_vehicles = rng.randint(2, 7)
    for _ in range(n_vehicles):
        vw = rng.randint(2, 4)
        vh = rng.randint(1, 3)
        vx = rng.randint(0, W - vw)
        vy = rng.randint(0, H - vh)
        if occ[vy : vy + vh, vx : vx + vw].sum() == 0:
            occ[vy : vy + vh, vx : vx + vw] = 1.0
            ref[vy : vy + vh, vx : vx + vw] = rng.uniform(0.3, 0.8)
            dyn[vy : vy + vh, vx : vx + vw] = 1.0

    # --- Small clutter ---
    n_clutter = rng.randint(0, 8)
    for _ in range(n_clutter):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        if occ[cy, cx] == 0:
            occ[cy, cx] = 1.0
            ref[cy, cx] = rng.uniform(0.1, 0.4)

    return np.stack([occ, ref, dyn], axis=0)  # [3, H, W]


def extract_observed_mask(
    measurements: torch.Tensor, threshold: float = 0.15
) -> tuple:
    """
    Extract high-confidence observed cells from thresholded measurements.
    Returns mask and observed scene values (occupancy-like from measurement magnitude).
    """
    mag = measurements[:, 2]  # [B, Nr, Na] magnitude channel
    # Normalize per sample
    mag_max = mag.reshape(mag.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    mag_norm = mag / (mag_max + 1e-8)

    # CFAR-like threshold
    obs_mask = (mag_norm > threshold).float()  # [B, Nr, Na]
    obs_values = mag_norm * obs_mask

    return obs_mask, obs_values


class ISACDataset(Dataset):
    """Pre-generated synthetic ISAC dataset."""

    def __init__(self, scenes, measurements, masks, obs_masks, obs_values):
        self.scenes = scenes  # [N, 3, H, W]
        self.measurements = measurements  # [N, 4, Nr, Na]
        self.masks = masks  # [N, Nr, Na]
        self.obs_masks = obs_masks  # [N, Nr, Na]
        self.obs_values = obs_values  # [N, Nr, Na]

    def __len__(self):
        return self.scenes.shape[0]

    def __getitem__(self, idx):
        return {
            "scene": self.scenes[idx],
            "measurement": self.measurements[idx],
            "sparsity_mask": self.masks[idx],
            "obs_mask": self.obs_masks[idx],
            "obs_values": self.obs_values[idx],
        }


def build_dataset(
    n_samples: int,
    scene_cfg: SceneConfig,
    sensor_cfg: SensorConfig,
    forward_model: ISACForwardModel,
    snr_db: float = None,
    missing_ratio: float = None,
    seed: int = 0,
):
    """Generate a full dataset of scenes and measurements."""
    if snr_db is None:
        snr_db = sensor_cfg.default_snr_db
    if missing_ratio is None:
        missing_ratio = sensor_cfg.default_missing_ratio

    rng = np.random.RandomState(seed)
    scenes = []
    for _ in range(n_samples):
        s = generate_scene(scene_cfg.H, scene_cfg.W, rng)
        scenes.append(s)
    scenes = torch.tensor(np.array(scenes), dtype=torch.float32)  # [N, 3, H, W]

    # Generate measurements in batches to avoid OOM
    all_meas, all_masks, all_obs_masks, all_obs_vals = [], [], [], []
    batch = 128
    device = next(forward_model.parameters()).device if len(list(forward_model.parameters())) > 0 else torch.device("cpu")
    # Use buffer device
    device = forward_model.K_full.device

    for i in range(0, n_samples, batch):
        s_batch = scenes[i : i + batch].to(device)
        with torch.no_grad():
            m = forward_model(s_batch)
            m = add_measurement_noise(m, snr_db)
            m_sparse, mask = apply_sparsity_mask(m, missing_ratio)
            obs_mask, obs_val = extract_observed_mask(m_sparse)
        all_meas.append(m_sparse.cpu())
        all_masks.append(mask.cpu())
        all_obs_masks.append(obs_mask.cpu())
        all_obs_vals.append(obs_val.cpu())

    measurements = torch.cat(all_meas, dim=0)
    masks = torch.cat(all_masks, dim=0)
    obs_masks = torch.cat(all_obs_masks, dim=0)
    obs_values = torch.cat(all_obs_vals, dim=0)

    return ISACDataset(scenes, measurements, masks, obs_masks, obs_values)
