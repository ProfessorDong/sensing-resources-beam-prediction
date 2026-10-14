"""
Configuration for S2S-Diff experiments.
"""
from dataclasses import dataclass, field
from typing import Tuple
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

@dataclass
class SceneConfig:
    H: int = 64
    W: int = 64
    C_x: int = 3  # occupancy, reflectivity, dynamic
    cell_size_m: float = 0.5  # meters per cell
    n_train: int = 5000
    n_val: int = 500
    n_test: int = 500

@dataclass
class SensorConfig:
    sensor_pos: Tuple[float, float] = (32.0, 0.0)  # pixel coords (center-bottom)
    n_range: int = 64
    n_angle: int = 64
    max_range_m: float = 45.0
    fov_deg: float = 120.0
    carrier_freq_ghz: float = 28.0
    bandwidth_mhz: float = 400.0
    n_antennas: int = 16
    default_snr_db: float = 10.0
    default_missing_ratio: float = 0.3

@dataclass
class ModelConfig:
    # Scene VAE
    vae_latent_ch: int = 8
    vae_base_ch: int = 32
    vae_kl_weight: float = 1e-5
    # Measurement encoder
    enc_base_ch: int = 32
    enc_context_dim: int = 128
    enc_n_heads: int = 4
    enc_n_layers: int = 2
    # Diffusion U-Net
    diff_base_ch: int = 64
    diff_ch_mults: Tuple[int, ...] = (1, 2, 4)
    diff_n_res_blocks: int = 2
    diff_attn_resolutions: Tuple[int, ...] = (8, 4)
    diff_n_heads: int = 4
    diff_dropout: float = 0.1
    # Diffusion process
    n_diffusion_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    n_inference_steps: int = 50  # DDIM steps
    n_posterior_samples: int = 8
    # Physics guidance
    physics_guidance_scale: float = 0.5

@dataclass
class TrainConfig:
    # VAE
    vae_epochs: int = 100
    vae_lr: float = 1e-3
    vae_batch_size: int = 64
    # Diffusion
    diff_epochs: int = 200
    diff_lr: float = 2e-4
    diff_batch_size: int = 32
    # Baselines
    baseline_epochs: int = 150
    baseline_lr: float = 1e-3
    baseline_batch_size: int = 64
    # General
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    lambda_seg: float = 1.0
    lambda_phys: float = 0.1
    lambda_obs: float = 1.0
    lambda_dice: float = 0.5
    lambda_ref: float = 0.5
    seed: int = 42
    num_workers: int = 4

@dataclass
class Config:
    scene: SceneConfig = field(default_factory=SceneConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
