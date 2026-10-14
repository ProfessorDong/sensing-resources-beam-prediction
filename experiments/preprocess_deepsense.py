"""
DeepSense 6G preprocessing: radar → range-Doppler, LiDAR → BEV occupancy, beam → conditioning.
Produces PyTorch-ready tensors for S2S-Diff training and evaluation.
"""
import os, sys, glob, logging
import numpy as np
import torch
import torch.nn.functional as torchF
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# LiDAR → BEV occupancy
# ============================================================

def read_ply_ascii(path):
    """Read ASCII PLY file with double x,y,z and ushort intensity. Fast numpy version."""
    with open(path, 'r') as f:
        n_pts = 0
        header_lines = 0
        for line in f:
            header_lines += 1
            if 'element vertex' in line:
                n_pts = int(line.strip().split()[-1])
            if 'end_header' in line:
                break

    if n_pts == 0:
        return np.zeros((0, 4), dtype=np.float32)

    try:
        data = np.loadtxt(path, skiprows=header_lines, max_rows=n_pts,
                          dtype=np.float32, usecols=(0, 1, 2, 3))
        return data  # [N, 4]
    except Exception:
        return np.zeros((0, 4), dtype=np.float32)


def lidar_to_bev(points, grid_size=64, x_range=(-32, 32), y_range=(0, 64),
                 z_range=(-2, 10)):
    """
    Project 3D LiDAR points to BEV occupancy grid.
    Args:
        points: [N, 4] (x, y, z, intensity)
        grid_size: output grid size
        x_range, y_range: physical extent in meters
        z_range: height filter
    Returns:
        occupancy: [H, W] binary grid
        reflectivity: [H, W] normalized intensity
    """
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    # Height filter
    mask = (z >= z_range[0]) & (z <= z_range[1])
    mask &= (x >= x_range[0]) & (x < x_range[1])
    mask &= (y >= y_range[0]) & (y < y_range[1])
    x, y, intensity = x[mask], y[mask], intensity[mask]

    if len(x) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32), \
               np.zeros((grid_size, grid_size), dtype=np.float32)

    # Map to grid indices
    cell_x = ((x - x_range[0]) / (x_range[1] - x_range[0]) * grid_size).astype(int)
    cell_y = ((y - y_range[0]) / (y_range[1] - y_range[0]) * grid_size).astype(int)
    cell_x = np.clip(cell_x, 0, grid_size - 1)
    cell_y = np.clip(cell_y, 0, grid_size - 1)

    occupancy = np.zeros((grid_size, grid_size), dtype=np.float32)
    ref_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    ref_count = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(len(x)):
        occupancy[cell_y[i], cell_x[i]] = 1.0
        ref_sum[cell_y[i], cell_x[i]] += intensity[i]
        ref_count[cell_y[i], cell_x[i]] += 1

    reflectivity = np.zeros_like(occupancy)
    valid = ref_count > 0
    reflectivity[valid] = ref_sum[valid] / ref_count[valid]
    # Normalize reflectivity to [0, 1]
    if reflectivity.max() > 0:
        reflectivity = reflectivity / reflectivity.max()

    return occupancy, reflectivity


# ============================================================
# Radar → range-Doppler image
# ============================================================

def radar_to_range_doppler(radar_complex, n_doppler_fft=250):
    """
    Convert raw radar tensor to range-Doppler magnitude image.
    Args:
        radar_complex: [4, 256, 250] complex64 (antennas, range, chirps)
    Returns:
        rd_image: [4, 256, 250] float32 (real channels from range-Doppler)
    """
    # FFT along chirps dimension for Doppler
    rd = np.fft.fftshift(np.fft.fft(radar_complex, n=n_doppler_fft, axis=2), axes=2)

    # Create 4-channel real representation: mag per antenna
    rd_mag = np.abs(rd).astype(np.float32)  # [4, 256, 250]

    return rd_mag


def resize_radar(rd_mag, target_h=64, target_w=64):
    """Resize range-Doppler image to target size using torch interpolation."""
    # rd_mag: [C, H, W]
    t = torch.from_numpy(rd_mag).unsqueeze(0)  # [1, C, H, W]
    t = torch.nn.functional.interpolate(t, size=(target_h, target_w),
                                         mode='bilinear', align_corners=False)
    return t.squeeze(0).numpy()  # [C, target_h, target_w]


# ============================================================
# Beam power → conditioning vector
# ============================================================

def load_beam_power(path):
    """Load 64-element beam power vector."""
    return np.loadtxt(path).astype(np.float32)  # [64]


# ============================================================
# DeepSense Dataset
# ============================================================

class DeepSenseDataset(Dataset):
    """PyTorch dataset for preprocessed DeepSense 6G data."""

    def __init__(self, scenes, measurements, beam_powers):
        """
        Args:
            scenes: [N, 3, H, W] (occupancy, reflectivity, dynamic)
            measurements: [N, 4, H, W] (radar range-Doppler channels)
            beam_powers: [N, 64] (mmWave beam power vectors)
        """
        self.scenes = scenes
        self.measurements = measurements
        self.beam_powers = beam_powers

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return {
            "scene": self.scenes[idx],
            "measurement": self.measurements[idx],
            "beam_power": self.beam_powers[idx],
            "obs_mask": torch.zeros(64, 64),  # placeholder
            "obs_values": torch.zeros(64, 64),  # placeholder
            "sparsity_mask": torch.ones(64, 64),  # no synthetic sparsity
        }


def preprocess_scenario(scenario_dir, grid_size=64, max_frames=None):
    """
    Preprocess one DeepSense scenario into tensors.
    Returns scenes, measurements, beam_powers as numpy arrays.
    """
    unit1_dir = os.path.join(scenario_dir, "unit1")

    radar_files = sorted(glob.glob(os.path.join(unit1_dir, "radar_data", "*.npy")))
    lidar_files = sorted(glob.glob(os.path.join(unit1_dir, "lidar_data", "*.ply")))
    beam_files = sorted(glob.glob(os.path.join(unit1_dir, "mmWave_data", "*.txt")))

    n = min(len(radar_files), len(lidar_files), len(beam_files))
    if max_frames:
        n = min(n, max_frames)

    log.info(f"  Processing {n} frames from {scenario_dir}")

    # First pass: determine LiDAR spatial extent
    log.info("  Determining LiDAR spatial extent...")
    sample_indices = np.linspace(0, n-1, min(50, n), dtype=int)
    all_x, all_y = [], []
    for idx in sample_indices:
        pts = read_ply_ascii(lidar_files[idx])
        if len(pts) > 0:
            all_x.extend(pts[:, 0].tolist())
            all_y.extend(pts[:, 1].tolist())

    if len(all_x) > 0:
        x_med = np.median(all_x)
        y_med = np.median(all_y)
        # Center the grid on the median point
        half = grid_size * 0.5  # 0.5m cells → 32m half-extent
        x_range = (x_med - half, x_med + half)
        y_range = (y_med - half, y_med + half)
        log.info(f"  LiDAR extent: x=[{x_range[0]:.1f},{x_range[1]:.1f}], "
                 f"y=[{y_range[0]:.1f},{y_range[1]:.1f}]")
    else:
        x_range = (-32, 32)
        y_range = (0, 64)

    # Second pass: process all frames
    scenes_list = []
    meas_list = []
    beam_list = []
    skipped = 0

    for i in range(n):
        if i % 500 == 0:
            log.info(f"  Frame {i}/{n}...")

        # LiDAR → BEV
        pts = read_ply_ascii(lidar_files[i])
        if len(pts) < 10:
            skipped += 1
            continue
        occ, ref = lidar_to_bev(pts, grid_size, x_range, y_range)

        # Skip frames with too few occupied cells
        if occ.sum() < 5:
            skipped += 1
            continue

        # Dynamic channel = 0 (no temporal info in single frame)
        dyn = np.zeros_like(occ)
        scene = np.stack([occ, ref, dyn], axis=0)  # [3, H, W]

        # Radar → range-Doppler
        radar_raw = np.load(radar_files[i])  # [4, 256, 250] complex
        rd_mag = radar_to_range_doppler(radar_raw)  # [4, 256, 250]
        rd_resized = resize_radar(rd_mag, grid_size, grid_size)  # [4, 64, 64]

        # Normalize radar
        rd_max = rd_resized.max()
        if rd_max > 0:
            rd_resized = rd_resized / rd_max

        # Beam power
        beam = load_beam_power(beam_files[i])  # [64]

        scenes_list.append(scene)
        meas_list.append(rd_resized)
        beam_list.append(beam)

    log.info(f"  Processed {len(scenes_list)} frames, skipped {skipped}")

    scenes = np.array(scenes_list, dtype=np.float32)
    measurements = np.array(meas_list, dtype=np.float32)
    beams = np.array(beam_list, dtype=np.float32)

    return scenes, measurements, beams


def build_deepsense_datasets(data_root, grid_size=64, max_frames=None):
    """
    Build train/test datasets from DeepSense scenarios.
    Train: Scenario 31
    Test: Scenario 33 (cross-scenario generalization)
    """
    log.info("=== Preprocessing Scenario 31 (train) ===")
    train_scenes, train_meas, train_beams = preprocess_scenario(
        os.path.join(data_root, "Scenario31"), grid_size, max_frames
    )

    log.info("=== Preprocessing Scenario 33 (test) ===")
    test_scenes, test_meas, test_beams = preprocess_scenario(
        os.path.join(data_root, "Scenario33"), grid_size, max_frames
    )

    # Split train into train/val (90/10)
    n_train = len(train_scenes)
    n_val = max(1, n_train // 10)
    perm = np.random.RandomState(42).permutation(n_train)

    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    train_ds = DeepSenseDataset(
        torch.tensor(train_scenes[tr_idx]),
        torch.tensor(train_meas[tr_idx]),
        torch.tensor(train_beams[tr_idx]),
    )
    val_ds = DeepSenseDataset(
        torch.tensor(train_scenes[val_idx]),
        torch.tensor(train_meas[val_idx]),
        torch.tensor(train_beams[val_idx]),
    )
    test_ds = DeepSenseDataset(
        torch.tensor(test_scenes),
        torch.tensor(test_meas),
        torch.tensor(test_beams),
    )

    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    log.info(f"Scene shape: {train_ds.scenes[0].shape}")
    log.info(f"Measurement shape: {train_ds.measurements[0].shape}")
    log.info(f"Beam shape: {train_ds.beam_powers[0].shape}")
    log.info(f"Train occupancy fill: {train_ds.scenes[:, 0].mean():.4f}")

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    data_root = os.path.join(os.path.dirname(__file__), "..", "datasets", "deepsense6g")
    train_ds, val_ds, test_ds = build_deepsense_datasets(data_root, max_frames=None)

    # Save preprocessed data
    save_dir = os.path.join(os.path.dirname(__file__), "preprocessed")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"scenes": train_ds.scenes, "meas": train_ds.measurements,
                "beams": train_ds.beam_powers}, os.path.join(save_dir, "deepsense_train.pt"))
    torch.save({"scenes": val_ds.scenes, "meas": val_ds.measurements,
                "beams": val_ds.beam_powers}, os.path.join(save_dir, "deepsense_val.pt"))
    torch.save({"scenes": test_ds.scenes, "meas": test_ds.measurements,
                "beams": test_ds.beam_powers}, os.path.join(save_dir, "deepsense_test.pt"))
    log.info(f"Saved preprocessed data to {save_dir}/")
