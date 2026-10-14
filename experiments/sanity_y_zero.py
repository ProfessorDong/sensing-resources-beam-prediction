"""
Y=0 sanity check on ISAC Sc.44 test set.

Loads the canonical S2S-Diff ISAC checkpoint and runs DDIM inference with
the measurement tensor set to all zeros. If the model is genuinely using Y,
IoU should collapse significantly. If the model has memorized a scene prior
and ignores Y, IoU stays close to the canonical 0.853.

Output: experiments/results/sanity_y_zero.json
"""
import os, sys, json, time, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, SimpleForwardModel, SimpleDS,
    _try_load, _clear_gpu,
)
from s2s_diff_v3 import evaluate_full, infer_diffusion
from torch.utils.data import DataLoader

CKPT = os.path.join(CKPT_DIR, "v2", "DeepSense_ISAC__Sc_42_43_44_")
SAVE = os.path.join(RESULTS_DIR, "sanity_y_zero.json")


def main():
    device = torch.device("cuda")
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    print("Loading ISAC Sc.44 test data ...")
    from train_all_real import preprocess_isac_scenario
    ds_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    sc_dir = os.path.join(ds_root, "deepsense6g", "Scenario44")
    scenes, meas = preprocess_isac_scenario(sc_dir, 44, max_frames=1500)
    print(f"  {len(scenes)} frames")

    diff_model = PixelDiffusionUNet().to(device)
    diffusion = DiffusionV2(n_steps=300, device=device)
    _try_load(diff_model, os.path.join(CKPT, "diff_v2.pt"), "diff_v2")
    diff_model.eval()

    test_ds = SimpleDS(scenes, meas)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    out = {}

    # Baseline: real Y
    print("\n[A] Inference with REAL Y ...")
    t = time.time()
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    pred, gt = infer_diffusion(diff_model, diffusion, test_loader,
                               n_steps=50, n_samples=4, eta=0.3, device=device)
    m_real = evaluate_full(pred, gt)
    out["real_Y"] = {**{k: float(v) for k, v in m_real.items()},
                     "elapsed": float(time.time() - t)}
    print(f"  IoU={m_real['IoU']:.4f}  ECE={m_real['ECE']:.4f}  NLL={m_real['NLL']:.4f}")

    # Y=0 sanity
    print("\n[B] Inference with Y=0 ...")
    zero_meas = np.zeros_like(meas, dtype=np.float32)
    test_ds_zero = SimpleDS(scenes, zero_meas)
    test_loader_zero = DataLoader(test_ds_zero, batch_size=8, shuffle=False, num_workers=0)
    t = time.time()
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    pred_z, gt_z = infer_diffusion(diff_model, diffusion, test_loader_zero,
                                   n_steps=50, n_samples=4, eta=0.3, device=device)
    m_zero = evaluate_full(pred_z, gt_z)
    out["zero_Y"] = {**{k: float(v) for k, v in m_zero.items()},
                     "elapsed": float(time.time() - t)}
    print(f"  IoU={m_zero['IoU']:.4f}  ECE={m_zero['ECE']:.4f}  NLL={m_zero['NLL']:.4f}")

    # Shuffled Y across frames
    print("\n[C] Inference with Y shuffled across frames ...")
    perm = np.random.RandomState(42).permutation(len(meas))
    shuf_meas = meas[perm]
    test_ds_shuf = SimpleDS(scenes, shuf_meas)
    test_loader_shuf = DataLoader(test_ds_shuf, batch_size=8, shuffle=False, num_workers=0)
    t = time.time()
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    pred_s, gt_s = infer_diffusion(diff_model, diffusion, test_loader_shuf,
                                   n_steps=50, n_samples=4, eta=0.3, device=device)
    m_shuf = evaluate_full(pred_s, gt_s)
    out["shuffled_Y"] = {**{k: float(v) for k, v in m_shuf.items()},
                         "elapsed": float(time.time() - t)}
    print(f"  IoU={m_shuf['IoU']:.4f}  ECE={m_shuf['ECE']:.4f}  NLL={m_shuf['NLL']:.4f}")

    # Spatially permuted Y (within each frame)
    print("\n[D] Inference with Y spatially permuted within each frame ...")
    spat_meas = meas.copy()
    rng = np.random.RandomState(123)
    for i in range(len(spat_meas)):
        # Permute the H*W cells of every channel using the same permutation
        H, W = spat_meas[i].shape[-2:]
        idx = rng.permutation(H * W)
        flat = spat_meas[i].reshape(spat_meas[i].shape[0], -1)
        spat_meas[i] = flat[:, idx].reshape(spat_meas[i].shape[0], H, W)
    test_ds_spat = SimpleDS(scenes, spat_meas)
    test_loader_spat = DataLoader(test_ds_spat, batch_size=8, shuffle=False, num_workers=0)
    t = time.time()
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    pred_p, gt_p = infer_diffusion(diff_model, diffusion, test_loader_spat,
                                   n_steps=50, n_samples=4, eta=0.3, device=device)
    m_spat = evaluate_full(pred_p, gt_p)
    out["spatial_perm_Y"] = {**{k: float(v) for k, v in m_spat.items()},
                              "elapsed": float(time.time() - t)}
    print(f"  IoU={m_spat['IoU']:.4f}  ECE={m_spat['ECE']:.4f}  NLL={m_spat['NLL']:.4f}")

    # Summary
    print("\n=== Summary (ISAC Sc.44) ===")
    print(f"  IoU    real={m_real['IoU']:.4f}  Y=0={m_zero['IoU']:.4f} "
          f" shuffled={m_shuf['IoU']:.4f}  spatial_perm={m_spat['IoU']:.4f}")
    print(f"  ECE    real={m_real['ECE']:.4f}  Y=0={m_zero['ECE']:.4f} "
          f" shuffled={m_shuf['ECE']:.4f}  spatial_perm={m_spat['ECE']:.4f}")
    drop_zero = m_real['IoU'] - m_zero['IoU']
    drop_shuf = m_real['IoU'] - m_shuf['IoU']
    drop_spat = m_real['IoU'] - m_spat['IoU']
    print(f"  IoU drop:  real - Y=0          = {drop_zero:+.4f}")
    print(f"             real - shuffled     = {drop_shuf:+.4f}")
    print(f"             real - spatial_perm = {drop_spat:+.4f}")

    out["summary"] = {
        "iou_drop_zero":         float(drop_zero),
        "iou_drop_shuffled":     float(drop_shuf),
        "iou_drop_spatial_perm": float(drop_spat),
        "uses_Y": bool(max(drop_zero, drop_shuf, drop_spat) > 0.05),
    }

    with open(SAVE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {SAVE}")


if __name__ == "__main__":
    main()
