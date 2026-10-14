"""
Training-seed variance on RADIal.

Re-trains S2S-Diff (PixelDiffusionUNet + SimpleForwardModel) on RADIal from
scratch under two additional seeds (43, 44) and evaluates on the canonical
RADIal test split. Combined with the existing canonical seed-42 run, this
gives 3 training seeds for genuine training-variance σ.

Output: experiments/results/radial_training_seeds.json
"""
import os, sys, json, time, gc, glob, logging
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, SimpleForwardModel, SimpleDS,
    train_s2s_v2, _try_load, _clear_gpu,
)
from s2s_diff_v3 import evaluate_full, infer_diffusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUT = os.path.join(RESULTS_DIR, "radial_training_seeds.json")
SEEDS = [43, 44]
EPOCHS = 150


def _seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    device = torch.device("cuda")

    # Load RADIal exactly as canonical run did
    from train_all_real import preprocess_radial
    ds_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    radial_base = os.path.join(ds_root, "radial")
    seq_dirs = sorted(glob.glob(os.path.join(radial_base, "extracted", "RECORD*")))
    labels_csv = os.path.join(radial_base, "labels_CVPR.csv")
    config_path = os.path.join(radial_base, "miscellaneous", "RADAR_CONFIG.m")
    log.info("Preprocessing RADIal (max 500 frames) ...")
    scenes, meas = preprocess_radial(seq_dirs[0], labels_csv, config_path, max_frames=500)
    log.info(f"  loaded {len(scenes)} frames")

    n_train = int(0.8 * len(scenes))
    train_scenes = scenes[:n_train]; train_meas = meas[:n_train]
    test_scenes = scenes[n_train:]; test_meas = meas[n_train:]
    # Same train/val split as run_on_dataset
    n = len(train_scenes); nv = max(1, n // 10)
    perm = np.random.RandomState(42).permutation(n)  # split is fixed across seeds
    tr_ds = SimpleDS(train_scenes[perm[nv:]], train_meas[perm[nv:]])
    val_ds = SimpleDS(train_scenes[perm[:nv]], train_meas[perm[:nv]])
    test_ds = SimpleDS(test_scenes, test_meas)
    log.info(f"  train={len(tr_ds)} val={len(val_ds)} test={len(test_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    results = {"per_seed": {}}

    # Seed 42 is the canonical row already in v3_results.json. Pull and re-cite.
    v3 = json.load(open(os.path.join(RESULTS_DIR, "v3", "v3_results.json")))
    seed42 = v3["RADIal"]["S2S-Diff"]
    results["per_seed"]["seed_42_canonical"] = {**seed42, "init": "canonical (v3)"}
    log.info(f"\n[Seed 42 canonical] IoU={seed42['IoU']:.4f} F1={seed42['F1']:.4f} ECE={seed42['ECE']:.4f}")

    for seed in SEEDS:
        log.info(f"\n=== Seed {seed} ===")
        _seed_all(seed)

        diff_model = PixelDiffusionUNet(base_ch=96).to(device)
        fwd_model = SimpleForwardModel().to(device)
        diffusion = DiffusionV2(n_steps=300, device=device)

        log.info(f"  training PixelDiffusionUNet for {EPOCHS} epochs (seed={seed}) ...")
        t0 = time.time()
        train_s2s_v2(diff_model, fwd_model, diffusion, tr_loader, val_loader,
                     epochs=EPOCHS, device=device)
        train_secs = time.time() - t0
        log.info(f"  training done in {train_secs:.1f}s")

        ckpt_dir = os.path.join(CKPT_DIR, "v2", f"RADIal_seed{seed}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(diff_model.state_dict(), os.path.join(ckpt_dir, "diff_v2.pt"))
        torch.save(fwd_model.state_dict(), os.path.join(ckpt_dir, "fwd_v2.pt"))

        log.info(f"  evaluating on RADIal test ...")
        _seed_all(42)  # match canonical eval seed for fair comparison
        pred, gt = infer_diffusion(diff_model, diffusion, test_loader,
                                   n_steps=50, n_samples=4, eta=0.3, device=device)
        m = evaluate_full(pred, gt)
        m["train_seconds"] = float(train_secs)
        m["init"] = f"random (seed={seed})"
        results["per_seed"][f"seed_{seed}"] = {k: float(v) if hasattr(v, "__float__") else v
                                                for k, v in m.items()}
        log.info(f"  Seed {seed}: IoU={m['IoU']:.4f} F1={m['F1']:.4f} NLL={m['NLL']:.4f} "
                 f"Brier={m['Brier']:.4f} ECE={m['ECE']:.4f}")

        del diff_model, fwd_model, diffusion
        _clear_gpu(); gc.collect()

        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  [partial save -> {OUT}]")

    # Aggregate across 3 training seeds
    keys = ["IoU", "F1", "NLL", "Brier", "ECE", "Chamfer", "RMSE"]
    summary = {}
    for k in keys:
        vals = np.array([results["per_seed"][s][k] for s in results["per_seed"]
                          if k in results["per_seed"][s]])
        if len(vals) > 0:
            summary[f"{k}_mean"] = float(vals.mean())
            summary[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            summary[f"{k}_min"] = float(vals.min())
            summary[f"{k}_max"] = float(vals.max())
    results["summary_3seeds"] = summary

    log.info("\n=== Training-seed summary on RADIal (3 seeds) ===")
    for k in keys:
        if f"{k}_mean" in summary:
            log.info(f"  {k}: {summary[f'{k}_mean']:.4f} ± {summary[f'{k}_std']:.4f}  "
                     f"(min {summary[f'{k}_min']:.4f}, max {summary[f'{k}_max']:.4f})")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nFinal: {OUT}")


if __name__ == "__main__":
    main()
