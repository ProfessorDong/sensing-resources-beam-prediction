"""
S2S-Diff v5: Cross-scenario robustness evaluation on ISAC.

Leave-one-scenario-out (LOSO) fine-tuning: for each holdout in {42, 43, 44},
fine-tune the existing diff_v2 checkpoint on the other two scenarios for
30 epochs (vs 150 for original training), then evaluate on the holdout.

Reports IoU/F1/NLL/Brier/ECE per fold and mean +/- std across folds.
Single seed (42) per fold for tractability; the multi-seed analysis (5 seeds)
is already in v4 for the canonical 42+43->44 split.

Output: experiments/results/v5/xscenario_results.json
"""
import os, sys, json, time, logging, math, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, SimpleForwardModel, SimpleDS,
    _try_load, _clear_gpu,
)
from s2s_diff_v3 import evaluate_full, infer_diffusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

V2_CKPT = os.path.join(CKPT_DIR, "v2", "DeepSense_ISAC__Sc_42_43_44_")
V5_RESULTS = os.path.join(RESULTS_DIR, "v5")
os.makedirs(V5_RESULTS, exist_ok=True)
SAVE_PATH = os.path.join(V5_RESULTS, "xscenario_results.json")

ISAC_SCENARIOS = [42, 43, 44]


def _seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def fine_tune_s2s(model, fwd_model, diffusion, train_loader, val_loader,
                  epochs=30, lr=1e-4, phys_weight=0.1, obs_weight=2.0,
                  device="cuda"):
    """Fine-tune diffusion + forward model from existing checkpoint."""
    params = list(model.parameters()) + list(fwd_model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float("inf")
    best_diff = None
    best_fwd = None

    for epoch in range(epochs):
        model.train(); fwd_model.train()
        epoch_loss = 0.0
        n_batches = 0
        for scene, meas in train_loader:
            scene, meas = scene.to(device), meas.to(device)
            B = scene.shape[0]
            t = torch.randint(0, diffusion.n_steps, (B,), device=device)
            noise = torch.randn_like(scene)
            x_t, _ = diffusion.q_sample(scene, t, noise)
            eps_pred = model(x_t, t, meas)
            loss_diff = F.mse_loss(eps_pred, noise)

            x0_pred = diffusion.predict_x0(x_t, t, eps_pred).clamp(0, 1)
            meas_pred = fwd_model(x0_pred.detach())
            meas_gt_proxy = fwd_model(scene.detach())
            loss_phys = F.mse_loss(meas_pred, meas_gt_proxy.detach())

            meas_from_gt = fwd_model(scene)
            loss_fwd = F.mse_loss(meas_from_gt, meas)

            occ_weight = 1.0 + (obs_weight - 1.0) * scene[:, 0:1]
            loss_weighted = (occ_weight * (eps_pred - noise).pow(2)).mean()

            pw = phys_weight * min(1.0, (epoch + 1) / 10.0)  # ramp over 10 fine-tune epochs
            loss = loss_diff + 0.3 * loss_weighted + pw * loss_phys + 0.1 * loss_fwd

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()

        model.eval()
        val_loss = 0.0; nv = 0
        with torch.no_grad():
            for scene, meas in val_loader:
                scene, meas = scene.to(device), meas.to(device)
                B = scene.shape[0]
                t = torch.randint(0, diffusion.n_steps, (B,), device=device)
                noise = torch.randn_like(scene)
                x_t, _ = diffusion.q_sample(scene, t, noise)
                eps_pred = model(x_t, t, meas)
                val_loss += F.mse_loss(eps_pred, noise).item()
                nv += 1
        val_loss = val_loss / max(nv, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_diff = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_fwd = {k: v.detach().cpu().clone() for k, v in fwd_model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(f"      epoch {epoch+1}/{epochs}: train={epoch_loss/max(n_batches,1):.4f}  val={val_loss:.4f}")

    if best_diff is not None:
        model.load_state_dict(best_diff)
        fwd_model.load_state_dict(best_fwd)
    return model, fwd_model


def run_fold(holdout_sc, all_scenes, all_meas, device="cuda", epochs=30):
    """Fold: holdout_sc is the test scenario; the other two are train."""
    log.info(f"\n--- Fold: holdout = Sc.{holdout_sc} ---")
    train_idxs = [i for i, sc in enumerate(ISAC_SCENARIOS) if sc != holdout_sc]
    test_idx = ISAC_SCENARIOS.index(holdout_sc)
    train_scenes = np.concatenate([all_scenes[i] for i in train_idxs], axis=0)
    train_meas = np.concatenate([all_meas[i] for i in train_idxs], axis=0)
    test_scenes = all_scenes[test_idx]
    test_meas = all_meas[test_idx]

    n_train = int(0.9 * len(train_scenes))
    train_ds = SimpleDS(train_scenes[:n_train], train_meas[:n_train])
    val_ds = SimpleDS(train_scenes[n_train:], train_meas[n_train:])
    test_ds = SimpleDS(test_scenes, test_meas)
    log.info(f"      train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    diff_model = PixelDiffusionUNet().to(device)
    fwd_model = SimpleForwardModel(scene_ch=3, meas_ch=4).to(device)
    diffusion = DiffusionV2(n_steps=300, device=device)

    # Start from canonical 42+43->44 checkpoints
    _try_load(diff_model, os.path.join(V2_CKPT, "diff_v2.pt"), "diff_v2 (init)")
    _try_load(fwd_model, os.path.join(V2_CKPT, "fwd_v2.pt"), "fwd_v2 (init)")

    log.info(f"   fine-tuning diff+fwd for {epochs} epochs ...")
    t_train = time.time()
    fine_tune_s2s(diff_model, fwd_model, diffusion, train_loader, val_loader,
                  epochs=epochs, device=device)
    train_secs = time.time() - t_train
    log.info(f"   training done in {train_secs:.1f}s")

    # save fold checkpoints
    fold_dir = os.path.join(V5_RESULTS, f"ckpt_holdout_{holdout_sc}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save(diff_model.state_dict(), os.path.join(fold_dir, "diff.pt"))
    torch.save(fwd_model.state_dict(), os.path.join(fold_dir, "fwd.pt"))

    # Evaluate
    log.info(f"   evaluating on Sc.{holdout_sc} test set ...")
    _seed_all(42)
    pred, gt = infer_diffusion(diff_model, diffusion, test_loader,
                               n_steps=50, n_samples=4, eta=0.3, device=device)
    metrics = evaluate_full(pred, gt)
    metrics["train_seconds"] = float(train_secs)
    metrics["n_train"] = int(len(train_ds))
    metrics["n_val"] = int(len(val_ds))
    metrics["n_test"] = int(len(test_ds))
    log.info(f"   Sc.{holdout_sc} held-out: IoU={metrics['IoU']:.4f}  "
             f"F1={metrics['F1']:.4f}  NLL={metrics['NLL']:.4f}  "
             f"Brier={metrics['Brier']:.4f}  ECE={metrics['ECE']:.4f}")

    del diff_model, fwd_model
    _clear_gpu(); gc.collect()
    return metrics


def main():
    device = torch.device("cuda")
    _seed_all(42)

    log.info("Loading ISAC data ...")
    from train_all_real import preprocess_isac_scenario
    ds_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    isac_scenes, isac_meas = [], []
    for sc in ISAC_SCENARIOS:
        sc_dir = os.path.join(ds_root, "deepsense6g", f"Scenario{sc}")
        s, m = preprocess_isac_scenario(sc_dir, sc, max_frames=1500)
        isac_scenes.append(s); isac_meas.append(m)
    log.info(f"  Sc.42={len(isac_scenes[0])}  Sc.43={len(isac_scenes[1])}  Sc.44={len(isac_scenes[2])}")

    results = {"folds": {}, "summary": {}}

    # Fold 0 is the canonical 42+43 -> 44, results already in v3 JSON.
    # We re-run it here for consistency with same evaluation pipeline.
    for holdout in ISAC_SCENARIOS:
        try:
            m = run_fold(holdout, isac_scenes, isac_meas, device=device, epochs=30)
            results["folds"][f"holdout_{holdout}"] = m
            with open(SAVE_PATH, "w") as f:
                json.dump(results, f, indent=2)
            log.info(f"   [partial save -> {SAVE_PATH}]")
        except Exception as e:
            log.exception(f"Fold holdout={holdout} failed: {e}")

    # Aggregate
    if results["folds"]:
        keys = ["IoU", "F1", "NLL", "Brier", "ECE", "Chamfer", "RMSE"]
        for k in keys:
            vals = np.array([results["folds"][f][k] for f in results["folds"]
                             if k in results["folds"][f]])
            if len(vals) > 0:
                results["summary"][f"{k}_mean"] = float(vals.mean())
                results["summary"][f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                results["summary"][f"{k}_min"] = float(vals.min())
                results["summary"][f"{k}_max"] = float(vals.max())

    log.info("\n=== Cross-scenario summary ===")
    for k, v in results["summary"].items():
        log.info(f"  {k}: {v:.4f}")

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Final results: {SAVE_PATH}")


if __name__ == "__main__":
    main()
