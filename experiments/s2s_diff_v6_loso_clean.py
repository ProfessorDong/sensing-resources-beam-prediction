"""
S2S-Diff v6: Leakage-free LOSO on ISAC.

The v5 protocol initializes every fold from the canonical Sc.42+43 ISAC
checkpoint. For the Sc.42 and Sc.43 holdouts this introduces leakage (the init
checkpoint already saw the held-out scenario). Here we redo those two folds
starting from the V2I checkpoint (no ISAC training data), then fine-tune on
the two non-holdout ISAC scenarios for 60 epochs and evaluate on the holdout.
The Sc.44 holdout fold from v5 is already leakage-free (the canonical Sc.42+43
init never saw Sc.44), so we re-use it.

Output: experiments/results/v6/loso_clean.json
"""
import os, sys, json, time, logging, gc
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
from s2s_diff_v5_xscenario import fine_tune_s2s

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

V2I_CKPT = os.path.join(CKPT_DIR, "v2", "DeepSense_V2I__Sc_31_33_")  # leakage-free init
V6_RESULTS = os.path.join(RESULTS_DIR, "v6")
os.makedirs(V6_RESULTS, exist_ok=True)
SAVE_PATH = os.path.join(V6_RESULTS, "loso_clean.json")
V5_PATH = os.path.join(RESULTS_DIR, "v5", "xscenario_results.json")

ISAC_SCENARIOS = [42, 43, 44]
LEAKAGE_FREE_FOLDS = [42, 43]  # need re-running; Sc.44 fold is already clean
RETRAIN_EPOCHS = 60


def _seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_fold_clean(holdout_sc, all_scenes, all_meas, device="cuda", epochs=60):
    log.info(f"\n--- Clean fold: holdout = Sc.{holdout_sc} (V2I-initialized) ---")
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

    # Init from V2I (no ISAC training data) -> guarantees leakage-free LOSO
    _try_load(diff_model, os.path.join(V2I_CKPT, "diff_v2.pt"), "diff_v2 (V2I init)")
    _try_load(fwd_model, os.path.join(V2I_CKPT, "fwd_v2.pt"), "fwd_v2 (V2I init)")

    log.info(f"   training {epochs} epochs from V2I init ...")
    t_train = time.time()
    fine_tune_s2s(diff_model, fwd_model, diffusion, train_loader, val_loader,
                  epochs=epochs, device=device)
    train_secs = time.time() - t_train
    log.info(f"   training done in {train_secs:.1f}s")

    fold_dir = os.path.join(V6_RESULTS, f"ckpt_holdout_{holdout_sc}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save(diff_model.state_dict(), os.path.join(fold_dir, "diff.pt"))
    torch.save(fwd_model.state_dict(), os.path.join(fold_dir, "fwd.pt"))

    log.info(f"   evaluating on Sc.{holdout_sc} test ...")
    _seed_all(42)
    pred, gt = infer_diffusion(diff_model, diffusion, test_loader,
                               n_steps=50, n_samples=4, eta=0.3, device=device)
    metrics = evaluate_full(pred, gt)
    metrics["train_seconds"] = float(train_secs)
    metrics["n_train"] = int(len(train_ds))
    metrics["n_test"] = int(len(test_ds))
    metrics["init"] = "V2I"
    metrics["epochs"] = int(epochs)
    log.info(f"   Sc.{holdout_sc} (clean): IoU={metrics['IoU']:.4f}  "
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

    # Pull the existing v5 Sc.44 fold (already leakage-free)
    results = {"folds": {}, "summary": {}, "protocol": "leakage-free LOSO"}
    if os.path.exists(V5_PATH):
        v5 = json.load(open(V5_PATH))
        sc44 = v5["folds"].get("holdout_44", {}).copy()
        sc44["init"] = "canonical Sc.42+43 (no Sc.44 leakage)"
        sc44["epochs"] = 30
        results["folds"]["holdout_44"] = sc44
        log.info(f"  Re-using v5 Sc.44 fold: IoU={sc44.get('IoU',0):.4f}  ECE={sc44.get('ECE',0):.4f}")

    for holdout in LEAKAGE_FREE_FOLDS:
        try:
            m = run_fold_clean(holdout, isac_scenes, isac_meas, device=device,
                               epochs=RETRAIN_EPOCHS)
            results["folds"][f"holdout_{holdout}"] = m
            with open(SAVE_PATH, "w") as f:
                json.dump(results, f, indent=2)
            log.info(f"   [partial save -> {SAVE_PATH}]")
        except Exception as e:
            log.exception(f"Fold holdout={holdout} failed: {e}")

    # Aggregate
    keys = ["IoU", "F1", "NLL", "Brier", "ECE", "Chamfer", "RMSE"]
    for k in keys:
        vals = np.array([results["folds"][f][k] for f in results["folds"]
                         if k in results["folds"][f]])
        if len(vals) > 0:
            results["summary"][f"{k}_mean"] = float(vals.mean())
            results["summary"][f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            results["summary"][f"{k}_min"] = float(vals.min())
            results["summary"][f"{k}_max"] = float(vals.max())

    log.info("\n=== Leakage-free LOSO summary ===")
    for k, v in results["summary"].items():
        log.info(f"  {k}: {v:.4f}")

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Final: {SAVE_PATH}")


if __name__ == "__main__":
    main()
