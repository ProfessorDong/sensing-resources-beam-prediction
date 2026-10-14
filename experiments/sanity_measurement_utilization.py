"""
Measurement-utilization ablation across ALL THREE datasets.

Generalizes sanity_y_zero.py (ISAC-only) to ISAC / V2I / RADIal. For each
dataset the canonical S2S-Diff v2 checkpoint is evaluated under four
conditioning settings:

    real_Y          the true measurement tensor
    zero_Y          Y := 0
    shuffled_Y      Y permuted across frames (destroys frame correspondence)
    spatial_perm_Y  Y permuted across cells within each frame (destroys
                    spatial structure, preserves the marginal histogram)

If the model genuinely conditions on Y, the three corrupted settings must
degrade. If the reconstruction is carried by the learned scene prior, all
four settings agree and the benchmark is prior-solvable.

This is the control experiment for the "calibrated but blind" thesis: the
ISAC result alone is confounded with a possible broken-conditioning bug.
Showing the SAME architecture and code path does use Y on another dataset
proves prior-domination is a property of the benchmark, not the model.

Inference protocol matches s2s_diff_v3.run_on_dataset exactly
(DDIM, n_steps=50, n_samples=4, eta=0.3, base_ch=96, DiffusionV2 n_steps=300).

Output: experiments/results/measurement_utilization.json
"""
import os, sys, json, time, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, SimpleDS, _try_load, _clear_gpu,
)
from s2s_diff_v3 import evaluate_full, infer_diffusion
from torch.utils.data import DataLoader

V2_CKPT = os.path.join(CKPT_DIR, "v2")
SAVE = os.path.join(RESULTS_DIR, "measurement_utilization.json")
DS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

# batch_size=8 reproduces the published sanity_y_zero.json ISAC numbers,
# which serves as a correctness check on this port.
BATCH = 8
SEED = 42


# ---------------------------------------------------------------- loaders
def load_isac():
    """Sc.42+43 -> 44. Test split = Scenario 44."""
    from train_all_real import preprocess_isac_scenario
    sc_dir = os.path.join(DS_ROOT, "deepsense6g", "Scenario44")
    scenes, meas = preprocess_isac_scenario(sc_dir, 44, max_frames=1500)
    return scenes, meas, "DeepSense_ISAC__Sc_42_43_44_"


def load_v2i():
    """Sc.31 -> 33. Test split = Scenario 33."""
    from preprocess_deepsense import preprocess_scenario as preprocess_v2i
    ts_dir = os.path.join(DS_ROOT, "deepsense6g", "Scenario33")
    scenes, meas, _beam = preprocess_v2i(ts_dir)
    return scenes, meas, "DeepSense_V2I__Sc_31_33_"


def load_radial():
    """80/20 split; test = last 20% (matches s2s_diff_v3)."""
    import glob
    from train_all_real import preprocess_radial
    base = os.path.join(DS_ROOT, "radial")
    seq_dirs = sorted(glob.glob(os.path.join(base, "extracted", "RECORD*")))
    labels_csv = os.path.join(base, "labels_CVPR.csv")
    config_path = os.path.join(base, "miscellaneous", "RADAR_CONFIG.m")
    scenes, meas = preprocess_radial(seq_dirs[0], labels_csv, config_path,
                                     max_frames=500)
    n_train = int(0.8 * len(scenes))
    return scenes[n_train:], meas[n_train:], "RADIal"


DATASETS = [
    ("DeepSense_ISAC", load_isac),
    ("DeepSense_V2I",  load_v2i),
    ("RADIal",         load_radial),
]


# ------------------------------------------------------------ corruptions
def make_variants(meas):
    """Return {setting_name: measurement_array} for the four conditions."""
    variants = {"real_Y": meas}

    variants["zero_Y"] = np.zeros_like(meas, dtype=np.float32)

    perm = np.random.RandomState(SEED).permutation(len(meas))
    variants["shuffled_Y"] = meas[perm]

    spat = meas.copy()
    rng = np.random.RandomState(123)
    for i in range(len(spat)):
        H, W = spat[i].shape[-2:]
        idx = rng.permutation(H * W)
        flat = spat[i].reshape(spat[i].shape[0], -1)
        spat[i] = flat[:, idx].reshape(spat[i].shape[0], H, W)
    variants["spatial_perm_Y"] = spat

    return variants


# ------------------------------------------------------------------ eval
def run_dataset(name, loader_fn, device):
    print(f"\n{'='*70}\nDATASET: {name}\n{'='*70}")
    scenes, meas, ckpt_name = loader_fn()
    print(f"  test frames: {len(scenes)}   meas shape: {meas.shape}")

    ckpt_dir = os.path.join(V2_CKPT, ckpt_name)
    model = PixelDiffusionUNet(base_ch=96).to(device)
    if not _try_load(model, os.path.join(ckpt_dir, "diff_v2.pt"), "diff_v2"):
        print(f"  !! checkpoint missing in {ckpt_dir}, skipping")
        return None
    model.eval()
    diffusion = DiffusionV2(n_steps=300, device=device)

    out = {}
    for setting, m in make_variants(meas).items():
        loader = DataLoader(SimpleDS(scenes, m), batch_size=BATCH,
                            shuffle=False, num_workers=0)
        t = time.time()
        np.random.seed(SEED); torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        pred, gt = infer_diffusion(model, diffusion, loader,
                                   n_steps=50, n_samples=4, eta=0.3,
                                   device=device)
        mt = evaluate_full(pred, gt)
        out[setting] = {**{k: float(v) for k, v in mt.items()},
                        "elapsed": float(time.time() - t)}
        print(f"  [{setting:>14}]  IoU={mt['IoU']:.4f}  ECE={mt['ECE']:.4f}  "
              f"NLL={mt['NLL']:.4f}  Brier={mt['Brier']:.6f}")

    # summary: drop of each corrupted setting vs real Y
    real = out["real_Y"]
    summary = {}
    for setting in ("zero_Y", "shuffled_Y", "spatial_perm_Y"):
        for metric in ("IoU", "F1", "ECE", "NLL", "Brier"):
            summary[f"{metric}_drop_{setting}"] = float(
                real[metric] - out[setting][metric])
    drops = [summary[f"IoU_drop_{s}"]
             for s in ("zero_Y", "shuffled_Y", "spatial_perm_Y")]
    summary["max_IoU_drop"] = float(max(drops))
    summary["max_IoU_drop_rel"] = float(max(drops) / real["IoU"]) if real["IoU"] else 0.0
    summary["uses_Y"] = bool(max(drops) > 0.05)
    out["summary"] = summary

    print(f"  --> max IoU drop = {summary['max_IoU_drop']:+.4f} "
          f"({100*summary['max_IoU_drop_rel']:+.1f}% rel)   "
          f"uses_Y = {summary['uses_Y']}")

    del model, diffusion
    _clear_gpu(); gc.collect()
    return out


def main():
    device = torch.device("cuda")
    results = {}
    for name, fn in DATASETS:
        try:
            r = run_dataset(name, fn, device)
            if r is not None:
                results[name] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {name} failed: {e}")
            results[name] = {"error": str(e)}

    print(f"\n{'='*70}\nMEASUREMENT UTILIZATION SUMMARY\n{'='*70}")
    print(f"{'dataset':<18}{'IoU(real)':>11}{'IoU(Y=0)':>11}"
          f"{'max drop':>11}{'rel':>9}  uses_Y")
    for name, r in results.items():
        if "summary" not in r:
            continue
        s = r["summary"]
        print(f"{name:<18}{r['real_Y']['IoU']:>11.4f}{r['zero_Y']['IoU']:>11.4f}"
              f"{s['max_IoU_drop']:>+11.4f}{100*s['max_IoU_drop_rel']:>8.1f}%"
              f"  {s['uses_Y']}")

    os.makedirs(os.path.dirname(SAVE), exist_ok=True)
    with open(SAVE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {SAVE}")


if __name__ == "__main__":
    main()
