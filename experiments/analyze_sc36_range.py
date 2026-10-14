"""
Range-stratified re-analysis of the Sc.36 experiment.

WHY
---
The headline "no arm beats the constant predictor (IoU 0.3919)" turns out to be
dominated by near-field LiDAR ground returns. Breaking the constant's score down
by range ring:

    0-8 m    208 cells (5% of grid)   GT occ 65.8%   constant IoU 0.8895
    8-16 m   604 cells                GT occ 52.0%   constant IoU 0.5151
    16-24 m  992 cells                GT occ 17.8%   constant IoU 0.0154
    24-32 m  1424 cells               GT occ  9.0%   constant IoU 0.0000

A constant map scores 0.89 where the sensor always returns and 0.00 where the
scene actually varies. Aggregate IoU therefore measures ground-return prediction,
not reconstruction. This script recomputes every arm's IoU per range ring so the
comparison is made where the task is real.

This is also a gap in verify_new_dataset.py: the screen catches globally
prior-solvable benchmarks, but not a benchmark whose METRIC is dominated by a
trivially predictable subregion. Worth adding as screen S5.

Output: experiments/results/sc36_range_stratified.json
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from s2s_diff_v2 import DiffusionV2, _clear_gpu
from exp_sc36_hbeam import ArmModel, SC36, load_cache, ARMS

GRID, EXTENT = 64, 32.0
RINGS = [(0, 8), (8, 16), (16, 24), (24, 32), (8, 32), (16, 32), (0, 64)]


def ring_masks():
    yy, xx = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    c = (GRID - 1) / 2.0
    rad = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) * (2 * EXTENT / GRID)
    return {f"{lo}-{hi}m": (rad >= lo) & (rad < hi) for lo, hi in RINGS}


def iou_masked(pred, gt, mask):
    p = pred[:, mask] > 0.5
    g = gt[:, mask] > 0.5
    i = np.logical_and(p, g).sum()
    u = np.logical_or(p, g).sum()
    return float(i / u) if u else 0.0


@torch.no_grad()
def predict(ck, arm, te, device, batch=16):
    model = ArmModel(arm).to(device)
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    model.eval()
    diff = DiffusionV2(n_steps=300, device=device)
    dl = DataLoader(SC36(*te), batch_size=batch, shuffle=False, num_workers=0)
    out = []
    for scene, meas, h in dl:
        meas, h = meas.to(device), h.to(device)
        cond = model.make_cond(meas, h)
        s = [diff.ddim_sample(model, cond, n_steps=50, eta=0.3) for _ in range(4)]
        out.append(torch.stack(s).mean(0)[:, 0].cpu().numpy())
    del model; _clear_gpu()
    return np.concatenate(out)


def main():
    device = torch.device("cuda")
    res = json.load(open(os.path.join(RESULTS_DIR, "sc36_hbeam.json")))
    tr, te, meta = load_cache(1)
    step = len(te[0]) / 1000
    sel = (np.arange(1000) * step).astype(int)
    te = tuple(x[sel] for x in te)
    gt = te[0][:, 0]
    masks = ring_masks()

    const = np.repeat(tr[0][:, 0].mean(0, keepdims=True), len(gt), 0)
    out = {"rings": {k: int(v.sum()) for k, v in masks.items()},
           "gt_occupancy_rate": {k: float((gt[:, v] > 0.5).mean()) for k, v in masks.items()},
           "constant": {k: iou_masked(const, gt, v) for k, v in masks.items()},
           "arms": {}}
    print(f"{'ring':>10}{'cells':>7}{'GT occ':>9}{'constant':>10}", end="")
    for a in ARMS:
        print(f"{a:>13}", end="")
    print()

    per_arm = {}
    for arm in ARMS:
        seeds = res["arms"].get(arm, {})
        acc = {k: [] for k in masks}
        for sd, rec in seeds.items():
            ck = rec.get("checkpoint")
            if not ck or not os.path.exists(ck):
                print(f"  ({arm}/{sd}: checkpoint missing)"); continue
            p = predict(ck, arm, te, device)
            for k, v in masks.items():
                acc[k].append(iou_masked(p, gt, v))
        per_arm[arm] = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                        for k, v in acc.items() if v}
        out["arms"][arm] = {k: {"mean": m, "sd": s} for k, (m, s) in per_arm[arm].items()}

    for k, v in masks.items():
        print(f"{k:>10}{int(v.sum()):>7}{(gt[:, v] > 0.5).mean():>9.3f}"
              f"{out['constant'][k]:>10.4f}", end="")
        for a in ARMS:
            m = per_arm.get(a, {}).get(k)
            print(f"{m[0]:>13.4f}" if m else f"{'--':>13}", end="")
        print()

    print("\nbest arm vs constant, per ring:")
    for k in masks:
        best = max(((a, per_arm[a][k][0]) for a in ARMS if k in per_arm.get(a, {})),
                   key=lambda x: x[1], default=None)
        if best:
            d = best[1] - out["constant"][k]
            print(f"  {k:>10}  best={best[0]:<12} {best[1]:.4f}  constant {out['constant'][k]:.4f}"
                  f"  delta={d:+.4f}  {'ARM WINS' if d > 0 else 'constant wins'}")
    p = os.path.join(RESULTS_DIR, "sc36_range_stratified.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
