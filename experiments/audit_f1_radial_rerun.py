"""Re-evaluate RADIal for Figure 1, because its preprocessing no longer reproduces.

WHAT WENT WRONG
---------------
`CLAUDE.md` records sha1 fingerprints for the RADIal tensors produced by the legacy
`train_all_real.preprocess_radial` loader: scene 67e26cfe7061276b, meas 3dd7bc53348de0d9.
Re-running that loader on 2026-09-28 gives scene 6a6436ff5172541d, meas a19096bfa037947b.
The tensors are not the same, so every RADIal number computed in August (model IoU 0.3802 in
`measurement_utilization.json`, threshold IoU 0.4931 in `trivial_baselines.json`) describes
data that the pipeline no longer produces. Those two numbers are mutually consistent with each
other and inconsistent with anything computed today.

DeepSense ISAC and V2I are unaffected: their baselines recompute to the published values
exactly (constant 0.8613, 0.1838; threshold 0.1912, 0.2202), so only the RADIal loader drifted.

WHAT THIS SCRIPT DOES
---------------------
Recomputes the whole RADIal triple on TODAY's tensors, so the three bars share one dataset and
one estimator: the canonical `diff_v2.pt` checkpoint under the same inference settings the
utilization battery used (50 DDIM steps, 4 samples, eta 0.3), and both trivial baselines with
hyperparameters refitted on train. IoU is `s2s_diff_v3.compute_iou` throughout, the per-frame
estimator that every model number in Figure 1 already uses.

Output: results/f1_radial_rerun.json
"""
import os, sys, glob, json, hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import PixelDiffusionUNet, DiffusionV2
from s2s_diff_v3 import evaluate_full, infer_diffusion, compute_iou
from sanity_measurement_utilization import SimpleDS, _try_load

DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
SEED = 42


def sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def iou_perframe(p, g):
    return compute_iou(torch.from_numpy(np.asarray(p, dtype=np.float32)),
                       torch.from_numpy(np.asarray(g, dtype=np.float32)))


def main():
    from train_all_real import preprocess_radial
    seq = sorted(glob.glob(os.path.join(DS, "radial", "extracted", "RECORD*")))
    s, m = preprocess_radial(seq[0], os.path.join(DS, "radial", "labels_CVPR.csv"),
                             os.path.join(DS, "radial", "miscellaneous", "RADAR_CONFIG.m"),
                             max_frames=500)
    fp = {"sequence": os.path.basename(seq[0]), "scene_sha1": sha(s), "meas_sha1": sha(m),
          "scene_sha1_recorded_2026_08": "67e26cfe7061276b",
          "meas_sha1_recorded_2026_08": "3dd7bc53348de0d9"}
    fp["reproduces_august"] = (fp["scene_sha1"] == fp["scene_sha1_recorded_2026_08"]
                               and fp["meas_sha1"] == fp["meas_sha1_recorded_2026_08"])
    print(f"sequence {fp['sequence']}")
    print(f"  scene sha1 {fp['scene_sha1']} (Aug: {fp['scene_sha1_recorded_2026_08']})")
    print(f"  meas  sha1 {fp['meas_sha1']} (Aug: {fp['meas_sha1_recorded_2026_08']})")
    print(f"  reproduces August: {fp['reproduces_august']}")

    n = int(0.8 * len(s))
    tr_s, tr_m, te_s, te_m = s[:n], m[:n], s[n:], m[n:]

    # ---- model, same checkpoint and inference settings as the utilization battery
    device = torch.device("cuda")
    model = PixelDiffusionUNet(base_ch=96).to(device)
    ck = os.path.join(CKPT_DIR, "v2", "RADIal", "diff_v2.pt")
    assert _try_load(model, ck, "diff_v2"), f"missing {ck}"
    model.eval()
    diffusion = DiffusionV2(n_steps=300, device=device)
    loader = DataLoader(SimpleDS(te_s, te_m), batch_size=8, shuffle=False, num_workers=0)
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    pred, gt = infer_diffusion(model, diffusion, loader, n_steps=50, n_samples=4,
                               eta=0.3, device=device)
    mt = evaluate_full(pred, gt)
    print(f"\nmodel (diff_v2, per-frame IoU): {mt['IoU']:.4f}   "
          f"(August value on the old tensors: 0.3802)")

    # ---- baselines, hyperparameters refitted on train, per-frame IoU
    gt_tr, gt_te = tr_s[:, 0], te_s[:, 0]
    const = np.repeat(gt_tr.mean(0, keepdims=True), len(gt_te), 0)
    const_iou = iou_perframe(const, gt_te)
    best = (-1.0, None, None)
    for c in range(tr_m.shape[1]):
        for t in np.arange(0.02, 1.0, 0.01):
            v = iou_perframe((tr_m[:, c] > t).astype(np.float32), gt_tr)
            if v > best[0]:
                best = (v, int(c), round(float(t), 2))
    train_iou, ch, tau = best
    thr_iou = iou_perframe((te_m[:, ch] > tau).astype(np.float32), gt_te)
    print(f"constant  {const_iou:.4f}")
    print(f"threshold ch{ch} > {tau:g} (train {train_iou:.4f}) -> test {thr_iou:.4f}")
    best_base = max(const_iou, thr_iou)
    print(f"\nmargin model - best rule = {mt['IoU'] - best_base:+.4f}")

    out = {"fingerprint": fp, "n_train": int(n), "n_test": int(len(te_s)),
           "estimator": "s2s_diff_v3.compute_iou (per-frame IoU, then mean)",
           "inference": {"ckpt": ck, "ddim_steps": 50, "n_samples": 4, "eta": 0.3,
                         "seed": SEED},
           "model": {k: float(v) for k, v in mt.items()},
           "constant": {"IoU": const_iou, "rule": "per-cell train occupancy mean > 0.5"},
           "threshold": {"IoU": thr_iou, "channel": ch, "tau": tau,
                         "train_IoU": float(train_iou)},
           "best_baseline": float(best_base),
           "margin": float(mt["IoU"] - best_base),
           "august_values_now_superseded": {"model": 0.3802, "threshold": 0.4931,
                                            "constant": 0.0}}
    p = os.path.join(RESULTS_DIR, "f1_radial_rerun.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
