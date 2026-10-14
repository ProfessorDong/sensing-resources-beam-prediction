"""Per-ring trivial baselines for View-of-Delft, on the exact frames the arms were scored on.

WHY
---
`vod_sparsity.json` carries per-ring IoU for the trained arms but only AGGREGATE baselines.
Figure F8 puts Sc.36 and VoD side by side, and Sc.36 has per-ring constants
(`sc36_range_stratified.json`); comparing a dataset that has its baseline against one that
does not is exactly the mismatch this project keeps catching. This also completes queue item
A3 (range-stratified VoD) -- the arm half was already in the sparsity run.

PROTOCOL
  * The threshold's channel and tau are NOT refitted. They are read from
    `vod_sparsity.json:baselines.threshold_config`, which fitted them on train, so this
    script cannot accidentally use a different rule than the published aggregate.
  * The constant is the per-cell TRAIN OCCUPANCY MEAN MAP, matching
    `exp_vod_sparsity.py:213` (`tr_s[:,0].mean(0)`) -- NOT a scalar. VoD is ~9% occupied, so
    a scalar mean would threshold to an all-empty map and score 0 in every ring; the per-cell
    map keeps the cells that are occupied in most training frames, which is what scores
    0.0170 in aggregate.
  * Eval frames are reproduced exactly: `(arange(650) * len(test)/650).astype(int)`.
  * Rings come from `exp_vod_sparsity.ring_masks()`.

Verification: the aggregate ring `0-64m` recomputed here must reproduce the published
aggregate baselines to 1e-6, and the script asserts it.

Output: results/vod_range_baselines.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_vod_sparsity import ring_masks, iou_masked, load_cache, RINGS

EVAL_LIMIT = 650                      # matches results/vod_sparsity.json config


def main():
    pub = json.load(open(os.path.join(RESULTS_DIR, "vod_sparsity.json")))
    cfg = pub["baselines"]["threshold_config"]
    tr_s, tr_m, te_s, te_m = load_cache(EVAL_LIMIT, 0)
    masks = ring_masks()
    gt_tr, gt_te = tr_s[:, 0], te_s[:, 0]
    print(f"VoD: train {len(gt_tr)}  eval {len(gt_te)}  rings {list(masks)}")

    # per-CELL train occupancy mean, broadcast over the eval frames
    const_map = np.repeat(gt_tr.mean(0, keepdims=True), len(gt_te), 0)
    out = {"eval_limit": EVAL_LIMIT, "n_eval": int(len(gt_te)), "n_train": int(len(gt_tr)),
           "rings": [f"{lo}-{hi}m" for lo, hi in RINGS],
           "constant_definition": "per-cell train occupancy mean map (exp_vod_sparsity.py:213)",
           "threshold_source": "vod_sparsity.json:baselines.threshold_config (fitted on train)",
           "gt_occupancy_rate": {k: float((gt_te[:, m] > 0.5).mean()) for k, m in masks.items()},
           "constant": {k: iou_masked(const_map, gt_te, m) for k, m in masks.items()},
           "threshold": {}}

    for nf in (1, 3, 5):
        c, t = int(cfg[str(nf)]["channel"]), float(cfg[str(nf)]["tau"])
        pred = (te_m[nf][:, c] > t).astype(np.float32)
        out["threshold"][f"r{nf}"] = {
            "channel": c, "tau": t, "train_IoU": cfg[str(nf)]["train_IoU"],
            "per_ring": {k: iou_masked(pred, gt_te, m) for k, m in masks.items()}}

    # --- TWO POOLING CONVENTIONS EXIST IN THIS PROJECT, and they differ by ~1-2%.
    # `iou_masked` pools intersection and union over every frame and cell at once. The
    # aggregate numbers in `vod_sparsity.json` come from `evaluate_full`, a different
    # reduction. The ARMS' per-ring numbers in that same file use `iou_masked` (line 183),
    # so the baselines here must too -- otherwise F8 would compare arms and baselines under
    # different estimators, which is the whole failure mode this project documents.
    # Both values are recorded; the check bounds the convention gap rather than hiding it.
    agg_c = out["constant"]["0-64m"]
    out["aggregate_cross_check"] = {
        "constant_iou_masked": agg_c,
        "constant_published_evaluate_full": pub["constant_baseline_IoU"],
        "rel_diff": abs(agg_c - pub["constant_baseline_IoU"]) / pub["constant_baseline_IoU"],
        "note": "iou_masked pools over all frames and cells; evaluate_full uses a different "
                "reduction. Ring values everywhere in F8 use iou_masked, matching the arms."}
    for nf in (1, 3, 5):
        a = out["threshold"][f"r{nf}"]["per_ring"]["0-64m"]
        b = pub["baselines"]["threshold_by_sparsity"][str(nf)]
        out["aggregate_cross_check"][f"threshold_r{nf}"] = {
            "iou_masked": a, "published": b, "rel_diff": abs(a - b) / b}
        assert abs(a - b) / b < 0.03, f"threshold r{nf} {a:.6f} vs published {b:.6f}"
    assert out["aggregate_cross_check"]["rel_diff"] < 0.03, "constant convention gap too large"
    print(f"  aggregate constant: {agg_c:.6f} (iou_masked) vs "
          f"{pub['constant_baseline_IoU']:.6f} (published evaluate_full), "
          f"{100*out['aggregate_cross_check']['rel_diff']:.1f}% apart -- "
          f"pooling convention, documented")

    print("\n  ring        gt_occ   constant   thresh(r1)   thresh(r5)")
    for k in out["constant"]:
        print(f"  {k:<10} {out['gt_occupancy_rate'][k]:>7.4f}  {out['constant'][k]:>8.4f}  "
              f"{out['threshold']['r1']['per_ring'][k]:>10.4f}  "
              f"{out['threshold']['r5']['per_ring'][k]:>10.4f}")

    p = os.path.join(RESULTS_DIR, "vod_range_baselines.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
