"""Read-only checks supporting PROJECT_REVIEW_2026-09-25.md.

Run with the project's w26 Python. Reads existing predictions and two small
legacy radar subsets; performs no training, inference, or artifact overwrites.
Prints JSON to stdout. AST extraction avoids training-module import side effects.
"""
import ast
import contextlib
import hashlib
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)


def functions_from(relative_path, names, namespace):
    tree = ast.parse((ROOT / relative_path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative_path, "exec"), namespace)
    return namespace


def sha1(array):
    return hashlib.sha1(array.tobytes()).hexdigest()[:16]


cal = functions_from("experiments/audit_ece_estimator.py", {"_bin_edges", "calib"}, {"np": np})
out = {"review_date": "2026-09-25", "cached_predictions": {}}
for dataset in ("DeepSense_ISAC", "DeepSense_V2I", "RADIal"):
    records = {}
    for method in ("S2S-Diff", "U-Net", "pix2pix", "constant"):
        with np.load(ROOT / "experiments/results/preds" / f"{dataset}__{method}.npz") as z:
            p, g = z["pred"], z["gt"]
        pb, gb = p > .5, g > .5
        inter, union = (pb & gb).sum((1, 2)), (pb | gb).sum((1, 2))
        rec = {"n_frames": len(g), "gt_sha1": sha1(g), "occupancy_rate": float(g.mean()),
               "mean_prediction": float(p.mean()),
               "macro_IoU": float(np.mean(inter / (union + 1e-8))),
               "micro_IoU": float(inter.sum() / max(union.sum(), 1)),
               "Brier": float(((p - g) ** 2).mean())}
        if method == "constant":
            rec["equals_test_mean"] = bool(np.allclose(p, np.broadcast_to(g.mean(0), p.shape)))
            assert rec["equals_test_mean"]
        else:
            rec["ECE_width15_inclusive"] = cal["calib"](p.ravel(), g.ravel(), 15, "width")[0]
            rec["ECE_mass15"] = cal["calib"](p.ravel(), g.ravel(), 15, "mass")[0]
        records[method] = rec
    out["cached_predictions"][dataset] = records

legacy = functions_from("experiments/train_all_real.py", {"preprocess_radial"},
                        {"np": np, "torch": torch, "F": F, "os": os,
                         "log": logging.getLogger("read_only_review")})
base = ROOT / "datasets/radial/extracted"
current_first = sorted(base.glob("RECORD*"))[0]
historical = base / "RECORD@2020-11-21_12.03.58"
cached_gt = np.load(ROOT / "experiments/results/preds/RADIal__S2S-Diff.npz")["gt"]
out["radial_sequence_checks"] = {}
for sequence in (current_first, historical):
    scenes, meas = legacy["preprocess_radial"](str(sequence), "", "", max_frames=500)
    rec = {"scene_sha1": sha1(scenes), "meas_sha1": sha1(meas),
           "test_occupancy_sha1": sha1(scenes[400:, 0]),
           "matches_cached_gt": bool(np.array_equal(scenes[400:, 0], cached_gt))}
    if sequence == historical:
        assert rec["scene_sha1"] == "67e26cfe7061276b"
        assert rec["meas_sha1"] == "3dd7bc53348de0d9"
        p, g = meas[400:, 0] > .51, scenes[400:, 0] > .5
        inter, union = (p & g).sum((1, 2)), (p | g).sum((1, 2))
        rec["threshold_micro_IoU"] = float(inter.sum() / union.sum())
        rec["threshold_macro_IoU"] = float(np.mean(inter / (union + 1e-8)))
    out["radial_sequence_checks"][sequence.name] = rec
assert out["radial_sequence_checks"][current_first.name]["matches_cached_gt"]
assert not out["radial_sequence_checks"][historical.name]["matches_cached_gt"]

screen = functions_from("experiments/verify_new_dataset.py", {"iou", "metrics", "screen"},
                        {"np": np, "DIVERSITY_WARN": .70})
g = np.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]], dtype=np.float32)
s, m = g[:, None], np.full_like(g[:, None], .3)
with contextlib.redirect_stdout(io.StringIO()):
    toy = screen["screen"]("constant_input_toy", s, m, s, m, save=False)
out["diagnostic_counterexamples"] = {
    "constant_input_reported_as_exact_threshold": toy["S2_channels_exactly_threshold"],
    "ECE_all_wrong_probability_one": screen["metrics"](np.ones_like(g), np.zeros_like(g))["ECE_equalwidth15"],
    "expected_ECE": 1.0,
}

tables = scipy.io.loadmat(ROOT / "datasets/deepsense6g/Scenario36/scenario36.mat")
powers = np.stack([tables[f"unit1_pwr{i}"] for i in (1, 2, 3, 4)], axis=1).reshape(-1, 256)
labels = tables["unit1_overall-beam"].ravel()
out["beam_label_check"] = {"n": len(labels), "argmax_agreement": float(np.mean(powers.argmax(1) == labels))}
assert out["beam_label_check"]["argmax_agreement"] == 1.0

sc36 = json.loads((ROOT / "experiments/results/sc36_hbeam.json").read_text())
out["sc36_summary"] = sc36["summary"]
out["sc36_contrasts"] = sc36["contrasts"]
out["sc36_constant_IoU"] = sc36["constant_baseline_IoU"]
for arm, records in sc36["arms"].items():
    mean = np.mean([r["eval"]["real"]["IoU"] for r in records.values()])
    assert np.isclose(mean, sc36["summary"][arm]["IoU_mean"])
out["source_sha256"] = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in ("experiments/preprocess_deepsense_sc36.py", "experiments/exp_sc36_hbeam.py",
                 "experiments/verify_new_dataset.py", "experiments/audit_ece_estimator.py",
                 "experiments/audit_ece_constant_fix.py", "experiments/s2s_diff_v3.py",
                 "experiments/train_all_real.py", "experiments/results/sc36_hbeam.json",
                 "experiments/results/ece_reestimation.json")
}
print(json.dumps(out, indent=2))
