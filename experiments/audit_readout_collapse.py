"""Is the "good" gt_scene readout learning, or emitting a constant class?

The Task A floor is the share of the TRAIN-modal beam in the test block (0.0631).
But train and test are different stretches of road, so the test-modal beam is a
DIFFERENT class with a much larger share (0.1876). A predictor that emits any single
constant class can therefore score up to 0.1876 while carrying zero information --
and gt_scene's best readout seed scored exactly 0.1876. Check directly: what is the
distribution of PREDICTED classes?
"""
import os, sys, json, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_beam_prediction import train_readout
from exp_sc36_hbeam import load_cache

N_CLASS, SEEDS, EPOCHS = 256, (0, 1, 2), 30

dev = torch.device("cuda")
tr, te, _ = load_cache(1, name="sc36_e100.npz")
btr = tr[2].reshape(len(tr[2]),-1).argmax(1); bte = te[2].reshape(len(te[2]),-1).argmax(1)
ytr = torch.from_numpy(btr).long(); yte = torch.from_numpy(bte).long()
tmb, tms = collections.Counter(bte.tolist()).most_common(1)[0]
print(f"test-modal class {tmb}, share {tms/len(bte):.4f}   train-modal floor "
      f"{np.mean(bte==collections.Counter(btr.tolist()).most_common(1)[0][0]):.4f}\n")

recon = np.load(os.path.join(RESULTS_DIR, "recon_cache_seed0_e100.npz"))
inputs = {"gt_scene": (tr[0], te[0]), "radar": (tr[1], te[1]),
          "recon": (recon["tr"], recon["te"])}

# The label distribution is the reference every predicted distribution is compared against.
lab_cnt = collections.Counter(bte.tolist())
out = {"n_test": int(len(bte)), "n_class": N_CLASS, "epochs": EPOCHS,
       "test_modal_class": int(tmb), "bound_const": tms / len(bte),
       "train_modal_class": int(collections.Counter(btr.tolist()).most_common(1)[0][0]),
       "floor_train_modal": float(np.mean(bte == collections.Counter(btr.tolist()).most_common(1)[0][0])),
       "label_distribution": {"counts_sorted_desc": [c for _, c in lab_cnt.most_common()],
                              "n_distinct": len(lab_cnt),
                              "entropy_nats": float(-sum((v/len(bte))*np.log(v/len(bte))
                                                         for v in lab_cnt.values()))},
       "cells": {}}

for name, (Xtr_np, Xte_np) in inputs.items():
    Xtr = torch.from_numpy(np.ascontiguousarray(Xtr_np)).float()
    Xte = torch.from_numpy(np.ascontiguousarray(Xte_np)).float()
    for seed in SEEDS:
        m, val, lg = train_readout(Xtr, ytr, Xte, yte, N_CLASS, seed, EPOCHS, dev, False,
                                   return_logits=True)
        pr = lg.argmax(1).numpy()
        cnt = collections.Counter(pr.tolist())
        top, topn = cnt.most_common(1)[0]
        H = -sum((v/len(pr))*np.log(v/len(pr)) for v in cnt.values())
        print(f"  {name:<9} seed{seed}  top1={m['top1']:.4f}  "
              f"distinct_preds={len(cnt):3d}  modal_pred={top:3d} "
              f"({topn/len(pr):.1%})  H={H:.3f} nats  "
              f"{'*** == test-modal class' if top == tmb else ''}", flush=True)
        out["cells"][f"{name}/seed{seed}"] = {
            "input": name, "seed": seed, "top1": float(m["top1"]),
            "top5": float(m.get("top5", float("nan"))),
            "n_distinct_predicted": len(cnt),
            "modal_predicted_class": int(top),
            "modal_predicted_share": topn / len(pr),
            "modal_pred_is_test_modal": bool(top == tmb),
            "pred_entropy_nats": float(H),
            # sorted descending: lets a cumulative-coverage curve be drawn without
            # storing 6,945 per-frame predictions
            "pred_counts_sorted_desc": [c for _, c in cnt.most_common()]}

p = os.path.join(RESULTS_DIR, "readout_collapse.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nSaved: {p}")
print(f"label distribution: {out['label_distribution']['n_distinct']} distinct classes, "
      f"H={out['label_distribution']['entropy_nats']:.3f} nats")
