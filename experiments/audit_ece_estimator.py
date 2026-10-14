"""
Re-estimate calibration error with defensible estimators.

The manuscript reports ECE 0.006 for S2S-Diff vs 0.020 for U-Net on ISAC
("3.3x better calibrated"). This is the only headline claim still standing
after the 2026-08 audits.

It is computed by compute_ece() (s2s_diff_v2.py:351) with 15 EQUAL-WIDTH bins
and the naive plugin estimator, over 5.59M cells whose predicted probabilities
pile up near zero. Kumar et al. (NeurIPS 2019, Prop. 3.3) prove the binned
estimate is always <= the true calibration error and that more bins reveal
more error; Vaicenavicius et al. (AISTATS 2019) identify exactly this regime
(few bins, many points) as prone to underestimation.

This script recomputes calibration four ways, for every method on every
dataset:

  E1_width   L1 ECE, equal-width bins      (the paper's estimator)
  E1_mass    L1 ECE, equal-MASS bins       (Kumar's recommendation)
  E2_plugin  L2 ECE, equal-mass, plugin
  E2_debias  L2 ECE, equal-mass, debiased  (Kumar Def. 5.2)

each swept over n_bins in {15, 30, 50, 100, 200, 500}, plus a bootstrap CI on
the S2S-Diff - U-Net gap at the paper's own setting.

Predictions are cached to results/preds/<dataset>__<method>.npz so any future
calibration analysis needs no re-inference.

Output: experiments/results/ece_reestimation.json
"""
import os, sys, json, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, UNetDirect, SimpleDS, _try_load, _clear_gpu,
)
from s2s_diff_v3 import (
    Pix2PixGenerator, infer_diffusion, infer_unet, infer_cvae, infer_pix2pix,
)
from models import CVAEBaseline
from torch.utils.data import DataLoader
from sanity_measurement_utilization import load_isac, load_v2i, load_radial

V2_CKPT = os.path.join(CKPT_DIR, "v2")
PRED_DIR = os.path.join(RESULTS_DIR, "preds")
SAVE = os.path.join(RESULTS_DIR, "ece_reestimation.json")
BIN_GRID = [15, 30, 50, 100, 200, 500]
PAPER_BINS = 15


# ----------------------------------------------------------------- estimators
def _bin_edges(p, n_bins, scheme):
    if scheme == "width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    e = np.quantile(p, q)
    e[0], e[-1] = 0.0, 1.0
    return np.unique(e)                      # drop collapsed quantiles


def calib(p, y, n_bins, scheme="width"):
    """Return (E1, E2_plugin, E2_debiased) for one binning."""
    edges = _bin_edges(p, n_bins, scheme)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    n = len(p)
    e1 = 0.0
    e2p = 0.0
    e2d = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        pbar = float(p[m].mean())
        ybar = float(y[m].mean())
        w = nb / n
        e1 += w * abs(pbar - ybar)
        e2p += w * (pbar - ybar) ** 2
        if nb > 1:                            # Kumar Def. 5.2 bias correction
            e2d += w * ((pbar - ybar) ** 2 - ybar * (1 - ybar) / (nb - 1))
        else:
            e2d += w * (pbar - ybar) ** 2
    return e1, float(np.sqrt(max(e2p, 0.0))), float(np.sqrt(max(e2d, 0.0)))


def bootstrap_gap(pA, yA, pB, yB, n_bins, scheme, n_boot=200, seed=0):
    """CI on E1(B) - E1(A), resampling frames."""
    rng = np.random.RandomState(seed)
    nf = pA.shape[0]
    out = []
    for _ in range(n_boot):
        i = rng.randint(0, nf, nf)
        a = calib(pA[i].ravel(), yA[i].ravel(), n_bins, scheme)[0]
        b = calib(pB[i].ravel(), yB[i].ravel(), n_bins, scheme)[0]
        out.append(b - a)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ----------------------------------------------------------------- prediction
def get_preds(ds_name, method, scenes, meas, ckpt_dir, device):
    """Occupancy-channel predictions [N,H,W] in [0,1], cached to npz."""
    os.makedirs(PRED_DIR, exist_ok=True)
    cache = os.path.join(PRED_DIR, f"{ds_name}__{method}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["pred"], z["gt"]

    loader = DataLoader(SimpleDS(scenes, meas), batch_size=8, shuffle=False, num_workers=0)
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    if method == "constant":
        gt = scenes[:, 0]
        pred = np.repeat(scenes[:, 0].mean(0, keepdims=True), len(scenes), 0)
        np.savez_compressed(cache, pred=pred.astype(np.float32), gt=gt.astype(np.float32))
        return pred, gt

    if method == "S2S-Diff":
        m = PixelDiffusionUNet(base_ch=96).to(device)
        if not _try_load(m, os.path.join(ckpt_dir, "diff_v2.pt"), "diff_v2"):
            return None, None
        p, g = infer_diffusion(m, DiffusionV2(n_steps=300, device=device), loader,
                               n_steps=50, n_samples=4, eta=0.3, device=device)
    elif method == "U-Net":
        m = UNetDirect(base_ch=96).to(device)
        if not _try_load(m, os.path.join(ckpt_dir, "unet_v2.pt"), "unet_v2"):
            return None, None
        p, g = infer_unet(m, loader, device=device)
    elif method == "CVAE":
        m = CVAEBaseline(meas_ch=4, scene_ch=3, latent_dim=64, base_ch=64).to(device)
        if not _try_load(m, os.path.join(ckpt_dir, "cvae_balanced.pt"), "cvae_balanced"):
            return None, None
        p, g = infer_cvae(m, loader, n_samples=4, device=device)
    elif method == "pix2pix":
        m = Pix2PixGenerator(in_ch=4, out_ch=3, base_ch=64).to(device)
        if not _try_load(m, os.path.join(ckpt_dir, "pix2pix.pt"), "pix2pix"):
            return None, None
        p, g = infer_pix2pix(m, loader, device=device)
    else:
        raise ValueError(method)

    pred = p[:, 0].cpu().numpy().astype(np.float32)
    gt = g[:, 0].cpu().numpy().astype(np.float32)
    np.savez_compressed(cache, pred=pred, gt=gt)
    del m; _clear_gpu(); gc.collect()
    return pred, gt


DATASETS = [("DeepSense_ISAC", load_isac), ("DeepSense_V2I", load_v2i), ("RADIal", load_radial)]
METHODS = ["S2S-Diff", "U-Net", "CVAE", "pix2pix", "constant"]


def main():
    device = torch.device("cuda")
    out = json.load(open(SAVE)) if os.path.exists(SAVE) else {}

    for ds_name, loader_fn in DATASETS:
        print(f"\n{'='*86}\n{ds_name}\n{'='*86}")
        scenes, meas, ckpt_name = loader_fn()
        ckpt_dir = os.path.join(V2_CKPT, ckpt_name)
        out.setdefault(ds_name, {})
        store = {}

        for meth in METHODS:
            pred, gt = get_preds(ds_name, meth, scenes, meas, ckpt_dir, device)
            if pred is None:
                print(f"  {meth}: checkpoint missing, skipped"); continue
            store[meth] = (pred, gt)
            p, y = pred.ravel(), gt.ravel()
            rec = {"n_cells": int(p.size), "mean_pred": float(p.mean()),
                   "base_rate": float(y.mean()), "sweep": {}}
            for nb in BIN_GRID:
                e1w = calib(p, y, nb, "width")[0]
                e1m, e2p, e2d = calib(p, y, nb, "mass")
                rec["sweep"][str(nb)] = {"E1_width": e1w, "E1_mass": e1m,
                                         "E2_plugin": e2p, "E2_debias": e2d}
            out[ds_name][meth] = rec
            s15 = rec["sweep"]["15"]; s500 = rec["sweep"]["500"]
            print(f"  {meth:<10} E1_width: {s15['E1_width']:.4f} (15 bins) -> "
                  f"{s500['E1_width']:.4f} (500)   "
                  f"E1_mass@15 {s15['E1_mass']:.4f}   E2_debias@15 {s15['E2_debias']:.4f}")

        # bootstrap the headline gap at the paper's own setting
        if "S2S-Diff" in store and "U-Net" in store:
            pa, ya = store["S2S-Diff"]; pb, yb = store["U-Net"]
            g = {}
            for scheme in ("width", "mass"):
                lo, hi = bootstrap_gap(pa, ya, pb, yb, PAPER_BINS, scheme)
                a = calib(pa.ravel(), ya.ravel(), PAPER_BINS, scheme)[0]
                b = calib(pb.ravel(), yb.ravel(), PAPER_BINS, scheme)[0]
                # one-sided reading in BOTH directions, explicitly named:
                #   lo > 0  -> S2S-Diff significantly BETTER calibrated
                #   hi < 0  -> S2S-Diff significantly WORSE calibrated
                g[scheme] = {"S2S": a, "UNet": b, "gap": b - a,
                             "ratio": (b / a) if a > 0 else None,
                             "ci95": [lo, hi],
                             "S2S_better_significant": bool(lo > 0),
                             "S2S_worse_significant": bool(hi < 0)}
                print(f"  gap ({scheme:<5} bins): U-Net {b:.4f} - S2S {a:.4f} = {b-a:+.4f}"
                      f"  ratio {g[scheme]['ratio']:.2f}x  CI[{lo:+.4f},{hi:+.4f}]"
                      f"  {'S2S BETTER' if lo > 0 else ('S2S WORSE' if hi < 0 else 'ns')}")
            out[ds_name]["_gap_vs_UNet"] = g

        del store; gc.collect()
        json.dump(out, open(SAVE, "w"), indent=2)

    print(f"\nSaved: {SAVE}")


if __name__ == "__main__":
    main()
