"""Timing + sanity probe before the full +-100 m beam experiment.

Two questions, both cheap:
  1. how fast is reconstruct() on this footprint -> honest ETA
  2. does the reconstruction actually VARY with the measurement? The +-100 m
     radar arm reported radar_util = +0.0005 IoU, i.e. essentially invariant.
     If recon is the same map for every frame it can carry no per-frame beam
     information, and the readout must land on the modal-beam floor. Measure it
     directly: per-frame spread of recon vs per-frame spread of the ground truth.
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/dong/Workspace/WritePaper/Submitted/SparseDiffusion/experiments")
from exp_beam_prediction import reconstruct
from exp_sc36_hbeam import load_cache

N = 48
tr, te, meta = load_cache(1, name="sc36_e100.npz")
te_s, te_m, te_h = te
sel = (np.arange(N) * (len(te_s) / N)).astype(int)
s, m, h = te_s[sel], te_m[sel], te_h[sel]
ck = "checkpoints/sc36/radar_seed42_e100.pt"

t0 = time.time()
r = reconstruct(ck, s, m, h, torch.device("cuda"))
dt = time.time() - t0
print(f"\n{N} frames in {dt:.1f} s -> {dt/N*1000:.0f} ms/frame")
for label, n in (("n_train=6000 + n_test=1500, 3 seeds", 7500*3),
                 ("full 17359 + 6945, 3 seeds", 24304*3)):
    print(f"  {label:38s} {dt/N*n/3600:.2f} h")

gt = s[:, 0]
rc = r[:, 0]
print(f"\nper-frame spread (sd across frames, mean over cells):")
print(f"  ground truth occupancy : {gt.std(axis=0).mean():.5f}")
print(f"  reconstruction         : {rc.std(axis=0).mean():.5f}")
print(f"  recon mean / gt mean   : {rc.mean():.4f} / {gt.mean():.4f}")
cs = [np.corrcoef(gt[i].ravel(), rc[i].ravel())[0, 1] for i in range(N)]
print(f"  per-frame corr(recon, gt): median {np.nanmedian(cs):+.4f}")
# is recon closer to its own frame than to another frame's?
own = np.mean([np.abs(rc[i]-gt[i]).mean() for i in range(N)])
oth = np.mean([np.abs(rc[i]-gt[(i+7) % N]).mean() for i in range(N)])
print(f"  MAE to own frame {own:.5f} vs shuffled frame {oth:.5f}  -> "
      f"{'frame-specific' if oth-own > 0.002 else 'NOT frame-specific'}")
