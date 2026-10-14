"""Is the image readout able to see WHERE something is? Instrument check for every
image-input beam-prediction result in this project.

WHY
---
`exp_beam_prediction.ImgReadout` ends in `nn.AdaptiveAvgPool2d(1)`, i.e. global average
pooling, which is translation-invariant by construction. Best-beam index is a function of the
target's BEARING, so a position-blind readout cannot solve it from any image, no matter what
the image contains. If that is what has been happening, then the image-input nulls in this
project measure the readout rather than the representation, and that includes the claim that
bird's-eye occupancy cannot expose the beam.

THE CHECK
---------
Rasterize unit 2's known position into a 64x64 map with the same shape and channel count as
the radar input, so the only thing that changes is the content. The beam is recoverable from
that image by construction: unit 1 is stationary, so position determines bearing determines
beam. Feed it to

  ImgReadout          the architecture actually used for every radar / occupancy / recon arm
  ImgReadoutSpatial   identical except the global average pool is replaced by a flatten, so
                      the classifier can see position

and compare both against the no-information bound and against the VecReadout result on the
same positions as raw coordinates.

Reading:
  blob -> ImgReadout FAILS, blob -> ImgReadoutSpatial CLEARS  => the pooling is the defect and
      every image-input beam null in this project is instrument-limited, not a finding.
  both clear                                                  => the image path is sound and
      the radar null stands.

Output: results/readout_spatial_audit.json
"""
import os, sys, json, collections
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_beam_prediction import ImgReadout, train_readout
import exp_beam_prediction as EBP

GRID = 64
TRAIN_FRAC, GAP_FRAC = 0.70, 0.02


class ImgReadoutSpatial(nn.Module):
    """ImgReadout with the global average pool replaced by a flatten of the 4x4 feature map.

    Same depth, same channel widths, same nonlinearities. The only difference is that this one
    can represent WHERE a feature is.
    """
    def __init__(self, in_ch, n_class, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base*2, 3, 2, 1), nn.GroupNorm(8, base*2), nn.SiLU(),
            nn.Conv2d(base*2, base*4, 3, 2, 1), nn.GroupNorm(8, base*4), nn.SiLU(),
            nn.Conv2d(base*4, base*4, 3, 2, 1), nn.GroupNorm(8, base*4), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(base*4*4*4, 256), nn.SiLU(), nn.Linear(256, n_class))
    def forward(self, x): return self.net(x)


def rasterize(gps, extent, sigma=1.6, n_ch=2):
    """unit2 position -> (n_ch, GRID, GRID) Gaussian blob, forward-North / right-East."""
    north, east = gps[:, 0], gps[:, 1]
    gx = (east + extent) / (2 * extent) * (GRID - 1)
    gy = (north + extent) / (2 * extent) * (GRID - 1)
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(np.float32)
    out = np.zeros((len(gps), n_ch, GRID, GRID), np.float32)
    for i, (cx, cy) in enumerate(zip(gx, gy)):
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        b = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        for c in range(n_ch):
            out[i, c] = b
    return out


def main():
    dev = torch.device("cuda")
    out = {}
    for sc in ("Scenario31", "Scenario33"):
        z = np.load(os.path.join(RESULTS_DIR, f"cache_{sc}_full.npz"), allow_pickle=True)
        gps, meas, beams = z["gps"], z["meas"], z["beams"]
        n = len(beams); ntr = int(TRAIN_FRAC * n); gap = int(GAP_FRAC * n)
        tr, te = slice(0, ntr), slice(ntr + gap, n)
        ytr, yte = beams[tr], beams[te]
        bmb, bn = collections.Counter(yte.tolist()).most_common(1)[0]
        bound = bn / len(yte)

        ext = float(np.percentile(np.abs(gps[tr, :2]), 99.5))
        blob = rasterize(gps, ext, n_ch=meas.shape[1])
        inside = float(np.mean((np.abs(gps[:, 0]) <= ext) & (np.abs(gps[:, 1]) <= ext)))
        print(f"\n=== {sc}: n {n}, test {len(yte)}, bound {bound:.4f}, "
              f"grid +-{ext:.0f} m, unit2 inside {100*inside:.1f}%")

        rec = {"n_test": int(len(yte)), "bound_const": bound, "extent_m": ext,
               "unit2_inside_grid": inside, "results": {}}
        yt = torch.from_numpy(ytr).long(); ye = torch.from_numpy(yte).long()

        def run(tag, Xtr, Xte, cls=None):
            if cls is not None:
                orig = EBP.ImgReadout
                EBP.ImgReadout = cls            # train_readout picks the class by ndim
            try:
                m, _ = train_readout(torch.from_numpy(np.ascontiguousarray(Xtr)).float(), yt,
                                     torch.from_numpy(np.ascontiguousarray(Xte)).float(), ye,
                                     64, 42, 60, dev, False)
            finally:
                if cls is not None:
                    EBP.ImgReadout = orig
            rec["results"][tag] = {**{k: float(v) for k, v in m.items()},
                                   "vs_bound": float(m["top1"] - bound)}
            print(f"  {tag:<34}top1 {m['top1']:.4f}  top5 {m['top5']:.4f}  "
                  f"vs bound {m['top1']-bound:+.4f}", flush=True)
            return m["top1"] - bound

        a = run("GPS blob -> ImgReadout (as used)", blob[tr], blob[te])
        b = run("GPS blob -> ImgReadoutSpatial", blob[tr], blob[te], ImgReadoutSpatial)
        c = run("GPS coords -> VecReadout (ref)", gps[tr], gps[te])
        d = run("radar -> ImgReadoutSpatial", meas[tr], meas[te], ImgReadoutSpatial)

        rec["pooling_is_the_defect"] = bool(a <= 0.05 and b > 0.05)
        rec["image_path_sound"] = bool(a > 0.05)
        out[sc] = rec
        print(f"  VERDICT: " + (
            "global average pooling is the defect; image-input beam nulls are "
            "instrument-limited" if rec["pooling_is_the_defect"] else
            "the image path can see position" if rec["image_path_sound"] else
            "neither image readout clears the bound; something else is wrong"))
        print(f"  radar with the spatial readout: {d:+.4f} vs bound")

    p = os.path.join(RESULTS_DIR, "readout_spatial_audit.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
