#!/usr/bin/env python
"""Save the per-frame beam shortlist of every run, so a wrong beam can be priced.

WHY THIS RUN EXISTS
-------------------
Every beam result in Paper B is a top-k inclusion rate: the fraction of test frames whose
ground-truth best beam is somewhere in the predicted shortlist. That is the wrong currency for
a communication decision in two ways, and neither can be recovered from the saved results.

  1. A 64-beam codebook on a 16-element array has beams that overlap. Missing the best index
     by one is not the same failure as missing it by thirty, and top-k cannot tell them apart.
     The recording ships the measured received power of ALL 64 beams per frame
     (`unit1_pwr_60ghz`), so the loss from probing a shortlist instead of the best beam is
     MEASURABLE rather than assumed.
  2. The resource sweep stores per-seed top-1 but only a MEAN top-5, and no per-frame output
     at all. So the airtime saving, which is linear in the top-5 inclusion rate, has no
     reproducible interval, and no claim about a stale decision, a single recording or an
     alternative estimator can be checked after the fact.

This script re-runs the sweep and saves, for every (architecture, setting, seed), the ordered
top-16 candidates for every test frame. Everything downstream is then an evaluation, not a
training run: `audit_beam_power_regret.py`.

TWO DEPARTURES FROM THE PUBLISHED SWEEP, BOTH DELIBERATE
--------------------------------------------------------
* DETERMINISM. The published sweep ran with cuDNN autotuning and nondeterministic kernels, so
  a seed fixes the initialization and the shuffling but NOT the result: re-running seed 42 at
  the full setting gave top-1 0.2271 and then 0.2363. The per-seed values in
  `sc31_resource_ablation_G_max.json` are therefore one draw from a run-to-run distribution
  and are not reproducible cell by cell. This run sets
  `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which
  costs nothing measurable (4.6 s versus 4.8 s per run) and makes every number here exactly
  reproducible. So this is also an independent replication of the resource trend, not only an
  instrumentation of it.
* ONE ALTERNATIVE ARCHITECTURE. `wide` is 2.1 M parameters against the published readout's
  0.78 M, with batch normalization and rectified-linear units instead of group normalization
  and SiLU, and dropout before the classifier. It is run at the two ends and the middle of the
  chirp axis plus the one-element control, which is enough to ask whether the resource
  ORDERING is a property of the data or of one modest network.

VALIDATION PREDICTIONS ARE SAVED TOO, and that is what makes a design rule auditable. The
inner validation block is the last tenth of the training region, exactly the rows the
checkpoint selector already uses, so an operating point can be chosen there and evaluated once
on test. Without them any (resource, shortlist) choice read off the test frontier is a
post-hoc selection.

Output: results/beam_shortlists_sc{31,33}.npz
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch imports CUDA
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from audit_collapse_fix import train as train_recipe
from audit_readout_spatial import ImgReadoutSpatial

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "datasets", "deepsense6g")
TRAIN_FRAC, GAP_FRAC = 0.70, 0.02
NBEAM, KSAVE = 64, 16
LR, EPOCHS = 3e-4, 40

# The published sweep's ten settings, keyed as the cache stores them.
SETTINGS = [(f"chirps={c}", (4, 256, c)) for c in (8, 16, 32, 64, 125, 250)] + \
           [(f"samples={s}", (4, s, 250)) for s in (64, 128)] + \
           [(f"rx={r}", (r, 256, 250)) for r in (1, 2)]
# Item C of the October 8 audit: the shortest, an intermediate and the full chirp setting,
# plus the one-element control. Expanded only if the ordering moves.
ALT_SETTINGS = ["chirps=8", "chirps=64", "chirps=250", "rx=1"]


class ImgReadoutWide(nn.Module):
    """A higher-capacity alternative with a different normalization and activation.

    Not a tuned competitor: the point is that it is a DIFFERENT estimator, so if the resource
    ordering is an artifact of the published readout's capacity, normalization or activation,
    it should move here. Nothing about it was chosen on test data.
    """
    def __init__(self, in_ch, n_class, base=64):
        super().__init__()
        def blk(a, b):
            return [nn.Conv2d(a, b, 3, 2, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(
            *blk(in_ch, base), *blk(base, base * 2), *blk(base * 2, base * 4),
            *blk(base * 4, base * 4), nn.Flatten(), nn.Dropout(0.1),
            nn.Linear(base * 4 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Linear(256, n_class))

    def forward(self, x):
        return self.net(x)


ARCHS = {"spatial": ImgReadoutSpatial, "wide": ImgReadoutWide}


def split_of(n):
    ntr = int(TRAIN_FRAC * n)
    return ntr, ntr + int(GAP_FRAC * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="31", choices=["31", "33"])
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--settings", default="all")
    a = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    sc = a.scenario

    rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}", f"scenario{sc}_dev.csv"))))
    z = np.load(os.path.join(RESULTS_DIR, f"cache_sc{sc}_ablation_max.npz"))
    beams = z["beams"]
    assert len(rows) == len(beams), (len(rows), len(beams))
    # The cache must describe the same rows as the CSV, or a shortlist cannot be joined to a
    # power vector, a timestamp or a sequence index.
    csv_beams = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    assert np.array_equal(csv_beams, beams), "cache labels disagree with the CSV"

    n = len(beams)
    ntr, start = split_of(n)
    yte = beams[start:]
    bmb, bn = collections.Counter(yte.tolist()).most_common(1)[0]
    bound = bn / len(yte)
    floor = float((yte == collections.Counter(beams[:ntr].tolist()).most_common(1)[0][0]).mean())
    yt = torch.from_numpy(beams[:ntr]).long()
    ye = torch.from_numpy(yte).long()
    print(f"Scenario {sc}: train {ntr}  gap {start - ntr}  test {len(yte)}  "
          f"bound {bound:.4f}  floor {floor:.4f}")

    want = [s for s, _ in SETTINGS] if a.settings == "all" else a.settings.split(",")
    # The inner validation block, as audit_collapse_fix.train slices it: the last tenth of
    # whatever training tensor it is handed. Recomputed here so the saved indices cannot
    # drift from the rows the trainer actually held out.
    nval = max(1, ntr // 10)
    vlo, vhi = ntr - nval, ntr
    Xval_rows = np.arange(vlo, vhi, dtype=np.int32)
    out = {"test_row_index": np.arange(start, n, dtype=np.int32),
           "val_row_index": Xval_rows,
           "beams_all": beams.astype(np.int16),
           "bound": np.float64(bound), "floor": np.float64(floor),
           "n_train": np.int32(ntr), "n_test": np.int32(len(yte)),
           "seeds": np.arange(42, 42 + a.seeds, dtype=np.int32)}
    t00 = time.time()
    for arch, net in ARCHS.items():
        todo = [s for s in want if arch == "spatial" or s in ALT_SETTINGS]
        for label in todo:
            key = dict(SETTINGS)[label]
            X = z["|".join(map(str, key))]
            Xtr = torch.from_numpy(np.ascontiguousarray(X[:ntr])).float()
            Xte = torch.from_numpy(np.ascontiguousarray(X[start:])).float()
            Xva = torch.from_numpy(np.ascontiguousarray(X[vlo:vhi])).float()
            sl = np.zeros((a.seeds, len(yte), KSAVE), np.uint8)
            vsl = np.zeros((a.seeds, nval, KSAVE), np.uint8)
            t1 = np.zeros(a.seeds); t5 = np.zeros(a.seeds)
            t0 = time.time()
            for i, s in enumerate(range(42, 42 + a.seeds)):
                o1, o5, lg, lv = train_recipe(Xtr, yt, Xte, ye, NBEAM, s, dev, lr=LR,
                                              epochs=EPOCHS, balanced=False, select="bal",
                                              net=net, return_logits=True, extra=Xva)
                ordk = lg.topk(KSAVE, 1).indices.numpy().astype(np.uint8)
                vsl[i] = lv.topk(KSAVE, 1).indices.numpy().astype(np.uint8)
                # the saved ordering must reproduce the scalars the trainer reports
                # the trainer reduces in float32, so compare at float32 resolution
                assert abs(float((ordk[:, 0] == yte).mean()) - o1) < 1e-6
                assert abs(float((ordk[:, :5] == yte[:, None]).any(1).mean()) - o5) < 1e-6
                sl[i], t1[i], t5[i] = ordk, o1, o5
            out[f"{arch}|{label}|shortlist"] = sl
            out[f"{arch}|{label}|val_shortlist"] = vsl
            out[f"{arch}|{label}|top1"] = t1
            out[f"{arch}|{label}|top5"] = t5
            ncoll = int(sum(abs(x - floor) < 0.005 or abs(x - bound) < 0.005 for x in t1))
            print(f"  {arch:<8}{label:<13}top1 med {np.median(t1):.4f}  "
                  f"top5 med {np.median(t5):.4f}  collapsed {ncoll}/{a.seeds}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    # Determinism is conditional on the software and hardware stack, so pin it in the file.
    import platform
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda or "", "cudnn": str(torch.backends.cudnn.version()),
           "gpu": torch.cuda.get_device_name(0),
           "deterministic_algorithms": "True", "cudnn_benchmark": "False",
           "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
           "lr": str(LR), "epochs": str(EPOCHS), "k_saved": str(KSAVE)}
    out["environment"] = np.array(json.dumps(env))
    p = os.path.join(RESULTS_DIR, f"beam_shortlists_sc{sc}.npz")
    np.savez_compressed(p, **out)
    print(f"\nSaved: {p}  ({os.path.getsize(p) / 1e6:.1f} MB, "
          f"{(time.time() - t00) / 60:.1f} min)")


if __name__ == "__main__":
    main()
