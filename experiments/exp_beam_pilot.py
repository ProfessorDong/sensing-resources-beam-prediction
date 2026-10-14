"""
Instrument validation for the beam-prediction experiment. NO reconstruction needed.

WHY THIS RUNS FIRST
-------------------
The full experiment needs ~3 h of diffusion inference to produce reconstructions.
Before spending that, check whether the readout can learn the beam AT ALL from
inputs already cached. Two gates:

  1. `hbeam` must score near 1.0. `overall-beam` IS the flat argmax of the 4x64
     `pwr` stack (verified, match 1.0000 on all 24,799 frames), so a readout that
     cannot recover it from h_beam is too weak to measure anything -- the
     instrument, not the label, is at fault. The first smoke test failed exactly
     here: 540 training samples and 2 epochs collapsed every predictor onto the
     modal class, `hbeam` included (0.2350).

  2. `gt_scene` must beat the modal-beam floor. It is the ceiling on what BEV
     geometry can offer. If ground-truth LiDAR occupancy cannot predict the beam,
     the task is not solvable from scene geometry, and a negative `recon` result
     would say nothing about reconstruction quality. Running the full experiment
     in that case would produce an uninterpretable null.

Only if both gates pass is the full experiment worth running.

Output: experiments/results/beam_pilot.json
"""
import os, sys, json, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_beam_prediction import train_readout
from exp_sc36_hbeam import load_cache

EPOCHS = 30


def main():
    device = torch.device("cuda")
    tr, te, _ = load_cache(1)
    tr_s, tr_m, tr_h = tr
    te_s, te_m, te_h = te

    # label = flat argmax of the 4x64 pwr stack (verified identity)
    btr = tr_h.reshape(len(tr_h), -1).argmax(1)
    bte = te_h.reshape(len(te_h), -1).argmax(1)
    atr, ate = btr // 64, bte // 64
    print(f"FULL cached data: train {len(tr_s)}  test {len(te_s)}")
    dom = collections.Counter(atr.tolist()).most_common(1)[0][0]
    print(f"dominant array in train: {dom+1}  (train {int((atr==dom).sum())}, "
          f"test {int((ate==dom).sum())})")

    tasks = [("A_256way", 256, False, btr, bte,
              np.ones(len(btr), bool), np.ones(len(bte), bool)),
             (f"B_array{dom+1}_64way", 64, True, btr % 64, bte % 64,
              atr == dom, ate == dom)]

    out, verdicts = {}, {}
    for tname, ncls, ordinal, ytr, yte, mtr, mte in tasks:
        mb = collections.Counter(ytr[mtr].tolist()).most_common(1)[0][0]
        floor = float((yte[mte] == mb).mean())
        print(f"\n=== {tname}  n_class={ncls}  train {int(mtr.sum())} "
              f"test {int(mte.sum())}  modal-beam floor {floor:.4f} ===")
        print(f"  {'input':<10}{'top1':>9}{'top5':>9}{'within2':>10}{'mae':>7}  vs floor")
        out[tname] = {"floor": floor, "n_class": ncls,
                      "n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                      "modal_beam": int(mb)}
        for name, Xtr_np, Xte_np in (("hbeam", tr_h, te_h),
                                     ("gt_scene", tr_s, te_s),
                                     ("radar", tr_m, te_m)):
            Xtr = torch.from_numpy(np.ascontiguousarray(Xtr_np[mtr])).float()
            Xte = torch.from_numpy(np.ascontiguousarray(Xte_np[mte])).float()
            m, _ = train_readout(Xtr, torch.from_numpy(ytr[mtr]).long(),
                                 Xte, torch.from_numpy(yte[mte]).long(),
                                 ncls, 42, EPOCHS, device, ordinal)
            w2 = m.get("within2", float("nan")); ma = m.get("mae_beams", float("nan"))
            print(f"  {name:<10}{m['top1']:>9.4f}{m['top5']:>9.4f}"
                  f"{w2:>10.4f}{ma:>7.2f}  {m['top1']-floor:+.4f}", flush=True)
            out[tname][name] = m

        h = out[tname]["hbeam"]["top1"]
        g = out[tname]["gt_scene"]["top1"]
        r = out[tname]["radar"]["top1"]
        gate1 = h > 0.80
        gate2 = (g - floor) > 0.05
        print(f"  GATE 1 instrument : hbeam {h:.4f}  "
              f"{'PASS' if gate1 else 'FAIL — readout cannot recover the argmax, too weak to measure anything'}")
        print(f"  GATE 2 learnable  : gt_scene {g:.4f} vs floor {floor:.4f}  "
              f"{'PASS' if gate2 else 'FAIL — beam is not predictable from BEV geometry'}")
        print(f"  (radar {r:.4f}, {r-floor:+.4f} vs floor)")
        verdicts[tname] = {"gate1_instrument": bool(gate1),
                           "gate2_learnable": bool(gate2),
                           "hbeam_top1": h, "gt_scene_top1": g, "radar_top1": r}
        out[tname]["verdict"] = verdicts[tname]

    ok = any(v["gate1_instrument"] and v["gate2_learnable"] for v in verdicts.values())
    print(f"\n{'='*78}")
    print(f"OVERALL: {'RUN the full experiment' if ok else 'DO NOT run the full experiment'}")
    for k, v in verdicts.items():
        print(f"  {k}: instrument={'PASS' if v['gate1_instrument'] else 'FAIL'}  "
              f"learnable={'PASS' if v['gate2_learnable'] else 'FAIL'}")
    if not ok:
        print("  -> reconstructing 3 x 7,500 frames would produce an uninterpretable null.")
    out["overall_proceed"] = bool(ok)
    p = os.path.join(RESULTS_DIR, "beam_pilot.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
