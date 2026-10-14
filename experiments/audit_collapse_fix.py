"""Why does the beam readout collapse, and which fix stops it?

THE PROBLEM
-----------
On DeepSense Sc.33 the readout collapses onto a constant class in 6.8 of 10 runs per setting;
on Sc.31 it is 0.8 of 10. While that holds, Sc.33 measures the optimizer rather than the data,
and Paper B cannot say whether its one-scenario result generalizes.

CANDIDATE CAUSES, each with a fix that isolates it
--------------------------------------------------
  A  baseline                     the current recipe: Adam 1e-3, 40 epochs, cosine schedule,
                                  unweighted cross-entropy, checkpoint chosen by validation
                                  accuracy
  B  learning rate too high       3e-4
  C  the modal class is too easy  class-balanced cross-entropy, weights ~ 1/frequency
  D  not enough epochs            120
  E  checkpoint SELECTION         <- the one that is a bug rather than a hyperparameter.
                                  The run keeps the epoch with the best validation ACCURACY.
                                  A collapsed model scores the validation modal share, which on
                                  Sc.33 is high; a learning model can dip below that early. So
                                  the selector can prefer a collapsed epoch and never leave it.
                                  Fix: select on balanced accuracy, which a constant predictor
                                  cannot win.
  F  B + C + E combined

Run on Sc.33 at full resources, 10 seeds per arm, and on Sc.31 as a control so a "fix" that
merely trades Sc.33 collapse for Sc.31 damage is visible. Collapse is the pre-existing rule: a
top-1 within 0.005 of the train-modal floor or of the no-information bound.

Output: results/collapse_fix_audit.json
"""
import os, sys, json, copy, collections, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from audit_readout_spatial import ImgReadoutSpatial

TRAIN_FRAC, GAP_FRAC = 0.70, 0.02
ARMS = {
    "A_baseline":        dict(lr=1e-3, epochs=40,  balanced=False, select="acc"),
    "B_lower_lr":        dict(lr=3e-4, epochs=40,  balanced=False, select="acc"),
    "C_class_balanced":  dict(lr=1e-3, epochs=40,  balanced=True,  select="acc"),
    "D_longer":          dict(lr=1e-3, epochs=120, balanced=False, select="acc"),
    "E_balanced_select": dict(lr=1e-3, epochs=40,  balanced=False, select="bal"),
    "F_combined":        dict(lr=3e-4, epochs=40,  balanced=True,  select="bal"),
    # Added after the first pass. F conflated the two helpful components with the harmful
    # one: class balancing destroys accuracy on both scenarios (Sc.31 -0.0229, Sc.33 -0.0896),
    # so F's zero collapse count is an artifact of being uniformly bad rather than a fix.
    # G is the combination actually indicated by the per-arm results.
    "G_lr_and_select":   dict(lr=3e-4, epochs=40,  balanced=False, select="bal"),
}


def balanced_acc(pred, y, n_class):
    """Mean per-class recall. A constant predictor scores 1/n_class whatever the imbalance."""
    tot = 0.0; seen = 0
    for c in range(n_class):
        m = y == c
        if m.any():
            tot += float((pred[m] == c).float().mean()); seen += 1
    return tot / max(seen, 1)


def train(X, y, Xte, yte, n_class, seed, dev, lr, epochs, balanced, select,
          net=ImgReadoutSpatial, return_logits=False, extra=None):
    """Recipe arm `select`/`lr` on one split. Returns (top1, top5), or (top1, top5, logits).

    `net` and `return_logits` were added 2026-10-09 for the shortlist experiment, which needs
    the per-frame ordering rather than two scalars and compares one alternative architecture.
    `extra` is a second feature tensor scored with the same selected checkpoint, added so that
    an operating point can be chosen on the INNER VALIDATION BLOCK and evaluated once on test;
    when given, its logits are appended to the return. All three default to the published
    behaviour, so every earlier caller is unchanged.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = net(X.shape[1], n_class).to(dev)
    nv = max(1, len(X) // 10)
    tl = DataLoader(TensorDataset(X[:-nv], y[:-nv]), batch_size=128, shuffle=True)
    vl = DataLoader(TensorDataset(X[-nv:], y[-nv:]), batch_size=256)
    w = None
    if balanced:
        cnt = torch.bincount(y[:-nv], minlength=n_class).float()
        w = (1.0 / cnt.clamp(min=1)); w = (w / w.sum() * n_class).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_sd = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in tl:
            loss = F.cross_entropy(model(xb.to(dev)), yb.to(dev), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for xb, yb in vl:
                P.append(model(xb.to(dev)).argmax(1).cpu()); Y.append(yb)
        P, Y = torch.cat(P), torch.cat(Y)
        score = balanced_acc(P, Y, n_class) if select == "bal" else float((P == Y).float().mean())
        if score >= best:
            best, best_sd = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        lg = torch.cat([model(xb.to(dev)).cpu()
                        for xb, _ in DataLoader(TensorDataset(Xte, yte), batch_size=256)])
    top1 = float((lg.argmax(1) == yte).float().mean())
    k = min(5, n_class)
    top5 = float((lg.topk(k, 1).indices == yte[:, None]).any(1).float().mean())
    if extra is None:
        return (top1, top5, lg) if return_logits else (top1, top5)
    with torch.no_grad():
        le = torch.cat([model(xb.to(dev)).cpu() for xb, in
                        DataLoader(TensorDataset(extra), batch_size=256)])
    return (top1, top5, lg, le)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {"arms": {k: v for k, v in ARMS.items()}, "seeds": a.seeds, "scenarios": {}}

    for tag in ("33", "31"):
        z = np.load(os.path.join(RESULTS_DIR, f"cache_sc{tag}_ablation.npz"))
        beams = z["beams"]; X = z["4|256|250"]
        n = len(beams); ntr = int(TRAIN_FRAC * n); gap = int(GAP_FRAC * n)
        tr, te = slice(0, ntr), slice(ntr + gap, n)
        ytr, yte = beams[tr], beams[te]
        bmb, bn = collections.Counter(yte.tolist()).most_common(1)[0]
        bound = bn / len(yte)
        floor = float((yte == collections.Counter(ytr.tolist()).most_common(1)[0][0]).mean())
        Xtr = torch.from_numpy(np.ascontiguousarray(X[tr])).float()
        Xte = torch.from_numpy(np.ascontiguousarray(X[te])).float()
        yt = torch.from_numpy(ytr).long(); ye = torch.from_numpy(yte).long()
        print(f"\n=== Sc.{tag}: train {ntr} test {len(yte)} bound {bound:.4f} floor {floor:.4f}")
        rec = {"bound": bound, "floor": floor, "n_train": int(ntr), "arms": {}}
        print(f"  {'arm':<20}{'median':>9}{'vs bound':>10}{'collapsed':>11}")
        for name, cfg in ARMS.items():
            vals = [train(Xtr, yt, Xte, ye, 64, 42 + s, dev, **cfg)[0] for s in range(a.seeds)]
            v = np.array(vals)
            coll = int(sum(abs(x - floor) < 0.005 or abs(x - bound) < 0.005 for x in v))
            rec["arms"][name] = {"seeds": [float(x) for x in v], "median": float(np.median(v)),
                                 "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                 "n_collapsed": coll,
                                 "median_vs_bound": float(np.median(v) - bound)}
            print(f"  {name:<20}{np.median(v):>9.4f}{np.median(v)-bound:>+10.4f}"
                  f"{coll:>8}/{a.seeds}", flush=True)
        out["scenarios"][f"Sc.{tag}"] = rec

    b33 = out["scenarios"]["Sc.33"]["arms"]; b31 = out["scenarios"]["Sc.31"]["arms"]
    base33 = b33["A_baseline"]["n_collapsed"]
    # Rank by median gain over the bound, NOT by collapse count. A uniformly bad arm has no
    # modal-class collapse because it collapses somewhere else; the first version of this
    # verdict chose exactly such an arm.
    cand = {k: v for k, v in b33.items() if k != "A_baseline"}
    best = max(cand, key=lambda k: cand[k]["median_vs_bound"])
    fixes = {k: v["n_collapsed"] for k, v in cand.items()}
    out["verdict"] = {
        "sc33_baseline_collapses": base33,
        "best_fix": best, "sc33_best_collapses": fixes[best],
        "sc33_best_median_vs_bound": b33[best]["median_vs_bound"],
        "sc31_cost_of_that_fix": b31[best]["median_vs_bound"] - b31["A_baseline"]["median_vs_bound"],
        "collapse_solved": bool(fixes[best] <= 2),
        "sc33_clears_bound_after_fix": bool(b33[best]["median_vs_bound"] > 0.05)}
    print(f"\nbest fix on Sc.33: {best}, collapses {base33}/{a.seeds} -> {fixes[best]}/{a.seeds}")
    print(f"  Sc.33 median vs bound after the fix: {b33[best]['median_vs_bound']:+.4f}")
    print(f"  cost on Sc.31: {out['verdict']['sc31_cost_of_that_fix']:+.4f} median vs bound")
    print(f"  collapse solved: {out['verdict']['collapse_solved']}   "
          f"Sc.33 now clears the bound: {out['verdict']['sc33_clears_bound_after_fix']}")
    p = os.path.join(RESULTS_DIR, "collapse_fix_audit.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
