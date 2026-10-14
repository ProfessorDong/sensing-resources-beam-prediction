#!/usr/bin/env python
"""Resources -> bearing-estimation quality -> beam accuracy, measured, with the CRB tested.

WHY THIS EXISTS
---------------
The earlier Paper B headline correlated "measurement utilization" with the beam-prediction
gain over the no-information bound and reported r = +0.97. That is near-definitional:
utilization is the real arm minus the shuffled arm, the shuffled arm sits at a nearly
constant offset below the bound (mean -0.0513, s.d. 0.0036 over all ten settings), so
utilization is the gain plus a constant and the correlation is a relabelling.

This script replaces it with a chain whose two links are measured independently.

  link 1   resources -> bearing error.  Stoica and Nehorai (1989) Thm 4.3b gives
           CRB(theta) ~ 1/(N*SNR*m(m^2-1)); folding the coherent gain of the range and
           Doppler FFTs into SNR leaves one resource product
                R = N_samples * N_chirps * m(m^2 - 1)
           and the bound predicts sigma_theta ~ R^(-1/2), i.e. a slope of EXACTLY -1/2 in
           log sigma vs log R. That is a falsifiable number, not a trend.

  link 2   bearing error -> beam accuracy.  unit1 is stationary, so bearing determines the
           beam: measured Spearman between GPS bearing and beam index is +0.9512, and the
           beam s.d. within a bearing ventile is 3.78 of 64 against 12.49 overall.

Neither link is definitional: bearing error is regressed against GPS ground truth and never
sees the beam label, and beam accuracy never sees the bearing.

NO-INFORMATION REFERENCE
------------------------
Per the project rule, the reference is the best constant EVALUATED ON TEST. For a regression
that is the test RMSE of the train-mean bearing. A run at or above it has learned nothing,
and is counted as a collapse rather than averaged in silently.

Output: results/sc31_bearing_mediation[_pool].json
"""
import argparse, copy, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_readout_spatial import ImgReadoutSpatial
from exp_sc31_resource_ablation import SETTINGS, TRAIN_FRAC, GAP_FRAC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def train_bearing(X, y, Xte, yte, seed, dev, lr, epochs):
    """Regress bearing in normalized units; return test RMSE in degrees."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ImgReadoutSpatial(X.shape[1], 1).to(dev)
    nv = max(1, len(X) // 10)
    mu, sd = float(y[:-nv].mean()), float(y[:-nv].std())
    yn = (y - mu) / sd
    tl = DataLoader(TensorDataset(X[:-nv], yn[:-nv]), batch_size=128, shuffle=True)
    vl = DataLoader(TensorDataset(X[-nv:], yn[-nv:]), batch_size=256)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_sd = np.inf, None
    for _ in range(epochs):
        model.train()
        for xb, yb in tl:
            loss = F.huber_loss(model(xb.to(dev)).squeeze(1), yb.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval(); se, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in vl:
                p = model(xb.to(dev)).squeeze(1).cpu()
                se += float(((p - yb) ** 2).sum()); n += len(yb)
        v = (se / n) ** 0.5
        if v <= best:
            best, best_sd = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        p = torch.cat([model(xb.to(dev)).squeeze(1).cpu()
                       for xb, _ in DataLoader(TensorDataset(Xte, yte), batch_size=256)])
    pred = p * sd + mu
    return float(torch.sqrt(((pred - yte) ** 2).mean())), float(mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="31", choices=["31", "33"])
    ap.add_argument("--pool", default="decimate", choices=["decimate", "max", "mean"])
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    dev = torch.device("cuda")
    sfx = "" if a.pool == "decimate" else f"_{a.pool}"

    sc = a.scenario
    z = np.load(os.path.join(RES, f"cache_sc{sc}_ablation{sfx}.npz"))
    lab = np.load(os.path.join(RES, f"sc{sc}_bearing_labels.npz"))
    brg = lab["bearing_deg"].astype(np.float32)
    n = len(brg); ntr = int(TRAIN_FRAC * n); gap = int(GAP_FRAC * n)
    tr, te = slice(0, ntr), slice(ntr + gap, n)
    assert len(z["beams"]) == n, (len(z["beams"]), n)
    yt = torch.from_numpy(brg[tr]).float(); ye = torch.from_numpy(brg[te]).float()
    # no-information reference: the train-mean bearing, scored on test
    bound = float(np.sqrt(np.mean((brg[te] - brg[tr].mean()) ** 2)))
    print(f"Scenario{sc}  pool={a.pool}  train {ntr}  test {n - ntr - gap}  seeds {a.seeds}")
    print(f"bearing: train mean {brg[tr].mean():.2f} deg, test mean {brg[te].mean():.2f} deg")
    print(f"NO-INFORMATION BEARING RMSE (train-mean constant on test): {bound:.3f} deg\n")

    out = {"scenario": f"Scenario{sc}", "pool": a.pool, "seeds": a.seeds, "epochs": a.epochs, "lr": a.lr,
           "bound_rmse_deg": bound, "n_train": int(ntr), "n_test": int(n - ntr - gap),
           "crb_note": "R = n_samples*n_chirps*n_rx*(n_rx^2-1); CRB predicts slope -0.5 "
                       "in log10 RMSE vs log10 R", "settings": {}}
    print(f"  {'setting':<14}{'R':>11}{'RMSE med':>10}{'sd':>7}{'vs bound':>10}{'coll':>6}")
    for key, label, sweep in SETTINGS:
        X = z["|".join(map(str, key))]
        Xtr = torch.from_numpy(np.ascontiguousarray(X[tr])).float()
        Xte = torch.from_numpy(np.ascontiguousarray(X[te])).float()
        r = []
        for s in range(a.seeds):
            rmse, _ = train_bearing(Xtr, yt, Xte, ye, 42 + s, dev, a.lr, a.epochs)
            r.append(rmse)
        arr = np.asarray(r)
        coll = int((arr >= 0.98 * bound).sum())
        m, ns, nc = key
        R = ns * nc * m * (m * m - 1)
        rec = {"n_rx": m, "n_samples": ns, "n_chirps": nc, "sweep": sweep,
               "resource_product_R": int(R),
               "rmse_deg_median": float(np.median(arr)),
               "rmse_deg_mean": float(arr.mean()),
               "rmse_deg_sd": float(arr.std(ddof=1)),
               "rmse_deg_seeds": [float(x) for x in arr],
               "vs_bound_deg": float(np.median(arr) - bound),
               "n_collapsed": coll, "n_seeds": len(arr)}
        out["settings"][label] = rec
        print(f"  {label:<14}{R:>11}{rec['rmse_deg_median']:>10.3f}"
              f"{rec['rmse_deg_sd']:>7.3f}{rec['vs_bound_deg']:>+10.3f}"
              f"{coll:>4}/{len(arr)}", flush=True)
    p = os.path.join(RES, f"sc{sc}_bearing_mediation{sfx}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
