"""
The measurement-sparsity experiment: does utilization rise with measurement density?

WHY THIS IS THE EXPERIMENT PAPER B NEEDS
----------------------------------------
Utilization theory predicts that measurement utilization increases with the
information content of the measurement. Nothing in this project could test that
before: ISAC indoor had utilization 0.0%, V2I 0.8%, and RADIal's target was a
threshold of its own input.

View-of-Delft can. It ships the same scenes with radar accumulated over 1, 3 or 5
frames -- ~356 / ~1064 / ~1769 points against ~178,995 LiDAR points -- so density
is the ONLY thing that varies. And its prior is nearly worthless (constant
predictor IoU 0.0206, scene diversity 0.0872), so the `none`->`radar` contrast is
not diluted by a benchmark the prior can already solve.

A one-line baseline already shows the effect without any training: the best
train-fitted threshold rises 0.1688 -> 0.2112 -> 0.2237 across the three levels
while the (measurement-independent) constant stays pinned at 0.0206. This
experiment asks whether a trained model shows the same monotone trend, and how
much of the achievable information it actually extracts.

DESIGN -- 2 arms x 3 sparsity x 3 seeds = 18 runs
-------------------------------------------------
  arms      `none` (unconditional, the prior-only floor) and `radar`
  sparsity  1 / 3 / 5 accumulated radar frames
  seeds     42, 43, 44

Identical backbone, identical parameter count, identical data and split in every
cell. Only the conditioning content and the measurement density vary.

There is no `hbeam` arm: VoD has no beam-power data. The h_beam question was
already answered on Sc.36 (clean null, p=0.804).

WHAT IS REPORTED
----------------
Per cell: IoU / ECE / NLL / Brier, the utilization battery (real / zero-radar /
shuffled-radar, with shuffled as the principled null), and range-stratified IoU
because aggregate BEV IoU proved misleading on Sc.36. Baselines are recomputed on
the EXACT evaluation frames. Contrasts are paired across seeds with 95% CIs, and
the monotonicity of utilization in density is tested directly.

Usage:
    python exp_vod_sparsity.py --seeds 3 --epochs 40
    python exp_vod_sparsity.py --seeds 1 --epochs 2 --train-limit 500 --eval-limit 48 --tag _smoke
"""
import os, sys, json, time, copy, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from s2s_diff_v2 import PixelDiffusionUNet, DiffusionV2, _clear_gpu
from s2s_diff_v3 import evaluate_full

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "vod.npz")
ARMS = ["none", "radar"]
SPARSITY = [1, 3, 5]
GRID, X_RANGE, Y_RANGE = 64, (0.0, 64.0), (-32.0, 32.0)
RINGS = [(0, 16), (16, 32), (32, 48), (48, 64), (0, 64)]


class VoD(Dataset):
    def __init__(self, s, m):
        self.s = torch.from_numpy(s); self.m = torch.from_numpy(m)
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i], self.m[i]


class ArmModel(nn.Module):
    """Identical backbone in both arms; only the conditioning CONTENT differs, so
    the comparison isolates information rather than capacity."""
    def __init__(self, arm, base_ch=96, meas_ch=4):
        super().__init__()
        self.arm = arm
        self.unet = PixelDiffusionUNet(base_ch=base_ch, meas_ch=meas_ch)
    def make_cond(self, meas):
        return torch.zeros_like(meas) if self.arm == "none" else meas
    def forward(self, x_t, t, cond):
        return self.unet(x_t, t, cond)


def ring_masks():
    """Forward grid: row index is x (forward), column is y (lateral)."""
    gx, gy = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    x = X_RANGE[0] + (gx + 0.5) * (X_RANGE[1] - X_RANGE[0]) / GRID
    y = Y_RANGE[0] + (gy + 0.5) * (Y_RANGE[1] - Y_RANGE[0]) / GRID
    r = np.sqrt(x ** 2 + y ** 2)
    return {f"{lo}-{hi}m": (r >= lo) & (r < hi) for lo, hi in RINGS}


def iou_masked(p, g, m):
    a, b = p[:, m] > 0.5, g[:, m] > 0.5
    i, u = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def load_cache(eval_limit=0, train_limit=0):
    if not os.path.exists(CACHE):
        raise SystemExit(f"cache missing: {CACHE}\nRun: python experiments/build_vod_cache.py")
    z = np.load(CACHE)
    tr_s = z["train_scenes"]; te_s = z["val_scenes"]
    tr_m = {n: z[f"train_meas_r{n}"] for n in SPARSITY}
    te_m = {n: z[f"val_meas_r{n}"] for n in SPARSITY}
    if train_limit:
        tr_s = tr_s[:train_limit]; tr_m = {n: v[:train_limit] for n, v in tr_m.items()}
    if eval_limit and eval_limit < len(te_s):
        # systematic sample across the whole val split, never head truncation
        sel = (np.arange(eval_limit) * (len(te_s) / eval_limit)).astype(int)
        te_s = te_s[sel]; te_m = {n: v[sel] for n, v in te_m.items()}
    return tr_s, tr_m, te_s, te_m


def train_one(arm, nf, seed, tr_s, tr_m, epochs, device, base_ch=96, lr=2e-4,
              batch=32, val_frac=0.1):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    m = tr_m[nf]
    nv = max(1, int(val_frac * len(tr_s)))
    g = torch.Generator(); g.manual_seed(seed)
    tl = DataLoader(VoD(tr_s[:-nv], m[:-nv]), batch_size=batch, shuffle=True,
                    num_workers=0, pin_memory=True, generator=g)
    vl = DataLoader(VoD(tr_s[-nv:], m[-nv:]), batch_size=batch, shuffle=False, num_workers=0)
    model = ArmModel(arm, base_ch).to(device)
    diff = DiffusionV2(n_steps=300, device=device)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_sd, hist = float("inf"), None, []
    for ep in range(epochs):
        model.train()
        for scene, meas in tl:
            scene, meas = scene.to(device), meas.to(device)
            t = torch.randint(0, diff.n_steps, (scene.shape[0],), device=device)
            noise = torch.randn_like(scene)
            x_t, _ = diff.q_sample(scene, t, noise)
            eps = model(x_t, t, model.make_cond(meas))
            occ_w = 1.0 + scene[:, 0:1]
            loss = (occ_w * (eps - noise).pow(2)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()
        model.eval(); v = 0.0
        with torch.no_grad():
            for scene, meas in vl:
                scene, meas = scene.to(device), meas.to(device)
                t = torch.randint(0, diff.n_steps, (scene.shape[0],), device=device)
                noise = torch.randn_like(scene)
                x_t, _ = diff.q_sample(scene, t, noise)
                v += F.mse_loss(model(x_t, t, model.make_cond(meas)), noise).item()
        v /= max(1, len(vl)); hist.append(v)
        if v < best:
            best, best_sd = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd)
    return model, {"best_val": best, "val_hist": hist, "argmin_epoch": int(np.argmin(hist))}


@torch.no_grad()
def predict(model, s, m, device, batch=16):
    diff = DiffusionV2(n_steps=300, device=device)
    dl = DataLoader(VoD(s, m), batch_size=batch, shuffle=False, num_workers=0)
    model.eval(); P, G = [], []
    for scene, meas in dl:
        cond = model.make_cond(meas.to(device))
        smp = [diff.ddim_sample(model, cond, n_steps=50, eta=0.3) for _ in range(4)]
        P.append(torch.stack(smp).mean(0).cpu()); G.append(scene)
    return torch.cat(P), torch.cat(G)


def evaluate(model, te_s, te_m, nf, device, masks):
    rng = np.random.RandomState(42)
    m = te_m[nf]
    variants = {"real": m, "zero_radar": np.zeros_like(m),
                "shuffled_radar": m[rng.permutation(len(m))]}
    out = {}
    for name, mm in variants.items():
        p, gt = predict(model, te_s, mm, device)
        out[name] = {k: float(v) for k, v in evaluate_full(p, gt).items()}
        if name == "real":
            pn, gn = p[:, 0].numpy(), gt[:, 0].numpy()
            out["rings"] = {k: iou_masked(pn, gn, v) for k, v in masks.items()}
    r = out["real"]["IoU"]
    out["summary"] = {"utilization_shuffled": float(r - out["shuffled_radar"]["IoU"]),
                      "utilization_zero": float(r - out["zero_radar"]["IoU"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--sparsity", default="1,3,5")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=650)
    ap.add_argument("--base-ch", type=int, default=96)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    sps = [int(x) for x in a.sparsity.split(",") if x]
    device = torch.device("cuda")
    masks = ring_masks()

    tr_s, tr_m, te_s, te_m = load_cache(a.eval_limit, a.train_limit)
    print(f"VoD cache: train {len(tr_s)}  eval {len(te_s)} (systematic sample of val)")
    print(f"  scene occupancy fill {100*(tr_s[:,0]>0.5).mean():.2f}%")
    for n in sps:
        print(f"  r{n}: measurement fill {100*(tr_m[n][:,0]>0).mean():.2f}%")

    # baselines on the EXACT eval frames
    const = np.repeat(tr_s[:, 0].mean(0, keepdims=True), len(te_s), 0)
    cm = evaluate_full(torch.from_numpy(np.stack([const]*3, 1)), torch.from_numpy(te_s))
    const_iou = float(cm["IoU"])
    thr, thr_cfg = {}, {}
    for n in sps:
        # channel and threshold BOTH fitted on train, then applied unchanged to eval.
        # iou_masked binarizes at >0.5, so the predictor must be a clean 0/1 map.
        ALL = np.ones((GRID, GRID), bool)
        best = (-1.0, None, None)
        for c in range(tr_m[n].shape[1]):
            for t in np.arange(0.02, 0.99, 0.02):
                v = iou_masked((tr_m[n][:, c] > t).astype(np.float32), tr_s[:, 0], ALL)
                if v > best[0]:
                    best = (v, c, round(float(t), 2))
        _, c, t = best
        thr[n] = iou_masked((te_m[n][:, c] > t).astype(np.float32), te_s[:, 0], ALL)
        thr_cfg[n] = {"channel": int(c), "tau": t, "train_IoU": float(best[0])}
    print(f"\nbaselines recomputed on these eval frames:")
    print(f"  constant  IoU {const_iou:.4f}  (ECE {cm['ECE']:.4f}, NLL {cm['NLL']:.4f})")
    for n in sps:
        print(f"  threshold r{n}  IoU {thr[n]:.4f}  "
              f"(ch{thr_cfg[n]['channel']}, tau={thr_cfg[n]['tau']:.2f}, "
              f"train {thr_cfg[n]['train_IoU']:.4f})")
    print(f"  ANY ARM MUST BEAT max(constant, threshold) TO MEAN ANYTHING\n")

    save = os.path.join(RESULTS_DIR, f"vod_sparsity{a.tag}.json")
    out = json.load(open(save)) if os.path.exists(save) else {}
    out.setdefault("config", vars(a)); out.setdefault("cells", {})
    out["baselines"] = {"constant": {k: float(v) for k, v in cm.items()},
                        "threshold_by_sparsity": {str(n): thr[n] for n in sps},
                        "threshold_config": {str(n): thr_cfg[n] for n in sps},
                        "note": "recomputed on the exact eval subset used here"}

    for nf in sps:
        for arm in arms:
            for si in range(a.seeds):
                seed = 42 + si
                key = f"r{nf}/{arm}/seed{seed}"
                if key in out["cells"]:
                    print(f"  [{key}] cached"); continue
                t0 = time.time()
                model, ti = train_one(arm, nf, seed, tr_s, tr_m, a.epochs, device, a.base_ch)
                ev = evaluate(model, te_s, te_m, nf, device, masks)
                out["cells"][key] = {"train": ti, "eval": ev, "wall_s": time.time() - t0}
                s = ev["summary"]
                print(f"  [{key:<22}] IoU={ev['real']['IoU']:.4f} ECE={ev['real']['ECE']:.4f} "
                      f"util={s['utilization_shuffled']:+.4f} "
                      f"vs_const={ev['real']['IoU']-const_iou:+.4f} "
                      f"ep*={ti['argmin_epoch']} [{time.time()-t0:.0f}s]")
                del model; _clear_gpu(); gc.collect()
                json.dump(out, open(save, "w"), indent=2)

    # ---------------- summary ----------------
    print(f"\n{'='*100}\nVoD sparsity sweep ({a.epochs} epochs, {a.seeds} seeds)   "
          f"constant {const_iou:.4f}\n{'='*100}")
    print(f"{'sparsity':>9}{'arm':>8}{'IoU':>9}{'sd':>8}{'util':>9}{'sd':>8}"
          f"{'vs const':>10}{'ECE':>8}")
    agg = {}
    for nf in sps:
        for arm in arms:
            rs = [v for k, v in out["cells"].items() if k.startswith(f"r{nf}/{arm}/")]
            if not rs: continue
            iou = [r["eval"]["real"]["IoU"] for r in rs]
            ut = [r["eval"]["summary"]["utilization_shuffled"] for r in rs]
            ece = [r["eval"]["real"]["ECE"] for r in rs]
            agg[(nf, arm)] = {"IoU": float(np.mean(iou)),
                              "IoU_sd": float(np.std(iou, ddof=1)) if len(iou) > 1 else 0.0,
                              "util": float(np.mean(ut)),
                              "util_sd": float(np.std(ut, ddof=1)) if len(ut) > 1 else 0.0,
                              "ECE": float(np.mean(ece)), "n": len(iou)}
            g = agg[(nf, arm)]
            print(f"{('r'+str(nf)):>9}{arm:>8}{g['IoU']:>9.4f}{g['IoU_sd']:>8.4f}"
                  f"{g['util']:>+9.4f}{g['util_sd']:>8.4f}"
                  f"{g['IoU']-const_iou:>+10.4f}{g['ECE']:>8.4f}")

    try:
        from scipy import stats
        print("\npaired contrasts, none->radar at each density:")
        deltas = {}
        for nf in sps:
            A = {k.split('/')[-1]: v for k, v in out["cells"].items() if k.startswith(f"r{nf}/none/")}
            B = {k.split('/')[-1]: v for k, v in out["cells"].items() if k.startswith(f"r{nf}/radar/")}
            ks = sorted(set(A) & set(B))
            if len(ks) < 2: continue
            x = np.array([A[k]["eval"]["real"]["IoU"] for k in ks])
            y = np.array([B[k]["eval"]["real"]["IoU"] for k in ks])
            d = y - x; t, p = stats.ttest_rel(y, x)
            ci = stats.t.interval(0.95, len(d)-1, loc=d.mean(), scale=stats.sem(d))
            deltas[nf] = float(d.mean())
            print(f"  r{nf}: delta={d.mean():+.4f}  t({len(d)-1})={t:.2f}  p={p:.4f}  "
                  f"95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  n={len(ks)}")
        if len(deltas) == 3:
            v = [deltas[n] for n in sps]
            mono = v[0] < v[1] < v[2]
            # Radar point counts per density. These were hardcoded; they are now read from
            # the degeneracy screen, which MEASURES them (`mean_radar_pts`). Checked
            # 2026-09-28: the hardcoded 356/1064/1769 vs the measured
            # 347.55/1039.95/1728.63 change the slope by 0.00001 IoU, so the published
            # +0.0190 stands -- but the value must be derived, not asserted. Note the
            # screen used its own frame subset; only the MEAN point count is borrowed, and
            # monotonicity and every p-value are independent of these x-values.
            scr = os.path.join(RESULTS_DIR, "vod_screen_summary.json")
            if os.path.exists(scr):
                sd = json.load(open(scr))
                pts = [sd[f"r{n}"]["mean_radar_pts"] for n in sps]
            else:
                pts = [356.0, 1064.0, 1769.0]
                print("  (screen summary missing -- falling back to nominal point counts)")
            print(f"  radar points per frame: {' / '.join(f'{x:.1f}' for x in pts)}")
            sl = np.polyfit(np.log(pts), v, 1)[0]
            print(f"\n  MONOTONE IN DENSITY: {mono}   "
                  f"({v[0]:+.4f} -> {v[1]:+.4f} -> {v[2]:+.4f})")
            print(f"  slope vs log(radar points) = {sl:+.4f} IoU per e-fold")
            print(f"  -> utilization theory predicts a positive slope; "
                  f"{'CONFIRMED' if sl > 0 else 'NOT confirmed'}")
            out["density_trend"] = {"deltas": {str(n): deltas[n] for n in sps},
                                    "monotone": bool(mono), "slope_per_log_pts": float(sl),
                                    "radar_pts_per_frame": [float(x) for x in pts],
                                    "radar_pts_source": ("measured, vod_screen_summary.json"
                                                         if os.path.exists(scr) else "nominal")}
        out["summary"] = {f"r{nf}/{arm}": g for (nf, arm), g in agg.items()}
        out["constant_baseline_IoU"] = const_iou
        json.dump(out, open(save, "w"), indent=2)
    except Exception as e:
        print(f"  (stats skipped: {e})")
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
