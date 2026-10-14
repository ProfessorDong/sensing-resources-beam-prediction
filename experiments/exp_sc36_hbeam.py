"""
Paper B's first honest experiment: does the model use its measurements when the
benchmark actually permits it -- and does communication-side side information
(h_beam) contribute anything?

WHY THIS IS THE EXPERIMENT THAT MATTERS
---------------------------------------
Every previous result in this project was measured on a degenerate benchmark:
ISAC indoor is solved better by a constant (scene diversity 0.9135), RADIal's
target was a threshold of its own input. Measurement utilization was 0.0% and
0.8% on two of three datasets, and no metric detected it.

Sc.36 passes the degeneracy screen (results/verify_DeepSense_Sc36.json):
4 distinct channels, no circularity, diversity 0.3027, constant IoU 0.3410.
So for the first time the question "does the model use Y?" can be asked fairly.

It also supplies `h_beam` (4x64 beam-power
vectors) as an EXPLICIT conditioning branch, and `unit1_overall-beam` as a
COMMUNICATION-side label. Editor points 1 and 9 and R1 #2 all turn on this.

ARMS -- one variable at a time, everything else identical
---------------------------------------------------------
  none          unconditional; the prior-only floor
  radar         radar polar measurement only
  hbeam         h_beam only (no radar)
  radar_hbeam   both; h_beam injected by FiLM on every U-Net stage

The contrast (none -> radar) is measurement utilization on a fair benchmark.
The contrast (radar -> radar_hbeam) is what the communication side is worth,
which is the ISAC claim.

REPORTED AGAINST THE TRIVIAL BASELINE, ALWAYS
---------------------------------------------
Sc.36's constant predictor scores IoU 0.3410. Any arm that does not clear that
has not demonstrated anything, and the summary prints the comparison so it
cannot be quietly omitted.

Usage:
    python exp_sc36_hbeam.py --seeds 3 --epochs 60
    python exp_sc36_hbeam.py --arms radar,radar_hbeam --seeds 1 --epochs 10   # smoke
"""
import os, sys, json, time, copy, argparse, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from s2s_diff_v2 import PixelDiffusionUNet, DiffusionV2, _clear_gpu
from s2s_diff_v3 import evaluate_full

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ARMS = ["none", "radar", "hbeam", "radar_hbeam", "radar_hbeam2"]
CONSTANT_IOU = 0.3410          # results/verify_DeepSense_Sc36.json, S4
N_BEAM = 256
GRID = 64


# =============================================================================
class SC36(Dataset):
    def __init__(self, scenes, meas, hbeam):
        self.s = torch.from_numpy(scenes)
        self.m = torch.from_numpy(meas)
        self.h = torch.from_numpy(hbeam)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i], self.m[i], self.h[i]


class HBeamFiLM(nn.Module):
    """h_beam -> per-channel (scale, shift) applied to the measurement stack.

    FiLM rather than concatenation because h_beam is a 256-dim vector with no
    spatial layout; tiling it into a 64x64 plane would assert a spatial structure
    it does not have. See Perez et al., FiLM (AAAI 2018).
    """
    def __init__(self, n_beam=N_BEAM, meas_ch=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_beam, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * meas_ch))
        nn.init.zeros_(self.net[-1].weight)      # start as identity: gamma=1, beta=0
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, meas, h):
        gb = self.net(h)
        g, b = gb.chunk(2, dim=1)
        g = (1.0 + g).unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)
        return meas * g + b


def hbeam_to_map(h, grid=GRID):
    """h_beam (B, 4*64) -> (B, 4, grid, grid) spatial map.

    Beam index is an ANGLE, and the radar measurement's column axis is azimuth,
    so each array's 64 beam powers are laid along the azimuth axis and broadcast
    along range. That gives h_beam genuine angular expression instead of the four
    broadcast scalars a FiLM bias would provide against a zero input.

    ASSUMPTION, not a verified fact: that the DeepSense 64-beam codebook and our
    64-bin azimuth FFT span comparable angular sectors in the same order. The two
    come from different arrays with different apertures and mountings, and no
    calibration ships with Sc.36. The network can absorb any monotone remapping,
    so this is an inductive bias rather than a claim -- do not describe the axes
    as physically aligned in the paper without checking with DeepSense.
    """
    B = h.shape[0]
    m = h.view(B, 4, 1, 64)                       # (B, arrays, 1, azimuth)
    if 64 != grid:
        m = F.interpolate(m, size=(1, grid), mode="nearest")
    return m.expand(B, 4, grid, grid).contiguous()


class HBeamFiLM2(nn.Module):
    """Per-AZIMUTH, bounded FiLM. Replaces HBeamFiLM, which collapsed.

    WHAT WENT WRONG WITH v1
    -----------------------
    v1 produced four per-channel scalars with unbounded gamma. Across all three
    seeds the model drove gamma to ~0.008 (checkpoint inspection), i.e. it
    multiplied the radar by ~1/125 and kept only beta, a spatially uniform
    constant. radar_hbeam therefore scored at hbeam-only level (0.2594) rather
    than anywhere near radar (0.3377). Classic multiplicative-gating collapse:
    the bias path is an easier early win, and once gamma -> 0 the gradient
    reaching the radar vanishes, so the collapse is self-reinforcing.

    TWO CHANGES
    -----------
    1. PER-AZIMUTH, not per-channel. h_beam is 4 arrays x 64 beams, and beam
       index is an angle; the measurement's column axis is azimuth. Modulating
       each azimuth column is the physically motivated operation -- beam power
       at an angle says how much energy was directed there. Four global scalars
       could not express that.
    2. BOUNDED gamma. gamma = 1 + 0.5*tanh(.) in [0.5, 1.5], so the radar can be
       attenuated at most 2x and can NEVER be erased. beta is bounded to +-0.5 so
       a large constant bias is not an escape route either.

    Still identity at init (zero-init final layer, tanh(0)=0), so the arm starts
    from the radar solution and must learn that beams help.
    """
    def __init__(self, n_beam=N_BEAM, meas_ch=4, n_az=64, hidden=128,
                 gamma_range=0.5, beta_range=0.5):
        super().__init__()
        self.meas_ch, self.n_az = meas_ch, n_az
        self.gr, self.br = gamma_range, beta_range
        self.net = nn.Sequential(
            nn.Linear(n_beam, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * meas_ch * n_az))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, meas, h):
        B = h.shape[0]
        gb = self.net(h).view(B, 2, self.meas_ch, self.n_az)
        gamma = 1.0 + self.gr * torch.tanh(gb[:, 0])          # [1-gr, 1+gr]
        beta = self.br * torch.tanh(gb[:, 1])                 # [-br, br]
        if self.n_az != meas.shape[-1]:
            gamma = F.interpolate(gamma, size=meas.shape[-1], mode="nearest")
            beta = F.interpolate(beta, size=meas.shape[-1], mode="nearest")
        return meas * gamma.unsqueeze(2) + beta.unsqueeze(2)  # broadcast over range

    @torch.no_grad()
    def gamma_stats(self, h):
        """Diagnostic: gamma must stay away from 0 or the radar has been erased."""
        B = h.shape[0]
        gb = self.net(h).view(B, 2, self.meas_ch, self.n_az)
        g = 1.0 + self.gr * torch.tanh(gb[:, 0])
        b = self.br * torch.tanh(gb[:, 1])
        return {"gamma_mean": float(g.mean()), "gamma_absmean": float(g.abs().mean()),
                "gamma_min": float(g.min()), "gamma_max": float(g.max()),
                "gamma_std_across_azimuth": float(g.std(dim=-1).mean()),
                "beta_absmean": float(b.abs().mean())}


class ArmModel(nn.Module):
    """Wraps PixelDiffusionUNet so every arm sees an identical 4-channel input
    tensor and an identical parameter count in the backbone. Only the CONTENT of
    that tensor differs, so the comparison isolates information, not capacity."""
    def __init__(self, arm, base_ch=96, meas_ch=4):
        super().__init__()
        self.arm = arm
        self.unet = PixelDiffusionUNet(base_ch=base_ch, meas_ch=meas_ch)
        if arm == "radar_hbeam":
            self.film = HBeamFiLM(meas_ch=meas_ch)            # v1, known to collapse
        elif arm == "radar_hbeam2":
            self.film = HBeamFiLM2(meas_ch=meas_ch)           # fixed: per-azimuth, bounded
        else:
            self.film = None

    def make_cond(self, meas, h):
        """Build the 4-channel conditioning tensor for this arm."""
        if self.arm == "none":
            return torch.zeros_like(meas)
        if self.arm == "radar":
            return meas
        if self.arm == "hbeam":
            return hbeam_to_map(h).to(meas.dtype)            # h_beam only, spatial
        return self.film(meas, h)                            # radar_hbeam / radar_hbeam2

    def forward(self, x_t, t, cond):
        """Signature matches DiffusionV2.ddim_sample's model(x, t, measurement),
        so the canonical sampler is reused unchanged and inference is identical
        to every other result in this project."""
        return self.unet(x_t, t, cond)


# =============================================================================
def load_cache(stride, name=None):
    p = os.path.join(CACHE, name) if name else os.path.join(CACHE, f"sc36_stride{stride}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"cache missing: {p}\nRun: python experiments/build_sc36_cache.py --stride {stride}")
    z = np.load(p)
    meta = json.load(open(p.replace(".npz", "_meta.json")))
    sp = meta["split"]
    sl = lambda k: slice(sp[k][0], sp[k][1])
    tr = (z["scenes"][sl("train")], z["meas"][sl("train")], z["hbeam"][sl("train")])
    te = (z["scenes"][sl("test")], z["meas"][sl("test")], z["hbeam"][sl("test")])
    return tr, te, meta


def train_arm(arm, seed, tr, epochs, device, base_ch=96, lr=2e-4, batch=32, val_frac=0.1):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    sc, me, hb = tr
    n = len(sc); nv = max(1, int(val_frac * n))
    # validation is the LAST contiguous block of train -- never a random subset
    trd = SC36(sc[:-nv], me[:-nv], hb[:-nv])
    vad = SC36(sc[-nv:], me[-nv:], hb[-nv:])
    g = torch.Generator(); g.manual_seed(seed)
    tl = DataLoader(trd, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    vl = DataLoader(vad, batch_size=batch, shuffle=False, num_workers=0)

    model = ArmModel(arm, base_ch=base_ch).to(device)
    diff = DiffusionV2(n_steps=300, device=device)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best, best_sd, hist = float("inf"), None, []
    for ep in range(epochs):
        model.train()
        for scene, meas, h in tl:
            scene, meas, h = scene.to(device), meas.to(device), h.to(device)
            t = torch.randint(0, diff.n_steps, (scene.shape[0],), device=device)
            noise = torch.randn_like(scene)
            x_t, _ = diff.q_sample(scene, t, noise)
            eps = model(x_t, t, model.make_cond(meas, h))
            occ_w = 1.0 + scene[:, 0:1]
            loss = (occ_w * (eps - noise).pow(2)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()
        model.eval(); v = 0.0
        with torch.no_grad():
            for scene, meas, h in vl:
                scene, meas, h = scene.to(device), meas.to(device), h.to(device)
                t = torch.randint(0, diff.n_steps, (scene.shape[0],), device=device)
                noise = torch.randn_like(scene)
                x_t, _ = diff.q_sample(scene, t, noise)
                v += F.mse_loss(model(x_t, t, model.make_cond(meas, h)), noise).item()
        v /= max(1, len(vl)); hist.append(v)
        if v < best:
            best, best_sd = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd)
    return model, {"best_val": best, "val_hist": hist}


@torch.no_grad()
def infer(model, sc, me, hb, device, n_steps=50, n_samples=4, eta=0.3, batch=16):
    """Uses DiffusionV2.ddim_sample -- the same sampler as every prior result."""
    diff = DiffusionV2(n_steps=300, device=device)
    dl = DataLoader(SC36(sc, me, hb), batch_size=batch, shuffle=False, num_workers=0)
    model.eval()
    preds, gts = [], []
    for scene, meas, h in dl:
        meas, h = meas.to(device), h.to(device)
        cond = model.make_cond(meas, h)
        samples = [diff.ddim_sample(model, cond, n_steps=n_steps, eta=eta)
                   for _ in range(n_samples)]
        preds.append(torch.stack(samples).mean(0).cpu())
        gts.append(scene)
    return torch.cat(preds), torch.cat(gts)


def utilization(model, sc, me, hb, device):
    """Four-way battery. shuffled-Y is the principled null (Y=0 confounds
    information removal with out-of-distribution input)."""
    rng = np.random.RandomState(42)
    out = {}
    variants = {
        "real": (me, hb),
        "zero_radar": (np.zeros_like(me), hb),
        "shuffled_radar": (me[rng.permutation(len(me))], hb),
        "shuffled_hbeam": (me, hb[rng.permutation(len(hb))]),
    }
    for name, (m, h) in variants.items():
        p, g = infer(model, sc, m, h, device)
        out[name] = {k: float(v) for k, v in evaluate_full(p, g).items()}
    r = out["real"]["IoU"]
    out["summary"] = {
        "radar_utilization": float(r - out["shuffled_radar"]["IoU"]),
        "hbeam_utilization": float(r - out["shuffled_hbeam"]["IoU"]),
        "zero_radar_drop": float(r - out["zero_radar"]["IoU"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--base-ch", type=int, default=96)
    ap.add_argument("--train-limit", type=int, default=0, help="cap train frames (smoke test)")
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="cap test frames by SYSTEMATIC sampling across the whole test block "
                         "(head-truncation would bias to one stretch of the route)")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    ap.add_argument("--cache", default=None, help="cache filename under experiments/cache/")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    device = torch.device("cuda")

    tr, te, meta = load_cache(a.stride, a.cache)
    if a.train_limit:
        tr = tuple(x[:a.train_limit] for x in tr)
    if a.eval_limit and a.eval_limit < len(te[0]):
        # Systematic sample spanning the ENTIRE test block. The vehicle travels ~18 km,
        # so the first N frames are one stretch of road; strided sampling preserves the
        # geographic and temporal coverage of the full split.
        step = len(te[0]) / a.eval_limit
        sel = (np.arange(a.eval_limit) * step).astype(int)
        te = tuple(x[sel] for x in te)
        print(f"  eval: systematic sample of {len(sel)} frames "
              f"(stride ~{step:.1f}) spanning the full test block")
    print(f"Sc.36 cache: train {len(tr[0])}  test {len(te[0])}  "
          f"(contiguous temporal split with gap; occupancy fill "
          f"{100*meta['occupancy_fill']:.2f}%)")
    # The 0.3410 bar came from a 120-frame screen. Recompute it on the EXACT frames
    # these models are evaluated on, otherwise the comparison is against a different set.
    const_pred = np.repeat(tr[0][:, 0].mean(0, keepdims=True), len(te[0]), 0)
    cm = evaluate_full(torch.from_numpy(np.stack([const_pred]*3, 1)),
                       torch.from_numpy(te[0]))
    const_iou = float(cm["IoU"])
    print(f"trivial baseline, recomputed on this eval set: constant IoU = {const_iou:.4f} "
          f"(ECE {cm['ECE']:.4f}, NLL {cm['NLL']:.4f}, Brier {cm['Brier']:.6f})")
    print(f"  [120-frame screen had reported {CONSTANT_IOU:.4f}]\n")

    save = os.path.join(RESULTS_DIR, f"sc36_hbeam{a.tag}.json")
    out = json.load(open(save)) if os.path.exists(save) else {"config": vars(a), "arms": {}}
    out.setdefault("arms", {})

    for arm in arms:
        out["arms"].setdefault(arm, {})
        for si in range(a.seeds):
            seed = 42 + si; key = f"seed{seed}"
            if key in out["arms"][arm]:
                print(f"  [{arm}/{key}] cached"); continue
            t0 = time.time()
            model, tinfo = train_arm(arm, seed, tr, a.epochs, device, base_ch=a.base_ch)
            u = utilization(model, te[0], te[1], te[2], device)
            ck_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "checkpoints", "sc36")
            os.makedirs(ck_dir, exist_ok=True)
            # tag is part of the filename: without it a +-100 m run overwrites the
            # +-32 m weights at the same path (it did, 2026-09-27). Untagged runs keep
            # their historical names so existing results JSONs stay valid.
            ck = os.path.join(ck_dir, f"{arm}_{key}{a.tag}.pt")
            torch.save(model.state_dict(), ck)
            gstats = None
            if model.film is not None and hasattr(model.film, "gamma_stats"):
                hb = torch.from_numpy(tr[2][:512]).to(device)
                gstats = model.film.gamma_stats(hb)
            rec = {"train": tinfo, "eval": u, "wall_s": time.time() - t0,
                   "checkpoint": ck, "film_gamma": gstats,
                   "n_params": sum(p.numel() for p in model.parameters())}
            out["arms"][arm][key] = rec
            s = u["summary"]
            print(f"  [{arm:<12}/{key}] IoU={u['real']['IoU']:.4f} ECE={u['real']['ECE']:.4f} "
                  f"radar_util={s['radar_utilization']:+.4f} hbeam_util={s['hbeam_utilization']:+.4f} "
                  f"vs_const={u['real']['IoU']-const_iou:+.4f} [{rec['wall_s']:.0f}s]")
            if gstats:
                collapsed = gstats["gamma_absmean"] < 0.3
                print(f"                 FiLM gamma: mean={gstats['gamma_mean']:.3f} "
                      f"|mean|={gstats['gamma_absmean']:.3f} "
                      f"range=[{gstats['gamma_min']:.3f},{gstats['gamma_max']:.3f}] "
                      f"az-std={gstats['gamma_std_across_azimuth']:.3f}"
                      f"{'  *** RADAR ERASED ***' if collapsed else '  (radar retained)'}")
            del model; _clear_gpu(); gc.collect()
            json.dump(out, open(save, "w"), indent=2)

    print(f"\n{'='*104}\nSc.36  ({a.epochs} epochs, {a.seeds} seeds)   "
          f"constant baseline IoU {const_iou:.4f} (this eval set)\n{'='*104}")
    print(f"{'arm':<14}{'IoU':>9}{'sd':>8}{'vs const':>10}{'ECE':>9}"
          f"{'radar util':>12}{'hbeam util':>12}")
    summ = {}
    for arm in ARMS:
        rs = out["arms"].get(arm, {})
        if not rs: continue
        iou = [r["eval"]["real"]["IoU"] for r in rs.values()]
        ece = [r["eval"]["real"]["ECE"] for r in rs.values()]
        ru = [r["eval"]["summary"]["radar_utilization"] for r in rs.values()]
        hu = [r["eval"]["summary"]["hbeam_utilization"] for r in rs.values()]
        sd = float(np.std(iou, ddof=1)) if len(iou) > 1 else 0.0
        summ[arm] = {"IoU_mean": float(np.mean(iou)), "IoU_sd": sd,
                     "ECE_mean": float(np.mean(ece)),
                     "radar_util_mean": float(np.mean(ru)),
                     "hbeam_util_mean": float(np.mean(hu)),
                     "beats_constant": bool(np.mean(iou) > const_iou), "n": len(iou)}
        s = summ[arm]
        print(f"{arm:<14}{s['IoU_mean']:>9.4f}{sd:>8.4f}"
              f"{s['IoU_mean']-const_iou:>+10.4f}{s['ECE_mean']:>9.4f}"
              f"{s['radar_util_mean']:>+12.4f}{s['hbeam_util_mean']:>+12.4f}")
    out["summary"] = summ
    out["constant_baseline_IoU"] = const_iou
    out["constant_baseline_full"] = {k: float(v) for k, v in cm.items()}
    out["constant_baseline_note"] = "recomputed on the exact eval subset used here"
    json.dump(out, open(save, "w"), indent=2)
    print(f"\nSaved: {save}")
    beat = [k for k, v in summ.items() if v["beats_constant"]]
    print(f"\narms beating the constant predictor: {beat if beat else 'NONE -- report that plainly'}")

    # Paired contrasts across seeds. These are the two questions the experiment exists for.
    try:
        from scipy import stats
        def paired(a_arm, b_arm, label):
            A, B = out["arms"].get(a_arm, {}), out["arms"].get(b_arm, {})
            ks = sorted(set(A) & set(B))
            if len(ks) < 2:
                print(f"  {label}: need >=2 shared seeds (have {len(ks)})"); return
            x = np.array([A[k]["eval"]["real"]["IoU"] for k in ks])
            y = np.array([B[k]["eval"]["real"]["IoU"] for k in ks])
            d = y - x
            t, p = stats.ttest_rel(y, x)
            ci = stats.t.interval(0.95, len(d)-1, loc=d.mean(), scale=stats.sem(d)) \
                 if d.std(ddof=1) > 0 else (d.mean(), d.mean())
            print(f"  {label:<34} delta={d.mean():+.4f}  t({len(d)-1})={t:.2f}  "
                  f"p={p:.3f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  n={len(ks)}")
            out.setdefault("contrasts", {})[label] = {
                "delta": float(d.mean()), "t": float(t), "p": float(p),
                "ci95": [float(ci[0]), float(ci[1])], "n": len(ks)}
        print("\npaired contrasts (the questions this experiment exists to answer):")
        paired("none", "radar", "measurement utilization (none->radar)")
        paired("radar", "radar_hbeam", "ISAC gain v1 (radar->radar_hbeam)")
        paired("radar", "radar_hbeam2", "ISAC gain FIXED (radar->radar_hbeam2)")
        paired("none", "hbeam", "h_beam alone (none->hbeam)")
        json.dump(out, open(save, "w"), indent=2)
    except Exception as e:
        print(f"  (paired stats skipped: {e})")


if __name__ == "__main__":
    main()
