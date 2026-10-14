"""
Controlled experiment: what does the training-time physics loss actually do?

audit_physics_loss.py established that L_phys contributes exactly zero gradient
to the diffusion model (246/246 tensors report grad=None), and that the two
published ablations differ from the full run in the occupancy-weight
coefficient, the loss scale, and the gradient-clipping scope -- so the
"+7.7 pp from physics" comparison is not controlled.

This experiment isolates one variable. Five arms:

  A  none          base loss only                       (controlled baseline)
  C  detached      + L_phys exactly as published         (positive control:
                     MSE(H(x0.detach()), H(gt).detach())  must equal A)
  B  connected     + MSE(H(x0), H(gt).detach())          (Eq. 18, correctly wired)
  D  to_Y          + MSE(H(x0), Y)                       (Eq. 17 form; R4 #4)
  E  pub_nophys    exact replication of
                     train_ablation_no_physics()         (reproduction control)

Arms A-D share: data, split, init-per-seed, optimizer parameter set
(model+surrogate in all arms), clip_grad_norm_ scope, base loss
L_diff + 0.3*L_weighted with obs_weight=2.0, surrogate supervision 0.1*L_fwd,
schedule, lr, weight decay, epochs, batch size, and checkpoint selection.
Only the physics term differs.

Arm E deliberately does NOT share those, because its purpose is to reproduce
the published ablation number and confirm the confound explanation.

Each trained model is then evaluated on (i) the full metric suite and (ii) the
four-way measurement-utilization battery, so we learn whether correctly wired
physics changes whether the model uses its measurements at all.

Usage:  python exp_physics_controlled.py RADIal [n_seeds] [epochs]
        python exp_physics_controlled.py DeepSense_ISAC [n_seeds] [epochs]
"""
import os, sys, json, time, gc, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, SimpleForwardModel, SimpleDS, _clear_gpu,
)
from s2s_diff_v3 import evaluate_full, infer_diffusion
from sanity_measurement_utilization import load_isac, load_radial, make_variants

ARMS = ["A_none", "C_detached", "B_connected", "D_to_Y", "E_pub_nophys"]
OBS_WEIGHT = 2.0
PHYS_WEIGHT = 0.1
LR = 2e-4
WD = 1e-5
BATCH = 32


def build_split(name):
    """Reproduce the canonical train/val/test split exactly."""
    import glob
    ds = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    if name == "RADIal":
        from train_all_real import preprocess_radial
        seq = sorted(glob.glob(os.path.join(ds, "radial", "extracted", "RECORD*")))
        s, m = preprocess_radial(seq[0], os.path.join(ds, "radial", "labels_CVPR.csv"),
                                 os.path.join(ds, "radial", "miscellaneous", "RADAR_CONFIG.m"),
                                 max_frames=500)
        n = int(0.8 * len(s))
        tr_s, tr_m, te_s, te_m = s[:n], m[:n], s[n:], m[n:]
    elif name == "DeepSense_ISAC":
        from train_all_real import preprocess_isac_scenario
        ps, pm = [], []
        for sc in (42, 43):
            a, b = preprocess_isac_scenario(
                os.path.join(ds, "deepsense6g", f"Scenario{sc}"), sc, max_frames=1500)
            ps.append(a); pm.append(b)
        tr_s, tr_m = np.concatenate(ps), np.concatenate(pm)
        te_s, te_m = preprocess_isac_scenario(
            os.path.join(ds, "deepsense6g", "Scenario44"), 44, max_frames=1500)
    else:
        raise ValueError(name)

    n = len(tr_s)
    nv = max(1, n // 10)
    perm = np.random.RandomState(42).permutation(n)          # canonical split seed
    return (tr_s[perm[nv:]], tr_m[perm[nv:]],
            tr_s[perm[:nv]], tr_m[perm[:nv]], te_s, te_m)


def phys_term(arm, fwd, x0_pred, scene, meas):
    """Return the physics term for this arm (or None)."""
    if arm == "A_none":
        return None
    if arm == "C_detached":                                   # as published
        return F.mse_loss(fwd(x0_pred.detach()), fwd(scene.detach()).detach())
    if arm == "B_connected":                                  # Eq. 18, wired
        return F.mse_loss(fwd(x0_pred), fwd(scene).detach())
    if arm == "D_to_Y":                                       # Eq. 17 form
        return F.mse_loss(fwd(x0_pred), meas)
    return None


def train_arm(arm, seed, data, epochs, device):
    tr_s, tr_m, va_s, va_m, _, _ = data
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)

    model = PixelDiffusionUNet(base_ch=96).to(device)
    fwd = SimpleForwardModel().to(device)
    diffusion = DiffusionV2(n_steps=300, device=device)

    published_arm = (arm == "E_pub_nophys")
    if published_arm:
        params = list(model.parameters())                     # published: model only
    else:
        params = list(model.parameters()) + list(fwd.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    g = torch.Generator(); g.manual_seed(seed)
    tl = DataLoader(SimpleDS(tr_s, tr_m), batch_size=BATCH, shuffle=True,
                    num_workers=0, pin_memory=True, generator=g)
    vl = DataLoader(SimpleDS(va_s, va_m), batch_size=BATCH, shuffle=False, num_workers=0)

    probe_epochs = {max(1, epochs // 4), max(2, (2 * epochs) // 3)}
    best, best_sd, hist = float("inf"), None, []
    n_nan = 0
    phys_grad_probe = []

    for ep in range(epochs):
        model.train(); fwd.train()
        for scene, meas in tl:
            scene, meas = scene.to(device), meas.to(device)
            B = scene.shape[0]
            t = torch.randint(0, diffusion.n_steps, (B,), device=device)
            noise = torch.randn_like(scene)
            x_t, _ = diffusion.q_sample(scene, t, noise)
            eps_pred = model(x_t, t, meas)

            if published_arm:
                occ_w = 1.0 + scene[:, 0:1]                   # published ablation
                loss = (occ_w * (eps_pred - noise).pow(2)).mean()
            else:
                loss_diff = F.mse_loss(eps_pred, noise)
                occ_w = 1.0 + (OBS_WEIGHT - 1.0) * scene[:, 0:1]
                loss_weighted = (occ_w * (eps_pred - noise).pow(2)).mean()
                x0_pred = diffusion.predict_x0(x_t, t, eps_pred).clamp(0, 1)
                loss_fwd = F.mse_loss(fwd(scene), meas)
                loss = loss_diff + 0.3 * loss_weighted + 0.1 * loss_fwd
                pt = phys_term(arm, fwd, x0_pred, scene, meas)
                if pt is not None:
                    pw = PHYS_WEIGHT * min(1.0, ep / 30.0)
                    # probe: does this term reach theta at all?
                    if ep in probe_epochs and len(phys_grad_probe) < 2:
                        gg = torch.autograd.grad(pt, list(model.parameters()),
                                                 retain_graph=True, allow_unused=True)
                        nrm = float(np.sqrt(sum(x.pow(2).sum().item()
                                                for x in gg if x is not None)))
                        phys_grad_probe.append({"epoch": ep, "grad_norm_theta": nrm,
                                                "n_none": sum(x is None for x in gg),
                                                "n_params": len(gg)})
                    loss = loss + pw * pt

            if not torch.isfinite(loss):
                n_nan += 1
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()

        model.eval()
        v = 0.0
        with torch.no_grad():
            for scene, meas in vl:
                scene, meas = scene.to(device), meas.to(device)
                t = torch.randint(0, diffusion.n_steps, (scene.shape[0],), device=device)
                noise = torch.randn_like(scene)
                x_t, _ = diffusion.q_sample(scene, t, noise)
                v += F.mse_loss(model(x_t, t, meas), noise).item()
        v /= max(1, len(vl))
        hist.append(v)
        if v < best:
            best = v
            best_sd = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_sd)
    return model, {"best_val": best, "val_hist": hist, "n_nonfinite_steps": n_nan,
                   "phys_grad_probe": phys_grad_probe}


def evaluate(model, te_s, te_m, device):
    diffusion = DiffusionV2(n_steps=300, device=device)
    res = {}
    for setting, mm in make_variants(te_m).items():
        loader = DataLoader(SimpleDS(te_s, mm), batch_size=8, shuffle=False, num_workers=0)
        np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        pred, gt = infer_diffusion(model, diffusion, loader, n_steps=50,
                                   n_samples=4, eta=0.3, device=device)
        res[setting] = {k: float(v) for k, v in evaluate_full(pred, gt).items()}
    drops = [res["real_Y"]["IoU"] - res[s]["IoU"]
             for s in ("zero_Y", "shuffled_Y", "spatial_perm_Y")]
    res["utilization"] = {"max_IoU_drop": float(max(drops)),
                          "shuffled_IoU_drop": float(drops[1]),
                          "uses_Y": bool(max(drops) > 0.05)}
    return res


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "RADIal"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    device = torch.device("cuda")
    save = os.path.join(RESULTS_DIR, f"physics_controlled_{name}.json")

    print(f"Loading {name} ...")
    data = build_split(name)
    print(f"  train={len(data[0])} val={len(data[2])} test={len(data[4])} "
          f"| {len(ARMS)} arms x {n_seeds} seeds x {epochs} epochs")

    out = {"config": {"dataset": name, "n_seeds": n_seeds, "epochs": epochs,
                      "batch": BATCH, "lr": LR, "weight_decay": WD,
                      "obs_weight": OBS_WEIGHT, "phys_weight": PHYS_WEIGHT,
                      "n_train": len(data[0]), "n_val": len(data[2]),
                      "n_test": len(data[4])},
           "arms": {}}
    if os.path.exists(save):
        out = json.load(open(save))

    for arm in ARMS:
        out["arms"].setdefault(arm, {})
        for si in range(n_seeds):
            seed = 42 + si
            key = f"seed{seed}"
            if key in out["arms"][arm]:
                print(f"  [{arm}/{key}] cached, skipping")
                continue
            t0 = time.time()
            model, tinfo = train_arm(arm, seed, data, epochs, device)
            ev = evaluate(model, data[4], data[5], device)
            out["arms"][arm][key] = {"train": tinfo, "eval": ev,
                                     "wall_s": time.time() - t0}
            u = ev["utilization"]
            print(f"  [{arm}/{key}] IoU={ev['real_Y']['IoU']:.4f} "
                  f"ECE={ev['real_Y']['ECE']:.4f} "
                  f"util_drop={u['max_IoU_drop']:+.4f} "
                  f"uses_Y={u['uses_Y']} "
                  f"({tinfo['n_nonfinite_steps']} nonfinite) "
                  f"[{time.time()-t0:.0f}s]")
            del model; _clear_gpu(); gc.collect()
            json.dump(out, open(save, "w"), indent=2)

    # summary
    print(f"\n{'='*88}\n{name}: {epochs} epochs, {n_seeds} seeds\n{'='*88}")
    print(f"{'arm':<16}{'IoU mean':>11}{'sd':>8}{'ECE':>9}{'util drop':>12}{'phys grad':>12}")
    summ = {}
    for arm in ARMS:
        rs = out["arms"].get(arm, {})
        if not rs:
            continue
        iou = [r["eval"]["real_Y"]["IoU"] for r in rs.values()]
        ece = [r["eval"]["real_Y"]["ECE"] for r in rs.values()]
        ut = [r["eval"]["utilization"]["max_IoU_drop"] for r in rs.values()]
        pg = [p["grad_norm_theta"] for r in rs.values()
              for p in r["train"].get("phys_grad_probe", [])]
        summ[arm] = {"IoU_mean": float(np.mean(iou)), "IoU_std": float(np.std(iou, ddof=1)) if len(iou) > 1 else 0.0,
                     "ECE_mean": float(np.mean(ece)),
                     "util_drop_mean": float(np.mean(ut)),
                     "phys_grad_theta_mean": float(np.mean(pg)) if pg else None,
                     "n": len(iou)}
        s = summ[arm]
        print(f"{arm:<16}{s['IoU_mean']:>11.4f}{s['IoU_std']:>8.4f}"
              f"{s['ECE_mean']:>9.4f}{s['util_drop_mean']:>+12.4f}"
              f"{('%.3e' % s['phys_grad_theta_mean']) if s['phys_grad_theta_mean'] is not None else 'n/a':>12}")
    out["summary"] = summ
    json.dump(out, open(save, "w"), indent=2)
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
