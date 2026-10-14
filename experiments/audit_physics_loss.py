"""
Rigorous audit of the training-time physics-consistency mechanism.

The manuscript's central design claim (Remark 1, Theorem 1, Sec. VIII) is that
S2S-Diff enforces measurement consistency on the DIFFUSION MODEL at training
time via L_phys, and that this yields +7.7 pp IoU on RADIal.

This script tests that claim six ways, on real data, with the canonical
checkpoints:

  T1  Gradient attribution. Replicates the exact training step and measures
      ||d L_term / d theta|| for every loss term, separately for the diffusion
      U-Net (theta) and the forward surrogate (phi). This is decisive: if
      L_phys yields zero gradient w.r.t. theta, it cannot constrain the
      diffusion model at all.

  T2  Effective-objective comparison. The full run and the "no physics"
      ablation are compared on the terms that actually reach theta, to see
      whether the ablation is a controlled comparison.

  T3  Surrogate fidelity. Is H~(X_gt) ~= Y? Compared against the trivial
      baseline of predicting the dataset-mean measurement.

  T4  Surrogate input sensitivity. Does H~ depend on its input at all?

  T5  Impulse response / receptive field. A real radar operator maps a BEV
      cell to a range bin determined by its distance to the sensor -- a global
      geometric transform (Eq. 6-7 defines K in R^{Ntau*Nphi x HW}). Measures
      what the learned surrogate actually does.

  T6  Gradient-clipping confound. clip_grad_norm_ is applied over
      model+fwd params jointly in the full run but model-only in the ablation.

Output: experiments/results/physics_loss_audit.json
"""
import os, sys, json, gc
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CKPT_DIR
from s2s_diff_v2 import (
    PixelDiffusionUNet, DiffusionV2, SimpleForwardModel, SimpleDS,
    _try_load, _clear_gpu,
)
from torch.utils.data import DataLoader
from sanity_measurement_utilization import load_isac, load_v2i, load_radial

V2_CKPT = os.path.join(CKPT_DIR, "v2")
SAVE = os.path.join(RESULTS_DIR, "physics_loss_audit.json")
SEED = 42
N_BATCH = 8          # batches averaged for gradient statistics
BATCH = 16
OBS_WEIGHT = 2.0     # as used in train_s2s_v2
PHYS_WEIGHT = 0.1    # phys_weight after ramp (epoch >= 30)


def gnorm(loss, params, retain=True):
    """||d loss / d params||_2 ; returns 0.0 if the graph is severed."""
    grads = torch.autograd.grad(loss, params, retain_graph=retain,
                                allow_unused=True, create_graph=False)
    tot, n_none = 0.0, 0
    for g in grads:
        if g is None:
            n_none += 1
            continue
        tot += g.pow(2).sum().item()
    return float(np.sqrt(tot)), n_none, len(grads)


def audit(name, loader_fn, device):
    print(f"\n{'='*72}\nDATASET: {name}\n{'='*72}")
    scenes, meas, ckpt_name = loader_fn()
    ckpt_dir = os.path.join(V2_CKPT, ckpt_name)

    model = PixelDiffusionUNet(base_ch=96).to(device)
    fwd = SimpleForwardModel().to(device)
    ok_m = _try_load(model, os.path.join(ckpt_dir, "diff_v2.pt"), "diff_v2")
    ok_f = _try_load(fwd, os.path.join(ckpt_dir, "fwd_v2.pt"), "fwd_v2")
    if not (ok_m and ok_f):
        print("  !! checkpoints missing, skipping")
        return None
    model.train(); fwd.train()
    diffusion = DiffusionV2(n_steps=300, device=device)

    th = [p for p in model.parameters() if p.requires_grad]
    ph = [p for p in fwd.parameters() if p.requires_grad]
    print(f"  theta params: {sum(p.numel() for p in th):,}   "
          f"phi params: {sum(p.numel() for p in ph):,}")

    loader = DataLoader(SimpleDS(scenes, meas), batch_size=BATCH,
                        shuffle=False, num_workers=0)

    # ---------------------------------------------------------------- T1
    acc = {k: [] for k in ("diff_th", "weighted_th", "phys_th", "fwd_th",
                           "phys_ph", "fwd_ph", "total_th",
                           "phys_val", "fwd_val", "diff_val")}
    torch.manual_seed(SEED); np.random.seed(SEED)
    for bi, (scene, m) in enumerate(loader):
        if bi >= N_BATCH:
            break
        scene, m = scene.to(device), m.to(device)
        B = scene.shape[0]
        t = torch.randint(0, diffusion.n_steps, (B,), device=device)
        noise = torch.randn_like(scene)
        x_t, _ = diffusion.q_sample(scene, t, noise)

        # --- exact replication of train_s2s_v2 inner loop ---
        eps_pred = model(x_t, t, m)
        loss_diff = F.mse_loss(eps_pred, noise)

        x0_pred = diffusion.predict_x0(x_t, t, eps_pred).clamp(0, 1)
        meas_pred = fwd(x0_pred.detach())
        meas_gt_proxy = fwd(scene.detach())
        loss_phys = F.mse_loss(meas_pred, meas_gt_proxy.detach())

        meas_from_gt = fwd(scene)
        loss_fwd = F.mse_loss(meas_from_gt, m)

        occ_weight = 1.0 + (OBS_WEIGHT - 1.0) * scene[:, 0:1]
        loss_weighted = (occ_weight * (eps_pred - noise).pow(2)).mean()
        # ----------------------------------------------------

        d_th, _, _ = gnorm(loss_diff, th)
        w_th, _, _ = gnorm(loss_weighted, th)
        p_th, p_none, p_tot = gnorm(loss_phys, th)
        f_th, f_none, f_tot = gnorm(loss_fwd, th)
        p_ph, _, _ = gnorm(loss_phys, ph)
        f_ph, _, _ = gnorm(loss_fwd, ph)

        total = loss_diff + 0.3 * loss_weighted + PHYS_WEIGHT * loss_phys + 0.1 * loss_fwd
        t_th, _, _ = gnorm(total, th, retain=False)

        acc["diff_th"].append(d_th);     acc["weighted_th"].append(w_th)
        acc["phys_th"].append(p_th);     acc["fwd_th"].append(f_th)
        acc["phys_ph"].append(p_ph);     acc["fwd_ph"].append(f_ph)
        acc["total_th"].append(t_th)
        acc["phys_val"].append(loss_phys.item())
        acc["fwd_val"].append(loss_fwd.item())
        acc["diff_val"].append(loss_diff.item())

        if bi == 0:
            print(f"  [T1] theta tensors with grad=None -- "
                  f"L_phys: {p_none}/{p_tot}   L_fwd: {f_none}/{f_tot}")

    T1 = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
              "max": float(np.max(v))} for k, v in acc.items()}
    print(f"  [T1] ||dL/dtheta||   L_diff={T1['diff_th']['mean']:.4e}  "
          f"L_weighted={T1['weighted_th']['mean']:.4e}")
    print(f"  [T1] ||dL/dtheta||   L_phys={T1['phys_th']['max']:.4e}  "
          f"L_fwd={T1['fwd_th']['max']:.4e}   <-- must be 0 to matter")
    print(f"  [T1] ||dL/dphi||     L_phys={T1['phys_ph']['mean']:.4e}  "
          f"L_fwd={T1['fwd_ph']['mean']:.4e}")
    print(f"  [T1] loss values     L_diff={T1['diff_val']['mean']:.4e}  "
          f"L_phys={T1['phys_val']['mean']:.4e}  L_fwd={T1['fwd_val']['mean']:.4e}")

    # ---------------------------------------------------------------- T3/T4
    fwd.eval()
    with torch.no_grad():
        sc = torch.from_numpy(scenes[:256]).float().to(device)
        ms = torch.from_numpy(meas[:256]).float().to(device)
        pred = fwd(sc)
        mean_meas = ms.mean(dim=0, keepdim=True).expand_as(ms)

        nmse_fwd = (F.mse_loss(pred, ms) / ms.pow(2).mean()).item()
        nmse_mean = (F.mse_loss(mean_meas, ms) / ms.pow(2).mean()).item()
        corr = float(np.corrcoef(pred.cpu().numpy().ravel(),
                                 ms.cpu().numpy().ravel())[0, 1])

        # T4: sensitivity of H~ to its input
        zero_in = fwd(torch.zeros_like(sc))
        perm = torch.randperm(sc.shape[0], device=device)
        shuf = fwd(sc[perm])
        var_across = pred.var(dim=0).mean().item()
        energy = pred.pow(2).mean().item()
        d_zero = (pred - zero_in).pow(2).mean().sqrt().item()
        d_shuf = (pred - shuf).pow(2).mean().sqrt().item()
        rms = pred.pow(2).mean().sqrt().item()

    T3 = {"nmse_surrogate": nmse_fwd, "nmse_mean_baseline": nmse_mean,
          "corr_surrogate_vs_Y": corr,
          "beats_mean_baseline": bool(nmse_fwd < nmse_mean)}
    T4 = {"output_rms": rms,
          "var_across_scenes_over_energy": var_across / max(energy, 1e-12),
          "rel_change_zero_input": d_zero / max(rms, 1e-12),
          "rel_change_shuffled_input": d_shuf / max(rms, 1e-12)}
    print(f"  [T3] NMSE surrogate={nmse_fwd:.4f}  mean-baseline={nmse_mean:.4f}"
          f"  beats={T3['beats_mean_baseline']}   corr={corr:.4f}")
    print(f"  [T4] rel change on zero input={T4['rel_change_zero_input']:.4f}"
          f"   on shuffled input={T4['rel_change_shuffled_input']:.4f}")

    # ---------------------------------------------------------------- T5
    with torch.no_grad():
        H, W = sc.shape[-2:]
        delta = torch.zeros(1, 3, H, W, device=device)
        base = fwd(delta)
        delta[0, :, H // 2, W // 2] = 1.0
        resp = (fwd(delta) - base).abs().sum(1)[0]          # [H,W]
        tot = resp.sum().item()
        if tot > 0:
            ys, xs = torch.meshgrid(torch.arange(H, device=device),
                                    torch.arange(W, device=device), indexing="ij")
            cy = (resp * ys).sum().item() / tot
            cx = (resp * xs).sum().item() / tot
            spread = float(torch.sqrt(
                (resp * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum() / tot).item())
            nz = int((resp > 1e-6 * resp.max()).sum().item())
        else:
            cy = cx = spread = float("nan"); nz = 0
    T5 = {"impulse_at": [H // 2, W // 2], "response_centroid": [cy, cx],
          "response_rms_spread_px": spread, "n_cells_responding": nz,
          "grid_cells": H * W,
          "theoretical_receptive_field": "7x7 (three 3x3 convs, no downsampling)"}
    print(f"  [T5] impulse at ({H//2},{W//2}) -> centroid ({cy:.2f},{cx:.2f}), "
          f"rms spread {spread:.2f}px, {nz}/{H*W} cells respond")

    del model, fwd, diffusion
    _clear_gpu(); gc.collect()
    return {"T1_gradient_attribution": T1, "T3_surrogate_fidelity": T3,
            "T4_surrogate_sensitivity": T4, "T5_impulse_response": T5}


def main():
    device = torch.device("cuda")
    out = {}
    for name, fn in [("DeepSense_ISAC", load_isac),
                     ("DeepSense_V2I", load_v2i),
                     ("RADIal", load_radial)]:
        try:
            r = audit(name, fn, device)
            if r:
                out[name] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            out[name] = {"error": str(e)}

    # ------------------------------------------------------------ T2 / T6
    # Terms that actually reach theta, full run vs no-physics ablation.
    # full:      L_diff + 0.3*L_weighted,  occ_weight = 1 + (2.0-1)*occ
    # ablation:  L_weighted_abl,           occ_w      = 1 + occ
    # Expand both as a*MSE + b*mean(occ*(eps-noise)^2):
    #   full     = 1.0*MSE + 0.3*(MSE + mean(occ*d^2)) = 1.3*MSE + 0.3*occ-term
    #   ablation = 1.0*MSE + 1.0*occ-term
    out["T2_effective_objective"] = {
        "full_model_facing_terms": "L_diff + 0.3*L_weighted",
        "ablation_model_facing_terms": "L_weighted (obs_weight fixed at 2.0)",
        "full_expanded": {"MSE_coeff": 1.3, "occ_term_coeff": 0.3,
                          "occ_rel_to_MSE": 0.3 / 1.3},
        "ablation_expanded": {"MSE_coeff": 1.0, "occ_term_coeff": 1.0,
                              "occ_rel_to_MSE": 1.0},
        "occ_weighting_ratio_ablation_over_full": 1.0 / (0.3 / 1.3),
        "loss_scale_ratio_full_over_ablation": 1.3,
        "note": ("The 'no physics' ablation differs from the full run in the "
                 "relative strength of the occupancy-weighted term (4.33x) and "
                 "in overall loss scale (1.3x), independent of any physics term."),
    }
    out["T6_clipping_confound"] = {
        "full_run": "clip_grad_norm_(model.params + fwd.params, 1.0)",
        "ablation": "clip_grad_norm_(model.params, 1.0)",
        "note": ("Joint clipping renormalizes using the combined norm, so the "
                 "diffusion model's effective step differs between the two runs "
                 "whenever the combined norm exceeds 1.0."),
    }

    os.makedirs(os.path.dirname(SAVE), exist_ok=True)
    with open(SAVE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {SAVE}")


if __name__ == "__main__":
    main()
