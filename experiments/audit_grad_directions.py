"""
Quantify how the three RADIal training configurations differ in the gradient
signal that actually reaches the diffusion model.

Effective theta-objectives (after removing terms whose gradient w.r.t. theta is
structurally zero, verified in audit_physics_loss.py T1):

  full        = L_diff + 0.3*L_weighted  = 1.3*MSE + 0.3*Occ
  no_physics  = L_weighted               = 1.0*MSE + 1.0*Occ
  no_obsw     = L_diff                   = 1.0*MSE + 0.0*Occ

where Occ = mean(occ * (eps_pred - noise)^2).
"""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CKPT_DIR, RESULTS_DIR
from s2s_diff_v2 import PixelDiffusionUNet, DiffusionV2, SimpleDS, _try_load
from torch.utils.data import DataLoader
from sanity_measurement_utilization import load_radial

dev = torch.device("cuda")
scenes, meas, ck = load_radial()
model = PixelDiffusionUNet(base_ch=96).to(dev)
_try_load(model, os.path.join(CKPT_DIR, "v2", ck, "diff_v2.pt"), "diff_v2")
model.train()
diff = DiffusionV2(n_steps=300, device=dev)
th = [p for p in model.parameters() if p.requires_grad]
loader = DataLoader(SimpleDS(scenes, meas), batch_size=16, shuffle=False)

def flat(loss, retain=True):
    g = torch.autograd.grad(loss, th, retain_graph=retain, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                      for x, p in zip(g, th)])

res = {k: [] for k in ("cos_full_nophys", "cos_full_noobsw", "cos_nophys_noobsw",
                       "norm_full", "norm_nophys", "norm_noobsw")}
torch.manual_seed(42); np.random.seed(42)
for bi, (scene, m) in enumerate(loader):
    if bi >= 6: break
    scene, m = scene.to(dev), m.to(dev)
    B = scene.shape[0]
    t = torch.randint(0, diff.n_steps, (B,), device=dev)
    noise = torch.randn_like(scene)
    x_t, _ = diff.q_sample(scene, t, noise)
    eps = model(x_t, t, m)
    d2 = (eps - noise).pow(2)
    occ = scene[:, 0:1]
    MSE = d2.mean()
    Occ = (occ * d2).mean()

    g_full   = flat(1.3*MSE + 0.3*Occ)
    g_nophys = flat(1.0*MSE + 1.0*Occ)
    g_noobsw = flat(1.0*MSE + 0.0*Occ, retain=False)

    cs = lambda a, b: float(F.cosine_similarity(a[None], b[None]).item())
    res["cos_full_nophys"].append(cs(g_full, g_nophys))
    res["cos_full_noobsw"].append(cs(g_full, g_noobsw))
    res["cos_nophys_noobsw"].append(cs(g_nophys, g_noobsw))
    res["norm_full"].append(float(g_full.norm()))
    res["norm_nophys"].append(float(g_nophys.norm()))
    res["norm_noobsw"].append(float(g_noobsw.norm()))

print(f"{'quantity':<26}{'mean':>10}{'std':>10}")
for k, v in res.items():
    print(f"{k:<26}{np.mean(v):>10.5f}{np.std(v):>10.5f}")

out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in res.items()}
out["effective_objectives"] = {
    "full":       {"MSE": 1.3, "Occ": 0.3, "Occ_over_MSE": 0.3/1.3},
    "no_physics": {"MSE": 1.0, "Occ": 1.0, "Occ_over_MSE": 1.0},
    "no_obsweight": {"MSE": 1.0, "Occ": 0.0, "Occ_over_MSE": 0.0},
}
out["radial_IoU"] = {"full": 0.38663, "no_physics": 0.30967, "no_obsweight": 0.33239}
p = os.path.join(RESULTS_DIR, "grad_direction_compare.json")
json.dump(out, open(p, "w"), indent=2)
print("\nSaved:", p)
