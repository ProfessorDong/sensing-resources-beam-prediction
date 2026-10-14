"""
Sensing-aided beam prediction: does the RECONSTRUCTED scene help choose the beam?

WHY THIS IS THE ISAC EXPERIMENT PAPER B NEEDS
---------------------------------------------
Editor points 1 and 9 and R1 #2 all said the paper reports no communication-side
result. The h_beam -> scene direction was tested on Sc.36 and gave a clean null
(radar->radar_hbeam2 = -0.0014, p=0.804). This is the reverse and operationally
meaningful direction: beam sweeping costs airtime, so predicting the best beam from
sensing alone is the actual ISAC value proposition.

THREE FACTS ESTABLISHED BEFORE DESIGNING THIS (2026-09-25)
----------------------------------------------------------
1. `unit1_overall-beam` IS the flat argmax of the 4x64 `unit1_pwr*` stack
   (0-indexed; match = 1.0000 on all 24,799 frames). **h_beam -> beam is therefore
   CIRCULAR.** h_beam must not be an input; it is kept only as a sanity check that
   must score ~100% and is NOT a result.
2. The task is physically well posed, but only WITHIN an array. Compensating for
   unit1's heading (it moves ~18 km), the bearing to unit2 correlates with beam
   index at spearman **+0.827** for array 1 (n=11,194). Pooled over all four arrays
   the correlation collapses to +0.020, because the arrays have different
   boresights. So the 256-way index is not a monotone function of angle.
3. Array occupancy is unbalanced: array 1 wins 14,588 frames, array 3 8,463,
   array 2 1,304, array 4 444.

TASKS -- both reported, because they answer different questions
---------------------------------------------------------------
  A  256-way `overall-beam`. The OPERATIONAL quantity ("which beam do I use?").
     Floor = modal-beam share = 0.1671. Confounded by array selection: a model can
     score by learning "array 1 usually wins" without any angular reasoning, so A
     alone is not evidence of geometric understanding.
  B  64-way beam-within-array, restricted to frames where array 1 wins. Geometrically
     well posed (spearman 0.827 vs bearing). Reported with top-1, within-+-2 beams,
     and angular error, so the result is physically interpretable.

PREDICTORS -- identical readout architecture and budget for every image input
----------------------------------------------------------------------------
  constant      modal beam                     the floor
  radar         raw 4-channel measurement      does sensing predict the beam at all?
  recon         RECONSTRUCTED scene            <-- THE ISAC CLAIM
  gt_scene      ground-truth LiDAR BEV         ceiling on what scene geometry offers
  hbeam         the 256-d power vector         sanity check only (circular, ~100%)

The decisive comparisons are recon vs radar (does reconstruction add value over the
raw measurement, i.e. is the scene a useful intermediate representation?) and
recon vs gt_scene (how much of the achievable information survives reconstruction).
If gt_scene itself is near the floor, the task is not solvable from scene geometry
and any negative result is about the task, not about the reconstruction.

Reconstructions come from the saved `radar`-arm checkpoints of exp_sc36_hbeam
(3 seeds), so reconstruction-seed variance is propagated into the error bars.

Usage:
    python exp_beam_prediction.py --recon-seeds 3 --readout-seeds 3
    python exp_beam_prediction.py --n-train 400 --n-test 150 --epochs 3 --tag _smoke
"""
import os, sys, json, time, copy, gc, argparse, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from s2s_diff_v2 import DiffusionV2, _clear_gpu
from exp_sc36_hbeam import ArmModel, load_cache as load_sc36_cache

SC36_RESULTS = os.path.join(RESULTS_DIR, "sc36_hbeam.json")
PREDICTORS = ["constant", "radar", "radar_shuffled", "recon", "recon_shuffled",
              "gt_scene", "hbeam"]
# *_shuffled feed inputs from OTHER frames (test-time permutation, labels untouched).
# They isolate frame-specific information from global statistics: a readout that scores
# well on shuffled inputs is exploiting the marginal distribution, not this frame's
# evidence. Same logic as the shuffled-Y null in the utilization battery -- and the
# reason that battery caught what Y=0 alone would have missed.


# ----------------------------------------------------------------- readout
class ImgReadout(nn.Module):
    """Same architecture for every image input; only in_ch differs (4 vs 3)."""
    def __init__(self, in_ch, n_class, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base*2, 3, 2, 1), nn.GroupNorm(8, base*2), nn.SiLU(),
            nn.Conv2d(base*2, base*4, 3, 2, 1), nn.GroupNorm(8, base*4), nn.SiLU(),
            nn.Conv2d(base*4, base*4, 3, 2, 1), nn.GroupNorm(8, base*4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base*4, 256), nn.SiLU(), nn.Linear(256, n_class))
    def forward(self, x): return self.net(x)


class VecReadout(nn.Module):
    def __init__(self, in_dim, n_class):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, n_class))
    def forward(self, x): return self.net(x)


def metrics(logits, y, n_class, ordinal):
    p = logits.argmax(1)
    out = {"top1": float((p == y).float().mean())}
    k = min(5, n_class)
    out["top5"] = float((logits.topk(k, 1).indices == y[:, None]).any(1).float().mean())
    if ordinal:
        out["within2"] = float(((p - y).abs() <= 2).float().mean())
        out["mae_beams"] = float((p - y).abs().float().mean())
    return out


def train_readout(X, y, Xte, yte, n_class, seed, epochs, device, ordinal, lr=1e-3, batch=128,
                  return_logits=False):
    torch.manual_seed(seed); np.random.seed(seed)
    img = X.ndim == 4
    model = (ImgReadout(X.shape[1], n_class) if img
             else VecReadout(X.shape[1], n_class)).to(device)
    nv = max(1, len(X) // 10)
    tl = DataLoader(TensorDataset(X[:-nv], y[:-nv]), batch_size=batch, shuffle=True)
    vl = DataLoader(TensorDataset(X[-nv:], y[-nv:]), batch_size=256)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_sd = 0.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in tl:
            loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval(); c = n = 0
        with torch.no_grad():
            for xb, yb in vl:
                c += (model(xb.to(device)).argmax(1).cpu() == yb).sum().item(); n += len(yb)
        if c / n >= best:
            best, best_sd = c / n, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        lg = torch.cat([model(xb.to(device)).cpu() for xb, _ in
                        DataLoader(TensorDataset(Xte, yte), batch_size=256)])
    if return_logits:
        return metrics(lg, yte, n_class, ordinal), float(best), lg
    return metrics(lg, yte, n_class, ordinal), float(best)


@torch.no_grad()
def reconstruct(ckpt, scenes, meas, hbeam, device, batch=16, n_samples=4):
    model = ArmModel("radar").to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    diff = DiffusionV2(n_steps=300, device=device)
    out = []
    for i in range(0, len(scenes), batch):
        mm = torch.from_numpy(meas[i:i+batch]).to(device)
        hh = torch.from_numpy(hbeam[i:i+batch]).to(device)
        cond = model.make_cond(mm, hh)      # the radar arm ignores h; signature needs it
        s = [diff.ddim_sample(model, cond, n_steps=50, eta=0.3) for _ in range(n_samples)]
        out.append(torch.stack(s).mean(0).cpu())
        if (i // batch) % 40 == 0:
            print(f"      recon {i}/{len(scenes)}", flush=True)
    del model; _clear_gpu(); gc.collect()
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--recon-seeds", type=int, default=3)
    ap.add_argument("--readout-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cache", default="",
                    help="cache .npz under experiments/cache (default sc36_stride1.npz)")
    ap.add_argument("--results", default="",
                    help="exp_sc36_hbeam results JSON holding the radar-arm checkpoints")
    a = ap.parse_args()
    device = torch.device("cuda")

    # ---- data: reuse the Sc.36 cache and its contiguous temporal split ----
    tr, te, meta = load_sc36_cache(1, name=a.cache or None)
    tr_s, tr_m, tr_h = tr
    te_s, te_m, te_h = te
    ntr = min(a.n_train, len(tr_s)); nte = min(a.n_test, len(te_s))
    # BOTH subsets are SYSTEMATICALLY sampled across their whole block. Head-truncating
    # the train subset is just as biased as head-truncating the eval subset: unit1 drives
    # ~18 km, so the first N frames are one stretch of route. Concretely, train[:6000]
    # is 54% array 3 while the full train block is 60% array 1 and the test block is
    # 56% array 1 -- head truncation would have induced a train/test shift in which
    # antenna array wins, on top of the geographic bias.
    sel_tr = (np.arange(ntr) * (len(tr_s) / ntr)).astype(int)
    sel_te = (np.arange(nte) * (len(te_s) / nte)).astype(int)
    tr_s, tr_m, tr_h = tr_s[sel_tr], tr_m[sel_tr], tr_h[sel_tr]
    te_s, te_m, te_h = te_s[sel_te], te_m[sel_te], te_h[sel_te]
    print(f"  train subset: systematic stride ~{len(tr_s) and (17359/max(ntr,1)):.1f} "
          f"across the whole train block")

    # ---- labels: overall-beam = flat argmax of the 4x64 pwr stack (verified) ----
    beam_tr = tr_h.reshape(len(tr_h), -1).argmax(1)
    beam_te = te_h.reshape(len(te_h), -1).argmax(1)
    arr_tr, arr_te = beam_tr // 64, beam_te // 64
    print(f"train {ntr}  test {nte}")
    print(f"  array distribution (train): {dict(collections.Counter(arr_tr.tolist()))}")

    tasks = {}
    # Task A: 256-way operational beam
    tasks["A_256way"] = {"n_class": 256, "ordinal": False,
                         "ytr": beam_tr, "yte": beam_te,
                         "mtr": np.ones(ntr, bool), "mte": np.ones(nte, bool)}
    # Task B: 64-way within-array, restricted to ONE array so the beam->angle
    # mapping is consistent. The dominant array is chosen FROM THE TRAINING DATA --
    # it is not always array 1. Globally array 1 wins most frames, but the winner
    # varies along the route (the first 400 train frames are 98% array 3), so
    # hardcoding an array silently empties this task on some subsets.
    dom = collections.Counter(arr_tr.tolist()).most_common(1)[0][0]
    n_dom_tr = int((arr_tr == dom).sum()); n_dom_te = int((arr_te == dom).sum())
    print(f"  dominant array in TRAIN: {dom+1} ({n_dom_tr}/{ntr} frames; "
          f"test {n_dom_te}/{nte})")
    if n_dom_tr >= 200 and n_dom_te >= 50:
        tasks[f"B_array{dom+1}_64way"] = {"n_class": 64, "ordinal": True,
                                          "ytr": beam_tr % 64, "yte": beam_te % 64,
                                          "mtr": arr_tr == dom, "mte": arr_te == dom}
    else:
        print(f"  -> Task B SKIPPED: too few frames for the dominant array "
              f"(need >=200 train / >=50 test)")
    for k, t in list(tasks.items()):
        ytr_m = t["ytr"][t["mtr"]]
        if len(ytr_m) == 0 or t["mte"].sum() == 0:
            print(f"  -> dropping {k}: empty after masking"); tasks.pop(k); continue
        mb = collections.Counter(ytr_m.tolist()).most_common(1)[0][0]
        t["floor"] = float((t["yte"][t["mte"]] == mb).mean())
        t["modal_beam"] = int(mb)
        # THE NO-INFORMATION BOUND. The floor above is the share of the TRAIN-modal beam
        # in the test block -- the right baseline PREDICTOR (no test peeking), but the
        # WRONG reference for information content. Train and test are contiguous blocks of
        # an 18 km drive, so they are different stretches of road with different dominant
        # beams (34 vs 162 on Sc.36 Task A). Any constant predictor can therefore score up
        # to the TEST-modal share -- 0.1876 vs a 0.0631 floor. A predictor must beat THIS
        # to have demonstrated that it uses the input at all. Failing to gate on it is what
        # made the 2026-09-26 grid sweep report a spurious +0.1246 "ceiling".
        yt = t["yte"][t["mte"]]
        cnt = collections.Counter(yt.tolist())
        bmb, bn = cnt.most_common(1)[0]
        t["bound_const"] = bn / len(yt)
        t["bound_class"] = int(bmb)
        print(f"  {k}: train {int(t['mtr'].sum())} test {int(t['mte'].sum())}  "
              f"train-modal beam {mb}  floor = {t['floor']:.4f}")
        print(f"      NO-INFORMATION BOUND (best constant on test, class {bmb}) = "
              f"{t['bound_const']:.4f}  <- beat this or the predictor shows nothing")

    save = os.path.join(RESULTS_DIR, f"beam_prediction{a.tag}.json")
    out = json.load(open(save)) if os.path.exists(save) else {}
    out["config"] = vars(a)
    out.setdefault("cells", {})
    out["tasks"] = {k: {"n_class": v["n_class"], "floor": v["floor"],
                        "modal_beam": v["modal_beam"],
                        "bound_const": v["bound_const"], "bound_class": v["bound_class"],
                        "n_train": int(v["mtr"].sum()), "n_test": int(v["mte"].sum())}
                    for k, v in tasks.items()}
    out["label_definition"] = ("overall-beam = flat argmax of the 4x64 pwr stack "
                               "(verified, match 1.0000) -> h_beam input is CIRCULAR")

    # ---- reconstructions from the saved radar-arm checkpoints ----
    # The checkpoint FILENAMES carry no footprint tag, so the +-32 m and +-100 m runs
    # write to the same paths and the later run silently overwrites the earlier one
    # (it did, on 2026-09-27). Reading checkpoint paths out of the wrong results JSON
    # would therefore load +-100 m weights while believing they are +-32 m. Assert that
    # the JSON we take the paths from was produced from the cache we just loaded.
    res_path = os.path.join(RESULTS_DIR, a.results) if a.results else SC36_RESULTS
    res = json.load(open(res_path))
    want = a.cache or "sc36_stride1.npz"
    got = res.get("config", {}).get("cache") or "sc36_stride1.npz"
    if got != want:
        raise SystemExit(f"cache mismatch: --cache={want} but {os.path.basename(res_path)} "
                         f"was trained on {got}. Checkpoint paths are not tagged by "
                         f"footprint, so this would load the wrong weights.")
    print(f"  checkpoints from {os.path.basename(res_path)} (cache {got})")
    cks = [r["checkpoint"] for k, r in sorted(res["arms"]["radar"].items())][:a.recon_seeds]
    recons = {}
    for i, ck in enumerate(cks):
        cf = os.path.join(RESULTS_DIR, f"recon_cache_seed{i}{a.tag}.npz")
        if os.path.exists(cf):
            z = np.load(cf); recons[i] = (z["tr"], z["te"]); print(f"  recon seed{i}: cached")
            continue
        print(f"  reconstructing with {os.path.basename(ck)} ...")
        t0 = time.time()
        rtr = reconstruct(ck, tr_s, tr_m, tr_h, device)
        rte = reconstruct(ck, te_s, te_m, te_h, device)
        np.savez_compressed(cf, tr=rtr, te=rte)
        recons[i] = (rtr, rte)
        print(f"    done in {(time.time()-t0)/60:.1f} min")

    # ---- inputs ----
    perm_rng = np.random.RandomState(7)
    def inputs(ri):
        rtr, rte = recons[ri]
        pt = perm_rng.permutation(len(te_m))          # test-time permutation only
        return {"radar":          (tr_m, te_m),
                "radar_shuffled": (tr_m, te_m[pt]),
                "recon":          (rtr, rte),
                "recon_shuffled": (rtr, rte[pt]),
                "gt_scene":       (tr_s, te_s),
                "hbeam":          (tr_h, te_h)}

    for tname, t in tasks.items():
        for pred in PREDICTORS:
            if pred == "constant":
                out["cells"][f"{tname}/constant"] = {"top1": t["floor"], "note": "modal beam"}
                continue
            for ri in range(len(recons) if pred.startswith("recon") else 1):
                for rs in range(a.readout_seeds):
                    key = f"{tname}/{pred}/recon{ri}/readout{rs}"
                    if key in out["cells"]:
                        continue
                    Xtr_np, Xte_np = inputs(ri)[pred]
                    Xtr = torch.from_numpy(np.ascontiguousarray(Xtr_np[t["mtr"]])).float()
                    Xte = torch.from_numpy(np.ascontiguousarray(Xte_np[t["mte"]])).float()
                    ytr = torch.from_numpy(t["ytr"][t["mtr"]]).long()
                    yte = torch.from_numpy(t["yte"][t["mte"]]).long()
                    t0 = time.time()
                    mt, bv = train_readout(Xtr, ytr, Xte, yte, t["n_class"], 42+rs,
                                           a.epochs, device, t["ordinal"])
                    out["cells"][key] = {**mt, "best_val": bv, "wall_s": time.time()-t0}
                    extra = (f" within2={mt['within2']:.4f} mae={mt['mae_beams']:.2f}"
                             if t["ordinal"] else "")
                    print(f"  [{key:<44}] top1={mt['top1']:.4f} top5={mt['top5']:.4f}{extra}")
                    _clear_gpu(); gc.collect()
                    json.dump(out, open(save, "w"), indent=2)

    # ---- summary ----
    print(f"\n{'='*104}")
    for tname, t in tasks.items():
        print(f"\nTASK {tname}   floor (modal beam) = {t['floor']:.4f}   "
              f"n_class={t['n_class']}  test n={int(t['mte'].sum())}")
        print(f"  {'predictor':<12}{'top1':>9}{'sd':>8}{'top5':>9}"
              f"{'within2':>10}{'mae':>8}   vs floor")
        agg = {}
        for pred in PREDICTORS:
            if pred == "constant":
                print(f"  {'constant':<12}{t['floor']:>9.4f}{'--':>8}{'--':>9}{'--':>10}{'--':>8}"
                      f"      ---"); continue
            rs = [v for k, v in out["cells"].items()
                  if k.startswith(f"{tname}/{pred}/")]
            if not rs: continue
            t1 = [r["top1"] for r in rs]; t5 = [r["top5"] for r in rs]
            w2 = [r.get("within2", np.nan) for r in rs]
            ma = [r.get("mae_beams", np.nan) for r in rs]
            sd = float(np.std(t1, ddof=1)) if len(t1) > 1 else 0.0
            agg[pred] = {"top1": float(np.mean(t1)), "top1_sd": sd,
                         "top5": float(np.mean(t5)),
                         "within2": float(np.nanmean(w2)), "mae_beams": float(np.nanmean(ma)),
                         "n": len(t1)}
            g = agg[pred]
            print(f"  {pred:<12}{g['top1']:>9.4f}{sd:>8.4f}{g['top5']:>9.4f}"
                  f"{g['within2']:>10.4f}{g['mae_beams']:>8.2f}   {g['top1']-t['floor']:+.4f}")
        out.setdefault("summary", {})[tname] = agg
        # unpaired Welch tests across readout/recon seeds on the decisive contrasts
        try:
            from scipy import stats as st
            def cmp(pa, pb, label):
                A = [r["top1"] for k, r in out["cells"].items() if k.startswith(f"{tname}/{pa}/")]
                B = [r["top1"] for k, r in out["cells"].items() if k.startswith(f"{tname}/{pb}/")]
                if len(A) < 2 or len(B) < 2:
                    print(f"    {label}: need >=2 runs each (have {len(A)},{len(B)})"); return
                t, pv = st.ttest_ind(A, B, equal_var=False)
                d = np.mean(A) - np.mean(B)
                print(f"    {label:<34} delta={d:+.4f}  t={t:+.2f}  p={pv:.4f}  "
                      f"n={len(A)}v{len(B)}")
                out.setdefault("contrasts", {}).setdefault(tname, {})[label] = {
                    "delta": float(d), "t": float(t), "p": float(pv),
                    "nA": len(A), "nB": len(B)}
            print("\n  decisive contrasts (Welch, across seeds):")
            cmp("recon", "radar", "recon vs radar (does recon help?)")
            cmp("recon", "recon_shuffled", "recon vs recon_shuffled (frame-specific?)")
            cmp("radar", "radar_shuffled", "radar vs radar_shuffled (frame-specific?)")
            cmp("gt_scene", "recon", "gt_scene vs recon (information lost)")
        except Exception as e:
            print(f"    (stats skipped: {e})")
        # Everything below is expressed as a gain OVER THE MODAL-BEAM FLOOR. Raw top1
        # ratios are meaningless here: if two predictors both collapse onto the modal
        # class they "agree" at 100% while carrying zero information. The smoke run
        # printed exactly that ("recon reaches 100.0% of the ceiling" with both at the
        # floor), so the ratio is now gated on the ceiling actually clearing the floor.
        fl = t["floor"]; bd = t["bound_const"]
        gain = lambda k: agg[k]["top1"] - fl if k in agg else None
        print(f"\n  gains over the train-modal floor ({fl:.4f}) and over the "
              f"NO-INFORMATION bound ({bd:.4f}):")
        print(f"    {'predictor':<16}{'top1':>8}{'vs floor':>10}{'vs BOUND':>10}   verdict")
        for k in ("radar", "radar_shuffled", "recon", "recon_shuffled", "gt_scene", "hbeam"):
            if k in agg:
                vb = agg[k]["top1"] - bd
                print(f"    {k:<16}{agg[k]['top1']:>8.4f}{gain(k):>+10.4f}{vb:>+10.4f}   "
                      + ("uses the input" if vb > 0.01 else
                         "NO demonstrated information -- a constant scores this"))
        if "recon" in agg and "radar" in agg:
            d = gain("recon") - gain("radar")
            print(f"  KEY: recon - radar = {d:+.4f}  -> reconstruction "
                  f"{'ADDS' if d > 0 else 'adds NOTHING'} over the raw measurement")
        if "recon" in agg and "gt_scene" in agg:
            cg = agg["gt_scene"]["top1"] - bd
            if cg > 0.01:
                print(f"  recon recovers "
                      f"{100*(agg['recon']['top1']-bd)/cg:.1f}% of the achievable gain "
                      f"(ceiling gt_scene {cg:+.4f} over the no-information bound)")
            else:
                print(f"  ceiling gt_scene is only {cg:+.4f} over the NO-INFORMATION bound "
                      f"-> BEV geometry does not predict the beam here, and no recon number "
                      f"in this table is interpretable")
        for sp, base in (("recon_shuffled", "recon"), ("radar_shuffled", "radar")):
            if sp in agg and base in agg:
                print(f"  {base} frame-specific gain over shuffled: "
                      f"{agg[base]['top1']-agg[sp]['top1']:+.4f} "
                      f"(shuffled {gain(sp):+.4f} over floor)")
        if "hbeam" in agg:
            # h_beam -> beam is circular (the label IS the argmax of h_beam, verified at
            # match 1.0000 on all 24,799 frames). A low score therefore never means the
            # label is wrong; it means the readout cannot express the argmax, which bounds
            # what every other predictor in the table could have reached.
            h = agg["hbeam"]["top1"]
            print(f"  instrument bound: hbeam top1={h:.4f} "
                  + ("OK (recovers the verified argmax)" if h > 0.95 else
                     "-- the readout, not the label, is the limit; every row below is "
                     "bounded by this"))
    json.dump(out, open(save, "w"), indent=2)
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
