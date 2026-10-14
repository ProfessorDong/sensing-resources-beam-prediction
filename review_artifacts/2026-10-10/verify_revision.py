"""Independent re-derivation of the claims added or changed on 2026-10-10.

Deliberately does NOT import the analysis it checks. Rebuilds the frontier, the validation
selection, the dedicated-cost margin, the Scenario 33 recording-level tail and the paired
fresh/stale delay comparison from the saved shortlists and the raw CSV and power files.
Exits non-zero on any failure.
"""
import collections, csv, itertools, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "results")
DS = os.path.join(ROOT, "datasets", "deepsense6g")
MAC = (open(os.path.join(ROOT, "paper_b", "numbers.tex")).read()
       + open(os.path.join(ROOT, "paper_b", "numbers_audit.tex")).read())
Q = json.load(open(os.path.join(RES, "beam_power_quality.json")))
TSEC, NB, K5, FS = 5.0, 64, 5, 100.0
CFG = {**{f"chirps={c}": (4, 256, c) for c in (8, 16, 32, 64, 125, 250)},
       **{f"samples={v}": (4, v, 250) for v in (64, 128)},
       **{f"rx={v}": (v, 256, 250) for v in (1, 2)}}
fails = []


def ck(cond, msg, got=None, want=None):
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}" + ("" if cond else f"   got {got} want {want}"))
    if not cond:
        fails.append(msg)


def mac(name):
    """Read one macro's value, balancing braces.

    A naive [^}]* truncates any value containing a group, such as chirps${=}$8, which is
    exactly how this checker first reported a false failure.
    """
    key = "\\newcommand{\\PB" + name + "}{"
    i = MAC.index(key) + len(key)
    depth, j = 1, i
    while depth:
        if MAC[j] == "{":
            depth += 1
        elif MAC[j] == "}":
            depth -= 1
            if not depth:
                break
        j += 1
    return MAC[i:j]


def load(sc):
    rows = list(csv.DictReader(open(os.path.join(
        DS, f"Scenario{sc}", f"scenario{sc}_dev.csv"))))
    n = len(rows); ntr = int(.7 * n); start = ntr + int(.02 * n)
    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    seq = np.array([int(r["seq_index"]) for r in rows])
    idx = np.array([int(r["index"]) for r in rows])
    t = []
    for r in rows:
        hms, fr = r["time_stamp"].split("-"); h, m, s_ = (int(x) for x in hms.split(":"))
        t.append((h * 3600 + m * 60 + s_) * 1e6 + int(fr))
    dt = np.diff(np.array(t, float)) / 1000.
    P = np.array([np.loadtxt(os.path.join(DS, f"Scenario{sc}",
                                          r["unit1_pwr_60ghz"].lstrip("./"))) for r in rows])
    return n, ntr, start, y, seq, idx, dt, P


for sc in ("31", "33"):
    tag = "One" if sc == "31" else "Three"
    print(f"\n=== Scenario {sc} ===")
    n, ntr, start, y, seq, idx, dt, P = load(sc)
    z = np.load(os.path.join(RES, f"beam_shortlists_sc{sc}.npz"))
    rec = Q["scenarios"][sc]
    Pt, yt = P[start:], y[start:]
    fin = np.isfinite(Pt).all(1)

    def reg(sl, mask, k):
        return 10 * np.log10(Pt[mask].max(1) / np.take_along_axis(
            Pt[mask], sl[mask][:, :k].astype(np.int64), 1).max(1))

    # 1. the validation block is the last tenth of the training region, as the trainer slices it
    nval = max(1, ntr // 10)
    ck(np.array_equal(z["val_row_index"], np.arange(ntr - nval, ntr)),
       "the saved validation rows are the trainer's own inner split")
    ck(len(set(z["val_row_index"].tolist()) & set(z["test_row_index"].tolist())) == 0,
       "validation and test rows are disjoint")
    ck(int(z["val_row_index"].max()) < start,
       "no validation row lies in the test block")

    # 2. the frontier: cost and quality rebuilt from scratch for every (setting, k)
    Pv, yv = P[z["val_row_index"]], y[z["val_row_index"]]
    vfin = np.isfinite(Pv).all(1)
    pts = {(r["setting"], r["k"]): r for r in rec["frontier"]["points"]}
    bad = 0
    for lab, (nrx, ns, nc) in CFG.items():
        S = z[f"spatial|{lab}|shortlist"]
        for k in (1, 2, 3, 5, 8, 16):
            r = pts[(lab, k)]
            want_cost = nc * (ns / FS) + k * TSEC
            if abs(r["cost_dedicated_us"] - want_cost) > 1e-9: bad += 1
            if abs(r["cost_reuse_us"] - k * TSEC) > 1e-9: bad += 1
            got = float(np.median([reg(S[i], fin, k).mean() for i in range(S.shape[0])]))
            if abs(got - r["test"]["mean_regret_db"]) > 1e-9: bad += 1
    ck(bad == 0, f"all {len(pts)} frontier points reproduce in cost and test regret", bad, 0)

    # 3. the selection rule, re-applied from the validation numbers alone
    sel = rec["frontier"]["selection"]["dedicated"]
    default = pts[("chirps=250", K5)]
    target = default["val"]["mean_regret_db"]
    ck(abs(sel["validation_target_db"] - target) < 1e-12,
       "the declared target is the default's own validation regret")
    elig = [r for r in pts.values() if r["val"]["mean_regret_db"] <= target + 1e-12]
    pick = min(elig, key=lambda r: (r["cost_dedicated_us"], r["k"]))
    ck((pick["setting"], pick["k"]) == (sel["selected"]["setting"], sel["selected"]["k"]),
       "the dedicated selection re-derives", (pick["setting"], pick["k"]),
       (sel["selected"]["setting"], sel["selected"]["k"]))
    ck(mac(f"QFrSetting{tag}").replace("${=}$", "=") == pick["setting"]
       and int(mac(f"QFrK{tag}")) == pick["k"], "the selection macros match")
    # the rule must be an improvement on test, not merely cheaper
    ck(pick["test"]["mean_regret_db"] < default["test"]["mean_regret_db"]
       and pick["cost_dedicated_us"] < default["cost_dedicated_us"],
       "the selected point is cheaper AND better on test")
    rsel = rec["frontier"]["selection"]["reuse"]
    rpick = min(elig, key=lambda r: (r["cost_reuse_us"], r["k"]))
    ck((rpick["setting"], rpick["k"]) == (rsel["selected"]["setting"], rsel["selected"]["k"]),
       "the reuse selection re-derives")
    ck(rpick["k"] == K5 and rpick["setting"] == "chirps=250",
       "under reuse the rule returns the paper's default", (rpick["setting"], rpick["k"]))

    # 4. the dedicated margin against the fixed prior, on Table III's convention
    prior = [b for b, _ in collections.Counter(y[:ntr].tolist()).most_common(K5)]
    psl = np.tile(np.array(prior), (len(yt), 1))
    ovh = lambda p: K5 * TSEC + (1 - p) * (NB - K5) * TSEC
    p5p = float((psl[:, :K5] == yt[:, None]).any(1).mean())
    for lab, nm in (("chirps=8", "Eight"), ("chirps=250", "Full")):
        S = z[f"spatial|{lab}|shortlist"]
        nc, ns = CFG[lab][2], CFG[lab][1]
        p5r = np.array([(S[i][:, :K5] == yt[:, None]).any(1).mean()
                        for i in range(S.shape[0])])
        d = nc * (ns / FS) + ovh(p5r) - ovh(p5p)
        # the generator prints 3 decimals below 100 us and 0 above, to avoid false precision
        shown = float(mac(f"QMargin{nm}{tag}"))
        tol = 5e-4 if abs(shown) < 100 else 0.5
        ck(abs(d.mean() - shown) < tol,
           f"macro QMargin{nm} (mean over seeds)", round(float(d.mean()), 4), shown)
        ck(int((d < 0).sum()) == int(mac(f"QMargin{nm}NBeat{tag}")),
           f"macro QMargin{nm}NBeat (seeds cheaper than the prior)")
    # the shortest frame on Sc.31 must be the borderline one and the full frame must not be
    if sc == "31":
        ck(int(mac("QMarginEightNBeatOne")) > 0 and int(mac("QMarginFullNBeatOne")) == 0,
           "Sc.31: the shortest frame is borderline and the full frame is not")
        lo, hi = float(mac("QMarginEightBootLoOne")), float(mac("QMarginEightBootHiOne"))
        ck(lo < 0 < hi, "the shortest-frame seed bootstrap straddles break-even", (lo, hi))

    # 5. the recording-level tail
    t_ = rec["tail_by_recording"]
    S250 = z["spatial|chirps=250|shortlist"]
    sq = seq[start:]
    diffs = {}
    for g in sorted(set(sq.tolist())):
        mm = (sq == g) & fin
        if mm.sum() < 20: continue
        r_ = float(np.median([(reg(S250[i], mm, K5) > 3).mean() for i in range(S250.shape[0])]))
        p_ = float((reg(psl, mm, K5) > 3).mean())
        diffs[g] = 100 * (r_ - p_)
    ck(set(diffs) == {r["sequence"] for r in t_["rows"]}, "the same recordings are scored")
    ck(all(abs(diffs[r["sequence"]] - r["difference_pp"]) < 1e-9 for r in t_["rows"]),
       "every per-recording exceedance difference reproduces")
    if sc == "33":
        w = max(diffs, key=lambda g: abs(diffs[g]))
        ck(w == int(mac("QTailSeqThree")), "the most influential recording is named", w)
        mm = fin & (sq != w)
        loo = 100 * (float(np.median([(reg(S250[i], mm, K5) > 3).mean()
                                      for i in range(S250.shape[0])]))
                     - float((reg(psl, mm, K5) > 3).mean()))
        ck(abs(loo - float(mac("QTailLooThree"))) < 5e-3,
           "macro QTailLoo (pooled difference without that recording)", round(loo, 3))
        ck(float(mac("QTailPooledThree")) > 0 > loo,
           "one recording flips the sign of the pooled tail difference")

    # 6. the paired fresh/stale comparison on identical future frames
    lag = max(int(k) for k in rec["delay"])
    ok = np.ones(len(yt), bool); el = np.zeros(len(yt))
    for j in range(len(yt)):
        i0 = start + j
        if i0 + lag >= n: ok[j] = False; continue
        tt = 0.
        for q in range(lag):
            a = i0 + q
            if not (seq[a+1] == seq[a] and idx[a+1] == idx[a]+1 and 0 < dt[a] <= 200):
                ok[j] = False; break
            tt += dt[a]
        el[j] = tt
    valid = ok & fin & np.r_[fin[lag:], np.zeros(lag, bool)]
    jj = np.arange(len(yt)) + lag
    Pl = Pt[jj[valid]]
    rg = lambda sl: 10*np.log10(Pl.max(1)/np.take_along_axis(Pl, sl.astype(np.int64),1).max(1))
    stale = [rg(S250[i][valid][:, :K5]) for i in range(S250.shape[0])]
    fresh = [rg(S250[i][jj[valid]][:, :K5]) for i in range(S250.shape[0])]
    si = np.median([(r <= 1e-12).mean() for r in stale])
    fi = np.median([(r <= 1e-12).mean() for r in fresh])
    ck(abs(fi - float(mac(f"QDelayFreshIncl{tag}"))) < 5e-5, "macro QDelayFreshIncl")
    ck(abs(si - float(mac(f"QDelayTopFive{tag}"))) < 5e-5, "macro QDelayTopFive (stale)")
    ck(fi > si, "the fresh shortlist beats the stale one on identical frames", (fi, si))
    pr = np.median([a.mean() - b.mean() for a, b in zip(stale, fresh)])
    ck(abs(pr - float(mac(f"QDelayPairedReg{tag}"))) < 5e-4, "macro QDelayPairedReg (paired)")
    ck(abs(el[valid].min() - float(mac(f"QDelayMsLo{tag}"))) < 0.6
       and abs(el[valid].max() - float(mac(f"QDelayMsHi{tag}"))) < 0.6,
       "the reported age span matches the eligible group",
       (round(el[valid].min(), 1), round(el[valid].max(), 1)))
    ck(el[valid].max() - el[valid].min() > 50,
       "the lag group spans a range of ages, so it is not a single elapsed time")

# 7. R does not order the full-aperture settings
M = json.load(open(os.path.join(RES, "sc31_bearing_mediation_max.json")))["settings"]
fp = sorted(((v["resource_product_R"], k, v["rmse_deg_median"], v["rmse_deg_sd"])
             for k, v in M.items() if v["n_rx"] == 4))
inv = [(a, b) for a, b in itertools.combinations(fp, 2) if a[0] < b[0] and a[2] < b[2]]
ties = [(a, b) for a, b in itertools.combinations(fp, 2) if a[0] == b[0]]
print("\n=== resource ordering ===")
ck(len(fp) == int(mac("NFullAperture")), "full-aperture setting count")
ck(len(inv) == int(mac("FullInversions")), "strict inversion count", len(inv))
ck(len(inv) >= 1 and len(ties) >= 1,
   "R neither orders the full-aperture settings nor separates the equal-R pair")
ck(abs(abs(ties[0][0][2] - ties[0][1][2]) - float(mac("FullTieGap"))) < 5e-3,
   "macro FullTieGap")

print(f"\n{'FAILED: ' + '; '.join(fails) if fails else 'all independent checks passed'}")
sys.exit(1 if fails else 0)
