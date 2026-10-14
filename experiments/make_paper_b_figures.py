#!/usr/bin/env python
"""Paper B figures. TikZ/PGFPlots source compiled to vector PDF.

Follows the Paper A discipline exactly:
  1. NO number is typed in. Every value is read from a JSON under results/ and injected.
     Literals that do appear are ASSERTIONS, so a drifted result file fails the build.
  2. A reference level ABOVE the measurement as well as below it on every accuracy panel. The
     no-information bound is the best constant on the test split. The upper reference is a
     position-based predictor whose family and hyperparameter are chosen on an inner validation
     block, never on test; it is one concrete predictor, not a bound on achievable accuracy.
     Reporting a gain over the lower bound alone overstates how much of the available
     information the radar recovers.
  3. Medians with bootstrap CIs, never means, because the readout collapses onto a constant
     class in a resource-dependent fraction of runs and the distribution is bimodal.
  4. Collapse counts are shown, not hidden.

Outputs into paper_b_figures/: B*.tex, B*.pdf, B*.png (review), figure_data.json, CAPTIONS.md.
Usage: python experiments/make_paper_b_figures.py [--no-compile] [--only B1,B4]
"""
import argparse, json, os, subprocess, sys, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_paper_a_figures import COLORS, PREAMBLE
from audit_readout_spatial import ImgReadoutSpatial as _IRS
NET_PARAMS = f"{sum(q.numel() for q in _IRS(2, 64).parameters() if q.requires_grad):,d}"

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(os.path.dirname(HERE), "paper_b_figures")
os.makedirs(OUT, exist_ok=True)
SEED = 20261004
rng = np.random.default_rng(SEED)


def reseed(tag):
    """Reseed from a per-figure stream.

    A single module-level generator makes every bootstrap depend on which other figures were
    built in the same process, so `--only B2` and a full build gave different intervals. Each
    bootstrap now draws from a stream keyed to its own figure, which makes the numbers a
    function of the data alone.
    """
    global rng
    # zlib.crc32, not hash(): Python randomizes string hashing per process, so hash()
    # would reseed differently on every run.
    rng = np.random.default_rng(zlib.crc32(tag.encode()) ^ SEED)
_c = {}


def J(name):
    if name not in _c:
        _c[name] = json.load(open(os.path.join(RES, name)))
    return _c[name]


def approx(a, b, tol=6e-4):
    assert abs(a - b) <= tol, f"ASSERTION FAILED: {a} vs {b} (tol {tol})"
    return a


def _sci(v, sig=1):
    """LaTeX scientific notation; Python's .1e renders as "7.9e-05" in a figure."""
    import math
    e = int(math.floor(math.log10(abs(v))))
    return rf"{v/10**e:.{sig}f}\times 10^{{{e}}}"


def bootci(x, n=20000):
    x = np.asarray(x, float)
    b = np.array([np.median(rng.choice(x, x.size, True)) for _ in range(n)])
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def write_tex(stem, body):
    p = os.path.join(OUT, f"{stem}.tex")
    open(p, "w").write(PREAMBLE + "\n\\begin{document}\n" + body + "\n\\end{document}\n")
    return p


def compile_tex(stem):
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            f"{stem}.tex"], cwd=OUT, capture_output=True, text=True)
        if r.returncode != 0:
            log = os.path.join(OUT, f"{stem}.log")
            msg = ""
            if os.path.exists(log):
                L = open(log, errors="replace").read().splitlines()
                h = [i for i, l in enumerate(L) if l.startswith("!")]
                msg = "\n".join(L[h[0]:h[0] + 18]) if h else "\n".join(L[-18:])
            raise RuntimeError(f"pdflatex failed on {stem}\n{msg}")
    subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile",
                    f"{stem}.pdf", stem], cwd=OUT, capture_output=True)
    for e in ("aux", "log"):
        f = os.path.join(OUT, f"{stem}.{e}")
        if os.path.exists(f):
            os.remove(f)


# ----------------------------------------------------------------- B1
def build_b1():
    """Chirp dose-response on both scenarios, with bound and position reference."""
    reseed("B1")
    out = {}
    CB = J("corrected_statistics.json")["clears_bound"]
    for tag, f in (("sc31", "sc31_resource_ablation_G_max.json"),
                   ("sc33", "sc33_resource_ablation_G_max.json")):
        d = J(f); S = d["settings"]
        assert d["pool"] == "max" and d["seeds"] == 15, (d.get("pool"), d["seeds"])
        # Which settings clear the no-information bound is a recorded determination, not
        # something to re-decide here: corrected_statistics.json owns it, under the rule that
        # the 2.5th bootstrap percentile of the median must exceed the bound. The figure reads
        # that list and checks its own bootstrap against it, so the panel cannot show a point
        # as clearing the bound while the statistics file says it does not.
        cb = CB[tag[2:]]
        assert cb["rule"].startswith("2.5th bootstrap percentile"), cb["rule"]
        misses = set(cb["not_clearing"])
        pts = []
        for c in (8, 16, 32, 64, 125, 250):
            key = f"chirps={c}"
            v = S[key]
            m, lo, hi = bootci(v["top1_seeds"])
            approx(m, v["top1_median"], 1e-9)
            clears = key not in misses
            assert clears == (lo > d["bound_const"]), (
                f"{tag} {key}: this figure's bootstrap and corrected_statistics.json "
                f"disagree on whether the setting clears the bound")
            pts.append({"n": c, "med": m, "lo": lo, "hi": hi, "clears": clears,
                        "coll": v["n_collapsed"], "nseed": v["n_seeds"]})
        R = J(f"{tag}_reference_predictors.json")
        g = [v for k, v in R["predictors"].items() if k.startswith("GPS reference")][0]
        best = pts[-1]["med"]
        out[tag] = {"bound": d["bound_const"], "posref": g["top1"],
                    "posref_top5": g["top5"], "posref_note": g["note"], "points": pts,
                    "n_fit": R["n_fit"], "n_val": R["n_val"], "n_test": R["n_test"],
                    "n_miss": len(misses & {f"chirps={c}" for c in (8,16,32,64,125,250)}),
                    # the same quantity as \PBShareGps*, by the same formula
                    "share": 100.0 * (best - d["bound_const"]) / (g["top1"] - d["bound_const"])}
        assert "validation" in g["note"], g["note"]
    approx(out["sc31"]["bound"], 0.0901, 5e-5)
    approx(out["sc33"]["bound"], 0.1371, 5e-5)
    approx(out["sc31"]["posref"], 0.5000, 5e-5)
    approx(out["sc33"]["posref"], 0.4543, 5e-5)
    approx(out["sc31"]["points"][-1]["med"], 0.2357, 5e-5)
    approx(out["sc33"]["points"][-1]["med"], 0.1828, 5e-5)
    approx(out["sc31"]["share"], 35.5, 0.05)
    # THE AXIS LIMITS ARE DERIVED FROM EVERYTHING DRAWN, shared by both panels so the two
    # scenarios stay comparable by eye. They were hardcoded at 0.02 and 0.56, which is the
    # defect the 2026-10-08 Figure 4 audit fixed one figure over: nothing fell outside the
    # box here, but 41 % of the Scenario 33 panel carried no data, and the compression that
    # buys costs the dose-response and the bootstrap intervals their legibility.
    _lo = min(min(p["lo"] for p in out[t]["points"]) for t in out)
    _hi = max(out[t]["posref"] for t in out)
    _bd = min(out[t]["bound"] for t in out)
    # The band under the lowest bound is NOT dead space: the bound's own label hangs below
    # its rule, inside the shading. Tightening ymin to the data pushed that label through
    # the bottom frame on Scenario 31, where the bound is lowest. The allowance is one
    # \scriptsize line plus a margin, about LBL_MM of the PANEL_MM panel, and because the
    # allowance is itself a fraction of the range this solves for ymin rather than guessing:
    #   ymin = bound - k (ymax - ymin),  k = LBL_MM / PANEL_MM.
    # Above the position reference only the rule needs sky, since its label hangs below too.
    LBL_MM, PANEL_MM = 3.1, 24.0
    k = LBL_MM / PANEL_MM
    out["ymax"] = round(_hi + 0.020, 4)
    out["ymin"] = round((_bd - k * out["ymax"]) / (1.0 - k), 4)
    assert out["ymin"] < min(_lo, _bd), (out["ymin"], _lo, _bd)
    out["label_allowance_mm"] = LBL_MM
    for t in ("sc31", "sc33"):
        for q in out[t]["points"]:
            assert out["ymin"] < q["lo"] and q["hi"] < out["ymax"], (t, q)
        assert out["ymin"] < out[t]["bound"] < out[t]["posref"] < out["ymax"], t
    approx(out["sc33"]["share"], 14.4, 0.05)
    assert out["sc31"]["n_miss"] == 0 and out["sc33"]["n_miss"] == 2
    return out


def emit_b1(d):
    # Drawn at the width it is placed at. A tikzpicture narrower than \textwidth is stretched
    # by \includegraphics, which scales the type with it: at 11.7 cm into 18.2 cm this figure
    # was magnified 1.55x and its axis labels rendered at 12.4 pt against 10 pt body copy.
    YMIN, YMAX = d["ymin"], d["ymax"]
    panels = []
    for i, (tag, title) in enumerate((("sc31", "Scenario 31"), ("sc33", "Scenario 33"))):
        z = d[tag]
        pos = "" if i == 0 else (r"at=(ax0.right of south east),"
                                 r"anchor=left of south west,xshift=10mm,")
        co = " ".join(f"({p['n']},{p['med']:.4f})" for p in z["points"])
        eb = "\n".join(f"      ({p['n']},{p['med']:.4f}) += (0,{p['hi']-p['med']:.4f}) "
                       f"-= (0,{p['med']-p['lo']:.4f})" for p in z["points"])
        hit = " ".join(f"({p['n']},{p['med']:.4f})" for p in z["points"] if p["clears"])
        miss = " ".join(f"({p['n']},{p['med']:.4f})" for p in z["points"] if not p["clears"])
        missplot = (rf"""  \addplot[only marks, mark=*, mark size=1.55pt, oiVerm,
           mark options={{fill=white, draw=oiVerm, line width=0.8pt}}, forget plot]
    coordinates {{{miss}}};""" if miss else "")
        # the collapse count goes above the upper whisker, except where the point sits high
        # enough that the label would run into the position reference
        lbl = "\n".join(
            rf"  \node[font=\scriptsize,gGrey,anchor=south,inner sep=2pt] "
            rf"at (axis cs:{p['n']},{p['hi']:.4f}) {{{p['coll']}/{p['nseed']}}};"
            for p in z["points"] if p["coll"] > 0)
        panels.append(rf"""
\begin{{axis}}[name=ax{i},{pos}
  scale only axis, width=70mm, height=24mm, xmode=log, log basis x=2,
  xmin=6.4, xmax=330, ymin={YMIN}, ymax={YMAX},
  xtick={{8,16,32,64,125,250}}, xticklabels={{8,16,32,64,125,250}},
  ytick={{0.1,0.2,0.3,0.4,0.5}},
  xlabel={{chirps per frame, the sensing airtime}},
  ylabel={{{'top-1 beam accuracy' if i == 0 else ''}}},
  label style={{font=\footnotesize}}, tick label style={{font=\footnotesize}},
  title={{\small\textbf{{{title}}}}},
  clip=false]
  \fill[gLight,opacity=0.45] (axis cs:6.4,{YMIN}) rectangle (axis cs:330,{z['bound']:.4f});
  \addplot[draw=gGrey, dashed, line width=0.6pt, forget plot]
    coordinates {{(6.4,{z['bound']:.4f}) (330,{z['bound']:.4f})}};
  \node[font=\scriptsize,gGrey,anchor=north east,inner sep=1.5pt]
    at (axis cs:330,{z['bound']:.4f}) {{no-information bound {z['bound']:.4f}}};
  \addplot[draw=goodLine, dotted, line width=0.8pt, forget plot]
    coordinates {{(6.4,{z['posref']:.4f}) (330,{z['posref']:.4f})}};
  \node[font=\scriptsize,goodLine,anchor=north west,inner sep=1.5pt]
    at (axis cs:6.6,{z['posref']:.4f}) {{position reference {z['posref']:.4f}}};
  \node[font=\scriptsize,gGrey,anchor=north,align=center,inner sep=1.5pt]
    at (axis cs:45,{z['posref'] - 0.055:.4f})
    {{the radar recovers {z['share']:.1f}\,\% of the gain\\the position reference supplies}};
  %% THE CAP IS SET ON THE PLOT, NOT INSIDE error bars/.cd. `error mark` and
  %% `error mark options` given after `error bars/.cd` are accepted and then silently
  %% ignored by this pgfplots: measured on the shipped figure, the whisker ends were
  %% 0.17 mm wide, which is the error bar's own line width, not a 2 pt dash. The error
  %% mark inherits the PLOT's mark, so `mark=-` here is what draws it. The dash is 1.4 mm
  %% wide against a 1.34 mm marker, so an interval shorter than the marker still shows its
  %% two ends. Verified in a standalone four-way test, 2026-10-14.
  \addplot[oiBlue, line width=1.0pt, mark=-, mark size=2.5pt, forget plot,
           error bars/.cd, y dir=both, y explicit,
           error bar style={{line width=0.6pt}}]
    coordinates {{
{eb}
    }};
  \addplot[only marks, mark=*, mark size=1.4pt, oiBlue, forget plot]
    coordinates {{{hit}}};
{missplot}
{lbl}
\end{{axis}}""")
    miss_total = d["sc31"]["n_miss"] + d["sc33"]["n_miss"]
    key = (rf"""
\node[font=\footnotesize,anchor=north west,align=left,inner sep=0pt,
      text width=168mm]
  at ([yshift=-2mm]ax0.below south west)
  {{\textcolor{{oiBlue}}{{$\bullet$}}~the bootstrap interval on the median excludes the bound.
   \textcolor{{oiVerm}}{{$\circ$}}~it does not, at {miss_total} of the
   {2 * len(d['sc31']['points'])} settings drawn. Shading marks accuracy at or below the
   bound. Grey counts are degenerate runs.}};""")
    return r"\begin{tikzpicture}" + "\n".join(panels) + key + "\n" + r"\end{tikzpicture}"


# ----------------------------------------------------------------- B4
def build_b4():
    """The instrument exhibit: decimate vs block-max, and which verdicts flipped."""
    reseed("B4")
    P = J("sc31_pooling_comparison.json")
    rows = sorted(P["per_setting"], key=lambda r: r["max"])
    assert P["n_verdicts"] == 10 and P["n_verdicts_unchanged"] == 6, (
        P["n_verdicts"], P["n_verdicts_unchanged"])
    assert all(r["delta"] > 0 for r in rows), "block-max did not win every setting"
    # These two summarize the whole exhibit and used to be computed inside emit_b4, where
    # nothing could check them: a value that exists only in the emitted string is a hand-typed
    # literal with extra steps. They are returned, so they reach figure_data.json.
    mean_delta = float(np.mean([r["delta"] for r in rows]))
    max_p = float(max(r["p"] for r in rows))
    worst = min(rows, key=lambda r: r["decimate"])
    lo = min([P["bound"]] + [r["decimate"] for r in rows])
    hi = max(r["max"] for r in rows)
    pad = 0.06 * (hi - lo)
    return {"bound": P["bound"], "rows": rows, "verdicts": P["verdicts"],
            "n_unchanged": P["n_verdicts_unchanged"], "n_total": P["n_verdicts"],
            "mean_delta": mean_delta, "max_p": max_p,
            "xmin": lo - pad, "xmax": hi + pad,
            # the shortest frame under the old resampler sat essentially on the bound; both
            # margins must be about the SAME setting, so take it by name, not by row order
            "worst": worst["setting"],
            "worst_dec_margin": worst["decimate"] - P["bound"],
            "worst_max_margin": worst["max"] - P["bound"]}


def emit_b4(d):
    # Single column since 2026-10-10. The full-width form carried a side table of the flipped
    # verdicts, which duplicated Table IV; the plot is the part a table cannot replace, so
    # only the plot survives and the flip list stays in the table.
    rows = d["rows"]
    n = len(rows)
    yt = ",".join(str(i) for i in range(n))
    yl = ",".join(r["setting"].replace("_", r"\_") for r in rows)
    seg = "\n".join(
        rf"  \draw[gLight, line width=2.4pt] (axis cs:{r['decimate']:.4f},{i}) -- "
        rf"(axis cs:{r['max']:.4f},{i});" for i, r in enumerate(rows))
    dec = " ".join(f"({r['decimate']:.4f},{i})" for i, r in enumerate(rows))
    mx = " ".join(f"({r['max']:.4f},{i})" for i, r in enumerate(rows))
    nflip = d["n_total"] - d["n_unchanged"]
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[name=ax, scale only axis, width=62mm, height=32mm,
  xmin={d['xmin']:.4f}, xmax={d['xmax']:.4f}, ymin=-0.8, ymax={n - 0.2},
  ytick={{{yt}}}, yticklabels={{{yl}}},
  label style={{font=\footnotesize}}, tick label style={{font=\scriptsize}},
  xlabel={{median top-1 beam accuracy}},
  title={{\small\textbf{{One line of the resampler}}}},
  legend style={{at={{(0.98,0.04)}}, anchor=south east, font=\scriptsize, draw=none,
                 fill=none, row sep=-2pt, legend cell align=left}}, clip=false]
  \addplot[draw=gGrey, dashed, line width=0.7pt, forget plot]
    coordinates {{({d['bound']:.4f},-0.8) ({d['bound']:.4f},{n - 0.2})}};
  \node[font=\scriptsize,gGrey,rotate=90,anchor=south,inner sep=2pt]
    at (axis cs:{d['bound']:.4f},{n / 2 - 0.6:.1f}) {{no-information bound {d['bound']:.4f}}};
{seg}
  \addplot[only marks, mark=o, mark size=1.7pt, oiVerm, line width=0.8pt]
    coordinates {{{dec}}};
  \addlegendentry{{decimate, the original}}
  \addplot[only marks, mark=*, mark size=1.9pt, oiBlue] coordinates {{{mx}}};
  \addlegendentry{{block maximum}}
\end{{axis}}
\node[anchor=north west, font=\scriptsize, align=left, text width=80mm]
  at ([yshift=-4mm]ax.below south west)
  {{Block maximum wins {n}/{n} settings, mean $+${d['mean_delta']:.4f}, every
    $p\le${d['max_p']:.4f}, and {nflip} of {d['n_total']} verdicts flip. Under decimation the
    shortest frame sat {d['worst_dec_margin']:.4f} above the bound, against
    {d['worst_max_margin']:.4f} under block maximum.}};
\end{{tikzpicture}}"""


def build_b2():
    """Link 1: measured bearing error against the resource product, and the CRB's exponent."""
    reseed("B2")
    from scipy import optimize
    M = J("sc31_bearing_mediation_max.json"); S = M["settings"]
    assert M["pool"] == "max" and M["seeds"] == 30
    bd = M["bound_rmse_deg"]
    approx(bd, 16.833, 5e-4)
    gain = [k for k in S if S[k]["n_rx"] == 4]            # processing gain only
    ap = [k for k in S if S[k]["n_rx"] != 4]              # aperture settings
    R = np.array([S[k]["resource_product_R"] for k in gain], float)
    msq = np.array([S[k]["rmse_deg_median"] ** 2 for k in gain])
    f = lambda x, a, b, c: a * np.power(x, -b) + c
    sc = R / R.max()
    pr, _ = optimize.curve_fit(f, sc, msq, p0=[msq.max(), 1.0, msq.min() * 0.5],
                               bounds=([0, 0.05, 0], [np.inf, 6, msq.max()]), maxfev=200000)
    # The interval is NOT recomputed here. An earlier version bootstrapped the MEAN while the
    # point estimate used the MEDIAN, which is an estimator mismatch and gave an interval
    # narrower than the estimator supports. audit_corrected_statistics.py computes both a
    # per-setting-independent and a joint-seed median bootstrap; we take the wider.
    E = J("corrected_statistics.json")["exponent"]
    # The joint-seed bootstrap is the wider of the two and is the one the manuscript quotes.
    wider = max(("independent_per_setting", "joint_seed"),
                key=lambda t: E[t]["ci95"][1] - E[t]["ci95"][0])
    lo, hi = E[wider]["ci95"]
    approx(E["point_b_median_based"], float(pr[1]), 2e-3)
    assert hi < 1.0, f"the fitted interval must exclude the CRB's b=1, got {hi}"
    a_, b_, c_ = pr
    approx(b_, 0.403, 3e-3)
    # An R^-1 reference curve, NOT a calibrated Cramer-Rao bound. It carries the fitted
    # asymptote c of the same model, and its amplitude is anchored so that it passes through
    # one measured point: argmax(msq) is the LARGEST measured mean square error, that is the
    # POOREST-resource setting and not the richest, as this comment used to claim. Anchoring
    # is a presentational choice that the caption now discloses; these data do not identify
    # the bound's absolute level.
    anchor_i = int(np.argmax(msq))
    a1 = (msq.max() - c_) / np.power(sc[anchor_i], -1.0)
    # OVER THE MEASURED RANGE ONLY. Drawn from 0.8 to 1.2 times it, both curves were
    # extrapolated a fifth of a decade past the data on each side, and the anchored
    # reference rose above the no-information bound in the top-left corner and struck
    # through the words "no-information bound" there. An anchored reference and a fit have
    # nothing to say beyond the resources that were measured. Audited 2026-10-14.
    xs = np.geomspace(R.min(), R.max(), 60)
    fit = [{"R": float(x), "rmse": float(np.sqrt(f(x / R.max(), a_, b_, c_)))} for x in xs]
    crb = [{"R": float(x), "rmse": float(np.sqrt(a1 * np.power(x / R.max(), -1.0) + c_))}
           for x in xs]
    # m = 1 has m(m^2-1) = 0, so R = 0 and no position on a logarithmic axis exists for it.
    # It is the sharpest evidence that the aperture factor is inert here, so it is reported
    # as an annotation rather than dropped in silence, as it was before 2026-10-08.
    zero = [k for k in ap if S[k]["resource_product_R"] == 0]
    assert len(zero) == 1, zero
    k0 = zero[0]
    # every drawn line and point must lie inside the axis box; the floor at 12.53 and the
    # b=1 curve's tail both fell below the old ymin of 12.8 and were drawn outside it
    ylo = min([c_ ** 0.5] + [z["rmse"] for z in crb] + [z["rmse"] for z in fit]
              + [S[k]["rmse_deg_median"] - S[k]["rmse_deg_sd"] for k in gain]) - 0.25
    # yhi takes the CURVES as well, which ylo already did. Omitting them left the anchored
    # reference's left end clearing the frame by 0.10 deg on the +0.55 margin alone, which
    # is the 2026-10-08 "drawn outside the box" defect surviving at the opposite end.
    yhi = max([bd] + [z["rmse"] for z in crb] + [z["rmse"] for z in fit]
              + [S[k]["rmse_deg_median"] + S[k]["rmse_deg_sd"] for k in gain]) + 0.55
    return {"bound": bd, "b": float(b_), "b_lo": float(lo), "b_hi": float(hi),
            "floor": float(np.sqrt(c_)), "fit": fit, "crb": crb,
            "anchor_key": gain[anchor_i], "anchor_R": float(R[anchor_i]),
            "anchor_rmse": float(np.sqrt(msq[anchor_i])),
            "ymin": float(ylo), "ymax": float(yhi),
            # derived like the y limits, for the same reason: a typed limit drifts when the
            # data moves and nothing notices
            "xmin": float(R.min() * 0.75), "xmax": float(R.max() * 1.35),
            "gain_pts": [{"k": k, "R": S[k]["resource_product_R"],
                          "rmse": S[k]["rmse_deg_median"],
                          "sd": S[k]["rmse_deg_sd"]} for k in gain],
            "ap_pts": [{"k": k, "R": S[k]["resource_product_R"],
                        "rmse": S[k]["rmse_deg_median"],
                        "sd": S[k]["rmse_deg_sd"]} for k in ap],
            "zero_key": k0, "zero_rmse": S[k0]["rmse_deg_median"],
            "zero_sd": S[k0]["rmse_deg_sd"]}


def emit_b2(d):
    # The symbols must be the manuscript's: R is written with \Nsamp\Nchirp there, which
    # expand to N_s and N_c, not the N_samp and N_chirp this figure used to draw.
    NS, NC = r"N_\mathrm{s}", r"N_\mathrm{c}"
    g = sorted(d["gain_pts"], key=lambda z: z["R"])
    eb = "\n".join(f"      ({z['R']},{z['rmse']:.3f}) += (0,{z['sd']:.3f}) -= (0,{z['sd']:.3f})"
                   for z in g)
    ft = " ".join(f"({z['R']:.0f},{z['rmse']:.3f})" for z in d["fit"])
    cb = " ".join(f"({z['R']:.0f},{z['rmse']:.3f})" for z in d["crb"])
    ebm = " ".join(f"({z['R']},{z['rmse']:.3f})" for z in g)
    rx2 = [z for z in d["ap_pts"] if z["R"] > 0]
    rx2e = "\n".join(f"      ({z['R']},{z['rmse']:.3f}) += (0,{z['sd']:.3f}) -= (0,{z['sd']:.3f})"
                     for z in rx2)
    rx2m = " ".join(f"({z['R']},{z['rmse']:.3f})" for z in rx2)
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[scale only axis, width=74mm, height=32mm, xmode=log,
  xmin={d['xmin']:.0f}, xmax={d['xmax']:.0f}, ymin={d['ymin']:.2f}, ymax={d['ymax']:.2f},
  xlabel={{resource product $R = {NS}\,{NC}\, m(m^2{{-}}1)$}},
  ylabel={{bearing RMSE (deg)}},
  label style={{font=\footnotesize}}, tick label style={{font=\footnotesize}},
  title={{\small\textbf{{Mean square error does not follow $R^{{-1}}$}}}},
  legend style={{at={{(0.5,-0.36)}}, anchor=north, legend columns=2, font=\scriptsize,
                 draw=none, fill=none, legend cell align=left,
                 /tikz/every even column/.append style={{column sep=3mm}}}}, clip=false]
  \addplot[draw=gGrey, dashed, line width=0.6pt, forget plot]
    coordinates {{({d['xmin']:.0f},{d['bound']:.3f}) ({d['xmax']:.0f},{d['bound']:.3f})}};
  \node[font=\scriptsize,gGrey,anchor=south west,inner sep=1.5pt]
    at (axis cs:{d['xmin'] * 1.03:.0f},{d['bound']:.3f}) {{no-information bound {d['bound']:.3f}}};
  \addplot[draw=goodLine, dotted, line width=0.8pt, forget plot]
    coordinates {{({d['xmin']:.0f},{d['floor']:.3f}) ({d['xmax']:.0f},{d['floor']:.3f})}};
  \node[font=\scriptsize,goodLine,anchor=south west,inner sep=1.5pt]
    at (axis cs:{d['xmin'] * 1.03:.0f},{d['floor']:.3f}) {{fitted asymptote {d['floor']:.2f}\,deg}};
  \addplot[oiVerm, dash dot, line width=0.9pt] coordinates {{{cb}}};
  \addlegendentry{{anchored $R^{{-1}}$ reference in MSE}}
  \addplot[oiBlue, line width=1.1pt] coordinates {{{ft}}};
  \addlegendentry{{measured fit, $b={d['b']:.3f}$}}
  %% The caps come from the PLOT's mark: `error mark` after `error bars/.cd` is accepted
  %% and ignored by this pgfplots, so these intervals had no ends at all. The error bars
  %% go down first with a dash mark, then the round markers that carry the legend entry.
  \addplot[oiBlue, mark=-, mark size=2.5pt, only marks, forget plot,
           error bars/.cd, y dir=both, y explicit,
           error bar style={{line width=0.5pt}}] coordinates {{
{eb}
  }};
  \addplot[only marks, mark=*, mark size=1.7pt, oiBlue] coordinates {{{ebm}}};
  \addlegendentry{{$m=4$, processing gain only}}
  \addplot[oiPurple, mark=-, mark size=2.5pt, only marks, forget plot,
           error bars/.cd, y dir=both, y explicit,
           error bar style={{line width=0.5pt}}] coordinates {{
{rx2e}
  }};
  \addplot[only marks, mark=square*, mark size=1.9pt, oiPurple] coordinates {{{rx2m}}};
  \addlegendentry{{$m=2$, aperture reduced}}
  %% align=right is NOT decoration. A \\ inside a TikZ node with no align key collapses
  %% silently, and this annotation printed "b = 1;reference anchored" run together in the
  %% rendered figure for as long as it existed. Any node containing \\ needs align.
  \node[font=\scriptsize,oiBlue,anchor=north east,align=right,inner sep=1.5pt]
    at (axis cs:{d['xmax'] * 0.965:.0f},{d['bound'] - 0.10:.2f})
    {{CI $[{d['b_lo']:.3f},{d['b_hi']:.3f}]$ excludes $b=1$;\\
      reference anchored at {d['anchor_key'].replace('=', '{{=}}')}}};
  \node[font=\scriptsize,oiPurple,anchor=north east,inner sep=1.5pt]
    %% clear of the blue annotation above, which is TWO lines: the 0.52 offset that was
    %% here fitted a one-line node and collided the moment align=right let it break.
    at (axis cs:{d['xmax'] * 0.965:.0f},{d['bound'] - 1.05:.2f})
    {{$m=1$: $R=0$, off axis, {d['zero_rmse']:.2f}\,deg}};
\end{{axis}}
\end{{tikzpicture}}"""


# ----------------------------------------------------------------- B3
def build_b3():
    """Link 2: beam gain against measured bearing error, two tasks, two labels."""
    A = J("sc31_mediation_analysis_max.json")
    assert A["pool"] == "max"
    approx(A["link2_pearson_r"], -0.9335, 5e-4)
    return {"r": A["link2_pearson_r"], "p": A["link2_pearson_p"],
            "rho": A["link2_spearman_r"], "rows": A["per_setting"],
            "partial_direct": A["partial_logR_gain_given_rmse"],
            "partial_mediator": A["partial_rmse_gain_given_logR"]}


def emit_b3(d):
    style = {"chirps": ("*", "oiBlue", "chirps (airtime)"),
             "samples": ("square*", "oiGreen", "samples (bandwidth)"),
             "antennas": ("triangle*", "oiPurple", "Rx (aperture)")}
    plots = []
    for sw, (mk, col, lab) in style.items():
        pts = [r for r in d["rows"] if r["sweep"] == sw]
        if not pts:
            continue
        co = " ".join(f"({r['rmse']:.3f},{r['gain']:.4f})" for r in pts)
        plots.append(rf"  \addplot[only marks, mark={mk}, mark size=1.9pt, {col}] "
                     rf"coordinates {{{co}}}; \addlegendentry{{{lab}}}")
    xs = [r["rmse"] for r in d["rows"]]; ys = [r["gain"] for r in d["rows"]]
    m, b = np.polyfit(xs, ys, 1)
    lo, hi = min(xs) - 0.15, max(xs) + 0.15
    # record what is drawn, as every other figure now does
    d["fit_slope"], d["fit_intercept"] = float(m), float(b)
    d["xmin"], d["xmax"], d["ymin"], d["ymax"] = float(lo), float(hi), 0.0, 0.165
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[scale only axis, width=71mm, height=34mm,
  xmin={lo:.2f}, xmax={hi:.2f}, ymin=0.0, ymax=0.165,
  ytick={{0,0.05,0.10,0.15}}, yticklabels={{0,0.05,0.10,0.15}},
  label style={{font=\footnotesize}}, tick label style={{font=\footnotesize}},
  xlabel={{bearing RMSE (deg), regressed against GPS}},
  ylabel={{beam-prediction gain over the bound}},
  title={{\small\textbf{{Two tasks, two labels, one relationship}}}},
  legend style={{at={{(0.03,0.03)}}, anchor=south west, font=\footnotesize, draw=none,
                 fill=none, row sep=-1.5pt, legend cell align=left}}, clip=false]
  \addplot[gGrey, dashed, line width=0.7pt, forget plot]
    coordinates {{({lo:.2f},{m*lo+b:.4f}) ({hi:.2f},{m*hi+b:.4f})}};
{chr(10).join(plots)}
  \node[font=\scriptsize, anchor=north east, align=right]
    at (axis cs:{hi:.2f},0.160)
    {{Pearson $r={d['r']:.4f}$, $p={_sci(d['p'])}$\\
      Spearman $={d['rho']:.4f}$}};
\end{{axis}}
\end{{tikzpicture}}"""


# ----------------------------------------------------------------- B5
def build_b5():
    """What the prediction is worth, separated by who pays for the radar frame."""
    d = J("alignment_cost_model.json")
    c = d["scenarios"]["31"]
    approx(d["T_sec_us"], 5.0, 1e-9)
    approx(c["airtime_exhaustive_us"], 320.0, 1e-9)
    # The reuse arm must beat both non-sensing comparators, and the dedicated arm must not.
    assert c["airtime_radar_full_us"] < c["airtime_prior_us"] < c["airtime_exhaustive_us"]
    reuse = [{"n": r["n_chirps"], "total": r["airtime_retained_us"],
              "restart": r["airtime_restart_us"]} for r in c["per_setting"]]
    ded = {}
    for tc, rows in c["dedicated_latency_plus_airtime"].items():
        assert not any(r["beats_prior"] for r in rows), tc
        if tc not in ("2.56", "10.24"):
            continue
        ded[tc] = {"Tc": float(tc), "adc": d["n_adc"] / float(tc),
                   "points": [{"n": r["n_chirps"], "total": r["total_us"],
                               "beats_exhaustive": r["beats_exhaustive"]} for r in rows]}
    # The verdict of the right-hand panel is that no dedicated setting reaches the shortlist.
    # Quantify how close the closest one comes, so the panel can state it rather than rely on
    # the reader measuring a gap against an unlabeled rule.
    allded = [(z["Tc"], p) for z in ded.values() for p in z["points"]]
    tc_c, near = min(allded, key=lambda zp: zp[1]["total"])
    return {"exhaustive": c["airtime_exhaustive_us"], "prior": c["airtime_prior_us"],
            "persistence": c["airtime_persistence_us"], "gps": c["airtime_gps_us"],
            "reuse": reuse, "dedicated": ded, "n_beam": d["n_beam"],
            "T_sec": d["T_sec_us"], "n_adc": d["n_adc"],
            "breakeven_Tc": c["dedicated_breakeven_T_chirp_vs_prior_us"],
            "breakeven_rate": d["n_adc"] / c["dedicated_breakeven_T_chirp_vs_prior_us"],
            "saving_vs_prior": c["reuse_budget_vs_prior_us"],
            "saving_vs_exhaustive": c["reuse_budget_vs_exhaustive_us"],
            "breakeven_n": c["per_setting"][-1]["n_chirps"],
            "breakeven_rate_short": c["dedicated_breakeven_by_predictor_quality"]["rows"][0]
                                     ["adc_measured_msps"],
            "breakeven_n_short": c["dedicated_breakeven_by_predictor_quality"]["rows"][0]
                                  ["n_chirps"],
            "near_gap": near["total"] - c["airtime_prior_us"],
            "near_n": near["n"], "near_Tc": tc_c, "near_adc": d["n_adc"] / tc_c,
            "n_beats_exhaustive": sum(p["beats_exhaustive"] for _, p in allded),
            "n_dedicated": len(allded)}


def emit_b5(d):
    # One shared logarithmic airtime axis across both panels. The quantity is the same in
    # both, the airtime the link spends before it can transmit; only what it includes differs,
    # and that is what the titles say. Drawn on two different scales, as this figure was until
    # 2026-10-08, the reference rules sit near the top of one panel and near the bottom of the
    # other and the two regimes cannot be compared by eye at all.
    YMIN, YMAX = 32, 3000
    PLOTW = 72          # mm, the plot area; every annotation must wrap inside it
    TXTW = PLOTW - 5
    def rules(side):
        """side: 'south east' puts the labels above their rules, 'north east' below."""
        return rf"""  \addplot[draw=gGrey, dashed, line width=0.7pt, forget plot]
    coordinates {{(6.4,{d['exhaustive']:.1f}) (330,{d['exhaustive']:.1f})}};
  \addplot[draw=oiPurple, dashdotted, line width=0.7pt, forget plot]
    coordinates {{(6.4,{d['prior']:.1f}) (330,{d['prior']:.1f})}};
  \node[font=\footnotesize,gGrey,anchor={side},inner sep=3pt]
    at (axis cs:330,{d['exhaustive']:.1f}) {{exhaustive sweep {d['exhaustive']:.1f}}};
  \node[font=\footnotesize,oiPurple,anchor={side},inner sep=3pt]
    at (axis cs:330,{d['prior']:.1f}) {{fixed training prior {d['prior']:.1f}}};"""
    co = " ".join(f"({p['n']},{p['total']:.1f})" for p in d["reuse"])
    cr = " ".join(f"({p['n']},{p['restart']:.1f})" for p in d["reuse"])
    cs = sorted(d["dedicated"].values(), key=lambda z: z["Tc"])
    col = ["oiBlue", "oiVerm"]
    plots, legs = [], []
    for i, z in enumerate(cs):
        pts = " ".join(f"({p['n']},{p['total']:.0f})" for p in z["points"])
        plots.append(rf"  \addplot[{col[i]}, mark=*, mark size=1.7pt, line width=1.0pt] "
                     rf"coordinates {{{pts}}};" + "\n"
                     rf"  \addlegendentry{{$T_\mathrm{{chirp}}={z['Tc']:g}\,\mu$s, "
                     rf"{z['adc']:.0f}\,Msps}}")
    axopts = (rf"scale only axis, width={PLOTW}mm, height=46mm, xmode=log, log basis x=2, ymode=log,"
              "\n  " rf"xmin=6.4, xmax=330, ymin={YMIN}, ymax={YMAX},"
              "\n  " r"xtick={8,16,32,64,125,250}, xticklabels={8,16,32,64,125,250},"
              "\n  " r"ytick={50,100,200,400,800,1600,3000},"
              r" yticklabels={50,100,200,400,800,1600,3000},"
              "\n  " r"label style={font=\footnotesize}, tick label style={font=\footnotesize},"
              "\n  " r"xlabel={chirps per frame, the sensing airtime},")
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[name=ax, {axopts}
  ylabel={{airtime before the link can transmit ($\mu$s)}},
  title={{\small\textbf{{Reuse, the frame is already paid for}}}},
  legend style={{at={{(0.03,0.72)}}, anchor=north west, font=\footnotesize, draw=none,
                 fill=none, row sep=-1pt}}, clip=false]
{rules('south east')}
  \addplot[draw=goodLine, dotted, line width=0.8pt, forget plot]
    coordinates {{(6.4,{d['persistence']:.1f}) (330,{d['persistence']:.1f})}};
  \node[font=\footnotesize,goodLine,anchor=south west,inner sep=1.5pt]
    at (axis cs:6.6,{d['persistence']:.1f}) {{previous beam {d['persistence']:.1f}}};
  \addplot[draw=goodLine, dotted, line width=0.8pt, forget plot]
    coordinates {{(6.4,{d['gps']:.1f}) (330,{d['gps']:.1f})}};
  \node[font=\footnotesize,goodLine,anchor=north east,inner sep=1.5pt]
    at (axis cs:330,{d['gps']:.1f}) {{partner position {d['gps']:.1f}}};
  \addplot[oiBlue, mark=*, mark size=1.9pt, line width=1.1pt] coordinates {{{co}}};
  \addlegendentry{{radar, retained shortlist}}
  \addplot[gGrey, mark=o, mark size=1.5pt, line width=0.7pt, densely dotted]
    coordinates {{{cr}}};
  \addlegendentry{{the sweep restarts on a miss}}
  \node[font=\footnotesize,oiBlue,anchor=north,align=center,inner sep=2.5pt,
        text width={TXTW}mm]
    at (axis cs:45,{YMAX * 0.80:.0f})
    {{at the full frame the radar saves {abs(d['saving_vs_prior']):.1f}\,$\mu$s against the
      prior and {abs(d['saving_vs_exhaustive']):.1f} against an exhaustive sweep}};
\end{{axis}}
\begin{{axis}}[name=bx, at=(ax.right of south east), anchor=left of south west, xshift=11mm,
  {axopts}
  ylabel={{}},
  title={{\small\textbf{{Dedicated, the link pays for the frame}}}},
  legend style={{at={{(0.03,0.97)}}, anchor=north west, font=\footnotesize, draw=none,
                 fill=none, row sep=-1pt}}, clip=false]
{rules('north east')}
{chr(10).join(plots)}
  \node[font=\footnotesize,black,anchor=north west,align=left,inner sep=2pt,
        text width={TXTW}mm]
    at (axis cs:6.6,{YMIN * 4.4:.0f})
    {{No setting reaches the prior. The closest, {d['near_n']} chirps at
      {d['near_adc']:.0f}\,Msps, is short by {d['near_gap']:.1f}\,$\mu$s. Break-even needs
      $T_\mathrm{{chirp}}<{d['breakeven_Tc']:.3f}\,\mu$s at {d['breakeven_n']} chirps, that is
      {d['breakeven_rate']:.0f}\,Msps, and {d['breakeven_rate_short']:.0f}\,Msps at
      {d['breakeven_n_short']}.}};
\end{{axis}}
\end{{tikzpicture}}"""


# ----------------------------------------------------------------- B0
# Four orders of magnitude separate one chirp from the interval over which the chosen beam
# stays usable, so one linear axis cannot carry them. Drawing two rows of boxes at whatever
# width fits makes the 320 us sweep look like a sizeable fraction of the 92.1 ms refresh,
# which is the reverse of what the cost argument rests on. The figure is therefore a zoom
# ladder: each tier is one magnified interval of the tier above it, each carries its own
# ticks, and every x position is computed from that tier's single scale, so the drawing
# cannot disagree with the durations it labels.
NBm, TSEC = r"N_\mathrm{B}", r"T_\mathrm{sec}"            # as \NB and \Tsec in PaperB.tex
TCHIRP, NSAMP, NCHIRP = r"T_\mathrm{chirp}", r"N_\mathrm{s}", r"N_\mathrm{c}"


def build_b0():
    """The three time scales, each drawn to its own scale."""
    TB = J("testbed_facts.json")
    C = J("alignment_cost_model.json")
    B = J("isac_airtime_budget_max.json")
    BC = J("beam_coherence.json")["scenarios"]["31"]
    assert TB["separate_radios"] is True
    s31 = C["scenarios"]["31"]
    n_beam, T_sec = C["n_beam"], C["T_sec_us"]
    n_rx, n_samp, n_chirp = TB["cube_shape"]

    # k is not a stored field: the model retains a top-5 shortlist and prices the miss.
    # Recover it from the airtime the model itself wrote out, so the k drawn here cannot
    # drift from the k behind the cost table.
    K = 5
    approx(s31["airtime_prior_us"],
           K * T_sec + (1.0 - s31["p5_prior"]) * (n_beam - K) * T_sec, 1e-9)
    approx(s31["airtime_exhaustive_us"], n_beam * T_sec, 1e-9)

    # The recording ships no radar timing, so the frame duration is not measured. It is
    # bracketed by the two chirp durations the dedicated regime is priced at; both must exist
    # in the cost model, or this figure and the cost table would describe different hardware.
    ded = s31["dedicated_latency_plus_airtime"]
    tc_lo, tc_hi = 2.56, 10.24
    for v in (tc_lo, tc_hi):
        assert f"{v:g}" in ded, f"chirp duration {v} us is not priced in the cost model"
    frame_lo, frame_hi = n_chirp * tc_lo, n_chirp * tc_hi                 # us

    period_ms = B["frame_period_us"] / 1000.0
    span_a, span_b, span_c = 100.0, 2600.0, 60.0                          # ms, us, us
    assert period_ms < span_a, "the logging period must fit inside the top tier"
    assert frame_hi <= span_b, "the frame must fit inside the middle tier"
    assert max(K * T_sec, 5 * tc_hi) <= span_c, "the bottom tier must show several of each"
    return {"n_rx": n_rx, "n_samp": n_samp, "n_chirp": n_chirp, "n_beam": n_beam, "k": K,
            "radar_band": TB["radar_band_ghz"], "link_band": TB["link_band_ghz"],
            "n_elem": TB["n_array_elements"], "T_sec": T_sec,
            "exhaustive": s31["airtime_exhaustive_us"],
            "T_chirp_lo": tc_lo, "T_chirp_hi": tc_hi,
            "frame_lo_us": frame_lo, "frame_hi_us": frame_hi,
            "frame_period_ms": period_ms,
            "coherence_ms": BC["beam_coherence_median_ms"],
            "coherence_ratio": BC["ratio_TB_over_frame_period_median"],
            "span_a_ms": span_a, "span_b_us": span_b, "span_c_us": span_c,
            "zoom_ab": span_a * 1000.0 / span_b, "zoom_bc": span_b / span_c,
            # what one refresh is actually spent on, which is what the old figure got wrong
            "duty_frame_pct": 100.0 * frame_hi / (period_ms * 1000.0),
            "duty_align_pct": 100.0 * s31["airtime_exhaustive_us"] / (period_ms * 1000.0)}


def emit_b0(d):
    xL, xR = 1.86, 9.92
    W = xR - xL
    XA = lambda t: xL + W * t / d["span_a_ms"]                            # ms
    XB = lambda t: xL + W * t / d["span_b_us"]                            # us
    XC = lambda t: xL + W * t / d["span_c_us"]                            # us
    yA, yB1, yB2, yC1, yC2 = 4.71, 3.60, 3.06, 1.71, 1.01                 # lane centers
    hc = 0.17
    o = []
    A = o.append

    def axis(y):
        A(rf"\draw[-{{Latex[length=3.4pt,width=2.6pt]}},line width=0.45pt] "
          rf"({xL - 0.07:.2f},{y:.2f}) -- ({xR + 0.17:.2f},{y:.2f});")

    def ticks(y, X, vals, fmt, unit, label=lambda v, i, n: True):
        # the tick row hangs below the blocks, which are hc tall, not below the lane center
        for i, v in enumerate(vals):
            A(rf"\draw[line width=0.4pt] ({X(v):.3f},{y - hc - 0.07:.2f}) -- "
              rf"({X(v):.3f},{y - hc:.2f});")
            if label(v, i, len(vals)):
                A(rf"\node[font=\tiny,anchor=north,inner sep=1pt] "
                  rf"at ({X(v):.3f},{y - hc - 0.09:.2f}) "
                  rf"{{{fmt(v)}{rf'~{unit}' if v == vals[-1] else ''}}};")

    def lane(y, name, band, col):
        A(rf"\node[font=\scriptsize,anchor=east,align=right,inner sep=1pt,text={col}] "
          rf"at ({xL - 0.16:.2f},{y:.2f}) {{{name}\\{band}\,GHz}};")

    def block(x0, x1, y, fill, draw, extra=""):
        A(rf"\draw[fill={fill},draw={draw},line width=0.4pt{extra}] "
          rf"({x0:.3f},{y - hc:.2f}) rectangle ({x1:.3f},{y + hc:.2f});")

    def dim(y, x0, x1, text):
        A(rf"\draw[{{Latex[length=2.6pt,width=2pt]}}-{{Latex[length=2.6pt,width=2pt]}},"
          rf"line width=0.4pt] ({x0:.3f},{y:.2f}) -- ({x1:.3f},{y:.2f});")
        A(rf"\node[font=\scriptsize,anchor=west,inner sep=2.5pt] at ({x1:.3f},{y:.2f}) "
          rf"{{{text}}};")

    def wedge(xa0, xa1, ya, yb, zoom):
        A(rf"\fill[gLight,opacity=0.5] ({xa0:.3f},{ya:.2f}) -- ({xa1:.3f},{ya:.2f}) -- "
          rf"({xR:.2f},{yb:.2f}) -- ({xL:.2f},{yb:.2f}) -- cycle;")
        A(rf"\node[font=\tiny,gGrey,anchor=west,inner sep=1pt] "
          rf"at ({xR + 0.20:.2f},{(ya + yb) / 2:.2f}) {{$\times${zoom:.0f}}};")

    # ---- (a) one refresh of the measurement ---------------------------------------------
    lane(yA, "radar", d["radar_band"], "oiBlue")
    axis(yA)
    ticks(yA, XA, [0, 25, 50, 75, 100], lambda v: f"{v:g}", "ms",
          label=lambda v, i, n: i in (0, n - 1))
    fw = XA(d["frame_hi_us"] / 1000.0) - XA(0)
    for t in (0.0, d["frame_period_ms"]):
        block(XA(t), XA(t) + fw, yA, "oiBlue!50", "oiBlue")
    A(rf"\draw[{{Latex[length=2.6pt,width=2pt]}}-{{Latex[length=2.6pt,width=2pt]}},"
      rf"line width=0.4pt] ({XA(0):.3f},{yA + 0.40:.2f}) -- "
      rf"({XA(d['frame_period_ms']):.3f},{yA + 0.40:.2f});")
    A(rf"\node[font=\scriptsize,anchor=south,inner sep=1.5pt] "
      rf"at ({(XA(0) + XA(d['frame_period_ms'])) / 2:.3f},{yA + 0.43:.2f}) "
      rf"{{logging period, median {d['frame_period_ms']:.1f}\,ms}};")
    yc, xbrk = yA + 0.92, xR - 0.62
    A(rf"\draw[goodLine,line width=0.8pt] ({XA(0):.3f},{yc:.2f}) -- "
      rf"({xbrk - 0.08:.2f},{yc:.2f});")
    A(rf"\draw[goodLine,line width=0.8pt,-{{Latex[length=3.4pt,width=2.6pt]}}] "
      rf"({xbrk + 0.08:.2f},{yc:.2f}) -- ({xR + 0.17:.2f},{yc:.2f});")
    for dx in (-0.09, 0.01):
        A(rf"\draw[goodLine,line width=0.5pt] ({xbrk + dx:.2f},{yc - 0.10:.2f}) -- "
          rf"({xbrk + dx + 0.10:.2f},{yc + 0.10:.2f});")
    A(rf"\node[font=\scriptsize,goodLine,anchor=south,inner sep=1.5pt] "
      rf"at ({(XA(0) + xR) / 2:.3f},{yc + 0.04:.2f}) {{ideal line-of-sight pointing time "
      rf"{d['coherence_ms'] / 1000.0:.2f}\,s, {d['coherence_ratio']:.1f} periods}};")
    wedge(XA(0), XA(d["frame_hi_us"] / 1000.0), yA - hc - 0.42, yB1 + hc + 0.12, d["zoom_ab"])

    # ---- (b) one frame against one alignment --------------------------------------------
    lane(yB1, "radar", d["radar_band"], "oiBlue")
    lane(yB2, "link", d["link_band"], "oiVerm")
    axis(yB1)
    axis(yB2)
    ticks(yB2, XB, [0, 500, 1000, 1500, 2000, 2500], lambda v: f"{v / 1000:g}", "ms")
    block(XB(0), XB(d["frame_lo_us"]), yB1, "oiBlue!50", "oiBlue")
    block(XB(d["frame_lo_us"]), XB(d["frame_hi_us"]), yB1, "oiBlue!16", "oiBlue",
          ",densely dashed")
    A(rf"\node[font=\scriptsize,anchor=west,inner sep=3pt] "
      rf"at ({XB(d['frame_lo_us']) + 0.05:.3f},{yB1:.2f}) {{one frame, ${NCHIRP}{TCHIRP}$, between "
      rf"{d['frame_lo_us'] / 1000.0:.2f} and {d['frame_hi_us'] / 1000.0:.2f}\,ms}};")
    block(XB(0), XB(d["exhaustive"]), yB2, "oiVerm!50", "oiVerm")
    A(rf"\node[font=\scriptsize,anchor=west,inner sep=3pt,fill=white] "
      rf"at ({XB(d['exhaustive']):.3f},{yB2:.2f}) {{exhaustive sweep, "
      rf"${NBm}{TSEC}={d['exhaustive']:.0f}\,\mu$s, {d['duty_align_pct']:.2f}\,\% of the "
      rf"period}};")
    wedge(XB(0), XB(d["span_c_us"]), yB2 - hc - 0.40, yC1 + hc + 0.34, d["zoom_bc"])

    # ---- (c) inside the frame and inside the sweep ---------------------------------------
    lane(yC1, "radar", d["radar_band"], "oiBlue")
    lane(yC2, "link", d["link_band"], "oiVerm")
    axis(yC1)
    axis(yC2)
    ticks(yC2, XC, [0, 10, 20, 30, 40, 50, 60], lambda v: f"{v:g}", r"$\mu$s")
    i = 0
    while i * d["T_chirp_hi"] < d["span_c_us"]:
        a, b = i * d["T_chirp_hi"], min((i + 1) * d["T_chirp_hi"], d["span_c_us"])
        block(XC(a), XC(b), yC1, "oiBlue!16", "oiBlue")
        A(rf"\draw[oiBlue,line width=0.8pt] ({XC(a) + 0.05:.3f},{yC1 - hc + 0.05:.2f}) -- "
          rf"({XC(b) - 0.05:.3f},{yC1 + hc - 0.05:.2f});")
        i += 1
    dim(yC1 + hc + 0.17, XC(0), XC(d["T_chirp_hi"]),
        rf"${TCHIRP}$, ${NSAMP}={d['n_samp']}$ samples on each of {d['n_rx']} elements")
    j = 0
    while j * d["T_sec"] < d["span_c_us"]:
        a, b = j * d["T_sec"], min((j + 1) * d["T_sec"], d["span_c_us"])
        block(XC(a), XC(b), yC2, "oiVerm!50" if j < d["k"] else "oiVerm!16", "oiVerm")
        j += 1
    dim(yC2 + hc + 0.17, XC(0), XC(d["T_sec"]),
        rf"${TSEC}={d['T_sec']:g}\,\mu$s per sector, ${NBm}={d['n_beam']}$ in all, "
        rf"the shaded $k={d['k']}$ retained")
    # Record the drawn geometry so the verifier can check it. Every one of the defects the
    # previous version of this figure carried was geometric, and a text-layer scan cannot see
    # geometry: the two rows were drawn at scales 362 apart, and the arrow labeled with the
    # logging period spanned the idle gap between frames rather than frame start to frame start.
    d["geom"] = {
        "a_x0": XA(0.0), "a_cm_per_ms": (XA(1.0) - XA(0.0)),
        "b_x0": XB(0.0), "b_cm_per_us": (XB(1.0) - XB(0.0)),
        "c_x0": XC(0.0), "c_cm_per_us": (XC(1.0) - XC(0.0)),
        "a_frame_w": XA(d["frame_hi_us"] / 1000.0) - XA(0.0),
        "a_period_arrow": [XA(0.0), XA(d["frame_period_ms"])],
        "a_frame2_x0": XA(d["frame_period_ms"]),
        "b_frame_w": XB(d["frame_hi_us"]) - XB(0.0),
        "b_sweep_w": XB(d["exhaustive"]) - XB(0.0),
        "c_chirp_w": XC(d["T_chirp_hi"]) - XC(0.0),
        "c_slot_w": XC(d["T_sec"]) - XC(0.0),
        "c_shortlist_w": XC(d["k"] * d["T_sec"]) - XC(0.0)}
    A(rf"\node[font=\tiny,gGrey,anchor=north west,align=left,inner sep=1pt] "
      rf"at ({0.10:.2f},{yC2 - hc - 0.37:.2f}) {{A miss still costs the remaining "
      rf"${NBm}-k$. The two bands are on separate radios, so the sensing\\and the alignment "
      rf"intervals overlap without contending; in a single-radio system they would be "
      rf"sequential.}};")
    return "\\begin{tikzpicture}[font=\\footnotesize]\n" + "\n".join(o) + "\n\\end{tikzpicture}"


# ----------------------------------------------------------------- B6
def build_b6():
    """Aperture does little on either task, because the route supplies the bearing."""
    A = J("sc31_resource_ablation_G_max.json"); S = A["settings"]
    M = J("sc31_bearing_mediation_max.json"); T = M["settings"]
    _R = J("sc31_reference_predictors.json")["predictors"]
    G = {"ceiling_top1": [v for k, v in _R.items()
                          if k.startswith("GPS reference")][0]["top1"]}
    RO = J("bearing_from_range_only.json")["sc31"]
    # m = 4 is the full configuration, which the sweep files store under chirps=250. This is
    # only an aperture ablation if the other two resources are held fixed across the three
    # points, so check it rather than assume it.
    keys = {1: "rx=1", 2: "rx=2", 4: "chirps=250"}
    ref = (S[keys[4]]["n_samples"], S[keys[4]]["n_chirps"])
    for m_, k in keys.items():
        assert S[k]["n_rx"] == m_, (k, S[k]["n_rx"], m_)
        assert (S[k]["n_samples"], S[k]["n_chirps"]) == ref, (
            f"{k} is not at the full {ref} payload, so this is not a one-factor aperture sweep")
    beam = [{"m": m_, "v": S[k]["top1_median"], "sd": S[k]["top1_sd"]}
            for m_, k in sorted(keys.items())]
    bear = [{"m": m_, "v": T[k]["rmse_deg_median"], "sd": T[k]["rmse_deg_sd"]}
            for m_, k in sorted(keys.items())]
    bb, b1, b4 = M["bound_rmse_deg"], bear[0]["v"], bear[2]["v"]
    approx(bear[0]["v"], 13.924, 5e-4)
    # y limits derived from everything drawn, so no rule or whisker can fall outside the box
    bl, bh = A["bound_const"], G["ceiling_top1"]
    beam_lo = min([bl] + [z["v"] - z["sd"] for z in beam])
    beam_hi = max([bh] + [z["v"] + z["sd"] for z in beam])
    bear_lo = min([RO["rmse_from_range_only_deg"]] + [z["v"] - z["sd"] for z in bear])
    bear_hi = max([bb] + [z["v"] + z["sd"] for z in bear])
    return {"beam": beam, "bear": bear, "beam_bound": bl,
            "ceiling": bh, "bear_bound": bb,
            "range_only": RO["rmse_from_range_only_deg"],
            "beam_ymin": beam_lo - 0.055 * (beam_hi - beam_lo),
            "beam_ymax": beam_hi + 0.14 * (beam_hi - beam_lo),
            "bear_ymin": bear_lo - 0.055 * (bear_hi - bear_lo),
            "bear_ymax": bear_hi + 0.14 * (bear_hi - bear_lo),
            "n_samp": ref[0], "n_chirp": ref[1],
            # both shares are the figure's own claim and were computed and then dropped
            "share_bear": 100 * (bb - b1) / (bb - b4),
            "share_beam": 100 * (beam[2]["v"] - beam[0]["v"])
                          / (beam[2]["v"] - A["bound_const"])}


def emit_b6(d):
    def bars(pts, fmt):
        return "\n".join(
            f"      ({z['m']},{z['v']:{fmt}}) += (0,{z['sd']:{fmt}}) -= (0,{z['sd']:{fmt}})"
            for z in pts)

    def marks(pts, fmt):
        return " ".join(f"({z['m']},{z['v']:{fmt}})" for z in pts)

    pts_beam, pts_bear = marks(d["beam"], ".4f"), marks(d["bear"], ".3f")
    ax = (r"scale only axis, width=30mm, height=25mm, xmode=log, log basis x=2,"
          "\n  " r"xmin=0.82, xmax=4.9, xtick={1,2,4}, xticklabels={1,2,4},"
          "\n  " r"label style={font=\footnotesize}, tick label style={font=\footnotesize},"
          "\n  " r"xlabel={receive elements $m$},")
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[name=ax0, {ax}
  ymin={d['beam_ymin']:.4f}, ymax={d['beam_ymax']:.4f},
  ylabel={{top-1 beam accuracy}},
  title={{\small\textbf{{Beam task}}}}, clip=false]
  \addplot[draw=gGrey, dashed, line width=0.6pt, forget plot]
    coordinates {{(0.82,{d['beam_bound']:.4f}) (4.9,{d['beam_bound']:.4f})}};
  \node[font=\scriptsize,gGrey,anchor=south east,inner sep=1.5pt]
    at (axis cs:4.9,{d['beam_bound']:.4f}) {{bound {d['beam_bound']:.4f}}};
  \addplot[draw=goodLine, dotted, line width=0.8pt, forget plot]
    coordinates {{(0.82,{d['ceiling']:.4f}) (4.9,{d['ceiling']:.4f})}};
  \node[font=\scriptsize,goodLine,anchor=north east,inner sep=1.5pt]
    at (axis cs:4.9,{d['ceiling']:.4f}) {{position reference {d['ceiling']:.4f}}};
  %% THE CAP COMES FROM THE PLOT'S OWN MARK. `error mark` and `error mark options` written
  %% after `error bars/.cd` are accepted and silently ignored by this pgfplots, so these
  %% whiskers had no caps at all: measured, their ends were the bar's own line width. The
  %% error mark inherits the plot's mark, so the line and the bars are drawn with `mark=-`
  %% and the round data markers go on afterwards in their own pass, as in B1. The marker
  %% also comes down from 1.7 to 1.4 pt, because on the beam task at m=4 the whole
  %% interval is 0.95 mm against what was a 1.20 mm marker and sat entirely inside it.
  %% Audited 2026-10-14.
  \addplot[oiBlue, mark=-, mark size=2.5pt, line width=1.0pt, forget plot,
           error bars/.cd, y dir=both, y explicit,
           error bar style={{line width=0.5pt}}] coordinates {{
{bars(d['beam'], '.4f')}
  }};
  \addplot[only marks, mark=*, mark size=1.4pt, oiBlue, forget plot]
    coordinates {{{pts_beam}}};
  \node[font=\scriptsize,oiBlue,anchor=center,align=center,inner sep=1.5pt, text width=26mm]
    at (axis cs:2,{(d['ceiling'] + max(z['v'] for z in d['beam'])) / 2:.4f})
    {{the array buys {d['share_beam']:.1f}\,\% of the gain over the bound}};
\end{{axis}}
\begin{{axis}}[name=ax1, at=(ax0.right of south east), anchor=left of south west, xshift=11mm,
  {ax}
  ymin={d['bear_ymin']:.3f}, ymax={d['bear_ymax']:.3f},
  ylabel={{bearing RMSE (deg), lower is better}},
  title={{\small\textbf{{Bearing task}}}}, clip=false]
  \addplot[draw=gGrey, dashed, line width=0.6pt, forget plot]
    coordinates {{(0.82,{d['bear_bound']:.3f}) (4.9,{d['bear_bound']:.3f})}};
  \node[font=\scriptsize,gGrey,anchor=north east,inner sep=1.5pt]
    at (axis cs:4.9,{d['bear_bound']:.3f}) {{bound {d['bear_bound']:.3f}}};
  \addplot[draw=oiVerm, dash dot, line width=0.8pt, forget plot]
    coordinates {{(0.82,{d['range_only']:.3f}) (4.9,{d['range_only']:.3f})}};
  \node[font=\scriptsize,oiVerm,anchor=south east,inner sep=1.5pt]
    at (axis cs:4.9,{d['range_only']:.3f}) {{range alone, no radar: {d['range_only']:.3f}}};
  %% THE CAP COMES FROM THE PLOT'S OWN MARK. `error mark` and `error mark options` written
  %% after `error bars/.cd` are accepted and silently ignored by this pgfplots, so these
  %% whiskers had no caps at all: measured, their ends were the bar's own line width. The
  %% error mark inherits the plot's mark, so the line and the bars are drawn with `mark=-`
  %% and the round data markers go on afterwards in their own pass, as in B1. The marker
  %% also comes down from 1.7 to 1.4 pt, because on the beam task at m=4 the whole
  %% interval is 0.95 mm against what was a 1.20 mm marker and sat entirely inside it.
  %% Audited 2026-10-14.
  \addplot[oiBlue, mark=-, mark size=2.5pt, line width=1.0pt, forget plot,
           error bars/.cd, y dir=both, y explicit,
           error bar style={{line width=0.5pt}}] coordinates {{
{bars(d['bear'], '.3f')}
  }};
  \addplot[only marks, mark=*, mark size=1.4pt, oiBlue, forget plot]
    coordinates {{{pts_bear}}};
  \node[font=\scriptsize,oiBlue,anchor=center,align=center,inner sep=1.5pt, text width=26mm]
    at (axis cs:2,{(d['range_only'] + min(z['v'] for z in d['bear'])) / 2:.3f})
    {{$m=1$ already reaches {d['share_bear']:.0f}\,\% of the reduction}};
\end{{axis}}
\end{{tikzpicture}}"""


# ----------------------------------------------------------------- B7
def _exact_spearman_p(x, y):
    """Two-sided p by enumerating every permutation of y against x, ties kept as observed."""
    import itertools
    from scipy import stats as _st
    r0 = abs(_st.spearmanr(x, y).statistic)
    perms = list(itertools.permutations(y))
    return sum(1 for q in perms if abs(_st.spearmanr(x, q).statistic) >= r0 - 1e-12) / len(perms)


def build_b7():
    """Collapse onto a constant class is itself a function of the sensing resource."""
    from scipy import stats
    ALPHA = 0.05
    out = {}
    for key, f, lab in (("beam31", "sc31_resource_ablation_G_max.json", "Sc.31, beam"),
                        ("beam33", "sc33_resource_ablation_G_max.json", "Sc.33, beam"),
                        ("bear31", "sc31_bearing_mediation_max.json", "Sc.31, bearing"),
                        ("bear33", "sc33_bearing_mediation_max.json", "Sc.33, bearing")):
        S = J(f)["settings"]
        n = J(f)["seeds"]
        pts = [{"n": c, "coll": S[f"chirps={c}"]["n_collapsed"], "of": n}
               for c in (8, 16, 32, 64, 125, 250)]
        rho = stats.spearmanr([p["n"] for p in pts], [p["coll"] for p in pts])
        # Sec. V-B promises EXACT enumeration for rank correlations, "rather than
        # asymptotically, which is optimistic at that sample size". This figure used
        # scipy's asymptotic p until 2026-10-08 and the paper's own warning was borne out:
        # the asymptotic values are 2 to 20 times smaller, and they changed the count of
        # significant trends from two to three.
        p_exact = _exact_spearman_p([q["n"] for q in pts], [q["coll"] for q in pts])
        out[key] = {"label": lab, "points": pts, "seeds": n,
                    "spearman": float(rho.statistic), "p": float(p_exact),
                    "p_asymptotic": float(rho.pvalue),
                    # A rank correlation over a column that is zero at most settings is
                    # carried by the few that are not. Count them, so the figure can say which
                    # curve has the least to order rather than presenting four equal verdicts.
                    "n_nonzero": sum(1 for q in pts if q["coll"] > 0),
                    "significant": bool(p_exact < ALPHA)}
    approx(out["bear33"]["spearman"],
           J("sc33_mediation_restricted.json")["collapse_spearman"][0], 1e-9)
    n_sig = sum(z["significant"] for z in out.values())
    weak = sorted(k for k, z in out.items() if not z["significant"])
    # Assert the DATA, not the conclusion. The previous version asserted n_sig == 3 and
    # weak == ["beam31"], which protected the asymptotic answer from being corrected.
    assert 0 < n_sig < len(out), (n_sig, weak)
    assert all(out[k]["p"] >= out[k]["p_asymptotic"] for k in out), \
        "the exact p should not fall below the asymptotic one at this sample size"
    out["_meta"] = {"alpha": ALPHA, "n_sig": n_sig, "n_curves": 4, "weak": weak,
                    "n_weak": len(weak), "test": "exact enumeration over 6! permutations",
                    "weak_nonzero": {k: out[k]["n_nonzero"] for k in weak}}
    return out


def _pfmt(p):
    """A p below the format's resolution must not be printed as 0.00."""
    return r"$p<0.001$" if p < 0.001 else rf"$p={p:.3f}$"


def emit_b7(d):
    style = {"beam31": ("*", "oiBlue"), "beam33": ("square*", "oiGreen"),
             "bear31": ("triangle*", "oiPurple"), "bear33": ("diamond*", "oiVerm")}
    m = d["_meta"]
    plots = []
    for k, (mk, col) in style.items():
        z = d[k]
        co = " ".join(f"({p['n']},{100.0 * p['coll'] / p['of']:.2f})" for p in z["points"])
        # the one curve that does not clear alpha is drawn dashed, so the exception is
        # visible and not merely readable
        dash = "" if z["significant"] else ", densely dashed"
        plots.append(
            rf"  \addplot[{col}, mark={mk}, mark size=1.7pt, line width=1.0pt{dash}] "
            rf"coordinates {{{co}}};" + "\n"
            rf"  \addlegendentry{{{z['label']} ($\rho={z['spearman']:.2f}$, "
            rf"{_pfmt(z['p'])})}}")
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[name=ax, scale only axis, width=62mm, height=38mm, xmode=log, log basis x=2,
  xmin=6.4, xmax=330, ymin=-5, ymax=106,
  xtick={{8,16,32,64,125,250}}, xticklabels={{8,16,32,64,125,250}},
  ytick={{0,25,50,75,100}},
  label style={{font=\footnotesize}}, tick label style={{font=\footnotesize}},
  xlabel={{chirps per frame, the sensing airtime}},
  ylabel={{degenerate runs (\%)}},
  title={{\small\textbf{{Even the failure mode is dose-dependent}}}},
  legend style={{at={{(0.5,-0.30)}}, anchor=north, legend columns=2, font=\scriptsize,
                 draw=none, fill=none, legend cell align=left,
                 /tikz/every even column/.append style={{column sep=3mm}}}}, clip=false]
{chr(10).join(plots)}
\end{{axis}}
\end{{tikzpicture}}"""


def captions(data):
    """Captions with every number injected from the same data the figures were built from."""
    out = ["# Paper B figure captions",
           "",
           "Generated by `make_paper_b_figures.py`. Every number here is injected from the",
           "same results JSON the figure was built from, so a caption cannot drift from its",
           "panel. Do not edit by hand.", ""]
    if "B1" in data:
        d = data["B1"]; a, b = d["sc31"], d["sc33"]
        rise31 = a["points"][-1]["med"] - a["points"][0]["med"]
        rise33 = b["points"][-1]["med"] - b["points"][0]["med"]
        out += [
            "## B1",
            "",
            f"**Beam-prediction accuracy rises with sensing airtime on both scenarios, and on "
            f"neither does it reach what the partner's position alone supplies.** Median top-1 "
            f"over 15 seeds against chirps per frame, with bootstrap 95% confidence intervals. "
            f"The dashed rule and the shaded band below it are the no-information bound, the "
            f"best constant predictor evaluated on the test split ({a['bound']:.4f} on "
            f"Scenario 31, {b['bound']:.4f} on Scenario 33). Filled markers are settings whose "
            f"bootstrap interval excludes that bound; open markers are the "
            f"{a['n_miss'] + b['n_miss']} that do not, both on Scenario 33 at the two shortest "
            f"frames, where the degenerate-run counts are also highest. The dotted rule is the "
            f"position-based reference predictor, built from unit 2's position through the same "
            f"label and split, with its family and hyperparameter chosen on an inner VALIDATION "
            f"block and evaluated once ({a['posref']:.4f} and {b['posref']:.4f}). It is one "
            f"concrete predictor and a control, not an upper bound on achievable accuracy: "
            f"position-based beam selection is established prior art. Accuracy rises by "
            f"{rise31:.4f} on Scenario 31 and {rise33:.4f} on Scenario 33 across a 31-fold "
            f"increase in airtime, with no saturation inside the range the recording provides, "
            f"reaching {a['share']:.1f}% and {b['share']:.1f}% of the gain the position "
            f"reference supplies over the bound. Grey labels give the number of degenerate "
            f"runs, shown only where non-zero; that count itself falls as resources rise. Both "
            f"panels use the block-maximum readout.",
            ""]
    if "B0" in data:
        d = data["B0"]
        out += ["## B0", "",
                f"**Three time scales, each tier a magnified interval of the one above it.** "
                f"Tier (a) is one refresh of the measurement: radar frames recur at a logging "
                f"period whose median over all timestamps is {d['frame_period_ms']:.1f} ms, and "
                f"the chosen beam stays within the half-power beamwidth for "
                f"{d['coherence_ms']:.0f} ms, {d['coherence_ratio']:.1f} periods. Tier (b) "
                f"magnifies one frame by {d['zoom_ab']:.0f} and places it beside one alignment: "
                f"an exhaustive sweep costs {d['exhaustive']:.0f} us, "
                f"{d['duty_align_pct']:.2f}% of the period. Tier (c) magnifies a further "
                f"{d['zoom_bc']:.0f} to one chirp of N_s = {d['n_samp']} samples on each of "
                f"{d['n_rx']} elements and one sector slot of T_sec = {d['T_sec']:g} us, with "
                f"the k = {d['k']} shortlist shaded. T_chirp is not logged with the recording, "
                f"so the frame is drawn across the two chirp durations the dedicated regime is "
                f"priced at, {d['T_chirp_lo']:g} and {d['T_chirp_hi']:g} us, giving "
                f"{d['frame_lo_us']/1000:.2f} to {d['frame_hi_us']/1000:.2f} ms; every other "
                f"interval is measured or quoted from its source. The radar occupies "
                f"{d['radar_band']} GHz and the link {d['link_band']} GHz on separate radios, "
                f"so the two intervals overlap without contending; in a single-radio system "
                f"they would be sequential.", ""]
    if "B5" in data:
        d = data["B5"]
        out += ["## B5", "",
                f"**What the prediction is worth depends on who pays for the radar frame.** "
                f"Both panels share one logarithmic axis giving the airtime the link spends "
                f"before it can transmit, and the same two reference rules, an exhaustive "
                f"sweep at {d['exhaustive']:.1f} us and a fixed shortlist drawn from the "
                f"training prior at {d['prior']:.1f} us. They differ only in whether the "
                f"frame's own transmission is charged. Left, the reuse regime: the frame is "
                f"acquired for another function, so the predictor pays only for beam "
                f"training, and at the full frame it costs "
                f"{d['reuse'][-1]['total']:.1f} us, saving {abs(d['saving_vs_prior']):.1f} us "
                f"against the shortlist and {abs(d['saving_vs_exhaustive']):.1f} against an "
                f"exhaustive sweep; the saving survives the more pessimistic model that "
                f"restarts the sweep on a miss. Two comparators that need information a "
                f"radar-only predictor does not have are drawn for scale, the previous beam "
                f"at {d['persistence']:.1f} us and the partner's position at "
                f"{d['gps']:.1f} us. Right, the dedicated regime: a single radio must "
                f"transmit the frame in the link's own band first, shown at two chirp "
                f"durations bracketing automotive front ends. No setting reaches the "
                f"shortlist rule; the closest, {d['near_n']} chirps at "
                f"{d['near_adc']:.0f} Msps, is short by {d['near_gap']:.1f} us, and "
                f"break-even would need a chirp shorter than {d['breakeven_Tc']:.3f} us, "
                f"that is {d['breakeven_rate']:.0f} Msps. "
                f"{d['n_beats_exhaustive']} of the {d['n_dedicated']} dedicated settings do "
                f"beat an exhaustive sweep, all of them at the shortest frames.", ""]
    if "B2" in data:
        d = data["B2"]
        out += ["## B2", "",
                f"**The benchmark orders the settings and overstates the rate.** Measured "
                f"bearing error against the resource product R = N_s N_c m(m^2-1), fitted with "
                f"RMSE^2 = a R^-b + c so that the irreducible floor cannot be absorbed into the "
                f"exponent. The fitted b = {d['b']:.3f}, 95% interval "
                f"[{d['b_lo']:.3f}, {d['b_hi']:.3f}], excludes the b = 1 the Cramer-Rao bound "
                f"predicts, and the floor is {d['floor']:.2f} degrees against a no-information "
                f"reference of {d['bound']:.3f}. The interval is the joint-seed median "
                f"bootstrap from corrected_statistics.json, not a bootstrap computed here: an "
                f"earlier version resampled the MEAN while the point estimate used the MEDIAN, "
                f"which gave an interval narrower than the estimator supports. The "
                f"reduced-aperture point lies below the fit because the aperture factor inflates "
                f"R without buying accuracy on these recordings, which is why the fit holds "
                f"m = 4 fixed; it carries the same seed spread as every other point, so the "
                f"departure can be judged. The m = 1 setting has m(m^2-1) = 0, so R = 0 and it "
                f"has no position on a logarithmic axis at all; it is annotated rather than "
                f"dropped, because reaching {d['zero_rmse']:.2f} degrees with no angular "
                f"information whatever is the sharpest evidence that the aperture term is "
                f"inert here. Error bars are the seed standard deviation.", ""]
    if "B3" in data:
        d = data["B3"]
        out += ["## B3", "",
                f"**Two tasks, two labels, one relationship.** Beam-prediction gain over the "
                f"no-information bound against measured bearing error, one point per resource "
                f"setting, Pearson r = {d['r']:+.4f} and Spearman {d['rho']:+.4f}. The bearing "
                f"regressor never sees the beam label and the beam readout never sees the "
                f"bearing, so this is an association between two measurements rather than "
                f"between a quantity and a transform of itself, which is what the retired "
                f"measurement-utilization correlation was. It is a cross-task association and "
                f"not an established mediation: it rests on ten settings, and it does not "
                f"replicate on Scenario 33 once the settings whose bearing instrument failed "
                f"are excluded.", ""]
    if "B4" in data:
        d = data["B4"]
        flipped = [v for v in d["verdicts"] if not v["verdict_unchanged"]]
        out += ["## B4", "",
                f"**Changing one line of the input resampler moves every setting and reverses "
                f"{len(flipped)} of {d['n_total']} conclusions.** The original reduced both "
                f"radar axes by selecting evenly spaced indices and discarding every bin "
                f"between them; the replacement block-reduces by taking the maximum. Below the "
                f"target size the two coincide, so this is a single-factor comparison. "
                f"Block-max wins all {len(d['rows'])}/{len(d['rows'])} settings, mean "
                f"+{d['mean_delta']:.4f}, every p <= {d['max_p']:.4f}, against a "
                f"no-information bound of {d['bound']:.4f}. The setting that sat lowest under "
                f"decimation, {d['worst']}, was {d['worst_dec_margin']:.4f} above that bound "
                f"and is {d['worst_max_margin']:.4f} above it under block-max. The mean and "
                f"the worst p are RETURNED by build_b4, not computed inside the emitter where "
                f"nothing could check them. Rows are ordered by their block-max median, so the "
                f"decimate values are deliberately not monotone.", ""]
    if "B6" in data:
        d = data["B6"]
        out += ["## B6", "",
                f"**Aperture buys little on either task.** Left, top-1 beam accuracy against "
                f"the number of receive elements, between the no-information bound "
                f"({d['beam_bound']:.4f}) and the validation-selected position reference "
                f"({d['ceiling']:.4f}); the array accounts for {d['share_beam']:.1f}% of the "
                f"gain over the bound. Right, bearing error over the same sweep, against its "
                f"own bound ({d['bear_bound']:.3f} deg) and a predictor using the partner's "
                f"RANGE alone with no radar at all ({d['range_only']:.3f} deg, dash-dotted). "
                f"A single element, whose azimuthal structure is exactly zero and whose "
                f"bearing information is identically zero, already reaches "
                f"{d['share_bear']:.0f}% of the reduction below the bound. The three points "
                f"differ only in m: all are at {d['n_samp']} samples and {d['n_chirp']} chirps, "
                f"which is what makes this a one-factor aperture ablation rather than a "
                f"resource sweep. Error bars on BOTH panels are the seed standard deviation; "
                f"the beam panel carried none until 2026-10-08, which left the flatness "
                f"unjudgeable. On a repeated route the geometry supplies the bearing, so "
                f"removing the array barely moves either metric.", ""]
    if "B7" in data:
        d = data["B7"]; m = d["_meta"]
        rows = ", ".join(
            f"{d[k]['label']} rho {d[k]['spearman']:+.2f} (p {d[k]['p']:.3f}, "
            f"{d[k]['n_nonzero']} of {len(d[k]['points'])} settings non-zero)"
            for k in ("beam31", "beam33", "bear31", "bear33"))
        out += ["## B7", "",
                f"**Even the failure mode is dose-dependent.** Percentage of training runs "
                f"landing within tolerance of a no-information reference, against sensing "
                f"airtime, for both tasks on both scenarios. Counts are out of "
                f"{d['beam31']['seeds']} seeds on the beam task and {d['bear31']['seeds']} on "
                f"the bearing task, which is why the axis is a percentage. {rows}. "
                f"{m['n_sig']} of the {m['n_curves']} decline significantly at alpha = "
                f"{m['alpha']:g} by the EXACT enumeration Sec. V-B specifies. The "
                f"{m['n_weak']} that do not are drawn DASHED: "
                + "; ".join(f"{d[k]['label']} (p {d[k]['p']:.3f}, degenerate runs at "
                            f"{d[k]['n_nonzero']} of {len(d[k]['points'])} settings)"
                            for k in m['weak'])
                + f". Treated as one confirmatory family, a Holm correction leaves only "
                f"{d['bear33']['label']} significant. Until 2026-10-08 this figure printed "
                f"scipy's ASYMPTOTIC p, which Sec. V-B itself warns is optimistic here: it "
                f"gave 0.158, 0.008, 0.034 and 0.000 against the exact 0.333, 0.022, 0.067 "
                f"and 0.006, and made three trends look significant instead of two. The "
                f"optimizer's "
                f"ability to find any solution better than a constant is itself a function of "
                f"the sensing resource, which supports the dose-response without requiring the "
                f"degenerate runs' scores to be interpretable.", ""]
    if "B9" in data:
        d = data["B9"]
        out += [
            "## B9",
            "",
            f"**The deployment and the measurement.** (a) Unit 1 is stationary. Its "
            f"{d['n_elem']}-element array carries a {d['n_beam']}-beam receive codebook over "
            f"{d['fov']:g} degrees. The beam directions are SCHEMATIC: the dataset "
            f"establishes neither uniform angular spacing nor calibrated beam centers, so no "
            f"angular width is claimed, and the {d['k']} highlighted directions are drawn "
            f"apart because a learned top-{d['k']} is not a contiguous sector. The link "
            f"arrow points INTO unit 1, because the "
            f"codebook combines on receive. The radar is a second radio on the same mast in a "
            f"second band ({d['band_radar']} GHz against {d['band_link']}), which is why "
            f"sensing time is not link airtime here and the cost model has three regimes. The "
            f"partner is {d['rng_lo']:.1f} to {d['rng_hi']:.1f} m away over the recording, "
            f"inside the {d['reach_m']:g} m rated reach. (b) One knob is cut per setting; the "
            f"shortlist is scored three ways and read between the no-information bound "
            f"({d['bound']:.4f}) and the validation-selected position reference "
            f"({d['gps']:.4f}).",
            "",
        ]
    # The blocks above are written in whatever order the code happens to run; emit them in
    # figure order so the file reads alongside the paper.
    head, blocks, cur = [], {}, None
    for line in out:
        if line.startswith("## B"):
            cur = line.split()[1]
            blocks[cur] = [line]
        elif cur is None:
            head.append(line)
        else:
            blocks[cur].append(line)
    for k in sorted(blocks):
        head += blocks[k]
    return "\n".join(head)


# ----------------------------------------------------------------- B8
# Sensing duration and shortlist length are two knobs on one budget, and until 2026-10-09 the
# paper varied only the first at a fixed k=5. Under the fixed-shortlist policy, which probes k
# beams and keeps the best measured one, both cost and quality are computable for every
# (setting, k) already saved, with no oracle anywhere. The point of the figure is that the
# cheap-frame/long-shortlist corner dominates the expensive-frame/short-shortlist corner in
# the dedicated regime, and that the regime decides which corner wins.
def build_b8():
    LBL_MM, PANEL_MM = 2.7, 25.0      # one \scriptsize line, and the axis height emit_b8 uses
    Q = J("beam_power_quality.json")["scenarios"]
    ST = J("static_shortlist.json")["scenarios"]
    out = {}
    for sc in ("31", "33"):
        F = Q[sc]["frontier"]
        pts = F["points"]
        sel = F["selection"]["dedicated"]
        S = ST[sc]
        # one polyline per chirp count over k, so the two knobs stay visually separable,
        # and the remaining settings as unconnected marks. Drawing only the chirp families
        # would make the word "frontier" false: the sample-count and aperture settings are
        # candidates for the selection and one of them is nondominated on Scenario 33.
        # EVERY length the selection searches, read from the grid the selector itself
        # built. The six stored frontier points are a sample of that grid; drawing the
        # sample made the prose's "over the same shortlist lengths" false and left the
        # two radar policies Section VI-D compares off the figure. Caught 2026-10-14.
        grid = list(S["radar_grid"].values())
        chirp, other = {}, {}
        for r in grid:
            tgt_ = chirp if (r["n_samples"] == 256 and r["n_rx"] == 4) else other
            tgt_.setdefault(r["n_chirps"] if tgt_ is chirp else r["setting"],
                            []).append(r)
        for d_ in (chirp, other):
            for v in d_.values():
                v.sort(key=lambda r: r["k"])
        assert sum(len(v) for v in chirp.values()) \
            + sum(len(v) for v in other.values()) == len(grid) == 160, len(grid)
        # the six stored points must lie on the drawn curves, so the denser grid cannot
        # silently disagree with the frontier JSON it replaces
        _by = {(r["setting"], r["k"]): r for r in grid}
        for r in pts:
            assert abs(_by[(r["setting"], r["k"])]["test"]["mean_regret_db"]
                       - r["test"]["mean_regret_db"]) < 1e-9, r["setting"]
            assert abs(_by[(r["setting"], r["k"])]["cost_dedicated_us"]
                       - r["cost_dedicated_us"]) < 1e-9, r["setting"]
        refs = {k: sorted(v, key=lambda r: r["k"])
                for k, v in F["reference_curves"].items()}
        # THE PRIOR THE SELECTOR USES, not the one the earlier JSON carries. The two are
        # both correct and describe different policies: this one is fitted on the finite
        # optimization rows, the reference-curve one on the whole training region, and on
        # Scenario 33 at k=16 they differ by 0.06 dB. Drawing one while selecting with the
        # other put two policies under one name. Caught by the 2026-10-13 audit.
        pri = [{"k": int(k), "cost_us": float(k) * F["T_sec_us"],
                "test_mean_regret_db": S["prior_test"][str(k)]["mean_regret_db"]}
               for k in sorted(int(x) for x in S["prior_test"])]
        assert len(pri) == 16, len(pri)
        assert all(r["cost_us"] == r["k"] * F["T_sec_us"]
                   for v in refs.values() for r in v)
        assert all(r["cost_us"] == r["k"] * F["T_sec_us"] for r in pri)
        # the power-aware static policy: no frame to acquire, so its dedicated cost is the
        # probing cost alone, which is what places it on this figure's x axis
        stat = [{"k": int(k), "cost_us": float(k) * F["T_sec_us"],
                 "regret": S["test"][str(k)]["mean_regret_db"]}
                for k in sorted(int(x) for x in S["test"])]
        picked = S["selection_dedicated"]["selected"]
        default = sel["default"]
        # Assert the data, not the conclusion. An earlier version asserted that the selected
        # point beats the default on test, which makes a hypothesis a precondition of a
        # valid build: a rule fixed on validation is entitled to lose on test.
        assert picked["val_mean_regret_db"] <= sel["validation_target_db"] + 1e-12
        assert picked["cost_us"] <= min(
            r["cost_dedicated_us"] for r in pts
            if r["val"]["mean_regret_db"] <= sel["validation_target_db"] + 1e-12)
        xs = ([r["cost_dedicated_us"] for r in grid] + [r["cost_us"] for r in pri]
              + [r["cost_us"] for r in stat])
        ys = ([r["test"]["mean_regret_db"] for r in grid]
              + [r["test_mean_regret_db"] for r in pri] + [r["regret"] for r in stat])
        out[sc] = {"chirp": {str(k): v for k, v in sorted(chirp.items())},
                   "other": {k: v for k, v in sorted(other.items())},
                   "n_radar_points": len(grid),
                   "prior": pri, "refs": refs, "static": stat,
                   "selected": picked, "default": default,
                   "sel_regret": picked["test"]["mean_regret_db"],
                   "sel_cost": picked["cost_us"], "sel_policy": picked["policy"],
                   "ratio": default["cost_dedicated_us"] / picked["cost_us"],
                   "target_db": sel["validation_target_db"],
                   "n_meeting": S["selection_dedicated"]["n_meeting_target"],
                   "n_points": len(pts), "n_cand": S["selection_dedicated"]["n_candidates"],
                   "n_other": len(other),
                   "xmin": min(xs) * 0.7, "xmax": max(xs) * 1.45,
                   # THE HEADROOM HOLDS A LABEL, so it is solved for rather than set as a
                   # fixed multiple. The chirp-family labels and the prior's label are
                   # anchored south at the topmost point they describe and rise one
                   # \scriptsize line above it. A flat *1.12 was tuned when this panel was
                   # 38 mm, where that line is 7.1 % of the range; at 25 mm it is 10.8 %
                   # against 10.7 % available, and the Scenario 31 prior label came within
                   # one pixel of the frame. Audited 2026-10-14.
                   "ymin": 0.0,
                   "ymax": round(max(ys) / (1.0 - (LBL_MM + 0.5) / PANEL_MM), 4),
                   "label_allowance_mm": LBL_MM, "panel_mm": PANEL_MM,
                   "tsec": F["T_sec_us"], "fs": F["fs_msps"]}
    return out


def emit_b8(d):
    def panel(sc, name, at):
        z = d[sc]
        segs, marks = [], []
        # every drawn radar point, so a family label can be lifted clear of its NEIGHBORS
        # and not merely of its own curve. The aperture settings run at 250 chirps and 256
        # samples, so their dedicated cost coincides with the chirps=250 family's exactly,
        # and samples=128 coincides with chirps=125: the equal-payload pairs of Table IV
        # are equal-cost here too, which puts other families' marks at a label's own x.
        allpts = [r for rows in z["chirp"].values() for r in rows] \
            + [r for rows in z["other"].values() for r in rows]
        for nc, rows in z["chirp"].items():
            co = " ".join(f"({r['cost_dedicated_us']:.2f},{r['test']['mean_regret_db']:.4f})"
                          for r in rows)
            segs.append(rf"  \addplot[oiBlue, line width=0.7pt, mark=*, mark size=1.0pt,"
                        rf" forget plot] coordinates {{{co}}};")
            hi = rows[0]           # the k=1 end, the top of each line, where the panel is empty
            near = [r["test"]["mean_regret_db"] for r in allpts
                    if 0.87 <= r["cost_dedicated_us"] / hi["cost_dedicated_us"] <= 1.15]
            marks.append(rf"  \node[font=\scriptsize,oiBlue,anchor=south,inner sep=2.6pt] at "
                         rf"(axis cs:{hi['cost_dedicated_us']:.2f},"
                         rf"{max(near):.4f}) {{{nc}}};")
        # the sample-count and aperture settings, each now a family over k like the chirp
        # families, drawn lighter so the two knobs stay visually separable
        for _s, rows in z["other"].items():
            oc = " ".join(
                f"({r['cost_dedicated_us']:.2f},{r['test']['mean_regret_db']:.4f})"
                for r in rows)
            segs.append(rf"  \addplot[oiSky, line width=0.5pt, mark=diamond,"
                        rf" mark size=1.2pt, forget plot] coordinates {{{oc}}};")
        st = " ".join(f"({r['cost_us']:.2f},{r['regret']:.4f})" for r in z["static"])
        segs.append(rf"  \addplot[goodLine, line width=0.9pt, mark=square*, mark size=1.1pt,"
                    rf" forget plot] coordinates {{{st}}};")
        # Two no-sensing policies, differing only in the objective their fixed probe set was
        # chosen for. The previous-beam and partner-position comparators need information a
        # radar-only predictor does not have and are reported in Table I, not here.
        rl = []
        # THE SELECTOR'S OWN PRIOR. `z["refs"]` carries the older policy fitted on the
        # whole training region; on Scenario 33 at k=16 the two differ by 0.062 dB, so
        # drawing one while selecting with the other put two policies under one name.
        # build_b8 was corrected on 2026-10-13 and THIS LINE WAS NOT, so the repair never
        # reached the rendered figure. Caught 2026-10-14. The verifier now reads the
        # emitted coordinates back and compares them with S["prior_test"].
        pr = z["prior"]
        assert len(pr) == 16, len(pr)
        co = " ".join(f"({r['cost_us']:.2f},{r['test_mean_regret_db']:.4f})" for r in pr)
        rl.append(rf"  \addplot[draw=gGrey, dashed, line width=0.7pt, mark=square,"
                  rf" mark size=1.0pt, forget plot] coordinates {{{co}}};")
        e = pr[0]
        rl.append(rf"  \node[font=\scriptsize,gGrey,anchor=south west,inner sep=1.2pt] at "
                  rf"(axis cs:{e['cost_us']:.2f},{e['test_mean_regret_db']:.4f}) "
                  rf"{{frequency prior}};")
        # the label goes in the empty band under the static curve, not on its last point,
        # where the selection marker already is
        rl.append(rf"  \node[font=\scriptsize,goodLine,anchor=east,inner sep=1.2pt] at "
                  rf"(axis cs:{z['sel_cost'] / 1.32:.2f},{0.055 * z['ymax']:.4f}) "
                  rf"{{power-aware static}};")
        f_ = z["default"]
        return rf"""\begin{{axis}}[name={name}, {at} scale only axis, width=66mm, height=25mm,
  xmode=log, xmin={z['xmin']:.2f}, xmax={z['xmax']:.2f},
  ymin={z['ymin']:.3f}, ymax={z['ymax']:.3f},
  xlabel={{modeled dedicated airtime ($\mu$s)}}, ylabel={{mean regret (dB)}},
  label style={{font=\footnotesize}}, tick label style={{font=\footnotesize}},
  title={{\small\textbf{{Scenario {sc}}}}}, clip=false]
{chr(10).join(rl)}
{chr(10).join(segs)}
{chr(10).join(marks)}
  \addplot[only marks, mark=o, mark size=3.2pt, oiVerm, line width=1.0pt, forget plot]
    coordinates {{({f_['cost_dedicated_us']:.2f},{f_['test']['mean_regret_db']:.4f})}};
  %% The star marks what the selection returns, which is this figure's conclusion. It sits
  %% ON the green static curve and was drawn in the curve's own color, so the paper's answer
  %% was the least visible mark on the panel while the default comparator circle stood out.
  %% A white disc lifts it off the line before the star is drawn over it.
  \addplot[only marks, mark=*, mark size=3.4pt, white, forget plot]
    coordinates {{({z['sel_cost']:.2f},{z['sel_regret']:.4f})}};
  \addplot[only marks, mark=star, mark size=4.2pt, goodLine, line width=1.3pt, forget plot]
    coordinates {{({z['sel_cost']:.2f},{z['sel_regret']:.4f})}};
\end{{axis}}"""
    return rf"""\begin{{tikzpicture}}
{panel('31', 'p0', '')}
{panel('33', 'p1', 'at=(p0.right of south east), anchor=left of south west, xshift=14mm,')}
\end{{tikzpicture}}"""


# ----------------------------------------------------------------- B9
# Figure 1: what the deployment is and what the measurement chain does to it. Every quantity
# drawn here is read from a results JSON, so the orienting figure cannot drift from the
# experiment it orients. The band plan is the part most often got wrong about this testbed:
# radar and link are separate radios, so radar time is not communication airtime, and that is
# why the cost model of Section III-B has three regimes rather than one.
def build_b9():
    T = J("testbed_facts.json")
    A = J("sc31_resource_ablation_G_max.json")
    R = J("sc31_reference_predictors.json")["predictors"]
    C = J("alignment_cost_model.json")
    gps = [v for k_, v in R.items() if k_.startswith("GPS reference")][0]
    nrx, nadc, nch = T["cube_shape"]
    assert T["separate_radios"] is True
    assert T["n_beams"] == C["n_beam"] and nadc == C["n_adc"]
    nb, fov, tsec = T["n_beams"], T["array_fov_deg"], C["T_sec_us"]
    # k is not a free literal: it is the shortlist length the cost model was priced at, and
    # the identity below recovers it from the priced airtime, so a change to the cost model
    # fails the build instead of leaving the figure drawing a shortlist nobody priced.
    k = 5
    S = C["scenarios"]["31"]
    approx(k * tsec + (1.0 - S["p5_prior"]) * (nb - k) * tsec, S["airtime_prior_us"], 1e-6)
    # NO ANGULAR CLAIM IS MADE ABOUT THE SHORTLIST. An earlier draft drew it as a sector
    # of exactly k * fov / n_beam degrees. A 64-beam codebook over a 90-degree field of
    # view establishes neither uniform angular spacing nor calibrated beam centers, and a
    # learned top-5 is five consecutive indices on a minority of frames, so a contiguous
    # wedge asserted two things the data does not support. The panel now highlights k
    # illustrative candidate directions, spread, and says it is schematic.
    # The partner's measured range over the whole recording, against the radar's rated reach.
    z = np.load(os.path.join(RES, "sc31_bearing_labels.npz"), allow_pickle=True)
    rlo, rhi = float(z["range_m"].min()), float(z["range_m"].max())
    assert rhi < T["radar_max_range_m"], "the measured range must lie inside the rated reach"
    return {"n_elem": T["n_array_elements"], "n_beam": nb, "fov": fov,
            "band_link": T["link_band_ghz"], "band_radar": T["radar_band_ghz"],
            "radar_lo": 76.0, "radar_hi": 81.0, "rate_hz": T["modality_rate_hz"],
            "n_rx": nrx, "n_adc": nadc, "n_chirp": nch,
            "reach_m": T["radar_max_range_m"], "rng_lo": rlo, "rng_hi": rhi,
            "chirps": sorted({A["settings"][k_]["n_chirps"] for k_ in A["settings"]
                              if A["settings"][k_]["sweep"] == "chirps"}),
            "samples": sorted({A["settings"][k_]["n_samples"] for k_ in A["settings"]
                               if A["settings"][k_]["sweep"] == "samples"} | {nadc}),
            "rxs": sorted({A["settings"][k_]["n_rx"] for k_ in A["settings"]
                           if A["settings"][k_]["sweep"] == "antennas"} | {nrx}),
            "bound": A["bound_const"], "gps": gps["top1"], "grid": 64,
            "params": NET_PARAMS, "tsec": tsec, "k": k, "probe_sep_deg": 8.0}


def emit_b9(d):
    import math
    # Three rules this figure is built to, each of them a defect found in its first draft.
    #  1. EVERY text node carries an explicit text width. Without one a node runs to its
    #     natural single-line width, silently stretching the canvas and dropping the placed
    #     scale; the first draft came out at 0.70 for exactly that reason.
    #  2. NO node is placed where another one already is. The first draft put a 62 mm note
    #     under a 34 mm box that was taller than it looked, and the last line of the box was
    #     printed underneath the first line of the note in the submitted PDF. The figure
    #     verifier now re-reads the word boxes out of the compiled file and asserts that no
    #     two of them overlap, because a text-layer check cannot see a collision.
    #  3. The drawing may not contradict the prose. The array is broadside to its own fan,
    #     not endfire along it; the link arrow points INTO unit 1, because the codebook is a
    #     receive combiner and Section III-A says so in as many words; and the shortlist
    #     wedge subtends k of n_beam beams and nothing wider.
    AX, AY = 2.05, 3.35                      # array phase center, boresight toward the road
    RY = 2.28                                # the radar aperture, lower on the same mast
    VX, VY = 5.85, 3.95                      # the vehicle, on the road
    aim = math.degrees(math.atan2(VY - AY, VX - AX))
    half, r0, r1 = d["fov"] / 2, 1.52, 1.74
    # k illustrative candidate directions, drawn apart so that the shortlist does not read
    # as one contiguous sector, which it is not
    sep = d["probe_sep_deg"]
    off = [(i - (d["k"] - 1) / 2.0) * sep for i in range(d["k"])]
    probe = chr(10).join(
        rf"  \draw[oiBlue, line width={1.5 if abs(o) < 1e-9 else 0.9:.1f}pt] "
        rf"({AX:.2f},{AY:.2f}) -- "
        rf"({AX + (r1 + 0.10) * math.cos(math.radians(aim + o)):.3f},"
        rf"{AY + (r1 + 0.10) * math.sin(math.radians(aim + o)):.3f});" for o in off)
    # the arc band is the codebook; a countable set of rays would state a beam count the
    # drawing does not have room for, so the codebook is drawn as a graduated band instead.
    ticks = "\n".join(
        rf"  \draw[oiBlue!45, line width=0.3pt] "
        rf"({AX + r0 * math.cos(math.radians(aim - half + i * d['fov'] / 8)):.3f},"
        rf"{AY + r0 * math.sin(math.radians(aim - half + i * d['fov'] / 8)):.3f}) -- "
        rf"({AX + r1 * math.cos(math.radians(aim - half + i * d['fov'] / 8)):.3f},"
        rf"{AY + r1 * math.sin(math.radians(aim - half + i * d['fov'] / 8)):.3f});"
        for i in range(9))
    el = "\n".join(
        rf"  \draw[oiBlue, line width=0.45pt] ({AX - 0.05:.2f},{AY - 0.45 + 0.06 * i:.2f}) -- "
        rf"({AX + 0.05:.2f},{AY - 0.45 + 0.06 * i:.2f});" for i in range(d["n_elem"]))
    rx = "\n".join(
        rf"  \draw[oiPurple, line width=0.5pt] ({AX - 0.06:.2f},{RY - 0.17 + 0.11 * i:.2f}) -- "
        rf"({AX + 0.06:.2f},{RY - 0.17 + 0.11 * i:.2f});" for i in range(d["n_rx"]))
    fx = lambda f: 0.15 + (f - 55.0) * 0.1467          # the band strip's frequency axis
    blk = ("draw=gGrey!65, fill=white, rounded corners=1.5pt, inner sep=2.5pt, "
           "font=\\scriptsize, align=center")
    hot = ("rounded corners=1.5pt, inner sep=2.5pt, font=\\scriptsize, align=center")
    rng = lambda v: ", ".join(str(x) for x in v)
    return rf"""\begin{{tikzpicture}}[>=latex, line cap=round]
%% ---- (a) the deployment ---------------------------------------------------
\begin{{scope}}
  \node[font=\small\bfseries, anchor=south west, text width=42mm] at (-0.05,4.74)
    {{(a)~Two radios, two bands}};
  %% road and vehicle
  \fill[gGrey!12] (5.45,1.52) rectangle (6.25,4.34);
  \draw[gGrey!45, dashed, line width=0.4pt] (5.85,1.62) -- (5.85,4.24);
  \draw[oiBlue, fill=oiBlue!30, rounded corners=1pt, line width=0.5pt]
    ({VX - 0.14:.2f},{VY - 0.25:.2f}) rectangle ({VX + 0.14:.2f},{VY + 0.25:.2f});
  %% the codebook, drawn as a graduated band, and the shortlist at its true angular width
  \fill[oiBlue!7] ({AX:.2f},{AY:.2f})
    -- ++({aim - half:.2f}:{r1:.2f}) arc ({aim - half:.2f}:{aim + half:.2f}:{r1:.2f}) -- cycle;
  \draw[oiBlue!35, line width=0.4pt]
    ({AX + r0 * math.cos(math.radians(aim - half)):.3f},{AY + r0 * math.sin(math.radians(aim - half)):.3f})
    arc ({aim - half:.2f}:{aim + half:.2f}:{r0:.2f});
  \draw[oiBlue!35, line width=0.4pt]
    ({AX + r1 * math.cos(math.radians(aim - half)):.3f},{AY + r1 * math.sin(math.radians(aim - half)):.3f})
    arc ({aim - half:.2f}:{aim + half:.2f}:{r1:.2f});
{ticks}
{probe}
  %% the link, drawn from the vehicle INTO the array: the codebook combines on receive
  \draw[oiBlue, line width=1.0pt, ->]
    ({VX - 0.18:.2f},{VY - 0.03:.2f}) -- ({AX + r1 + 0.16:.2f},{AY + 0.10:.2f});
  %% the radar, a second radio: out and back on the same aperture
  \draw[oiPurple, line width=0.8pt, ->] ({AX + 0.14:.2f},{RY + 0.04:.2f})
    to[bend left=7] ({VX - 0.19:.2f},{VY - 0.34:.2f});
  \draw[oiPurple, line width=0.8pt, dashed, ->] ({VX - 0.22:.2f},{VY - 0.46:.2f})
    to[bend left=7] ({AX + 0.12:.2f},{RY - 0.12:.2f});
  %% the mast
  \draw[gGrey!70, line width=0.8pt] ({AX:.2f},1.52) -- ({AX:.2f},{AY + 0.52:.2f});
  \draw[gGrey!70, line width=0.8pt] ({AX - 0.22:.2f},1.52) -- ({AX + 0.22:.2f},1.52);
  \draw[oiBlue, line width=1.2pt] ({AX:.2f},{AY - 0.47:.2f}) -- ({AX:.2f},{AY + 0.47:.2f});
{el}
  \draw[oiPurple, line width=0.9pt] ({AX:.2f},{RY - 0.19:.2f}) -- ({AX:.2f},{RY + 0.19:.2f});
{rx}
  %% labels, each attached to the object it names
  \node[font=\scriptsize\bfseries, anchor=south, text width=34mm] at ({AX:.2f},{AY + 0.60:.2f})
    {{unit~1, stationary}};
  \node[font=\scriptsize, oiBlue, anchor=east, align=right, text width=21mm]
    at ({AX - 0.16:.2f},{AY + 0.06:.2f})
    {{{d['n_elem']}-element array\\ \textbf{{{d['n_beam']} beams}} over {d['fov']:g}$^\circ$}};
  \node[font=\scriptsize, oiPurple, anchor=east, align=right, text width=21mm]
    at ({AX - 0.16:.2f},{RY - 0.02:.2f})
    {{FMCW radar\\ ${d['n_rx']}{{\times}}{d['n_adc']}{{\times}}{d['n_chirp']}$ per frame\\
      at {d['rate_hz']}\,Hz, {d['reach_m']:g}\,m reach}};
  \node[font=\scriptsize, anchor=west, align=left, text width=19mm] at (6.38,{VY:.2f})
    {{\textbf{{unit~2}}\\ vehicle,\\ quasi-omni}};
  \node[font=\scriptsize, oiBlue, anchor=south east, align=right, text width=30mm]
    at (5.55,4.12) {{probe $k{{=}}{d['k']}$ of {d['n_beam']},
      ${d['tsec']:g}\,\mu$s each, assumed\\ (directions schematic)}};
  \node[font=\scriptsize, oiPurple, anchor=north west, align=left, text width=23mm]
    at (3.05,2.50) {{partner {d['rng_lo']:.0f} to {d['rng_hi']:.0f}\,m,
      Scenario 31}};
  %% the two bands, drawn rather than asserted
  \draw[gGrey!70, line width=0.5pt, ->] (0.15,0.92) -- (4.72,0.92);
  \fill[oiBlue] ({fx(60) - 0.055:.3f},0.92) rectangle ({fx(60) + 0.055:.3f},1.16);
  \fill[oiPurple] ({fx(d['radar_lo']):.3f},0.92) rectangle ({fx(d['radar_hi']):.3f},1.16);
  \node[font=\scriptsize, oiBlue, anchor=north, text width=22mm]
    at ({fx(60):.3f},0.86) {{{d['band_link']}\,GHz\\ link}};
  \node[font=\scriptsize, oiPurple, anchor=north, text width=22mm]
    at ({(fx(d['radar_lo']) + fx(d['radar_hi'])) / 2:.3f},0.86)
    {{{d['band_radar']}\,GHz\\ radar}};
  \node[font=\scriptsize, gGrey, anchor=west, text width=12mm] at (4.78,0.92) {{GHz}};
\end{{scope}}
%% ---- (b) the measurement chain --------------------------------------------
\begin{{scope}}[xshift=8.30cm]
  \node[font=\small\bfseries, anchor=south west, text width=60mm] at (-0.05,4.74)
    {{(b)~From cube to shortlist}};
  \node[{blk}, text width=19mm, minimum height=10mm, anchor=west] (cube) at (-0.05,4.00)
    {{raw cube\\ ${d['n_rx']}{{\times}}{d['n_adc']}{{\times}}{d['n_chirp']}$}};
  \node[draw=oiVerm, fill=oiVerm!8, {hot}, text width=33mm, minimum height=11mm,
        anchor=west] (cut) at (2.85,4.00)
    {{\textbf{{one knob at a time}}\\[1pt]
      $N_\mathrm{{c}}$ {rng(d['chirps'])}\\
      $N_\mathrm{{s}}$ {rng(d['samples'])} \quad $m$ {rng(d['rxs'])}}};
  \node[{blk}, text width=20mm, minimum height=10mm, anchor=west] (tf) at (7.15,4.00)
    {{range, Doppler\\ and angle\\ transforms}};
  \node[{blk}, text width=19mm, minimum height=8.5mm, anchor=west] (maps) at (-0.05,2.66)
    {{two ${d['grid']}{{\times}}{d['grid']}$\\ maps}};
  \node[draw=oiBlue!70, fill=oiBlue!7, {hot}, text width=26mm, minimum height=8.5mm,
        anchor=west] (cnn) at (2.85,2.66)
    {{learned readout\\ {d['params']} weights}};
  \node[draw=oiBlue!70, fill=oiBlue!7, {hot}, text width=20mm, minimum height=8.5mm,
        anchor=west] (sl) at (7.15,2.66) {{shortlist of\\ $k$ beams}};
  \draw[->, gGrey!75, line width=0.5pt] (cube) -- (cut);
  \draw[->, gGrey!75, line width=0.5pt] (cut) -- (tf);
  %% the offset sets this connector's FINAL DROP, which carries an arrowhead: at
  %% -0.25 against the raised middle row it was 1.65 mm, shorter than the fanout
  %% drops below it. -0.20 puts the horizontal run 1.5 mm under the box above and
  %% leaves a 2.15 mm drop.
  \draw[->, gGrey!75, line width=0.5pt] (tf.south) -- ++(0,-0.20) -| (maps.north);
  \draw[->, gGrey!75, line width=0.5pt] (maps) -- (cnn);
  \draw[->, gGrey!75, line width=0.5pt] (cnn) -- (sl);
  %% The shortlist is scored three ways, and the three are different currencies.
  %% THE SCORING BOXES ARE DECLARED BEFORE THE CONNECTORS, for two reasons. A filled node
  %% drawn afterwards paints over an arrowhead that lands on its border, which is half of
  %% why these three arrows did not show; and every endpoint below is then taken from a
  %% node anchor instead of a typed coordinate, so the bus cannot overshoot the junction
  %% or the boxes drift out from under it. The drop is \BUSDROP, long enough that a latex
  %% arrowhead has visible shaft behind it: at 0.9 mm the head was longer than its own
  %% segment. Audited 2026-10-14.
  \node[draw=oiVerm!75, fill=oiVerm!8, {hot}, text width=23mm, minimum height=5.5mm,
        anchor=north] (s1) at (1.35,1.73) {{top-$k$ inclusion}};
  \node[draw=oiVerm!75, fill=oiVerm!8, {hot}, text width=25mm, minimum height=5.5mm,
        anchor=north] (s2) at (4.45,1.73) {{power regret $L_k$}};
  \node[draw=oiVerm!75, fill=oiVerm!8, {hot}, text width=27mm, minimum height=5.5mm,
        anchor=north] (s3) at (7.75,1.73) {{alignment airtime $A(k)$}};
  \draw[gGrey!75, line width=0.5pt] (sl.south) -- (sl.south |- 0,1.98);
  \draw[gGrey!75, line width=0.5pt] (s1.north |- 0,1.98) -- (sl.south |- 0,1.98);
  \draw[->, gGrey!75, line width=0.5pt] (s1.north |- 0,1.98) -- (s1.north);
  \draw[->, gGrey!75, line width=0.5pt] (s2.north |- 0,1.98) -- (s2.north);
  \draw[->, gGrey!75, line width=0.5pt] (s3.north |- 0,1.98) -- (s3.north);
  %% and every learned number is read between two references on the same split
  \draw[goodLine, line width=0.7pt, <->, >=latex] (1.05,0.98) -- (8.30,0.98);
  \node[font=\scriptsize, anchor=north west, align=left, text width=32mm] at (0.95,0.88)
    {{Scenario 31: bound\\ {d['bound']:.4f} top-1}};
  \node[font=\scriptsize, anchor=north east, align=right, text width=32mm] at (8.40,0.88)
    {{position reference\\ {d['gps']:.4f} top-1}};
\end{{scope}}
\end{{tikzpicture}}"""


FIGS = {"B1": (build_b1, emit_b1), "B2": (build_b2, emit_b2),
        "B3": (build_b3, emit_b3), "B4": (build_b4, emit_b4),
        "B5": (build_b5, emit_b5), "B6": (build_b6, emit_b6),
        "B7": (build_b7, emit_b7), "B0": (build_b0, emit_b0),
        "B8": (build_b8, emit_b8),
        "B9": (build_b9, emit_b9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    want = [x.strip() for x in a.only.split(",") if x.strip()] or list(FIGS)
    data = {}
    for k in want:
        b, e = FIGS[k]
        print(f"building {k} ...", end=" ", flush=True)
        d = b(); data[k] = d
        write_tex(k, e(d))
        if not a.no_compile:
            compile_tex(k)
        print("ok")
    p = os.path.join(OUT, "figure_data.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(data); json.dump(old, open(p, "w"), indent=2)
    open(os.path.join(OUT, "CAPTIONS.md"), "w").write(captions(old))
    print(f"\nwrote {', '.join(want)} + CAPTIONS.md into {OUT}")


if __name__ == "__main__":
    main()
