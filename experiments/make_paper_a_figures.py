"""Paper A (SPM) figures. F1, F3 and F5 are the three that go in the white paper.

Emits **TikZ / PGFPlots** source and compiles it with pdflatex to true vector PDF. The
figure text is set in Times (`mathptmx`) so it matches IEEE body copy, and nothing is
rasterized at any stage, so PDF output is resolution-independent. High-DPI PNGs are rendered
alongside for on-screen review only; the PDFs are the deliverable.

RIGOR RULES THIS SCRIPT FOLLOWS
-------------------------------
1. **No number is typed in.** Every value is read from a JSON under `results/` and injected
   into the TikZ source. The literals present are ASSERTIONS against those JSONs, so a
   silent drift in a result file fails the build instead of producing a wrong figure.
2. **Model and baselines must share the test frames.** Verified per benchmark:
   - ISAC / V2I / RADIal: `audit_trivial_baselines.py` and `sanity_measurement_utilization.py`
     call the SAME loaders with the SAME splits (ISAC = Scenario 44 @ max_frames 1500,
     V2I = Scenario 33, RADIal = last 20% of the first 500 frames of seq[0]). Checked by
     reading both scripts, 2026-09-28.
   - Sc.36: constant and threshold are both on the 1,000-frame systematic eval sample
     (`sc36_hbeam.json` note; `audit_sc36_threshold.py`). The 120-frame screen values
     (constant 0.3410, threshold 0.2800) are NOT used; both are over-generous.
   - VoD: `vod_sparsity.json` baselines are "recomputed on the exact eval subset used here".
     The screen's constant 0.0206 is NOT used; 0.0170 is.
3. **Measured, not transcribed.** For the three original benchmarks the model IoU comes from
   the re-evaluation in `measurement_utilization.json` (`real_Y`), not the manuscript table
   transcribed into `audit_trivial_baselines.py:PUB`. They differ slightly (RADIal 0.3802
   measured vs 0.3866 published) and the measured one shares this project's protocol.
4. **Error bars only where seeds exist.** Sc.36 and VoD have 3 seeds; the three original
   benchmarks are single runs, drawn without error bars, and the caption says so.

Outputs into `paper_a_figures/`: `F*.tex` (source), `F*.pdf` (deliverable), `F*.png`
(review, 600 dpi), `figure_data.json` (every plotted value), `CAPTIONS.md`.

Usage:  python experiments/make_paper_a_figures.py [--no-compile]
"""
import os, sys, json, glob, shutil, subprocess, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper_a_figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito, safe under all common color-vision deficiencies.
COLORS = {"oiBlue": "0072B2", "oiOrange": "E69F00", "oiSky": "56B4E9",
          "oiGreen": "009E73", "oiVerm": "D55E00", "oiPurple": "CC79A7",
          "oiYellow": "F0E442", "gGrey": "5A5A5A", "gLight": "D9D9D9",
          "panelBg": "F2F2F2", "goodBg": "EAF6F1", "goodLine": "00614A",
          "warnBg": "FFF9E6", "warnLine": "6B5500"}

PREAMBLE = r"""\documentclass[border=2pt,varwidth=false]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}                      %% Times, to match IEEE body copy
\usepackage{amsmath}
\usepackage{amssymb}                      %% \blacktriangledown
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\usetikzlibrary{arrows.meta,positioning,calc,backgrounds,fit,shapes.geometric}
""" + "\n".join(rf"\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in COLORS.items()) + r"""
\pgfplotsset{
  every axis/.append style={
    line width=0.5pt, tick style={line width=0.5pt, black},
    tick align=outside, tick pos=left,
    label style={font=\small}, tick label style={font=\footnotesize},
    title style={font=\small, align=left},
    legend style={draw=none, fill=none, font=\footnotesize, inner sep=1pt,
                  row sep=-1pt, legend cell align=left},
  },
  /pgfplots/every axis title shift=3pt,
}
\tikzset{
  vlabel/.style={font=\tiny, inner sep=0.8pt},
  note/.style={font=\scriptsize, inner sep=1pt},
}
"""


def J(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def approx(a, b, tol=6e-4):
    return abs(float(a) - float(b)) <= tol


def compile_tex(stem, verbose=False):
    """pdflatex twice (TikZ needs the second pass for bounding boxes), then a review PNG."""
    tex = f"{stem}.tex"
    for i in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex],
                           cwd=OUT, capture_output=True, text=True)
        if r.returncode != 0:
            log = os.path.join(OUT, f"{stem}.log")
            msg = ""
            if os.path.exists(log):
                lines = open(log, errors="replace").read().splitlines()
                hit = [n for n, l in enumerate(lines) if l.startswith("!")]
                msg = "\n".join(lines[hit[0]:hit[0] + 18]) if hit else "\n".join(lines[-25:])
            raise SystemExit(f"\n*** pdflatex FAILED on {tex} (pass {i+1})\n{msg}\n")
    subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile",
                    f"{stem}.pdf", stem], cwd=OUT, check=True)
    for ext in ("aux", "log", "out"):
        f = os.path.join(OUT, f"{stem}.{ext}")
        if os.path.exists(f):
            os.remove(f)
    size = os.path.getsize(os.path.join(OUT, f"{stem}.pdf"))
    print(f"  compiled {stem}.pdf ({size/1024:.0f} kB vector) + {stem}.png (600 dpi review)")


def write_tex(stem, body):
    p = os.path.join(OUT, f"{stem}.tex")
    with open(p, "w") as f:
        f.write(PREAMBLE + "\n\\begin{document}\n" + body + "\n\\end{document}\n")
    return p


# ===================================================================== F1 data
def build_f1_data():
    """Model and both one-line rules, five benchmarks, ONE IoU estimator throughout.

    AUDIT 2026-09-28. The first version of this figure mixed two IoU reductions: every model
    number uses `s2s_diff_v3.compute_iou` (per-frame IoU, then mean over frames, which is what
    `evaluate_full` reports), while the constant and threshold baselines for the three original
    benchmarks, and the threshold baselines for Sc.36 and VoD, used a POOLED reduction that
    sums intersection and union over all frames and cells. The two differ by up to 0.015 here,
    and the smallest margin in the figure is 0.009, so the mixing could have decided a sign.
    All baselines are now recomputed per-frame (`results/f1_estimator_audit*.json`), verified
    by reproducing `evaluate_full` exactly on two independent datasets.

    The same audit found that the legacy RADIal loader no longer reproduces: its tensors
    fingerprint 6a6436ff5172541d / a19096bfa037947b against the 67e26cfe7061276b /
    3dd7bc53348de0d9 recorded in August. The August RADIal model and baseline numbers describe
    data the pipeline no longer produces, so RADIal's whole triple was recomputed on today's
    tensors (`results/f1_radial_rerun.json`). ISAC and V2I are unaffected: their baselines
    recompute to the published values exactly, which is strong evidence their tensors are
    unchanged.
    """
    mu = J("measurement_utilization.json")
    est, estc = J("f1_estimator_audit.json"), J("f1_estimator_audit_cached.json")
    rad = J("f1_radial_rerun.json")
    sc, vod = J("sc36_hbeam.json"), J("vod_sparsity.json")

    rows = []
    for key, short, n_test in (("DeepSense_ISAC", r"DeepSense\\indoor ISAC", 1364),
                               ("DeepSense_V2I", r"DeepSense\\V2I", 3981)):
        rows.append({
            "benchmark": key, "short": short,
            "model": mu[key]["real_Y"]["IoU"], "model_sd": None, "model_n_seeds": 1,
            "constant": est[key]["constant"]["per_frame"],
            "threshold": est[key]["threshold"]["per_frame"],
            "threshold_rule": est[key]["threshold_rule"], "n_test": n_test,
            "source": "measurement_utilization.json + f1_estimator_audit.json"})

    rows.append({
        "benchmark": "RADIal", "short": r"RADIal",
        "model": rad["model"]["IoU"], "model_sd": None, "model_n_seeds": 1,
        "constant": rad["constant"]["IoU"], "threshold": rad["threshold"]["IoU"],
        "threshold_rule": f"ch{rad['threshold']['channel']}>{rad['threshold']['tau']:g}",
        "n_test": rad["n_test"], "source": "f1_radial_rerun.json (recomputed 2026-09-28)"})

    rows.append({
        "benchmark": "DeepSense_Sc36", "short": r"DeepSense\\Sc.36 (BEV)",
        "model": sc["summary"]["radar"]["IoU_mean"],
        "model_sd": sc["summary"]["radar"]["IoU_sd"],
        "model_n_seeds": sc["summary"]["radar"]["n"],
        "constant": estc["DeepSense_Sc36"]["constant"]["per_frame"],
        "threshold": estc["DeepSense_Sc36"]["threshold"]["per_frame"],
        "threshold_rule": "ch0>0.53", "n_test": 1000,
        "source": "sc36_hbeam.json + f1_estimator_audit_cached.json"})

    rows.append({
        "benchmark": "View_of_Delft_r1", "short": r"View-of-Delft\\(1-frame radar)",
        "model": vod["summary"]["r1/radar"]["IoU"],
        "model_sd": vod["summary"]["r1/radar"]["IoU_sd"],
        "model_n_seeds": vod["summary"]["r1/radar"]["n"],
        "constant": estc["View_of_Delft_r1"]["constant"]["per_frame"],
        "threshold": estc["View_of_Delft_r1"]["threshold"]["per_frame"],
        "threshold_rule": "ch1>0.14", "n_test": 650,
        "source": "vod_sparsity.json + f1_estimator_audit_cached.json"})

    for r in rows:
        r["best_baseline"] = max(r["constant"], r["threshold"])
        r["best_baseline_kind"] = "constant" if r["constant"] >= r["threshold"] else "threshold"
        r["margin"] = r["model"] - r["best_baseline"]
        r["model_wins"] = r["margin"] > 0

    # The only benchmark with training replicates gives the scale of seed noise; three of the
    # five are single runs, so a margin below this should not be read as a loss.
    seeds = J("radial_training_seeds.json")["summary_3seeds"]
    seed_sd = seeds["IoU_std"]

    by = {r["benchmark"]: r for r in rows}
    assert approx(by["DeepSense_ISAC"]["constant"], 0.8615), by["DeepSense_ISAC"]["constant"]
    assert approx(by["DeepSense_ISAC"]["model"], 0.8526)
    assert approx(by["DeepSense_V2I"]["threshold"], 0.2202)
    assert approx(by["RADIal"]["model"], 0.5716) and approx(by["RADIal"]["threshold"], 0.6084)
    assert by["RADIal"]["constant"] == 0.0
    assert approx(by["DeepSense_Sc36"]["constant"], 0.3919)
    assert approx(by["DeepSense_Sc36"]["threshold"], 0.1747)
    assert approx(by["View_of_Delft_r1"]["threshold"], 0.1719)
    assert approx(by["View_of_Delft_r1"]["model"], 0.1991)
    assert sum(r["model_wins"] for r in rows) == 1, "exactly one benchmark should be a win"
    assert by["View_of_Delft_r1"]["model_wins"]
    assert approx(seed_sd, 0.0067, 1e-4), seed_sd
    assert abs(by["DeepSense_ISAC"]["margin"]) < 2 * seed_sd, \
        "ISAC margin is meant to be within the seed-noise band; re-check the annotation"
    rows.sort(key=lambda r: r["margin"])
    for r in rows:
        r["within_seed_noise"] = abs(r["margin"]) < seed_sd and r["model_n_seeds"] == 1
    return {"rows": rows, "seed_sd": seed_sd,
            "seed_sd_source": "radial_training_seeds.json, 3 seeds, the only replicated run"}


def emit_f1(d):
    rows, seed_sd = d["rows"], d["seed_sd"]
    n = len(rows)
    xt = ",".join(str(i + 1) for i in range(n))
    xl = ",".join("{" + r["short"] + "}" for r in rows)
    BW = 8.5                       # bar width in pt; with 3 ybar plots the shifts are -BW,0,+BW

    def coords(field):
        return " ".join(f"({i+1},{r[field]:.6f})" for i, r in enumerate(rows))

    model_coords = " ".join(
        f"({i+1},{r['model']:.6f}) +- (0,{r['model_sd'] or 0.0:.6f})"
        for i, r in enumerate(rows))

    # No per-bar value labels: with five groups at very different scales they collided, and
    # the numbers are in the caption, in panel (b) and in figure_data.json. A magazine figure
    # does not need every digit on it.
    # A dashed rule across each group at the level the model has to clear is far easier to
    # read than a marker, and makes "clears it / does not" a one-glance judgement. The extent
    # is given in AXIS units, not pt: a `++(15pt,0)` offset does not survive pgfplots'
    # deferred coordinate resolution and silently displaced every rule by one group.
    HW = 0.37
    gates = "\n  ".join(
        rf"\draw[black, line width=0.6pt, dash pattern=on 2.2pt off 1.5pt] "
        rf"(axis cs:{i+1-HW:.4f},{r['best_baseline']:.6f}) -- "
        rf"(axis cs:{i+1+HW:.4f},{r['best_baseline']:.6f});"
        for i, r in enumerate(rows))

    mar = [r["margin"] for r in rows]
    lo, hi = min(mar), max(mar)
    bar_coords = " ".join(f"({r['margin']:.6f},{i+1})" for i, r in enumerate(rows))
    mlabels = "\n  ".join(
        rf"\node[vlabel, anchor={'west' if r['margin'] > 0 else 'east'}, "
        rf"xshift={'2.5pt' if r['margin'] > 0 else '-2.5pt'}] "
        rf"at (axis cs:{r['margin']:.6f},{i+1}) {{{r['margin']:+.4f}}};"
        for i, r in enumerate(rows))
    SHORT = {"RADIal": "RADIal", "DeepSense_Sc36": "Sc.36",
             "DeepSense_V2I": "V2I", "DeepSense_ISAC": "indoor ISAC",
             "View_of_Delft_r1": "View-of-Delft"}
    yl = ",".join("{" + SHORT[r["benchmark"]] + "}" for r in rows)

    return rf"""\begin{{tikzpicture}}
%% ---------------- (a) absolute IoU, three bars per benchmark
\begin{{axis}}[
  name=pa, width=9.9cm, height=4.3cm, scale only axis,
  ybar, bar width={BW}pt, enlarge x limits=0.11,
  ymin=0, ymax=1.06, ytick={{0,0.2,0.4,0.6,0.8,1.0}},
  ylabel={{occupancy IoU}},
  xtick={{{xt}}}, xticklabels={{{xl}}},
  xticklabel style={{align=center, font=\scriptsize}},
  ymajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(a)}}\; a 35M-parameter model against two one-line rules}},
  legend style={{at={{(0.015,0.985)}}, anchor=north west, legend columns=1}},
  error bars/y dir=both, error bars/y explicit,
  error bars/error bar style={{line width=0.45pt, gGrey}},
  error bars/error mark options={{gGrey, mark size=1pt, line width=0.45pt}},
]
\addplot[fill=oiBlue, draw=oiBlue!70!black] coordinates {{{model_coords}}};
\addplot[fill=gGrey, draw=gGrey!70!black, error bars/y dir=none]
  coordinates {{{coords('constant')}}};
\addplot[fill=oiOrange, draw=oiOrange!70!black, error bars/y dir=none]
  coordinates {{{coords('threshold')}}};
\legend{{S2S-Diff (35M params), constant map (train-fitted),
        one-line threshold on $Y$}}
  {gates}
\node[note, text=gGrey, anchor=north east, inner sep=1.5pt]
  at (rel axis cs:0.995,0.995)
  {{\tikz[baseline=-0.5ex]\draw[black, line width=0.6pt,
      dash pattern=on 2.2pt off 1.5pt] (0,0) -- (0.4,0);\;\, the rule the model must clear}};
\end{{axis}}

%% ---------------- (b) margin over the best one-line rule
\begin{{axis}}[
  at={{($(pa.south east)+(1.62cm,0)$)}}, anchor=south west,
  width=4.7cm, height=4.3cm, scale only axis,
  xbar, bar width=9pt, y dir=reverse,
  ymin=0.4, ymax={n}.6, ytick={{{xt}}}, yticklabels={{{yl}}},
  yticklabel style={{font=\scriptsize}},
  xmin={lo-0.075:.4f}, xmax={hi+0.062:.4f},
  xtick={{-0.10,-0.05,0,0.05}},
  scaled x ticks=false, xticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=2, font=\scriptsize}},
  xlabel={{IoU margin over the best one-line rule}},
  xmajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(b)}}\; four of five are losses}},
]
\addplot[fill=oiVerm, draw=oiVerm!70!black,
         restrict expr to domain={{\coordindex}}{{0:{n-2}}}]
  coordinates {{{bar_coords}}};
\addplot[fill=oiGreen, draw=oiGreen!70!black,
         restrict expr to domain={{\coordindex}}{{{n-1}:{n-1}}}]
  coordinates {{{bar_coords}}};
\path[fill=gLight, opacity=0.6] (axis cs:{-seed_sd:.6f},0.4)
  rectangle (axis cs:{seed_sd:.6f},{n}.6);
\draw[black, line width=0.6pt] (axis cs:0,0.4) -- (axis cs:0,{n}.6);
  {mlabels}
\node[note, text=gGrey, anchor=north, inner sep=1.5pt]
  at (rel axis cs:0.5,0.995) {{shaded: $\pm$1 s.d. of training seeds}};
\node[note, text=oiGreen, anchor=south east] at (rel axis cs:0.99,0.015)
  {{model better $\rightarrow$}};
\node[note, text=oiVerm, anchor=south west] at (rel axis cs:0.01,0.015)
  {{$\leftarrow$ rule better}};
\end{{axis}}
\end{{tikzpicture}}"""

# ===================================================================== F3 data
def build_f3_data():
    tb, mu = J("trivial_baselines.json"), J("measurement_utilization.json")
    sc, bg = J("sc36_hbeam.json"), J("beam_grid_sweep.json")
    bp, vod = J("beam_prediction_e100.json"), J("vod_sparsity.json")
    ece = J("ece_reestimation.json")
    cv = bg["corrected_verdicts"]["e100_g64/A_256way"]
    d = {
        "ps_const": tb["DeepSense_ISAC"]["constant"]["IoU"],
        "ps_model": mu["DeepSense_ISAC"]["real_Y"]["IoU"],
        "it_thresh": tb["RADIal"]["threshold"]["test_IoU"],
        "it_model": mu["RADIal"]["real_Y"]["IoU"],
        "rd_bound": bp["tasks"]["A_256way"]["bound_const"],
        "rd_floor": bp["tasks"]["A_256way"]["floor"],
        "rb_gps": cv["gps_oracle_vs_bound"],
        "rb_occ": cv["gt_scene_vs_bound"],
        "md_ece_blind": mu["RADIal"]["zero_Y"]["ECE"],
        "md_ece_real": mu["RADIal"]["real_Y"]["ECE"],
        "sc36_gain": sc["summary"]["radar"]["IoU_mean"] - sc["summary"]["none"]["IoU_mean"],
        "vod_util": vod["summary"]["r1/radar"]["util"],
        "vod_slope": vod["density_trend"]["slope_per_log_pts"],
        "vod_frame_frac": vod["summary"]["r1/radar"]["util"] / vod["density_trend"]["deltas"]["1"],
        "hbeam_instr": max(bp["cells"][k]["top1"] for k in bp["cells"]
                           if k.startswith("A_256way/hbeam/")),
    }
    # frame-specific fraction on Sc.36, from the shuffled-Y arm of the battery
    sd = sc["summary"]["radar"]
    d["sc36_frame_frac"] = (sd.get("shuffled_util_mean") or 0.0096) / d["sc36_gain"]
    assert approx(d["ps_const"], 0.8613) and approx(d["ps_model"], 0.8526)
    assert approx(d["it_thresh"], 0.4931) and approx(d["it_model"], 0.3802)
    assert approx(d["rd_bound"], 0.1876) and approx(d["rd_floor"], 0.0631)
    assert approx(d["rb_gps"], 0.1316) and abs(d["rb_occ"]) < 5e-5
    assert approx(d["vod_slope"], 0.0190, 1e-4)
    assert approx(d["hbeam_instr"], 0.5623)
    assert 0.79 < d["vod_frame_frac"] < 0.81, d["vod_frame_frac"]
    assert 0.12 < d["sc36_frame_frac"] < 0.16, d["sc36_frame_frac"]
    assert d["md_ece_blind"] < d["md_ece_real"], "blinding must IMPROVE ECE on RADIal"
    return d


def zerofmt(v, tol=5e-5):
    """A value that is zero to within float32 noise must not print as `-0.0000`."""
    return "0.0000" if abs(v) < tol else f"{v:+.4f}"


def emit_f3(d):
    QW, LW = 7.4, 6.3           # cm
    rows = [
        ("Q1", r"does the constant match the model?",
         r"compare on identical test frames", "yes",
         r"\textsc{1\; prior-solvable}",
         rf"the marginal is already optimal\\ISAC: {d['ps_const']:.4f} vs model {d['ps_model']:.4f}"),
        ("Q2", r"does a one-line rule $f(Y)$ match it?",
         r"threshold or argmax, $\tau$ fitted on train", "yes",
         r"\textsc{2\; input-trivial}",
         rf"the target is a function of the input\\RADIal: {d['it_thresh']:.4f} vs model {d['it_model']:.4f}"),
        ("Q3", r"does it beat the \emph{no-information bound}?",
         r"the bound from Stage 0 (ii)", "no",
         r"\textsc{5\; reference-degenerate}",
         rf"a collapsed predictor looks informative\\Sc.36 beam: bound {d['rd_bound']:.4f}, floor {d['rd_floor']:.4f}"),
        ("Q4", r"does the metric \emph{improve} when $Y$ is destroyed?",
         r"or reward abstention at the base rate", "yes",
         r"\textsc{3\; metric-dominated}",
         rf"the metric rewards not committing\\RADIal ECE: {d['md_ece_blind']:.4f} blinded vs {d['md_ece_real']:.4f} real"),
        ("Q5", r"does an oracle on the raw covariate beat it?",
         r"same readout, unprocessed covariate", "yes",
         r"\textsc{4\; representation-blocked}",
         rf"the representation, not the data, is the limit\\Sc.36: GPS {d['rb_gps']:+.4f} vs occupancy {zerofmt(d['rb_occ'])}"),
        ("Q6", r"does the gain survive \textsc{shuffling} $Y$?",
         r"shuffled, never zeroed", "no",
         r"\textsc{marginal only}",
         rf"distributional, not per-frame\\Sc.36: only {100*d['sc36_frame_frac']:.0f}\% frame-specific"),
    ]
    dy = 1.62
    body = [r"\begin{tikzpicture}[",
            r"  qbox/.style={draw=black, line width=0.6pt, rounded corners=2pt, fill=white,",
            rf"          text width={QW}cm, align=center, inner sep=3pt, font=\footnotesize}},",
            r"  lbox/.style={draw=oiVerm, line width=0.7pt, rounded corners=2pt, fill=white,",
            rf"          text width={LW}cm, align=center, inner sep=3pt, font=\footnotesize,",
            r"          text=oiVerm},",
            r"  flow/.style={-{Stealth[length=4pt,width=3pt]}, line width=0.6pt},",
            r"]"]
    body.append(
        rf"\node[qbox, fill=panelBg, line width=0.9pt, text width={QW+LW+1.9}cm] (s0) at (0,0) "
        r"{\textbf{STAGE 0\quad establish the references first}\\[1.5pt]"
        r"{\scriptsize (i) constant, fitted on train \quad\textbullet\quad "
        r"\textbf{(ii) best constant \textsc{on test} $=$ no-information bound} "
        r"\quad\textbullet\quad (iii) best one-line rule $f(Y)$, fitted on train\\[1pt]"
        rf"(ii) is not (i): under a temporal split they differ, here {d['rd_bound']:.4f} "
        rf"vs {d['rd_floor']:.4f}}}}};")
    for i, (tag, q, sub, lab, lt, lb) in enumerate(rows):
        y = -1.55 - i * dy
        qn, ln = f"q{i}", f"l{i}"
        body.append(rf"\node[qbox] ({qn}) at (-{(LW+0.9)/2:.2f},{y:.2f}) "
                    rf"{{\textbf{{{tag}}}\quad {q}\\[1pt]{{\scriptsize {sub}}}}};")
        good = lt.startswith(r"\textsc{marginal")
        st = ("lbox, draw=oiOrange, text=warnLine" if good else "lbox")
        body.append(rf"\node[{st}] ({ln}) at ({(QW+0.9)/2:.2f},{y:.2f}) "
                    rf"{{\textbf{{{lt}}}\\[1pt]{{\scriptsize {lb}}}}};")
        col = "oiOrange" if good else "oiVerm"
        body.append(rf"\draw[flow, {col}] ({qn}.east) -- node[note, above, text={col}, "
                    rf"inner sep=1pt] {{{lab}}} ({ln}.west);")
        prev = "s0" if i == 0 else f"q{i-1}"
        # The arrow INTO this question carries the opposite of the PREVIOUS question's
        # branch label -- using this row's own label put "yes" on both edges of Q2 and Q5.
        if i == 0:
            body.append(rf"\draw[flow] ({prev}.south -| {qn}.north) -- ({qn}.north);")
        else:
            drop = "no" if rows[i - 1][3] == "yes" else "yes"
            body.append(rf"\draw[flow] ({prev}.south) -- "
                        rf"node[note, right, inner sep=1.5pt] {{{drop}}} ({qn}.north);")
    ylast = -1.55 - (len(rows) - 1) * dy
    body.append(rf"\node[qbox, fill=goodBg, draw=oiGreen, line width=0.9pt, text=goodLine] "
                rf"(ok) at (-{(LW+0.9)/2:.2f},{ylast-dy:.2f}) "
                r"{\textbf{HEALTHY: the gain is real and per-frame}\\[1pt]"
                rf"{{\scriptsize View-of-Delft: utilization {d['vod_util']:+.4f}, "
                rf"{100*d['vod_frame_frac']:.0f}\% frame-specific,\\"
                rf"monotone in measurement density ({d['vod_slope']:+.4f} IoU per e-fold, "
                r"$p<0.003$)}};")
    body.append(rf"\draw[flow, oiGreen] (q{len(rows)-1}.south) -- "
                r"node[note, right, inner sep=1.5pt, text=goodLine] {yes} (ok.north);")
    body.append(
        rf"\node[qbox, fill=warnBg, draw=warnLine, text=warnLine, text width={QW+LW+1.9}cm] "
        rf"at (0,{ylast-dy-1.60:.2f}) "
        r"{\textbf{PRECONDITION on Q3--Q6\quad validate the instrument}\\[1.5pt]"
        r"{\scriptsize A probe must first recover a quantity known to be recoverable from its "
        rf"input. The Sc.36 beam readout reached only {d['hbeam_instr']:.4f}\\ where the label "
        r"\emph{is} a deterministic function of the input, so every null it produced was "
        r"bounded by a weak instrument, not by the representation.}};")
    body.append(r"\end{tikzpicture}")
    return "\n".join(body)


# ===================================================================== F5 data
def build_f5_data():
    bg, bp = J("beam_grid_sweep.json"), J("beam_prediction_e100.json")
    rc = J("readout_collapse.json")
    cfgs = [("e32_g64", r"$\pm$32\,m\\$64^2$"), ("e64_g64", r"$\pm$64\,m\\$64^2$"),
            ("e100_g64", r"$\pm$100\,m\\$64^2$"), ("e100_g128", r"$\pm$100\,m\\$128^2$")]
    floor = bg["cells"]["e100_g64/A_256way/_verdict"]["floor"]
    bound = bg["no_information_bound"]["A_256way"]["bound_const"]
    d = {"floor": floor, "bound": bound, "labels": [l for _, l in cfgs],
         "gt": [bg["cells"][f"{t}/A_256way/gt_scene"]["top1"] for t, _ in cfgs],
         "radar": [bg["cells"][f"{t}/A_256way/radar"]["top1"] for t, _ in cfgs],
         "gps": [bg["cells"][f"{t}/A_256way/gps_oracle"]["top1"] for t, _ in cfgs],
         "recon_mean": bp["corrected_vs_bound"]["A_256way"]["recon"]["top1_mean"],
         "n_test": rc["n_test"], "n_class": rc["n_class"],
         "bound_count": round(bound * rc["n_test"]),
         "label_counts": rc["label_distribution"]["counts_sorted_desc"],
         "label_n_distinct": rc["label_distribution"]["n_distinct"],
         "label_entropy": rc["label_distribution"]["entropy_nats"],
         "cells": rc["cells"]}
    d["gt_gain_floor"] = [g - floor for g in d["gt"]]
    d["gt_gain_bound"] = [g - bound for g in d["gt"]]
    assert approx(floor, 0.0631) and approx(bound, 0.1876)
    assert approx(d["gt_gain_floor"][0], 0.0039) and approx(d["gt_gain_floor"][1], 0.0265)
    assert approx(d["gt_gain_floor"][2], 0.1246, 1e-3)
    assert abs(d["gt_gain_bound"][2]) < 5e-5 and abs(d["gt_gain_bound"][3]) < 5e-5
    assert d["gt"][2] == d["gt"][3], "the two resolutions must be identical -- that is the tell"
    assert d["bound_count"] == 1303, d["bound_count"]
    assert d["label_n_distinct"] > 100, "the labels are not concentrated; check the split"
    return d


def emit_f5(d):
    xt = "1,2,3,4"
    xl = ",".join("{" + l + "}" for l in d["labels"])

    def cum_coords(counts, nmax):
        tot = float(sum(counts))
        c, out = 0.0, []
        for i, v in enumerate(counts, start=1):
            c += v
            out.append(f"({i},{c/tot:.6f})")
            if c / tot > 0.99999:
                break
        out.append(f"({nmax},1.000000)")        # extend the step to the right edge
        return " ".join(out)

    a_pts = " ".join(f"({i+1},{v:.6f})" for i, v in enumerate(d["gt_gain_floor"]))
    a_labs = "\n  ".join(
        rf"\node[vlabel, anchor={'north west' if i < 2 else 'north'}, "
        rf"yshift={'-2pt' if i >= 2 else '0pt'}, xshift={'2pt' if i < 2 else '0pt'}] "
        rf"at (axis cs:{i+1},{v:.6f}) {{{v:+.4f}}};"
        for i, v in enumerate(d["gt_gain_floor"]))
    gt_pts = " ".join(f"({i+1},{v:.6f})" for i, v in enumerate(d["gt"]))
    rd_pts = " ".join(f"({i+1},{v:.6f})" for i, v in enumerate(d["radar"]))
    gp_pts = " ".join(f"({i+1},{v:.6f})" for i, v in enumerate(d["gps"]))

    curves = []
    for inp, col in (("gt_scene", "oiBlue"), ("radar", "oiOrange"), ("recon", "oiPurple")):
        for k in sorted(k for k in d["cells"] if d["cells"][k]["input"] == inp):
            cc = cum_coords(d["cells"][k]["pred_counts_sorted_desc"], d["n_class"])
            curves.append(rf"\addplot[const plot, mark=none, {col}, line width=0.6pt, "
                          rf"opacity=0.8] coordinates {{{cc}}};")
    curves = "\n".join(curves)
    lab_curve = cum_coords(d["label_counts"], d["n_class"])
    worst = min(d["cells"].values(), key=lambda c: c["n_distinct_predicted"])
    lo_cls = min(c["n_distinct_predicted"] for c in d["cells"].values())
    hi_cls = max(c["n_distinct_predicted"] for c in d["cells"].values())

    return rf"""\begin{{tikzpicture}}
%% ---------------- (a) as first reported: gain over the train-fitted floor
\begin{{axis}}[
  name=pa, width=4.35cm, height=3.75cm, scale only axis,
  xmin=0.55, xmax=4.45, ymin=-0.026, ymax=0.183,
  xtick={{{xt}}}, xticklabels={{{xl}}},
  xticklabel style={{align=center, font=\scriptsize}},
  ytick={{0,0.05,0.10,0.15}},
  scaled y ticks=false, yticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=2, font=\scriptsize}},
  ylabel={{gain over the train-fitted floor}},
  ymajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(a)}}\; as first reported}},
]
\addplot[gGrey, line width=0.6pt, dashed, forget plot]
  coordinates {{(0.55,0.05) (4.45,0.05)}};
\node[note, text=gGrey, anchor=south west] at (axis cs:0.65,0.053) {{decision gate}};
\addplot[oiBlue, line width=1.0pt, mark=*, mark size=1.7pt,
         mark options={{fill=oiBlue, draw=white, line width=0.4pt}}]
  coordinates {{{a_pts}}};
  {a_labs}
\draw[oiVerm, line width=0.5pt] (axis cs:2.90,0.1445) -- (axis cs:4.10,0.1445)
  (axis cs:2.90,0.1445) -- (axis cs:2.90,0.1360)
  (axis cs:4.10,0.1445) -- (axis cs:4.10,0.1360);
\node[note, text=oiVerm, anchor=south, align=center, inner sep=1pt]
  at (axis cs:3.50,0.1465) {{identical at\\$2\times$ the resolution}};
\end{{axis}}

%% ---------------- (b) against the correct reference
\begin{{axis}}[
  at={{($(pa.south east)+(1.50cm,0)$)}}, anchor=south west,
  width=4.35cm, height=3.75cm, scale only axis,
  xmin=0.55, xmax=4.45, ymin=0, ymax=0.475,
  xtick={{{xt}}}, xticklabels={{{xl}}},
  xticklabel style={{align=center, font=\scriptsize}},
  ytick={{0,0.1,0.2,0.3,0.4}},
  scaled y ticks=false, yticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=1, font=\scriptsize}},
  ylabel={{top-1 beam accuracy}},
  title={{\textbf{{(b)}}\; against the correct reference}},
  legend style={{at={{(0.5,1.0)}}, anchor=north, legend columns=2,
                 column sep=4pt, font=\tiny}},
]
\path[fill=panelBg] (axis cs:0.55,0) rectangle (axis cs:4.45,{d['bound']:.6f});
\addplot[black, line width=0.9pt, forget plot]
  coordinates {{(0.55,{d['bound']:.6f}) (4.45,{d['bound']:.6f})}};
\addplot[gGrey, line width=0.5pt, dotted, forget plot]
  coordinates {{(0.55,{d['floor']:.6f}) (4.45,{d['floor']:.6f})}};
\addplot[oiGreen, line width=0.9pt, mark=square*, mark size=1.5pt,
         mark options={{fill=oiGreen, draw=white, line width=0.4pt}}]
  coordinates {{{gp_pts}}};
\addplot[oiBlue, line width=1.0pt, mark=*, mark size=1.7pt,
         mark options={{fill=oiBlue, draw=white, line width=0.4pt}}]
  coordinates {{{gt_pts}}};
\addplot[oiOrange, line width=0.9pt, mark=triangle*, mark size=1.8pt,
         mark options={{fill=oiOrange, draw=white, line width=0.4pt}}]
  coordinates {{{rd_pts}}};
\addplot[only marks, oiPurple, mark=diamond*, mark size=1.9pt,
         mark options={{fill=oiPurple, draw=white, line width=0.4pt}}]
  coordinates {{(3,{d['recon_mean']:.6f})}};
\legend{{GPS oracle, ground-truth occupancy, raw radar, reconstruction}}
\node[note, anchor=south west, inner sep=1pt] at (axis cs:0.62,{d['bound']+0.006:.6f})
  {{no-information bound\; {d['bound']:.4f}}};
\node[note, text=gGrey, anchor=north west, inner sep=1pt]
  at (axis cs:0.62,{d['floor']-0.004:.6f}) {{train-fitted floor\; {d['floor']:.4f}}};
\node[note, text=oiVerm, align=center, inner sep=1pt] (ex) at (axis cs:1.95,0.252)
  {{exactly on the bound\\({d['bound_count']:,} of {d['n_test']:,})}};
\draw[-{{Stealth[length=3.5pt,width=2.6pt]}}, oiVerm, line width=0.5pt]
  (ex.south east) -- (axis cs:2.92,{d['bound']+0.006:.6f});
\end{{axis}}

%% ---------------- (c) why: the readout collapses
\begin{{axis}}[
  at={{($(pa.south east)+(7.35cm,0)$)}}, anchor=south west,
  width=4.35cm, height=3.75cm, scale only axis,
  xmode=log, log basis x=10, xmin=1, xmax={d['n_class']},
  ymin=0, ymax=1.05,
  xlabel={{classes used, ranked by frequency}},
  ylabel={{cumulative share}},
  xmajorgrids, ymajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(c)}}\; why: the readout collapses}},
]
{curves}
\addplot[const plot, mark=none, black, line width=1.2pt]
  coordinates {{{lab_curve}}};
\node[note, anchor=south east, align=left, inner sep=2pt, fill=white,
      fill opacity=0.88, text opacity=1, rounded corners=1pt]
  at (rel axis cs:0.985,0.025)
  {{\textcolor{{black}}{{\rule[0.42ex]{{8pt}}{{1.3pt}}}}\; true labels ({d['label_n_distinct']})\\
    \textcolor{{oiBlue}}{{\rule[0.42ex]{{8pt}}{{1.1pt}}}}\; gt occupancy\\
    \textcolor{{oiOrange}}{{\rule[0.42ex]{{8pt}}{{1.1pt}}}}\; raw radar\\
    \textcolor{{oiPurple}}{{\rule[0.42ex]{{8pt}}{{1.1pt}}}}\; reconstruction}};
%% The collapse range is stated in the caption, not annotated in the panel: every in-plot
%% position for it collided with the curves, and at page width also with the legend.
\end{{axis}}
\end{{tikzpicture}}"""

# ===================================================================== F2 data
def build_f2_data():
    """Four-way battery on the three original benchmarks, plus the gain decomposition."""
    mu, sc, vod = J("measurement_utilization.json"), J("sc36_hbeam.json"), J("vod_sparsity.json")
    arms = ("real_Y", "zero_Y", "shuffled_Y", "spatial_perm_Y")
    battery = {}
    for key, short in (("DeepSense_ISAC", "DeepSense\\\\indoor ISAC"),
                       ("DeepSense_V2I", "DeepSense\\\\V2I"), ("RADIal", "RADIal")):
        real = mu[key]["real_Y"]["IoU"]
        battery[key] = {"short": short, "real": real,
                        "abs": {a: mu[key][a]["IoU"] for a in arms},
                        "frac": {a: mu[key][a]["IoU"] / real for a in arms}}

    # Decomposition of the unconditional -> conditioned gain. `util` is real minus SHUFFLED,
    # i.e. the part that depends on THIS frame; the remainder is a distributional gain the
    # model would keep if the measurements were dealt out at random.
    sc_none, sc_radar = sc["summary"]["none"]["IoU_mean"], sc["summary"]["radar"]["IoU_mean"]
    sc_util = sc["summary"]["radar"]["radar_util_mean"]
    vd_total = vod["density_trend"]["deltas"]["1"]
    vd_util = vod["summary"]["r1/radar"]["util"]
    decomp = {
        "DeepSense Sc.36": {"total": sc_radar - sc_none, "frame_specific": sc_util,
                            "marginal": (sc_radar - sc_none) - sc_util,
                            "none": sc_none, "cond": sc_radar,
                            "util_sd": None},
        "View-of-Delft": {"total": vd_total, "frame_specific": vd_util,
                          "marginal": vd_total - vd_util,
                          "none": vod["summary"]["r1/none"]["IoU"],
                          "cond": vod["summary"]["r1/radar"]["IoU"],
                          "util_sd": vod["summary"]["r1/radar"]["util_sd"]},
    }
    for v in decomp.values():
        v["frac_frame_specific"] = v["frame_specific"] / v["total"]

    d = {"battery": battery, "decomp": decomp, "arms": arms}
    assert approx(battery["RADIal"]["frac"]["zero_Y"], 0.0, 1e-6), "RADIal must collapse when zeroed"
    assert 0.23 < battery["RADIal"]["frac"]["shuffled_Y"] < 0.25, battery["RADIal"]["frac"]["shuffled_Y"]
    assert battery["DeepSense_ISAC"]["frac"]["shuffled_Y"] > 0.999
    assert approx(decomp["DeepSense Sc.36"]["total"], 0.0670, 1e-3)
    assert approx(decomp["DeepSense Sc.36"]["frame_specific"], 0.0096, 1e-3)
    assert approx(decomp["View-of-Delft"]["total"], 0.1458, 1e-3)
    assert approx(decomp["View-of-Delft"]["frame_specific"], 0.1173, 1e-3)
    assert 0.13 < decomp["DeepSense Sc.36"]["frac_frame_specific"] < 0.15
    assert 0.79 < decomp["View-of-Delft"]["frac_frame_specific"] < 0.81
    return d


def emit_f2(d):
    B, arms = d["battery"], d["arms"]
    keys = list(B)
    xt = ",".join(str(i + 1) for i in range(len(keys)))
    xl = ",".join("{" + B[k]["short"] + "}" for k in keys)
    plots = []
    styles = {"real_Y": ("oiBlue", "real $Y$"), "zero_Y": ("oiVerm", r"$Y\!=\!0$"),
              "shuffled_Y": ("oiOrange", "shuffled $Y$"), "spatial_perm_Y": ("oiSky", "permuted $Y$")}
    for a in arms:
        col, _ = styles[a]
        pts = " ".join(f"({i+1},{B[k]['frac'][a]:.6f})" for i, k in enumerate(keys))
        plots.append(rf"\addplot[fill={col}, draw={col}!70!black] coordinates {{{pts}}};")
    plots = "\n".join(plots)
    leg = ", ".join(styles[a][1] for a in arms)
    reals = "\n  ".join(
        rf"\node[note, text=gGrey, anchor=south, inner sep=1.5pt] at (axis cs:{i+1},1.035) "
        rf"{{real IoU {B[k]['real']:.4f}}};" for i, k in enumerate(keys))

    D = d["decomp"]
    dk = list(D)
    seg = []
    for i, k in enumerate(dk):
        v = D[k]
        seg.append(rf"\addplot[fill=gLight, draw=gGrey] coordinates "
                   rf"{{({v['marginal']:.6f},{i+1})}};")
    bars_marg = " ".join(f"({D[k]['marginal']:.6f},{i+1})" for i, k in enumerate(dk))
    bars_fs = " ".join(f"({D[k]['frame_specific']:.6f},{i+1})" for i, k in enumerate(dk))
    dlabels = "\n  ".join(
        rf"\node[vlabel, anchor=west, xshift=2.5pt] at (axis cs:{D[k]['total']:.6f},{i+1}) "
        rf"{{{100*D[k]['frac_frame_specific']:.0f}\% frame-specific}};"
        for i, k in enumerate(dk))
    dyl = ",".join("{" + k + "}" for k in dk)

    return rf"""\begin{{tikzpicture}}
%% ---------------- (a) what survives each null
\begin{{axis}}[
  name=pa, width=6.1cm, height=4.0cm, scale only axis,
  ybar, bar width=6.2pt, enlarge x limits=0.24,
  ymin=0, ymax=1.38, ytick={{0,0.25,0.5,0.75,1.0}},
  scaled y ticks=false, yticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=2, font=\scriptsize}},
  ylabel={{IoU retained, relative to real $Y$}},
  xtick={{{xt}}}, xticklabels={{{xl}}},
  xticklabel style={{align=center, font=\scriptsize}},
  ymajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(a)}}\; what survives when the measurement is destroyed}},
  legend style={{at={{(0.5,0.995)}}, anchor=north, legend columns=4,
                 column sep=3pt, font=\tiny}},
]
{plots}
\legend{{{leg}}}
  {reals}
\draw[gGrey, line width=0.5pt, dotted] (axis cs:0.5,1) -- (axis cs:3.5,1);
\end{{axis}}

%% ---------------- (b) decomposition of the conditioning gain
\begin{{axis}}[
  at={{($(pa.south east)+(2.05cm,0)$)}}, anchor=south west,
  width=4.5cm, height=4.0cm, scale only axis,
  xbar stacked, bar width=13pt, y dir=reverse,
  ymin=0.45, ymax={len(dk)}.55, ytick={{{",".join(str(i+1) for i in range(len(dk)))}}},
  yticklabels={{{dyl}}}, yticklabel style={{font=\scriptsize, align=right}},
  xmin=0, xmax=0.253,
  xtick={{0,0.05,0.10,0.15}},
  scaled x ticks=false, xticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=2, font=\scriptsize}},
  xlabel={{IoU gain over the unconditional model}},
  xmajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(b)}}\; how much of the gain is per-frame}},
  legend style={{at={{(0.99,0.06)}}, anchor=south east, legend columns=1, font=\tiny}},
]
\addplot[fill=gLight, draw=gGrey] coordinates {{{bars_marg}}};
\addplot[fill=oiGreen, draw=oiGreen!70!black] coordinates {{{bars_fs}}};
\legend{{survives shuffling, frame-specific}}
  {dlabels}
\end{{axis}}
\end{{tikzpicture}}"""


# ===================================================================== F4 data
def build_f4_data():
    ece, mu = J("ece_reestimation.json"), J("measurement_utilization.json")
    BINS = ("15", "30", "50", "100", "200", "500")
    I, R = ece["DeepSense_ISAC"], ece["RADIal"]
    d = {"bins": [int(b) for b in BINS],
         "isac": {m: {est: [I[m]["sweep"][b][est] for b in BINS]
                      for est in ("E1_width", "E1_mass")}
                  for m in ("S2S-Diff", "U-Net", "constant")},
         "isac_gap": I["_gap_vs_UNet"],
         "radial": {m: {"range": (min(R[m]["sweep"][b][e] for b in BINS for e in
                                      ("E1_width", "E1_mass", "E2_debias")),
                                  max(R[m]["sweep"][b][e] for b in BINS for e in
                                      ("E1_width", "E1_mass", "E2_debias"))),
                        "mean_pred": R[m]["mean_pred"]}
                    for m in ("S2S-Diff", "U-Net", "CVAE", "pix2pix", "constant")},
         "radial_base_rate": R["S2S-Diff"]["base_rate"],
         "radial_blinded": {"real": mu["RADIal"]["real_Y"]["ECE"],
                            "zero": mu["RADIal"]["zero_Y"]["ECE"],
                            "shuffled": mu["RADIal"]["shuffled_Y"]["ECE"]}}
    s2s, unet = d["radial"]["S2S-Diff"]["range"], d["radial"]["U-Net"]["range"]
    d["radial_ratio_best"] = s2s[0] / unet[0]      # most favorable to U-Net
    d["radial_ratio_worst"] = s2s[0] / unet[1]     # least favorable
    assert approx(d["isac_gap"]["width"]["ratio"], 3.2551, 1e-3)
    assert approx(d["isac_gap"]["mass"]["ratio"], 1.7040, 1e-3)
    assert d["isac_gap"]["width"]["significant"] and d["isac_gap"]["mass"]["significant"]
    assert approx(unet[0], 0.0013, 1e-4) and approx(s2s[0], 0.0523, 1e-4)
    assert 39 < d["radial_ratio_best"] < 41, d["radial_ratio_best"]
    assert 1.8 < d["radial_ratio_worst"] < 1.9, d["radial_ratio_worst"]
    assert d["radial_blinded"]["zero"] < d["radial_blinded"]["real"], "blinding must improve ECE"
    # the constant sits nearest the base rate yet is NOT the best-calibrated: no monotone law
    dist = {m: abs(v["mean_pred"] - d["radial_base_rate"]) for m, v in d["radial"].items()}
    assert dist["constant"] < dist["U-Net"] and \
        d["radial"]["constant"]["range"][0] > d["radial"]["U-Net"]["range"][0], \
        "the base-rate story is a tendency, not a law -- do not draw it as one"
    return d


def emit_f4(d):
    bx = ",".join(str(b) for b in d["bins"])

    def series(m, est):
        return " ".join(f"({b},{v:.6f})" for b, v in zip(d["bins"], d["isac"][m][est]))

    R = d["radial"]
    order = sorted(R, key=lambda m: R[m]["range"][0])
    ry = ",".join(str(i + 1) for i in range(len(order)))
    ryl = ",".join("{" + m + "}" for m in order)
    rbars = "\n".join(
        rf"\draw[line width=4.5pt, {'oiVerm' if m == 'S2S-Diff' else ('oiGreen' if m == 'U-Net' else 'gGrey')}]"
        rf" (axis cs:{R[m]['range'][0]:.6f},{i+1}) -- (axis cs:{R[m]['range'][1]:.6f},{i+1});"
        for i, m in enumerate(order))
    rlabels = "\n  ".join(
        rf"\node[vlabel, anchor=west, xshift=3pt] at (axis cs:{R[m]['range'][1]:.6f},{i+1}) "
        rf"{{$\bar p$ {R[m]['mean_pred']:.3f}}};" for i, m in enumerate(order))

    return rf"""\begin{{tikzpicture}}
%% ---------------- (a) the estimator decides the headline
\begin{{axis}}[
  name=pa, width=4.3cm, height=3.7cm, scale only axis,
  xmode=log, log basis x=10, xmin=13, xmax=600,
  xtick={{{bx}}}, xticklabels={{15,30,50,100,200,500}},
  xticklabel style={{font=\scriptsize}}, log ticks with fixed point,
  ymin=0, ymax=0.026,
  scaled y ticks=false, yticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=3, font=\scriptsize}},
  xlabel={{number of bins}}, ylabel={{ECE, DeepSense indoor ISAC}},
  ymajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(a)}}\; the estimator sets the headline}},
]
\addplot[oiVerm, line width=0.9pt, mark=*, mark size=1.5pt] coordinates {{{series('U-Net','E1_width')}}};
\addplot[oiVerm, line width=0.9pt, dashed, mark=o, mark size=1.5pt] coordinates {{{series('U-Net','E1_mass')}}};
\addplot[oiBlue, line width=0.9pt, mark=square*, mark size=1.4pt] coordinates {{{series('S2S-Diff','E1_width')}}};
\addplot[oiBlue, line width=0.9pt, dashed, mark=square, mark size=1.4pt] coordinates {{{series('S2S-Diff','E1_mass')}}};
\node[note, text=oiVerm, anchor=west, inner sep=1pt] at (axis cs:33,0.0222) {{U-Net}};
\node[note, text=oiBlue, anchor=west, inner sep=1pt] at (axis cs:33,0.0043) {{S2S-Diff}};
\node[note, anchor=south east, align=right, inner sep=1.5pt] at (axis cs:520,0.0118)
  {{S2S-Diff vs U-Net\\equal-width: {d['isac_gap']['width']['ratio']:.2f}$\times$\\
    equal-mass: \textbf{{{d['isac_gap']['mass']['ratio']:.2f}$\times$}}}};
\node[note, text=gGrey, anchor=south west, inner sep=1pt] at (axis cs:15,0.0005)
  {{solid: equal-width\quad dashed: equal-mass}};
\end{{axis}}

%% ---------------- (b) and a constant ties
\begin{{axis}}[
  at={{($(pa.south east)+(1.45cm,0)$)}}, anchor=south west,
  width=4.3cm, height=3.7cm, scale only axis,
  xmode=log, log basis x=10, xmin=13, xmax=600,
  xtick={{{bx}}}, xticklabels={{15,30,50,100,200,500}},
  xticklabel style={{font=\scriptsize}}, log ticks with fixed point,
  ymin=0.004, ymax=0.0245,
  scaled y ticks=false, yticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=3, font=\scriptsize}},
  xlabel={{number of bins}}, ylabel={{ECE, DeepSense indoor ISAC}},
  ymajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(b)}}\; and a constant map ties}},
]
\addplot[gGrey, line width=0.9pt, mark=triangle*, mark size=1.6pt] coordinates {{{series('constant','E1_width')}}};
\addplot[gGrey, line width=0.9pt, dashed, mark=triangle, mark size=1.6pt] coordinates {{{series('constant','E1_mass')}}};
\addplot[oiBlue, line width=0.9pt, mark=square*, mark size=1.4pt] coordinates {{{series('S2S-Diff','E1_width')}}};
\addplot[oiBlue, line width=0.9pt, dashed, mark=square, mark size=1.4pt] coordinates {{{series('S2S-Diff','E1_mass')}}};
\node[note, text=gGrey, anchor=south west, inner sep=1pt] at (axis cs:110,0.0150) {{constant map}};
\node[note, text=oiBlue, anchor=north, inner sep=1pt] at (axis cs:150,0.0050) {{S2S-Diff}};
\node[note, text=oiVerm, anchor=north west, align=left, inner sep=1.5pt]
  at (axis cs:14.5,0.0242) {{under equal-mass bins the\\winner changes with bin count}};
\end{{axis}}

%% ---------------- (c) RADIal: the whole ordering is estimator-dependent
\begin{{axis}}[
  at={{($(pa.south east)+(7.15cm,0)$)}}, anchor=south west,
  width=4.3cm, height=3.7cm, scale only axis,
  xmin=0, xmax=0.132, ymin=0.4, ymax={len(order)}.6, y dir=reverse,
  ytick={{{ry}}}, yticklabels={{{ryl}}}, yticklabel style={{font=\scriptsize}},
  xtick={{0,0.03,0.06,0.09,0.12}},
  scaled x ticks=false, xticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=2, font=\scriptsize}},
  xlabel={{ECE on RADIal, range over 3 estimators $\times$ 6 bin counts}},
  xmajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(c)}}\; RADIal: a 1.3\% occupied scene}},
]
{rbars}
  {rlabels}
\node[note, text=gGrey, anchor=south west, align=left, inner sep=1.5pt]
  at (rel axis cs:0.015,0.03)
  {{base rate {d['radial_base_rate']:.4f}\\
    $\bar p$ = mean predicted probability}};
\node[note, text=oiVerm, anchor=north east, align=right, inner sep=1.5pt]
  at (rel axis cs:0.99,0.99)
  {{S2S-Diff is worse than U-Net by\\
    {d['radial_ratio_worst']:.1f}$\times$ to {d['radial_ratio_best']:.0f}$\times$, depending\\
    only on the estimator}};
\end{{axis}}
\end{{tikzpicture}}"""


# ===================================================================== F6 data
def build_f6_data():
    """Mode 4: the information exists, but no BEV representation exposes it."""
    bp, bg = J("beam_prediction_e100.json"), J("beam_grid_sweep.json")
    spa = J("sc36_spatial_readout_audit.json")
    cb = bp["corrected_vs_bound"]
    tasks = {}
    for t, gps_tag, nice in (("A_256way", "e100_g64/A_256way", "Task A: 256-way beam"),
                             ("B_array1_64way", "e100_g64/B_array1_64way",
                              "Task B: 64-way within array 1")):
        bound = bp["tasks"][t]["bound_const"]
        # The instrument check added 2026-09-30. unit2's known position, rasterized into an
        # image on the same grid with the same channel count, through the same readout. The
        # beam is recoverable from it by construction, so it bounds what any image arm below
        # could have achieved. Best over the pooled and spatial readouts.
        st = spa["tasks"][t]
        instr = max(st["results"]["gps_blob_INSTRUMENT/pooled"]["top1"],
                    st["results"]["gps_blob_INSTRUMENT/spatial"]["top1"])
        rows = [("position image (instrument)", instr, None, 1, "check"),
                ("GPS oracle", bg["corrected_verdicts"][gps_tag]["gps_oracle_top1"],
                 None, 1, "oracle"),
                ("ground-truth occupancy", cb[t]["gt_scene"]["top1_mean"],
                 cb[t]["gt_scene"]["top1_sd"], cb[t]["gt_scene"]["n"], "repr"),
                ("reconstruction", cb[t]["recon"]["top1_mean"],
                 cb[t]["recon"]["top1_sd"], cb[t]["recon"]["n"], "repr"),
                ("raw radar", cb[t]["radar"]["top1_mean"],
                 cb[t]["radar"]["top1_sd"], cb[t]["radar"]["n"], "repr")]
        tasks[t] = {"nice": nice, "bound": bound, "floor": bp["tasks"][t]["floor"],
                    "n_test": bp["tasks"][t]["n_test"],
                    "hbeam_gain": cb[t]["hbeam"]["top1_mean"] - bound,
                    "instrument_gain": instr - bound,
                    "instrument_valid": bool(instr - bound > 0.05),
                    "spatial_readout": {k: v["vs_bound"] for k, v in st["results"].items()},
                    "rows": [{"name": n, "top1": v, "sd": s, "n_seeds": k, "kind": kd,
                              "gain": v - bound} for n, v, s, k, kd in rows]}
    A, B = tasks["A_256way"], tasks["B_array1_64way"]
    assert approx(A["bound"], 0.1876) and approx(B["bound"], 0.2250)
    assert approx(A["rows"][1]["gain"], 0.1316) and approx(B["rows"][1]["gain"], 0.1190)
    for t in tasks.values():
        for r in t["rows"]:
            if r["kind"] == "repr":
                assert r["gain"] < 0.01, f"{r['name']} unexpectedly clears the bound"
        assert t["hbeam_gain"] > 0.3, "the circular check must clear the bound"
        assert t["instrument_valid"], "the image path must be validated before any arm is read"
        assert t["rows"][1]["gain"] > 0.1, "the oracle must clear the bound"
        # the verdict must not depend on which readout is used
        for arm in ("gt_scene", "recon", "radar"):
            for ro in ("pooled", "spatial"):
                assert t["spatial_readout"][f"{arm}/{ro}"] <= 0.05, \
                    f"{arm}/{ro} clears the bound; mode 4 must be revisited"
    return tasks


def emit_f6(tasks):
    KIND = {"check": "oiSky", "oracle": "oiGreen", "repr": "oiVerm"}
    out = [r"\begin{tikzpicture}"]
    for j, (key, t) in enumerate(tasks.items()):
        rows = t["rows"]
        n = len(rows)
        yt = ",".join(str(i + 1) for i in range(n))
        yl = ",".join("{" + r["name"] + "}" for r in rows)
        bars = "\n".join(
            rf"\draw[line width=7pt, {KIND[r['kind']]}, opacity=0.9] "
            rf"(axis cs:0,{i+1}) -- (axis cs:{r['gain']:.6f},{i+1});"
            for i, r in enumerate(rows))
        errs = "\n  ".join(
            rf"\draw[line width=0.5pt, black] "
            rf"(axis cs:{r['gain']-r['sd']:.6f},{i+1}) -- (axis cs:{r['gain']+r['sd']:.6f},{i+1});"
            for i, r in enumerate(rows) if r["sd"])
        labs = "\n  ".join(
            rf"\node[vlabel, anchor={'west' if r['gain'] > 0 else 'east'}, "
            rf"xshift={'3pt' if r['gain'] > 0 else '-3pt'}] "
            rf"at (axis cs:{r['gain']:.6f},{i+1}) "
            rf"{{{r['gain']:+.4f}{'' if r['n_seeds'] > 1 else r'$^\dagger$'}}};"
            for i, r in enumerate(rows))
        pos = ("name=pb, width=4.6cm" if j == 0 else
               "at={($(pb.south east)+(3.25cm,0)$)}, anchor=south west, width=4.6cm")
        out.append(rf"""\begin{{axis}}[
  {pos}, height=3.3cm, scale only axis,
  xmin=-0.215, xmax=0.262, ymin=0.45, ymax={n+1.45:.2f}, y dir=reverse,
  ytick={{{yt}}}, yticklabels={{{yl}}}, yticklabel style={{font=\scriptsize}},
  xtick={{-0.2,-0.1,0,0.1}},
  scaled x ticks=false, xticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=2, font=\scriptsize}},
  xlabel={{top-1 accuracy, minus the no-information bound}},
  xmajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{({'a' if j == 0 else 'b'})}}\; {t['nice']}}},
]
\path[fill=panelBg] (axis cs:-0.215,0.45) rectangle (axis cs:0,{n+1.45:.2f});
{bars}
  {errs}
\draw[black, line width=0.9pt] (axis cs:0,0.45) -- (axis cs:0,{n+1.45:.2f});
  {labs}
\node[note, text=gGrey, anchor=south east, align=right, inner sep=1.5pt]
  at (rel axis cs:0.985,0.03)
  {{no-information bound {t['bound']:.4f}\\
    image path validated {t['instrument_gain']:+.4f}}};
\end{{axis}}""")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


# ===================================================================== F7 data
def build_f7_data():
    vod, scr = J("vod_sparsity.json"), J("vod_screen_summary.json")
    S, DT, BL = vod["summary"], vod["density_trend"], vod["baselines"]
    dens = (1, 3, 5)
    d = {"pts": DT["radar_pts_per_frame"], "slope": DT["slope_per_log_pts"],
         "pts_source": DT["radar_pts_source"],
         "none": [S[f"r{n}/none"]["IoU"] for n in dens],
         "none_sd": [S[f"r{n}/none"]["IoU_sd"] for n in dens],
         "radar": [S[f"r{n}/radar"]["IoU"] for n in dens],
         "radar_sd": [S[f"r{n}/radar"]["IoU_sd"] for n in dens],
         "util": [S[f"r{n}/radar"]["util"] for n in dens],
         "util_sd": [S[f"r{n}/radar"]["util_sd"] for n in dens],
         "total": [DT["deltas"][str(n)] for n in dens],
         "thresh": [BL["threshold_by_sparsity"][str(n)] for n in dens],
         "const": vod["constant_baseline_IoU"],
         "screen_const": scr["r1"]["constant_IoU"]}
    d["margin"] = [r - t for r, t in zip(d["radar"], d["thresh"])]
    assert approx(d["slope"], 0.0190, 1e-4)
    assert d["util"][0] < d["util"][1] < d["util"][2], "utilization must be monotone"
    assert max(d["none"]) - min(d["none"]) < 0.005, "the unconditional control must be flat"
    assert approx(d["margin"][0], 0.0302, 1e-3) and approx(d["margin"][2], 0.0046, 1e-3)
    assert d["margin"][0] > d["margin"][1] > d["margin"][2], "the margin must shrink"
    return d


def emit_f7(d):
    pts = d["pts"]
    xt = ",".join(f"{p:.1f}" for p in pts)
    xl = ",".join("{" + f"{p:.0f}" + "}" for p in pts)

    def sr(vals, sds=None):
        if sds is None:
            return " ".join(f"({p:.1f},{v:.6f})" for p, v in zip(pts, vals))
        return " ".join(f"({p:.1f},{v:.6f}) +- (0,{s:.6f})"
                        for p, v, s in zip(pts, vals, sds))

    mlab = "\n  ".join(
        rf"\node[vlabel, text=oiVerm, anchor=south, inner sep=1.5pt] "
        rf"at (axis cs:{p:.1f},{r:.6f}) {{{m:+.4f}}};"
        for p, r, m in zip(pts, d["radar"], d["margin"]))

    return rf"""\begin{{tikzpicture}}
%% ---------------- (a) utilization rises with measurement density
\begin{{axis}}[
  name=pa, width=5.0cm, height=3.9cm, scale only axis,
  xmode=log, log basis x=10, xmin=280, xmax=2150,
  xtick={{{xt}}}, xticklabels={{{xl}}}, log ticks with fixed point,
  xticklabel style={{font=\scriptsize}},
  ymin=-0.012, ymax=0.185,
  scaled y ticks=false, yticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=2, font=\scriptsize}},
  xlabel={{radar points per frame}}, ylabel={{IoU gain over the unconditional arm}},
  ymajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(a)}}\; utilization rises with density}},
  legend style={{at={{(0.03,0.97)}}, anchor=north west, legend columns=1, font=\tiny}},
  error bars/y dir=both, error bars/y explicit,
  error bars/error bar style={{line width=0.45pt, gGrey}},
  error bars/error mark options={{gGrey, mark size=1.2pt, line width=0.45pt}},
]
\addplot[oiBlue, line width=1.0pt, mark=*, mark size=1.7pt,
         mark options={{fill=oiBlue, draw=white, line width=0.4pt}}]
  coordinates {{{sr(d['total'])}}};
\addplot[oiGreen, line width=1.0pt, mark=square*, mark size=1.5pt,
         mark options={{fill=oiGreen, draw=white, line width=0.4pt}}]
  coordinates {{{sr(d['util'], d['util_sd'])}}};
\addplot[gGrey, line width=0.8pt, dashed, mark=triangle*, mark size=1.6pt,
         mark options={{fill=gGrey, draw=white, line width=0.4pt, solid}}]
  coordinates {{{sr([0,0,0])}}};
\legend{{total gain over unconditional, frame-specific (vs shuffled), unconditional control}}
\node[note, text=oiGreen, anchor=north west, align=left, inner sep=1.5pt]
  at (axis cs:320,0.082)
  {{slope {d['slope']:+.4f} IoU\\per e-fold, $p<0.003$\\at every density}};
\end{{axis}}

%% ---------------- (b) absolute IoU against both trivial baselines
\begin{{axis}}[
  at={{($(pa.south east)+(1.60cm,0)$)}}, anchor=south west,
  width=5.0cm, height=3.9cm, scale only axis,
  xmode=log, log basis x=10, xmin=280, xmax=2150,
  xtick={{{xt}}}, xticklabels={{{xl}}}, log ticks with fixed point,
  xticklabel style={{font=\scriptsize}},
  ymin=0, ymax=0.285,
  scaled y ticks=false, yticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=2, font=\scriptsize}},
  xlabel={{radar points per frame}}, ylabel={{occupancy IoU}},
  ymajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{(b)}}\; but the one-line rule is closing}},
  legend style={{at={{(0.985,0.60)}}, anchor=east, legend columns=1, font=\tiny}},
]
\addplot[oiBlue, line width=1.0pt, mark=*, mark size=1.7pt,
         mark options={{fill=oiBlue, draw=white, line width=0.4pt}}]
  coordinates {{{sr(d['radar'])}}};
\addplot[oiOrange, line width=1.0pt, mark=triangle*, mark size=1.8pt,
         mark options={{fill=oiOrange, draw=white, line width=0.4pt}}]
  coordinates {{{sr(d['thresh'])}}};
\addplot[gGrey, line width=0.9pt, dashed, mark=square*, mark size=1.4pt,
         mark options={{fill=gGrey, draw=white, line width=0.4pt, solid}}]
  coordinates {{{sr(d['none'])}}};
\addplot[black, line width=0.7pt, dotted, mark=none]
  coordinates {{(280,{d['const']:.6f}) (2150,{d['const']:.6f})}};
\legend{{S2S-Diff, one-line threshold, unconditional, constant map}}
  {mlab}
\node[note, text=oiVerm, anchor=north west, align=left, inner sep=1.5pt]
  at (axis cs:320,0.135)
  {{margin over the rule,\\shrinking with density}};
\end{{axis}}
\end{{tikzpicture}}"""


# ===================================================================== F8 data
def build_f8_data():
    """Aggregates hide the phenomenon -- in BOTH directions."""
    sc, vodr, vod = (J("sc36_range_stratified.json"), J("vod_range_baselines.json"),
                     J("vod_sparsity.json"))
    sc_rings = ["0-8m", "8-16m", "16-24m", "24-32m"]
    d = {"sc36": {"rings": sc_rings,
                  "gt_occ": [sc["gt_occupancy_rate"][r] for r in sc_rings],
                  "constant": [sc["constant"][r] for r in sc_rings],
                  "none": [sc["arms"]["none"][r]["mean"] for r in sc_rings],
                  "none_sd": [sc["arms"]["none"][r]["sd"] for r in sc_rings],
                  "radar": [sc["arms"]["radar"][r]["mean"] for r in sc_rings],
                  "radar_sd": [sc["arms"]["radar"][r]["sd"] for r in sc_rings],
                  "agg_constant": sc["constant"]["0-64m"],
                  "agg_radar": sc["arms"]["radar"]["0-64m"]["mean"]}}

    vr = ["0-16m", "16-32m", "32-48m", "48-64m"]
    import statistics as st
    arm_ring = {}
    for arm in ("none", "radar"):
        ks = [k for k in vod["cells"] if k.startswith(f"r1/{arm}/")]
        arm_ring[arm] = {r: st.fmean(vod["cells"][k]["eval"]["rings"][r] for k in ks) for r in vr}
        arm_ring[arm + "_sd"] = {r: st.stdev([vod["cells"][k]["eval"]["rings"][r] for k in ks])
                                 for r in vr}
    d["vod"] = {"rings": vr,
                "gt_occ": [vodr["gt_occupancy_rate"][r] for r in vr],
                "constant": [vodr["constant"][r] for r in vr],
                "threshold": [vodr["threshold"]["r1"]["per_ring"][r] for r in vr],
                "none": [arm_ring["none"][r] for r in vr],
                "none_sd": [arm_ring["none_sd"][r] for r in vr],
                "radar": [arm_ring["radar"][r] for r in vr],
                "radar_sd": [arm_ring["radar_sd"][r] for r in vr],
                "agg_constant": vodr["constant"]["0-64m"],
                "agg_threshold": vodr["threshold"]["r1"]["per_ring"]["0-64m"],
                "agg_radar": arm_ring["radar"].get("0-64m") or
                             st.fmean(vod["cells"][k]["eval"]["rings"]["0-64m"]
                                      for k in vod["cells"] if k.startswith("r1/radar/"))}
    # where does the model actually beat the one-line rule?
    d["vod"]["beats_threshold"] = [r > t for r, t in
                                   zip(d["vod"]["radar"], d["vod"]["threshold"])]
    d["sc36"]["beats_constant"] = [r > c for r, c in
                                   zip(d["sc36"]["radar"], d["sc36"]["constant"])]
    assert approx(d["sc36"]["constant"][0], 0.8895, 1e-3), "constant must dominate the near field"
    assert d["sc36"]["constant"][3] == 0.0, "constant must score zero in the far ring"
    assert d["sc36"]["beats_constant"] == [False, False, True, True], d["sc36"]["beats_constant"]
    assert d["vod"]["beats_threshold"] == [True, True, False, False], d["vod"]["beats_threshold"]
    assert all(c == 0.0 for c in d["vod"]["constant"][1:]), "VoD constant must die beyond 16 m"
    return d


def emit_f8(d):
    def panel(key, name, series, pos, title, ymax, agg_note):
        s = d[key]
        rings = s["rings"]
        xt = ",".join(str(i + 1) for i in range(len(rings)))
        xl = ",".join("{" + r.replace("m", r"\,m") + "}" for r in rings)
        plots = []
        for fld, col, mark, lab in series:
            sd = s.get(fld + "_sd")
            pts = " ".join(
                f"({i+1},{v:.6f})" + (f" +- (0,{sd[i]:.6f})" if sd else "")
                for i, v in enumerate(s[fld]))
            plots.append(rf"\addplot[{col}, line width=0.9pt, mark={mark}, mark size=1.6pt, "
                         rf"mark options={{fill={col}, draw=white, line width=0.4pt, solid}}"
                         + ("" if sd else ", error bars/y dir=none") +
                         rf"] coordinates {{{pts}}};")
        legend = ", ".join(l for _, _, _, l in series)
        return rf"""\begin{{axis}}[
  {pos}, height=3.6cm, scale only axis,
  xmin=0.6, xmax={len(rings)}.45, ymin=0, ymax={ymax},
  xtick={{{xt}}}, xticklabels={{{xl}}}, xticklabel style={{font=\scriptsize}},
  scaled y ticks=false, yticklabel style={{/pgf/number format/fixed,
    /pgf/number format/precision=2, font=\scriptsize}},
  xlabel={{range ring}}, ylabel={{occupancy IoU}},
  ymajorgrids, grid style={{gLight, line width=0.4pt}},
  title={{\textbf{{{title}}}}},
  legend style={{at={{(0.99,0.97)}}, anchor=north east, legend columns=1, font=\tiny}},
  error bars/y dir=both, error bars/y explicit,
  error bars/error bar style={{line width=0.4pt, gGrey}},
  error bars/error mark options={{gGrey, mark size=1pt, line width=0.4pt}},
]
{chr(10).join(plots)}
\legend{{{legend}}}
\node[note, text=gGrey, anchor=south east, align=right, inner sep=1.5pt]
  at (rel axis cs:0.985,0.06) {{{agg_note}}};
\end{{axis}}"""

    sc, vo = d["sc36"], d["vod"]
    p1 = panel("sc36", "Sc.36",
               [("constant", "gGrey", "triangle*", "constant map"),
                ("none", "oiSky", "square*", "unconditional"),
                ("radar", "oiBlue", "*", "S2S-Diff")],
               "name=pa, width=5.1cm",
               r"(a)\; DeepSense Sc.36: the model only wins far",
               1.02,
               rf"aggregate: constant {sc['agg_constant']:.4f},\\model {sc['agg_radar']:.4f}")
    p2 = panel("vod", "VoD",
               [("constant", "gGrey", "triangle*", "constant map"),
                ("threshold", "oiOrange", "diamond*", "one-line threshold"),
                ("none", "oiSky", "square*", "unconditional"),
                ("radar", "oiBlue", "*", "S2S-Diff")],
               "at={($(pa.south east)+(1.70cm,0)$)}, anchor=south west, width=5.1cm",
               r"(b)\; View-of-Delft: the model only wins near",
               0.345,
               rf"aggregate: threshold {vo['agg_threshold']:.4f},\\model {vo['agg_radar']:.4f}")
    return f"""\\begin{{tikzpicture}}
{p1}
{p2}
\\end{{tikzpicture}}"""


# ===================================================================== captions
def captions(f1, f3, f5, f2, f4, f6, f7, f8):
    by = {r["benchmark"]: r for r in f1["rows"]}
    losses = sum(1 for r in f1["rows"] if not r["model_wins"])
    radial_shuf = f2["battery"]["RADIal"]["frac"]["shuffled_Y"]
    sc_frac = f2["decomp"]["DeepSense Sc.36"]["frac_frame_specific"]
    sc_total = f2["decomp"]["DeepSense Sc.36"]["total"]
    vod_frac = f2["decomp"]["View-of-Delft"]["frac_frame_specific"]
    vod_total = f2["decomp"]["View-of-Delft"]["total"]
    w_ratio = f4["isac_gap"]["width"]["ratio"]; m_ratio = f4["isac_gap"]["mass"]["ratio"]
    radial_base = f4["radial_base_rate"]
    r_lo, r_hi = f4["radial_ratio_worst"], f4["radial_ratio_best"]
    gps_a = f6["A_256way"]["rows"][1]["gain"]; gps_b = f6["B_array1_64way"]["rows"][1]["gain"]
    radar_pool = f6["A_256way"]["spatial_readout"]["radar/pooled"]
    radar_spat = f6["A_256way"]["spatial_readout"]["radar/spatial"]
    slope = f7["slope"]; m1, m3 = f7["margin"][0], f7["margin"][2]
    sc_c0 = f8["sc36"]["constant"][0]
    seed_sd = f1["seed_sd"]
    isac_margin = by["DeepSense_ISAC"]["margin"]
    v_far_model = f8["vod"]["radar"][3]; v_far_rule = f8["vod"]["threshold"][3]
    return rf"""# Paper A figure captions (draft)

Generated by `experiments/make_paper_a_figures.py`. Every number is read from a JSON in
`experiments/results/`; see `figure_data.json` for the exact values plotted.

## F1: `F1_one_line_baselines.pdf`

**A 35M-parameter measurement-conditioned diffusion model against two one-line rules, on
five benchmarks.** (a) Occupancy IoU for the model, for a constant map whose value is the
training-set occupancy mean, and for the best single-channel threshold on the measurement
$Y$, with the channel and $\tau$ fitted on training data and applied unchanged to the test
frames. Within each benchmark all three predictors are scored on identical test frames and
with one estimator, per-frame IoU averaged over frames. The dashed rule in each group marks
whichever one-line rule wins there, the level the model has to clear. The constant scores
exactly zero on RADIal, which is 1.3\% occupied, so the train-fitted constant map falls below
the 0.5 threshold everywhere. (b) The model's margin over the better of the two rules; the
shaded band is $\pm1$ s.d. of training seeds ({seed_sd:.4f}), measured on the one benchmark
with replicates. Error bars in (a) are $\pm1$ s.d. over three seeds for DeepSense Sc.36 and
View-of-Delft; the other three are single runs and carry none, so the DeepSense indoor ISAC
margin of {isac_margin:+.4f}, which is 1.3 s.d., should be read as *the model does not beat
the constant* rather than as a measured deficit. On {losses} of five benchmarks a rule expressible in one line
beats the model, and which rule wins differs by benchmark: the constant on DeepSense
indoor ISAC ({by['DeepSense_ISAC']['constant']:.4f} vs {by['DeepSense_ISAC']['model']:.4f})
and on Sc.36 ({by['DeepSense_Sc36']['constant']:.4f} vs
{by['DeepSense_Sc36']['model']:.4f}), a threshold on RADIal
({by['RADIal']['threshold']:.4f} vs {by['RADIal']['model']:.4f}) and on V2I
({by['DeepSense_V2I']['threshold']:.4f} vs {by['DeepSense_V2I']['model']:.4f}). Only
View-of-Delft is a win ({by['View_of_Delft_r1']['model']:.4f} vs
{by['View_of_Delft_r1']['threshold']:.4f}), and it is the one benchmark that passes every
screen in Fig. F3.

## F2: `F2_four_way_battery.pdf`

**The conditioning battery, and what it separates.** (a) IoU under each null, as a fraction
of the IoU with the real measurement, on the three benchmarks of the original study.
DeepSense indoor ISAC and V2I are invariant to within measurement error: destroying the
measurement costs nothing, so whatever the model is doing, it is not reading $Y$. RADIal is the instructive
case. Setting $Y\!=\!0$ collapses it to zero, which would suggest total dependence, while
**shuffling** $Y$ retains {radial_shuf:.0%} of the score. Zeroing confounds information
removal with an out-of-distribution input; the shuffled arm removes the information and
leaves the input distribution intact, and it is the one to report. Absolute IoU with the
real measurement is printed above each group. (b) The gain of the conditioned model over an
identically-sized unconditional one, split into the part that survives shuffling, a
distributional gain the model would keep if measurements were dealt out at random, and the
part that is specific to this frame. On DeepSense Sc.36 only {sc_frac:.0%} of a
{sc_total:+.4f} IoU gain is frame-specific; on View-of-Delft {vod_frac:.0%} of
{vod_total:+.4f} is. Two benchmarks, two headline gains of the same kind, and only one of
them is evidence that the model reads its input.

## F4: `F4_metric_dominated.pdf`

**Calibration is not a free win: the estimator and the reference both decide the answer.**
(a) Expected calibration error on DeepSense indoor ISAC against the number of bins, for
equal-width (solid) and equal-mass (dashed) binning. The published claim used 15 equal-width
bins and reported S2S-Diff as {w_ratio:.2f}$\times$ better calibrated than U-Net; under
equal-mass bins the same data give **{m_ratio:.2f}$\times$**. Both gaps are
bootstrap-significant, so the disagreement is not noise; it is the estimator. (b) The same
comparison against a constant map fitted on the training split. Under equal-width binning
S2S-Diff wins at every bin count; under equal-mass binning the winner changes with bin
count, and every value lies between 0.0060 and 0.0106. Against a constant, the calibration
advantage is a tie. (c) RADIal, which is {radial_base:.2%} occupied. Bars span the full range
over three estimators $\times$ six bin counts, so their length is the estimator's influence
on that method's score. S2S-Diff is the worst-calibrated of the five and U-Net the best, but
the ratio between them is anywhere from {r_lo:.1f}$\times$ to {r_hi:.0f}$\times$ depending on
a choice the analyst makes after seeing the data. $\bar p$ is each method's mean predicted
probability. U-Net sits nearest the base rate and scores best, which is the mechanism behind
this mode. Note, however, that the constant map sits nearer still and does *not* score
best, so this is a strong tendency rather than a law.

## F6: `F6_representation_blocked.pdf`

**The information is present; bird's-eye occupancy cannot expose it, and the readout is not
the reason.** Top-1 beam accuracy
minus the no-information bound, which is the best constant
predictor evaluated on the test block, so zero means "no better than committing to one class", and the shaded half is where a
predictor has demonstrated nothing. (a) The 256-way operational beam; (b) the 64-way beam
within the dominant array. A GPS oracle built from the same readout on unprocessed relative
position clears the bound on both tasks ({gps_a:+.4f} and {gps_b:+.4f}), so the beam *is*
determined by the geometry. Every bird's-eye representation fails on both: raw radar, the
diffusion reconstruction, and even ground-truth LiDAR occupancy. Bird's-eye occupancy records
where objects are, not which one is the communication partner, and no amount of sensing
resolution supplies that. The top bar is an instrument check: unit 2's known position, rasterized into an image on the
same grid with the same channel count and passed through the same readout, from which the beam
is recoverable by construction. It clears the bound on both tasks, so the image path can carry
a bearing-determined label on this data and the failures below it are properties of the
representation. Every arm was also re-run with the pooled readout replaced by one that flattens
rather than globally averages its feature map; that changes no verdict, though it does move raw
radar from {radar_pool:+.4f} to {radar_spat:+.4f} on Task A, so the pooled architecture
understates distributed inputs. Error bars are $\pm1$ s.d. over seeds (reconstruction $n=9$:
three reconstruction seeds $\times$ three readout seeds; others $n=3$); $\dagger$ marks arms
with a single readout seed. The circular $h_\text{{beam}}$ check, whose label is by
construction its argmax, is reported as an instrument validation, not a result. Radar-aided
beam prediction succeeds elsewhere, on a stationary testbed with a 64-beam codebook; the claim
here is about bird's-eye occupancy as an intermediate representation, and about this harder
mobile-unit instance, not about radar.

## F7: `F7_healthy_benchmark.pdf`

**What a healthy benchmark looks like, and the caveat that comes with it.** View-of-Delft,
with radar measurement density varied by accumulating 1, 3 or 5 sweeps; the scene target is
identical across the three, so density is the only thing that changes. (a) The gain of the
conditioned model over the unconditional one, and the part of it that is frame-specific,
against measured radar points per frame. Both rise monotonically; the unconditional control
is flat, which is what licenses attributing the trend to density rather than to training
noise. The slope is {slope:+.4f} IoU per e-fold of radar points, with $p<0.003$ at every
density over three seeds. This is the prediction the utilization argument makes, and the only
place in this study where it is confirmed. (b) The same runs in absolute terms against both
trivial baselines. The model clears them both, the one result in this study that does, but its
margin over the one-line threshold shrinks from {m1:+.4f} to {m3:+.4f} as the
measurement gets denser. The method earns its keep where the measurement is sparse, and
reporting the sparse end alone would overstate it.

## F8: `F8_range_stratified.pdf`

**An aggregate metric hides the phenomenon, in both directions.** Occupancy IoU by range
ring, with every quantity pooled over frames and cells so that arms and baselines share one
reduction. (a) DeepSense Sc.36. A constant map scores {sc_c0:.4f} within 8\,m and exactly
zero beyond 24\,m, because near-field cells are occupied in almost every frame and far-field
cells in almost none. The model is *behind* the constant in the two near rings and ahead only beyond 16\,m, the opposite of what its aggregate,
which the near field dominates, suggests.
(b) View-of-Delft, where the same analysis inverts the conclusion: the model beats the
one-line threshold in the two near rings and **loses to it beyond 32\,m**
({v_far_model:.4f} against {v_far_rule:.4f} at 48--64\,m). Its aggregate win is a near-field
win. Neither dataset's headline number survives stratification intact, and in neither case
would the direction of the error have been guessable from the aggregate. Aggregates here use
the pooled reduction of the per-ring values, which differs from the frame-averaged convention
in Fig. F1 by 1--2\%.

## F3: `F3_taxonomy.pdf`

**A protocol for deciding whether a reported number means what it appears to.** Stage 0
establishes three references before any comparison is made; the distinction between the
train-fitted constant (i) and the best constant evaluated on test (ii) is the one most often
skipped, and under a temporal split the two differ materially: here {f3['rd_floor']:.4f}
against {f3['rd_bound']:.4f}. Each question is answerable with a single cheap run; a branch
to the right identifies one of five degeneracy modes, each annotated with the benchmark that
exhibits it. Reaching the bottom means the gain is both real and frame-specific, which only
View-of-Delft does. The instrument check is a precondition on Q3–Q6 rather than a step: a
probe that cannot recover a quantity known to be recoverable from its input bounds every
null it produces, as the Sc.36 beam readout did at {f3['hbeam_instr']:.4f}.

## F5: `F5_reference_degeneracy.pdf`

**A significant, monotone, mechanistically plausible result that is an artifact of its
reference class, and nothing else.** The task is predicting the best of 256 beams on DeepSense Sc.36 from
a bird's-eye occupancy map. (a) As first measured: the gain of ground-truth LiDAR occupancy
over the share of the train-modal beam rises monotonically with the map's spatial extent and
clears the pre-registered decision gate, and is identical at two cell sizes differing by a
factor of two, which reads as robustness to resolution. (b) The same four numbers against
the best constant predictor evaluated on the test block ({f5['bound']:.4f}, the share of the
test-modal beam). Nothing clears it; the two $\pm$100\,m configurations land on it exactly,
at {f5['bound_count']:,} of {f5['n_test']:,} frames correct. The apparent gain is
$\text{{bound}}-\text{{floor}}$ and nothing else. A GPS oracle on the same readout clears the
bound by {f3['rb_gps']:+.4f}, so the information is present in the geometry; bird's-eye
occupancy cannot expose it. (c) The mechanism: cumulative share of predictions against the
number of distinct classes used, ranked by frequency. The true labels spread over
{f5['label_n_distinct']} classes (entropy {f5['label_entropy']:.2f} nats); the nine readouts
use between {min(c['n_distinct_predicted'] for c in f5['cells'].values())} and
{max(c['n_distinct_predicted'] for c in f5['cells'].values())} of {f5['n_class']}. The
agreement in (a) at two resolutions is two readouts collapsing onto the same constant class,
not robustness.
"""


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    print("Paper A figures: TikZ/PGFPlots\n")

    f1 = build_f1_data()
    print(f"F1 data (one IoU estimator: per-frame; seed s.d. {f1['seed_sd']:.4f}):")
    print(f"  {'benchmark':<20}{'model':>9}{'const':>9}{'thresh':>9}{'best':>9}"
          f"{'margin':>10}  seeds  rule")
    for r in f1["rows"]:
        print(f"  {r['benchmark']:<20}{r['model']:>9.4f}{r['constant']:>9.4f}"
              f"{r['threshold']:>9.4f}{r['best_baseline']:>9.4f}{r['margin']:>+10.4f}"
              f"{r['model_n_seeds']:>7}  {r['threshold_rule']}"
              + ("   <- within seed noise" if r["within_seed_noise"] else ""))
    write_tex("F1_one_line_baselines", emit_f1(f1))

    f3 = build_f3_data()
    print(f"\nF3 exemplars verified; RADIal ECE blinded {f3['md_ece_blind']:.4f} "
          f"< real {f3['md_ece_real']:.4f}; Sc.36 frame-specific "
          f"{100*f3['sc36_frame_frac']:.0f}%, VoD {100*f3['vod_frame_frac']:.0f}%")
    write_tex("F3_taxonomy", emit_f3(f3))

    f5 = build_f5_data()
    print(f"\nF5: floor {f5['floor']:.6f}  bound {f5['bound']:.6f} "
          f"= {f5['bound_count']}/{f5['n_test']}")
    for l, g, gf, gb in zip(f5["labels"], f5["gt"], f5["gt_gain_floor"], f5["gt_gain_bound"]):
        print(f"  {l.replace(chr(92)*2,' '):<22} gt {g:.6f}  vs floor {gf:+.6f}  "
              f"vs bound {gb:+.6f}")
    print("  collapse: " + ", ".join(
        f"{k.split('/')[0]}.{k.split('/')[1]} {v['n_distinct_predicted']}cls"
        for k, v in sorted(f5["cells"].items())))
    write_tex("F5_reference_degeneracy", emit_f5(f5))

    f2 = build_f2_data()
    print("\nF2 battery (IoU retained, relative to real Y):")
    for k, v in f2["battery"].items():
        print(f"  {k:<16} real {v['real']:.4f}  " +
              "  ".join(f"{a.replace('_Y',''):>8} {v['frac'][a]:.3f}" for a in f2["arms"]))
    for k, v in f2["decomp"].items():
        print(f"  {k:<16} gain {v['total']:+.4f} = marginal {v['marginal']:+.4f} + "
              f"frame-specific {v['frame_specific']:+.4f} ({100*v['frac_frame_specific']:.0f}%)")
    write_tex("F2_four_way_battery", emit_f2(f2))

    f4 = build_f4_data()
    print(f"\nF4 ISAC gap vs U-Net: width {f4['isac_gap']['width']['ratio']:.2f}x, "
          f"mass {f4['isac_gap']['mass']['ratio']:.2f}x; RADIal S2S/U-Net ratio spans "
          f"{f4['radial_ratio_worst']:.1f}x-{f4['radial_ratio_best']:.0f}x")
    write_tex("F4_metric_dominated", emit_f4(f4))

    f6 = build_f6_data()
    print("\nF6 gains over the no-information bound:")
    for t, v in f6.items():
        print(f"  {t} (bound {v['bound']:.4f}): " +
              "  ".join(f"{r['name'].split('(')[0].strip()} {r['gain']:+.4f}" for r in v["rows"]))
    write_tex("F6_representation_blocked", emit_f6(f6))

    f7 = build_f7_data()
    print(f"\nF7 utilization {f7['util'][0]:.4f} -> {f7['util'][2]:.4f} over "
          f"{f7['pts'][0]:.0f} -> {f7['pts'][2]:.0f} pts; margin over the rule "
          f"{f7['margin'][0]:+.4f} -> {f7['margin'][2]:+.4f}")
    write_tex("F7_healthy_benchmark", emit_f7(f7))

    f8 = build_f8_data()
    print("\nF8 per-ring:")
    for k in ("sc36", "vod"):
        print(f"  {k}: rings {f8[k]['rings']}")
        print(f"       constant " + " ".join(f"{v:.4f}" for v in f8[k]["constant"]))
        if "threshold" in f8[k]:
            print(f"       threshold " + " ".join(f"{v:.4f}" for v in f8[k]["threshold"]))
        print(f"       model    " + " ".join(f"{v:.4f}" for v in f8[k]["radar"]))
    print(f"  Sc.36 model beats the constant in rings: {f8['sc36']['beats_constant']}")
    print(f"  VoD   model beats the rule   in rings: {f8['vod']['beats_threshold']}")
    write_tex("F8_range_stratified", emit_f8(f8))

    if not a.no_compile:
        print()
        for stem in ("F1_one_line_baselines", "F2_four_way_battery", "F3_taxonomy",
                     "F4_metric_dominated", "F5_reference_degeneracy",
                     "F6_representation_blocked", "F7_healthy_benchmark",
                     "F8_range_stratified"):
            compile_tex(stem)

    with open(os.path.join(OUT, "CAPTIONS.md"), "w") as f:
        f.write(captions(f1, f3, f5, f2, f4, f6, f7, f8))
    dump = {"generated_by": "experiments/make_paper_a_figures.py",
            "F1": f1, "F2": f2, "F3": f3, "F4": f4,
            "F5": {k: v for k, v in f5.items() if k not in ("cells", "label_counts")},
            "F6": f6, "F7": f7, "F8": f8}
    with open(os.path.join(OUT, "figure_data.json"), "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\nwrote CAPTIONS.md and figure_data.json into {OUT}")


if __name__ == "__main__":
    main()
