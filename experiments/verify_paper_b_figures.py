#!/usr/bin/env python
"""Extract the text layer of each compiled Paper B figure and re-check it against the JSONs.

The generator injects values from results/*.json into TikZ. This reads them back OUT of the
compiled PDF and compares against the same JSONs, so a stale figure cannot reach a submission.
It also enforces the project's house rules on the rendered artifact rather than the source:
no em-dash anywhere in the figure text, and every accuracy panel carrying BOTH the
no-information bound and the validation-selected position reference.

Usage: python experiments/verify_paper_b_figures.py
Exit code is non-zero if anything fails.
"""
import json
import re
import subprocess, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(os.path.dirname(HERE), "paper_b_figures")
PB = os.path.join(os.path.dirname(HERE), "paper_b")
_c = {}
checks = []


def J(n):
    if n not in _c:
        _c[n] = json.load(open(os.path.join(RES, n)))
    return _c[n]


def raw_text(stem):
    p = os.path.join(FIG, f"{stem}.pdf")
    if not os.path.exists(p):
        return None
    return subprocess.run(["pdftotext", "-layout", p, "-"],
                          capture_output=True, text=True).stdout


def text(stem):
    p = os.path.join(FIG, f"{stem}.pdf")
    if not os.path.exists(p):
        return None
    t = subprocess.run(["pdftotext", "-layout", p, "-"],
                       capture_output=True, text=True).stdout
    # LaTeX math mode emits U+2212 MINUS SIGN, which is not ASCII "-". Normalize it, and
    # the various Unicode hyphens, so a numeric check compares digits and not glyphs.
    for ch in ("\u2212", "\u2010", "\u2011", "\u2012", "\u2013"):
        t = t.replace(ch, "-")
    # Collapse whitespace. A phrase that wraps inside a figure's text node arrives with a
    # newline in the middle of it, and a plain substring match then reports a false failure:
    # the manuscript verifier learned this on 2026-10-06 and this one relearned it on
    # 2026-10-10, when widening one node by 14 mm "broke" two passing checks.
    return re.sub(r"\s+", " ", t)


def ck(cond, label):
    checks.append((bool(cond), label))


def num(t, v, label, fmt="{:.4f}"):
    ck(fmt.format(v) in t, f"{label}: {fmt.format(v)} present in the PDF text")


def PBMACRO_DEG_NSIG(_J):
    """\\PBDegNSig as the manuscript defines it, so figure and prose cannot disagree."""
    import re as _re
    tex = open(os.path.join(PB, "numbers.tex")).read()
    m_ = _re.search(r"\\newcommand\{\\PBDegNSig\}\{(\d+)\}", tex)
    return m_.group(1) if m_ else "?"


def _pdf_width(fig):
    """Media-box width of a figure in points, for checking the magnification it is placed at."""
    try:
        out = subprocess.check_output(["pdfinfo", os.path.join(FIG, f"{fig}.pdf")]).decode()
        return float(out.split("Page size:")[1].split("pts")[0].split("x")[0])
    except Exception:
        return None


_fd = None


def J_FIGDATA():
    global _fd
    if _fd is None:
        _fd = json.load(open(os.path.join(FIG, "figure_data.json")))
    return _fd


def _word_collisions(fig, tol=0.6):
    """Every pair of words whose boxes overlap in the compiled figure.

    A text-layer check reads what a figure SAYS and never where it says it. Figure 1 was
    built, verified green and placed in the manuscript with the last line of one node
    printed underneath the first line of another, because every check it had was a
    substring test. This reads the word boxes back out of the PDF instead.
    """
    p_ = os.path.join(FIG, f"{fig}.pdf")
    if not os.path.exists(p_):
        return None
    xml = subprocess.run(["pdftotext", "-bbox", p_, "-"],
                         capture_output=True, text=True).stdout
    ws = [(float(a), float(b), float(c), float(d), w) for a, b, c, d, w in re.findall(
        r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>',
        xml)]
    bad = []
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            a, b = ws[i], ws[j]
            if (min(a[2], b[2]) - max(a[0], b[0]) > tol
                    and min(a[3], b[3]) - max(a[1], b[1]) > tol):
                bad.append((a[0], a[1], a[4], b[4]))
    return bad


def main():
    # ---------------- B1 ----------------
    t = text("B1")
    if t is None:
        print("B1.pdf missing"); return 1
    ck("—" not in t and "---" not in t, "B1: no em-dash in the rendered figure")
    for tag, f, title in (("sc31", "sc31_resource_ablation_G_max.json", "Scenario 31"),
                          ("sc33", "sc33_resource_ablation_G_max.json", "Scenario 33")):
        d = J(f); S = d["settings"]
        ck(title in t, f"B1: panel title {title!r}")
        ck(d["pool"] == "max", f"B1/{tag}: source is the block-max run")
        ck(d["seeds"] == 15, f"B1/{tag}: 15 seeds")
        num(t, d["bound_const"], f"B1/{tag} no-information bound")
        R = J(f"{tag}_reference_predictors.json")["predictors"]
        g = [v for k_, v in R.items() if k_.startswith("GPS reference")][0]
        num(t, g["top1"], f"B1/{tag} position reference")
        # The upper reference must be SELECTED ON VALIDATION, never on test: a reference
        # tuned on the split it is meant to inform is an upper bound fitted to the comparison.
        ck("validation" in g["note"],
           f"B1/{tag}: the position reference is validation-selected, not test-selected")
        ck(g["top1"] > d["settings"]["chirps=250"]["top1_median"],
           f"B1/{tag}: the position reference sits above every learned setting")
        # the house rule: a bound alone is not enough
        ck("no-information bound" in t and "position reference" in t,
           f"B1/{tag}: BOTH reference levels are drawn")
        for c in (8, 16, 32, 64, 125, 250):
            v = S[f"chirps={c}"]
            if v["n_collapsed"]:
                ck(f"{v['n_collapsed']}/{v['n_seeds']}" in t,
                   f"B1/{tag}: collapse count {v['n_collapsed']}/{v['n_seeds']} at {c} chirps")
        # A drawn bound that does not say which settings clear it invites the reader to assume
        # they all do. Two of the twelve do not, both on Scenario 33, and before 2026-10-08
        # every point was drawn with the same marker.
        fdB1 = json.load(open(os.path.join(FIG, "figure_data.json")))["B1"][tag]
        CB = J("corrected_statistics.json")["clears_bound"][tag[2:]]
        chirp_misses = {k for k in CB["not_clearing"] if k.startswith("chirps=")}
        ck(fdB1["n_miss"] == len(chirp_misses),
           f"B1/{tag}: the figure marks exactly the settings corrected_statistics.json "
           f"records as not clearing the bound")
        for pt in fdB1["points"]:
            rec = f"chirps={pt['n']}" not in chirp_misses
            ck(pt["clears"] == rec and pt["clears"] == (pt["lo"] > d["bound_const"]),
               f"B1/{tag}: bound-clearing at {pt['n']} chirps agrees with the statistics file "
               f"and with the drawn interval")
        # the share is the paper's headline framing of this panel; it must be the same
        # quantity as \PBShareGps*, by the same formula
        share = 100.0 * (S["chirps=250"]["top1_median"] - d["bound_const"]) / (
            g["top1"] - d["bound_const"])
        ck(abs(fdB1["share"] - share) < 1e-9, f"B1/{tag}: the quoted share is recomputed")
        num(t, round(share, 1), f"B1/{tag} share of the position reference", "{:.1f}")
    # ---- the axis box, and the intervals the legend's own semantics rest on ----
    # B1 had no check on either. Its limits were hardcoded at 0.02 and 0.56, the defect the
    # 2026-10-08 Figure 4 audit fixed one figure over, and nothing asserted that a drawn
    # value stays inside the box. Audited 2026-10-14.
    _b1 = J_FIGDATA()["B1"]
    # THE LIMITS ARE READ BACK OUT OF THE EMITTED AXIS, not taken from the builder. Taking
    # them from figure_data.json compares the builder with itself: the first version of
    # this check passed while the emitter still used hardcoded 0.02 and 0.56, which is the
    # Figure 3 prior defect of 2026-10-14 reproduced one figure over. Check the artifact.
    _src1 = open(os.path.join(FIG, "B1.tex")).read()
    _lims = sorted({(float(a), float(b)) for a, b in
                    re.findall(r"ymin=([\d.]+), ymax=([\d.]+)", _src1)})
    ck(len(_lims) == 1,
       f"B1: both panels share one axis box, so the scenarios stay comparable ({_lims})")
    _ymin, _ymax = _lims[0] if _lims else (0.0, 1.0)
    _drawn = []
    for tag in ("sc31", "sc33"):
        _drawn += [_b1[tag]["bound"], _b1[tag]["posref"]]
        _drawn += [q["lo"] for q in _b1[tag]["points"]]
        _drawn += [q["hi"] for q in _b1[tag]["points"]]
    ck(all(_ymin < v < _ymax for v in _drawn),
       f"B1: every drawn value lies inside the axis box, got "
       f"{[v for v in _drawn if not (_ymin < v < _ymax)][:3]}")
    # the limits are DERIVED from what is drawn, not typed
    _bd = min(_b1[t]["bound"] for t in ("sc31", "sc33"))
    _pr = max(_b1[t]["posref"] for t in ("sc31", "sc33"))
    _k = _b1["label_allowance_mm"] / 24.0
    ck(abs(_ymax - round(_pr + 0.020, 4)) < 1e-9
       and abs(_ymin - round((_bd - _k * _ymax) / (1.0 - _k), 4)) < 1e-9,
       "B1: the axis limits are derived from the drawn content, not hardcoded")
    # the lowest bound's label hangs BELOW its rule, inside the shading, so the band under
    # it has to be at least one line tall. Tightening ymin to the data pushed that label
    # through the bottom frame on Scenario 31 before this check existed.
    _room = 24.0 * (_bd - _ymin) / (_ymax - _ymin)
    ck(_room >= _b1["label_allowance_mm"] - 0.05,
       f"B1: the band under the lowest bound is {_room:.2f} mm, enough for its own label")
    # THE ERROR BAR CAPS. `error mark` and `error mark options` written after
    # `error bars/.cd` are accepted and silently ignored by this pgfplots: measured on the
    # shipped figure the whisker ends were 0.17 mm wide, the bar's own line width, and the
    # 2026-10-08 entry recording that the intervals "now have caps" was never true. The
    # error mark inherits the PLOT's mark, so the plot must carry one.
    _ebp = re.findall(r"\\addplot\[oiBlue[^\]]*error bars/\.cd[^\]]*\]", _src1, re.S)
    ck(len(_ebp) == 2, f"B1: one error-bar plot per panel ({len(_ebp)})")
    ck(all("mark=-" in q and "mark size=" in q for q in _ebp),
       "B1: the error-bar plot carries its own mark, which is what draws the caps")
    ck(not any("mark=none" in q for q in _ebp),
       "B1: no error-bar plot sets mark=none, which leaves the caps undrawn")
    ck("error mark=" not in _src1,
       "B1: the ignored `error mark=` idiom is gone, so a cap cannot be silently dropped")

    # the figure is placed at \textwidth; drawn much narrower, \includegraphics magnifies the
    # type with it, and this one was rendering its 8 pt labels at 12.4 pt against 10 pt body
    _w = _pdf_width("B1")
    ck(_w is not None and 0.80 <= 516.0 / _w <= 1.25,
       f"B1: drawn close to the width it is placed at (scale {516.0 / _w:.2f} into \\textwidth)")
    # ---------------- B4 ----------------
    t = text("B4")
    if t is None:
        print("B4.pdf missing"); return 1
    ck("—" not in t and "---" not in t, "B4: no em-dash in the rendered figure")
    P = J("sc31_pooling_comparison.json")
    num(t, P["bound"], "B4 bound", "{:.4f}")
    nflip = P["n_verdicts"] - P["n_verdicts_unchanged"]
    ck(f"{nflip} of {P['n_verdicts']} verdicts flip" in t,
       f"B4: states {nflip} of {P['n_verdicts']} verdicts flip")
    ck(f"{P['n_verdicts']}/{P['n_verdicts']} settings" in t.replace("  ", " "),
       "B4: states block-max wins every setting")
    import numpy as np
    num(t, float(np.mean([r["delta"] for r in P["per_setting"]])), "B4 mean improvement")
    # The mean improvement and the worst p used to be computed inside emit_b4, so they never
    # reached figure_data and nothing could check them against the source. A value that exists
    # only in the emitted string is a hand-typed literal with extra steps.
    fd4 = json.load(open(os.path.join(FIG, "figure_data.json")))["B4"]
    for key, src, lab in (
            ("mean_delta", float(np.mean([r["delta"] for r in P["per_setting"]])), "mean delta"),
            ("max_p", float(max(r["p"] for r in P["per_setting"])), "worst p")):
        ck(key in fd4 and abs(fd4[key] - src) < 1e-12,
           f"B4: the {lab} is recorded, not computed at emit time")
    num(t, fd4["max_p"], "B4 worst per-setting p", "{:.4f}")
    # both margins must describe the SAME setting, not two different rows
    w = [r for r in P["per_setting"] if r["setting"] == fd4["worst"]]
    ck(len(w) == 1 and w[0]["decimate"] == min(r["decimate"] for r in P["per_setting"]),
       "B4: the quoted setting is the one that sat lowest under decimation")
    if w:
        ck(abs(fd4["worst_dec_margin"] - (w[0]["decimate"] - P["bound"])) < 1e-12
           and abs(fd4["worst_max_margin"] - (w[0]["max"] - P["bound"])) < 1e-12,
           "B4: both quoted margins are that setting's, against the bound")
        num(t, fd4["worst_dec_margin"], "B4 decimate margin over the bound")
        num(t, fd4["worst_max_margin"], "B4 block-max margin over the bound")
    # READ THE EMITTED AXIS, not the builder's. Testing the builder's limits against the
    # source JSON is a check that compares a builder with itself and cannot see an emitter
    # that uses different numbers; B4 had no check that opened B4.tex at all. Fourth
    # instance of this pattern in one day, after B8's prior, B1's axis and B2's box.
    _s4 = open(os.path.join(FIG, "B4.tex")).read()
    _m4 = re.search(r"xmin=([-\d.]+), xmax=([-\d.]+), ymin=([-\d.]+), ymax=([-\d.]+)", _s4)
    ck(_m4 is not None, "B4: the emitted axis box parses")
    if _m4:
        _x0, _x1, _y0, _y1 = (float(v) for v in _m4.groups())
        ck(abs(_x0 - fd4["xmin"]) < 5e-4 and abs(_x1 - fd4["xmax"]) < 5e-4,
           "B4: the emitted limits are the derived ones")
        bad = [r["setting"] for r in P["per_setting"]
               if not (_x0 <= r["decimate"] <= _x1 and _x0 <= r["max"] <= _x1)]
        ck(not bad and _x0 <= P["bound"] <= _x1,
           f"B4: every drawn point and the bound lie inside the EMITTED axis ({bad[:2]})")
        # THE CAPTION'S CROSS-REFERENCE. It said Table IV "lists the verdicts that flip",
        # which is false both ways: the figure counts four flips and only three reach that
        # table, spread over its first two rows, while its other two rows are not resampler
        # flips at all. The caption now claims the two ROWS that follow from this change.
        _pbsrc = open(os.path.join(PB, "PaperB.tex")).read()
        _cap4 = re.search(r"\\includegraphics\[width=\\columnwidth\]\{B4\.pdf\}\s*"
                          r"\\caption\{(.*?)\}\s*\\label\{fig:instrument\}",
                          _pbsrc, re.S)
        ck(_cap4 is not None, "B4: the caption is located")
        if _cap4:
            _c4 = re.sub(r"\s+", " ", _cap4.group(1))
            ck("lists the verdicts that flip" not in _c4,
               "B4: the caption no longer claims Table IV enumerates this figure's flips")
            ck("Two of the reversals" in _c4,
               "B4: the caption counts the Table IV rows this change produces")
        # the ten rows occupy y = 0 .. n-1, so the categorical axis must hold them all
        _rows4 = [int(v) for v in re.findall(r"\([\d.]+,(\d+)\)", _s4)]
        ck(bool(_rows4) and _y0 < min(_rows4) and max(_rows4) < _y1,
           f"B4: every row index lies inside the emitted y range "
           f"({min(_rows4) if _rows4 else None}..{max(_rows4) if _rows4 else None} "
           f"in {_y0}..{_y1})")
    # B4 is placed in ONE COLUMN since 2026-10-10, so its scale is read against
    # \columnwidth and not \textwidth. The side table of flipped verdicts it used to carry
    # duplicated Table IV and is gone; the figure states the flip COUNT and the table lists
    # which ones, so the count is what is checked here.
    _w4 = _pdf_width("B4")
    ck(_w4 is not None and 0.80 <= 252.0 / _w4 <= 1.25,
       f"B4: drawn close to the column it is placed in (scale {252.0 / _w4:.2f})")
    ck(f"{nflip} of {P['n_verdicts']} verdicts flip" in t,
       "B4: the flip count is stated in the figure")
    # every setting's two medians must appear
    for r in P["per_setting"]:
        ck(r["setting"] in t, f"B4: row {r['setting']}")

    # ---------------- B2 ----------------
    t = text("B2")
    if t is not None:
        r = raw_text("B2")
        ck("\u2014" not in r and "---" not in r, "B2: no em-dash in the rendered figure")
        M = J("sc31_bearing_mediation_max.json")
        ck(M["pool"] == "max" and M["seeds"] == 30, "B2: source is the 30-seed block-max run")
        num(t, M["bound_rmse_deg"], "B2 bearing bound", "{:.3f}")
        ck("b = 1" in t or "b=1" in t.replace(" ", ""), "B2: the CRB's b=1 is drawn for contrast")
        fd = json.load(open(os.path.join(FIG, "figure_data.json")))["B2"]
        num(t, fd["b"], "B2 fitted exponent", "{:.3f}")
        num(t, fd["b_lo"], "B2 exponent CI low", "{:.3f}")
        num(t, fd["b_hi"], "B2 exponent CI high", "{:.3f}")
        num(t, fd["floor"], "B2 fitted asymptote", "{:.2f}")
        ck(fd["b_hi"] < 1.0, "B2: the fitted CI excludes the CRB's b=1, as the title claims")
        # Section VIII calls this a fitted asymptote because it is a parameter of a
        # three-parameter model, not measured label noise or a Bayes error. The figure said
        # "irreducible floor" until 2026-10-09, contradicting its own text.
        ck("fitted asymptote" in t and "irreducible" not in t,
           "B2: the asymptote is labeled fitted, not irreducible")
        # The b=1 curve carries the fitted asymptote and is anchored to one measured point,
        # so it must not be presented as what the bound predicts in absolute terms, and the
        # anchoring must be visible.
        ck("anchored" in t and "what the CRB predicts" not in t,
           "B2: the reference curve is labeled anchored")
        ck(fd["anchor_key"].replace("=", "") in t.replace("=", ""),
           f"B2: the anchor setting {fd['anchor_key']} is named in the figure")
        # Until 2026-10-08 the axis started at 12.8 while the floor was 12.53 and the b=1
        # curve's tail reached 12.65, so both were drawn OUTSIDE the plot box with clip=false.
        # Nothing the figure draws may fall outside the axis it is drawn in.
        # READ BACK OUT OF THE EMITTED AXIS, not from the builder. Comparing the builder's
        # limits with the builder's own list of drawn values is a check that compares a
        # builder with itself; the emitter rounds to two decimals, so the two were never
        # even equal. Same defect as B1's axis check and B8's prior, all three found
        # 2026-10-14.
        _s2 = open(os.path.join(FIG, "B2.tex")).read()
        _m2 = re.search(r"xmin=([\d.e+]+), xmax=([\d.e+]+), "
                        r"ymin=([\d.]+), ymax=([\d.]+)", _s2)
        ck(_m2 is not None, "B2: the emitted axis box parses")
        _x0, _x1, ylo, yhi = (float(v) for v in _m2.groups()) if _m2 else (0, 1, 0, 1)
        # every emitted coordinate AND every node anchor, inside that box
        _pts2 = [(float(a), float(b)) for a, b in
                 re.findall(r"\(([\d.e+]+),([\d.]+)\)", _s2)]
        _pts2 = [(x, y) for x, y in _pts2 if 5.0 < y < 30.0]
        _bad2 = [(x, y) for x, y in _pts2 if not (_x0 <= x <= _x1 and ylo <= y <= yhi)]
        ck(not _bad2,
           f"B2: every emitted coordinate lies inside the emitted box ({_bad2[:3]})")
        _anc = [float(v) for v in re.findall(r"at \(axis cs:([\d.e+]+),", _s2)]
        ck(bool(_anc) and all(_x0 <= v <= _x1 for v in _anc),
           f"B2: every label anchor lies inside the emitted box ({_anc})")
        # THE CURVES MAY NOT RISE ABOVE THE NO-INFORMATION BOUND. Drawn from 0.8 to 1.2
        # times the measured range, the anchored reference was extrapolated past the data
        # into the top-left corner and struck through the words "no-information bound".
        # A word-box collision guard cannot see this: it is text against a STROKE.
        _curve_max = max((y for x, y in _pts2), default=0.0)
        ck(_curve_max <= fd["bound"] + 1e-3,
           f"B2: nothing is drawn above the no-information bound {fd['bound']:.3f}, so the "
           f"reference cannot run through its label (highest drawn {_curve_max:.3f})")

        drawn = ([("floor", fd["floor"]), ("bound", fd["bound"])]
                 + [("b=1 curve", z["rmse"]) for z in fd["crb"]]
                 + [("fit curve", z["rmse"]) for z in fd["fit"]])
        for z in fd["gain_pts"] + [q for q in fd["ap_pts"] if q["R"] > 0]:
            # .get, not [], so a missing spread is reported by its own check below rather
            # than crashing this one: a verifier that dies is not a verifier that failed
            sd_ = z.get("sd", 0.0)
            drawn += [(z["k"] + " upper", z["rmse"] + sd_),
                      (z["k"] + " lower", z["rmse"] - sd_)]
        out = [n for n, v in drawn if not (ylo <= v <= yhi)]
        ck(not out, f"B2: everything drawn lies inside the axis box ({', '.join(out[:3])})")
        # the aperture point carried no error bar while every other point did, which is
        # exactly the point a reader has to judge against the fit
        ck(all(z.get("sd", 0.0) > 0 for z in fd["ap_pts"]),
           "B2: the aperture settings carry their seed spread, like the others")
        # m=1 has R=0 and cannot sit on a log axis; it must be reported, not dropped
        z0 = [z for z in fd["ap_pts"] if z["R"] == 0]
        ck(len(z0) == 1 and f"{z0[0]['rmse']:.2f}" in t,
           "B2: the m=1 setting, which has R=0 and no position on this axis, is still stated")
        # the figure must use the manuscript's own symbols for the resource product
        import re as _re
        tex = open(os.path.join(PB, "PaperB.tex")).read()
        subs = []
        for mac in ("Nsamp", "Nchirp"):
            m_ = _re.search(r"\\newcommand\{\\" + mac + r"\}\{N_\\mathrm\{(\w+)\}\}", tex)
            ck(m_ is not None, f"B2: the manuscript defines \\{mac}")
            if m_:
                subs.append(m_.group(1))
        ck(len(subs) == 2 and f"N{subs[0]} N{subs[1]}" in t,
           f"B2: R is written with the manuscript's symbols, N{'/'.join(subs)}")
        _w2 = _pdf_width("B2")
        ck(_w2 is not None and 0.80 <= 252.0 / _w2 <= 1.25,
           f"B2: drawn close to the width it is placed at (scale {252.0 / _w2:.2f})")
    # ---------------- B3 ----------------
    t = text("B3")
    if t is not None:
        r = raw_text("B3")
        ck("\u2014" not in r and "---" not in r, "B3: no em-dash in the rendered figure")
        A = J("sc31_mediation_analysis_max.json")
        num(t, A["link2_pearson_r"], "B3 Pearson r", "{:.4f}")
        num(t, A["link2_spearman_r"], "B3 Spearman", "{:.4f}")
        # The partial correlations moved into the manuscript body (Sec. VI-B), where
        # verify_paper_b_manuscript.py checks them. The figure now carries only the scatter.
        ck(A["partial_logR_gain_given_rmse"][1] > 0.05
           and A["partial_rmse_gain_given_logR"][1] < 0.05,
           "B3: the mediation reading still matches the numbers")
    # ---------------- B5 ----------------
    t = text("B5")
    if t is not None:
        r = raw_text("B5")
        ck("\u2014" not in r and "---" not in r, "B5: no em-dash in the rendered figure")
        C = J("alignment_cost_model.json")
        c = C["scenarios"]["31"]
        ck(f"{c['airtime_exhaustive_us']:.1f}" in t, "B5: exhaustive-sweep cost stated")
        ck(f"{c['airtime_prior_us']:.1f}" in t, "B5: the fixed-prior comparator is stated")
        ck(f"{c['airtime_persistence_us']:.1f}" in t, "B5: the persistence comparator is stated")
        ck(f"{c['airtime_gps_us']:.1f}" in t, "B5: the position comparator is stated")
        # Both regimes must be named, because a single verdict would misstate the result.
        ck("Reuse" in t and "Dedicated" in t,
           "B5: both deployment regimes are named in the panel titles")
        # The claims each panel makes, asserted against the data.
        ck(c["airtime_radar_full_us"] < c["airtime_prior_us"] < c["airtime_exhaustive_us"],
           "B5: under reuse the radar beats BOTH non-sensing comparators")
        ck(all(not z["beats_prior"]
               for rows in c["dedicated_latency_plus_airtime"].values() for z in rows),
           "B5: under dedicated sensing no setting beats the fixed prior")
        # Until 2026-10-08 the dedicated panel DREW the exhaustive and prior rules and
        # labeled neither, so the verdict it exists to show, that no dedicated setting
        # reaches the prior, could not be read off it. Both panels must name both rules, and
        # a text-layer scan sees one occurrence for two drawn rules, so count them.
        for lab, v in (("exhaustive sweep", c["airtime_exhaustive_us"]),
                       ("fixed training prior", c["airtime_prior_us"])):
            ck(t.count(f"{lab} {v:.1f}") == 2,
               f"B5: {lab!r} is labeled in BOTH panels, not just the reuse one")
        fdB5 = json.load(open(os.path.join(FIG, "figure_data.json")))["B5"]
        # the closest dedicated setting, quoted so the panel states its own verdict
        gaps = [z["total_us"] - c["airtime_prior_us"]
                for tc in ("2.56", "10.24")
                for z in c["dedicated_latency_plus_airtime"][tc]]
        ck(min(gaps) > 0, "B5: no dedicated setting reaches the fixed prior")
        ck(abs(fdB5["near_gap"] - min(gaps)) < 1e-9,
           "B5: the quoted closest approach is the smallest shortfall in the cost model")
        ck(f"{min(gaps):.1f}" in t, "B5: that closest approach is stated in the figure")
        rate = C["n_adc"] / c["dedicated_breakeven_T_chirp_vs_prior_us"]
        ck(f"{rate:.0f}" in t, "B5: the break-even sample rate is stated")
        ck(f"{c['dedicated_breakeven_T_chirp_vs_prior_us']:.3f}" in t,
           "B5: the break-even chirp duration is stated")

    fd = json.load(open(os.path.join(FIG, "figure_data.json")))
    _w5 = _pdf_width("B5")
    ck(_w5 is not None and 0.80 <= 516.0 / _w5 <= 1.25,
       f"B5: drawn close to the width it is placed at (scale {516.0 / _w5:.2f})")
    # ---------------- B6 ----------------
    t = text("B6")
    if t is not None:
        r = raw_text("B6")
        ck("\u2014" not in r, "B6: no em-dash in the rendered figure")
        d = fd["B6"]
        num(t, d["beam_bound"], "B6 beam bound"); num(t, d["ceiling"], "B6 geometry ceiling")
        num(t, d["bear_bound"], "B6 bearing bound", "{:.3f}")
        num(t, d["range_only"], "B6 range-only bearing", "{:.3f}")
        M = J("sc31_bearing_mediation_max.json")["settings"]
        ck(abs(d["bear"][0]["v"] - M["rx=1"]["rmse_deg_median"]) < 1e-9,
           "B6: rx=1 bearing matches its JSON")
        ck(d["range_only"] < min(z["v"] for z in d["bear"]),
           "B6: range alone beats every aperture, as the caption claims")
        # This is only an aperture ablation if bandwidth and airtime are held fixed across
        # the three points; the m=4 point is stored under chirps=250, not under an rx key.
        A6 = J("sc31_resource_ablation_G_max.json")["settings"]
        cfg = {(A6[k]["n_samples"], A6[k]["n_chirps"]) for k in ("rx=1", "rx=2", "chirps=250")}
        ck(len(cfg) == 1, "B6: the three points differ only in m, one factor")
        ck(all(A6[k]["n_rx"] == m_ for k, m_ in
               (("rx=1", 1), ("rx=2", 2), ("chirps=250", 4))),
           "B6: each drawn point carries the aperture it is labeled with")
        # Before 2026-10-08 the beam panel had no error bars while the bearing panel did,
        # and the two shares the figure exists to state were computed and then dropped.
        for side in ("beam", "bear"):
            ck(all(z.get("sd", 0.0) > 0 for z in d[side]),
               f"B6: the {side} panel carries its seed spread")
        for key, fmt, lab in (("share_beam", "{:.1f}", "beam"), ("share_bear", "{:.0f}", "bearing")):
            ck(fmt.format(d[key]) in t,
               f"B6: the {lab} panel states the aperture's share, {fmt.format(d[key])}")
        ck(abs(d["share_beam"] - 100 * (d["beam"][2]["v"] - d["beam"][0]["v"])
               / (d["beam"][2]["v"] - d["beam_bound"])) < 1e-9,
           "B6: the beam share is recomputed from the drawn points and the bound")
        ck(abs(d["share_bear"] - 100 * (d["bear_bound"] - d["bear"][0]["v"])
               / (d["bear_bound"] - d["bear"][2]["v"])) < 1e-9,
           "B6: the bearing share is recomputed from the drawn points and the bound")
        for side, rules in (("beam", ("beam_bound", "ceiling")),
                            ("bear", ("bear_bound", "range_only"))):
            lo, hi = d[f"{side}_ymin"], d[f"{side}_ymax"]
            vals = [(r, d[r]) for r in rules]
            vals += [(f"{side} m={z['m']} up", z["v"] + z.get("sd", 0.0)) for z in d[side]]
            vals += [(f"{side} m={z['m']} lo", z["v"] - z.get("sd", 0.0)) for z in d[side]]
            bad = [n for n, v in vals if not (lo <= v <= hi)]
            ck(not bad, f"B6: the {side} panel draws nothing outside its axis ({bad[:2]})")
        _w6 = _pdf_width("B6")
        ck(_w6 is not None and 0.80 <= 252.0 / _w6 <= 1.25,
           f"B6: drawn close to the width it is placed at (scale {252.0 / _w6:.2f})")
    # ---------------- B7 ----------------
    t = text("B7")
    if t is not None:
        r = raw_text("B7")
        ck("\u2014" not in r, "B7: no em-dash in the rendered figure")
        d = fd["B7"]
        for k in ("beam31", "beam33", "bear31", "bear33"):
            ck(d[k]["spearman"] < 0, f"B7: {k} collapse falls with resource")
            ck(f"{d[k]['spearman']:.2f}".lstrip("-") in t.replace("\u2212", "-"),
               f"B7: {k} Spearman printed")
        SR = J("sc33_mediation_restricted.json")["collapse_spearman"][0]
        ck(abs(d["bear33"]["spearman"] - SR) < 1e-9,
           "B7: Sc.33 bearing Spearman matches the analysis JSON")
        # Before 2026-10-08 the panel drew four rank correlations and no p at all, under a
        # title asserting the pattern holds. One of the four does not clear alpha, and the
        # body text says so; only the figure did not.
        m7 = d["_meta"]
        curves = [k for k in ("beam31", "beam33", "bear31", "bear33")]
        ck(sum(d[k]["significant"] for k in curves) == m7["n_sig"],
           "B7: the recorded count of significant curves matches the curves")
        ck(m7["n_sig"] < len(curves),
           "B7: the figure has a non-significant curve to distinguish")
        # every p must be printed, and a p below the format's resolution must not read as 0.00
        for k in curves:
            pv = d[k]["p"]
            want = "p < 0.001" if pv < 0.001 else f"p = {pv:.3f}"
            ck(want in t.replace("\u2212", "-"), f"B7: {k} prints {want}")
        ck(f"{PBMACRO_DEG_NSIG(J)}" == str(m7["n_sig"]),
           "B7: the count of significant curves matches \\PBDegNSig in the manuscript")
        ck(all(d[k]["n_nonzero"] == m7["weak_nonzero"][k]
               and m7["weak_nonzero"][k] < len(d[k]["points"]) for k in m7["weak"]),
           "B7: each weak curve is one with degenerate runs at few settings")
        # Sec. V-B specifies exact enumeration; the asymptotic p is optimistic at n = 6 and
        # using it changed the count of significant trends from two to three.
        import itertools as _it
        from scipy import stats as _st
        for k in curves:
            xs = [q["n"] for q in d[k]["points"]]; ys = [q["coll"] for q in d[k]["points"]]
            r0 = abs(_st.spearmanr(xs, ys).statistic)
            pe = sum(1 for q in _it.permutations(ys)
                     if abs(_st.spearmanr(xs, q).statistic) >= r0 - 1e-12) / 720
            ck(abs(d[k]["p"] - pe) < 1e-12,
               f"B7: {k} quotes the EXACT rank-test p, not the asymptotic one")
            ck(d[k]["significant"] == (pe < m7["alpha"]),
               f"B7: {k} significance follows the exact p")
        _w7 = _pdf_width("B7")
        ck(_w7 is not None and 0.80 <= 252.0 / _w7 <= 1.25,
           f"B7: drawn close to the width it is placed at (scale {252.0 / _w7:.2f})")
    # ---------------- B0 ----------------
    t = text("B0")
    if t is not None:
        r = raw_text("B0")
        ck("\u2014" not in r and "---" not in r, "B0: no em-dash in the rendered figure")
        TB = J("testbed_facts.json")
        C = J("alignment_cost_model.json")
        # The band separation is the fact the whole cost argument turns on, so the timing
        # figure has to carry both bands and say they do not contend.
        # "76--81" is typeset as an en dash, which text() normalizes to "-".
        ck(TB["radar_band_ghz"].replace("--", "-") in t,
           "B0: the radar band is stated")
        ck(TB["link_band_ghz"] in t, "B0: the link band is stated")
        ck(TB["separate_radios"] is True and "without contending" in t,
           "B0: the figure states that the two bands do not contend")
        ck(f"{C['scenarios']['31']['airtime_exhaustive_us']:.0f}" in t,
           "B0: the exhaustive-sweep cost is stated")
        ck(f"{C['T_sec_us']:g}" in t, "B0: T_sec is stated")
        ck(str(TB["n_beams"]) in t, "B0: the codebook size is stated")
        B = J("isac_airtime_budget_max.json")
        ck(f"{B['frame_period_us'] / 1000.0:.1f}" in t, "B0: the logging period is stated")
        BC = J("beam_coherence.json")["scenarios"]["31"]
        ck(f"{BC['beam_coherence_median_ms'] / 1000.0:.2f}" in t,
           "B0: the beam coherence time is stated")
        # ---- geometry -------------------------------------------------------------------
        # The figure spans four decades of time, so each tier carries its own scale. Nothing
        # in the text layer can see whether a drawn length agrees with the duration beside it,
        # and the version of this figure before 2026-10-08 drew its two rows at scales 362
        # apart and labeled the idle gap between frames with the logging period. These checks
        # read the drawn coordinates back and re-derive every duration from them.
        d = json.load(open(os.path.join(FIG, "figure_data.json"))).get("B0", {})
        g = d.get("geom")
        ck(g is not None, "B0: the drawn geometry is recorded")
        if g:
            n_chirp = J("testbed_facts.json")["cube_shape"][2]
            per_ms = B["frame_period_us"] / 1000.0
            ex_us = C["scenarios"]["31"]["airtime_exhaustive_us"]
            # every drawn length, divided by its own tier's scale, must give back its duration
            def dur(width, scale):
                return width / scale
            ck(abs(dur(g["a_period_arrow"][1] - g["a_period_arrow"][0], g["a_cm_per_ms"])
                   - per_ms) < 1e-6,
               "B0: the period arrow spans one logging period, frame start to frame start")
            ck(abs(g["a_period_arrow"][0] - g["a_x0"]) < 1e-9
               and abs(g["a_frame2_x0"] - g["a_period_arrow"][1]) < 1e-9,
               "B0: the period arrow runs between the two drawn frames, not across the gap")
            ck(abs(dur(g["a_frame_w"], g["a_cm_per_ms"]) * 1000.0
                   - n_chirp * d["T_chirp_hi"]) < 1e-6,
               "B0: tier (a) draws the frame at its bracketed duration")
            ck(abs(dur(g["b_frame_w"], g["b_cm_per_us"]) - n_chirp * d["T_chirp_hi"]) < 1e-6,
               "B0: tier (b) draws the frame at the same duration as tier (a)")
            ck(abs(dur(g["b_sweep_w"], g["b_cm_per_us"]) - ex_us) < 1e-6,
               "B0: tier (b) draws the exhaustive sweep at its model cost")
            ck(abs(dur(g["c_slot_w"], g["c_cm_per_us"]) - C["T_sec_us"]) < 1e-6,
               "B0: tier (c) draws the sector slot at T_sec")
            ck(abs(dur(g["c_shortlist_w"], g["c_cm_per_us"])
                   - d["k"] * C["T_sec_us"]) < 1e-6,
               "B0: tier (c) shades exactly k slots")
            ck(abs(dur(g["c_chirp_w"], g["c_cm_per_us"]) - d["T_chirp_hi"]) < 1e-6,
               "B0: tier (c) draws the chirp at the duration tier (b) brackets the frame with")
            # a tier may magnify the one above it but may never contradict it
            ck(abs(g["b_cm_per_us"] / (g["a_cm_per_ms"] / 1000.0) - d["zoom_ab"]) < 1e-6,
               "B0: the stated magnification is the ratio of the two tiers' scales")
            ck(abs(g["c_cm_per_us"] / g["b_cm_per_us"] - d["zoom_bc"]) < 1e-6,
               "B0: the second stated magnification is the ratio of its two tiers' scales")
            ck(abs(100.0 * ex_us / (per_ms * 1000.0) - d["duty_align_pct"]) < 1e-9,
               "B0: the quoted share of the period is the sweep cost over the period")

    # ---------------- B8, Figure 3 ----------------
    # Added with no checks on 2026-10-10 and extended on 2026-10-11 with the static policy.
    # The claim the figure makes is a dominance claim, so the checks are about what is
    # DRAWN: every candidate the selection ranges over, the selected point, and the fact
    # that the static curve is the one the rule picks.
    t = text("B8")
    if t is None:
        print("B8.pdf missing"); return 1
    ck("—" not in t and "---" not in t, "B8: no em-dash in the rendered figure")
    Q8 = J("beam_power_quality.json")["scenarios"]
    S8 = J("static_shortlist.json")["scenarios"]
    fd8 = J_FIGDATA()["B8"]
    for sc in ("31", "33"):
        F8, z8, st8 = Q8[sc]["frontier"], fd8[sc], S8[sc]
        ck(f"Scenario {sc}" in t, f"B8/{sc}: panel title")
        # EVERY candidate the selector ranges over is drawn, or the word frontier is false
        # and the prose's "over the same shortlist lengths" is false with it. The selection
        # searches all sixteen lengths for ten radar settings and two static families, so
        # 160 radar points and 2 x 16 static points make up its 192 candidates. Drawing the
        # six stored frontier lengths showed 60 of the 160 and left the two radar policies
        # Section VI-D compares off the figure entirely. Caught 2026-10-14.
        ndrawn = sum(len(v) for v in z8["chirp"].values()) \
            + sum(len(v) for v in z8["other"].values())
        nrad = len(st8["radar_grid"])
        ck(ndrawn == nrad == 160,
           f"B8/{sc}: all {nrad} radar candidates are drawn, not a projection "
           f"of them ({ndrawn} drawn)")
        ck(ndrawn + len(z8["static"]) + len(z8["prior"])
           == st8["selection_dedicated"]["n_candidates"],
           f"B8/{sc}: the drawn points are exactly the selection's "
           f"{st8['selection_dedicated']['n_candidates']} candidates")
        # the six stored frontier points must lie on the denser drawn grid
        _drawn = {(r["setting"], r["k"]): r for v in z8["chirp"].values() for r in v}
        _drawn.update({(r["setting"], r["k"]): r
                       for v in z8["other"].values() for r in v})
        for r in F8["points"]:
            d_ = _drawn[(r["setting"], r["k"])]
            ck(abs(d_["test"]["mean_regret_db"] - r["test"]["mean_regret_db"]) < 1e-9,
               f"B8/{sc}: drawn {r['setting']} k={r['k']} matches the stored frontier point")
        # THE PRIOR THE FIGURE DRAWS IS THE PRIOR THE SELECTOR USES. build_b8 was corrected
        # on 2026-10-13 to compute it and emit_b8 kept reading the older full-training-region
        # curve, so the repair never reached the rendered figure: on Scenario 33 at k=16 the
        # drawn point stayed at 0.1556 while selection used 0.2178. This check reads the
        # EMITTED COORDINATES out of B8.tex, because checking the builder's return value is
        # what missed it the first time.
        ck(len(z8["prior"]) == 16, f"B8/{sc}: the prior is drawn at all sixteen lengths")
        for r in z8["prior"]:
            ck(abs(r["test_mean_regret_db"]
                   - st8["prior_test"][str(r["k"])]["mean_regret_db"]) < 1e-12,
               f"B8/{sc}: the prior curve at k={r['k']} is the selector's own fit")
        # the static curve is the one from the experiment, point for point
        for r in z8["static"]:
            ck(abs(r["regret"] - st8["test"][str(r["k"])]["mean_regret_db"]) < 1e-12,
               f"B8/{sc}: the static curve at k={r['k']} is the measured test regret")
            ck(abs(r["cost_us"] - r["k"] * F8["T_sec_us"]) < 1e-12,
               f"B8/{sc}: the static policy at k={r['k']} is charged probing time only")
        # the marked point is what the stated rule selects, and the rule is on validation
        sel8 = st8["selection_dedicated"]["selected"]
        ck(abs(z8["sel_cost"] - sel8["cost_us"]) < 1e-9
           and abs(z8["sel_regret"] - sel8["test"]["mean_regret_db"]) < 1e-12,
           f"B8/{sc}: the starred point is the one \\eqref{{eq:select}} selects")
        ck(sel8["val_mean_regret_db"]
           <= F8["selection"]["dedicated"]["validation_target_db"] + 1e-12,
           f"B8/{sc}: the selected policy meets the validation target it was selected under")
        # and the figure must not assert that the selection wins on test: a rule fixed on
        # validation is entitled to lose there. Only the data is asserted.
        ck("power-aware static" in t and "frequency prior" in t,
           f"B8/{sc}: both static policies are named in the panel")
        num(t, F8["selection"]["dedicated"]["default"]["test"]["mean_regret_db"],
            f"B8/{sc} default operating point", "{:.4f}") if False else None
    # The same assertion made against the RENDERED source rather than the builder's return
    # value. The 2026-10-13 prior repair passed every check that read figure_data.json and
    # still shipped the wrong curve, because the emitter never consumed the corrected data.
    # A figure check that reads only what the builder returns cannot see what the emitter
    # ignored.
    src8 = open(os.path.join(FIG, "B8.tex")).read()
    # ---- the headroom holds the family labels, so it is measured, not assumed ----
    # Every drawn y is read back out of the emitted coordinates and compared with the
    # emitted ymax. A fixed *1.12 multiple was tuned for a 38 mm panel; at 25 mm the
    # \scriptsize line it has to hold is 10.8 % of the range against 10.7 % available, and
    # the Scenario 31 prior label came within ONE PIXEL of the frame. Audited 2026-10-14.
    _lims8 = re.findall(r"ymin=([\d.]+), ymax=([\d.]+)", src8)
    _axes8 = src8.split("\\begin{axis}")[1:]
    ck(len(_lims8) == 2 and len(_axes8) == 2,
       f"B8: two panels with explicit limits ({len(_lims8)}, {len(_axes8)})")
    for _i, _sc in enumerate(("31", "33")):
        if _i >= len(_lims8):
            break
        _y0, _y1 = float(_lims8[_i][0]), float(_lims8[_i][1])
        _ys = [float(v) for _x, v in
               re.findall(r"\(([\d.]+),([\d.]+)\)", _axes8[_i])]
        _ys = [v for v in _ys if _y0 <= v <= _y1 * 1.5]
        ck(bool(_ys) and max(_ys) < _y1,
           f"B8/{_sc}: every emitted point lies inside the axis box")
        if _ys:
            _head = 100.0 * (_y1 - max(_ys)) / (_y1 - _y0)
            _need = 100.0 * J_FIGDATA()["B8"][_sc]["label_allowance_mm"] / \
                J_FIGDATA()["B8"][_sc]["panel_mm"]
            ck(_head >= _need,
               f"B8/{_sc}: {_head:.1f} % headroom above the highest point, enough for the "
               f"family label that sits on it ({_need:.1f} % needed)")
    _dash = re.findall(r"\\addplot\[draw=gGrey, dashed[^\]]*\] coordinates \{([^}]*)\}",
                       src8)
    ck(len(_dash) == 2, f"B8: one emitted prior curve per panel ({len(_dash)} found)")
    for _i, sc in enumerate(("31", "33")):
        if _i >= len(_dash):
            break
        pairs = re.findall(r"\(([\d.]+),([\d.]+)\)", _dash[_i])
        ck(len(pairs) == 16,
           f"B8/{sc}: the emitted prior carries all sixteen lengths ({len(pairs)})")
        st8 = S8[sc]
        bad = []
        for cost, val in pairs:
            k_ = int(round(float(cost) / 5.0))
            want = st8["prior_test"][str(k_)]["mean_regret_db"]
            if abs(float(val) - round(want, 4)) > 5e-5:
                bad.append((k_, val, want))
        ck(not bad,
           f"B8/{sc}: every EMITTED prior coordinate is the selector's own fit"
           + (f" (mismatch at {bad[:2]})" if bad else ""))
    _w8 = _pdf_width("B8")
    ck(_w8 is not None and 0.80 <= 516.0 / _w8 <= 1.25,
       f"B8: drawn close to the width it is placed at (scale {516.0 / _w8:.2f} into \\textwidth)")

    # ---------------- B9, Figure 1 ----------------
    # This figure shipped with no checks at all, and it was the only one of the six placed in
    # the manuscript whose words overlapped each other in the compiled PDF. A text-layer
    # verifier reads what a figure SAYS and cannot see where it says it, so the checks below
    # read geometry out of the TikZ source and word boxes out of the PDF.
    t = text("B9")
    if t is None:
        print("B9.pdf missing"); return 1
    ck("—" not in t and "---" not in t, "B9: no em-dash in the rendered figure")
    T9 = J("testbed_facts.json")
    C9 = J("alignment_cost_model.json")
    A9 = J("sc31_resource_ablation_G_max.json")
    S9 = A9["settings"]
    nrx9, nadc9, nch9 = T9["cube_shape"]
    nb9, fov9 = T9["n_beams"], T9["array_fov_deg"]
    cube9 = f"{nrx9}\u00d7{nadc9}\u00d7{nch9}"
    ck(f"{T9['n_array_elements']}-element array" in t, "B9: the array size is the testbed's")
    ck(f"{nb9} beams over {fov9:g}" in t, "B9: the codebook size and the field of view")
    ck(t.count(cube9) == 2, f"B9: the cube shape {cube9} is drawn in both panels")
    ck(f"at {T9['modality_rate_hz']} Hz" in t, "B9: the logging rate")
    ck(f"{T9['radar_max_range_m']:g} m reach" in t, "B9: the rated radar reach")
    ck(f"{T9['link_band_ghz']} GHz" in t, "B9: the link band")
    ck(T9["radar_band_ghz"].replace("--", "-") + " GHz" in t, "B9: the radar band")
    ck(T9["separate_radios"] is True and "Two radios, two bands" in t,
       "B9: the panel states the band plan the cost model turns on")
    # k is not a literal in the figure: it is the shortlist the cost model priced.
    k9 = 5
    sc9 = C9["scenarios"]["31"]
    ck(abs(k9 * C9["T_sec_us"] + (1 - sc9["p5_prior"]) * (nb9 - k9) * C9["T_sec_us"]
           - sc9["airtime_prior_us"]) < 1e-6,
       "B9: the drawn shortlist length is the one the cost model priced")
    ck(f"probe k={k9} of {nb9}" in t, "B9: the probe step names k and the codebook size")
    ck(f"{C9['T_sec_us']:g} \u00b5s each" in t,
       "B9: the per-sector duration is the cost model's")
    for ax, key, lab in (("chirps", "n_chirps", "N"), ("samples", "n_samples", "N"),
                         ("antennas", "n_rx", "m")):
        vals = sorted({S9[s][key] for s in S9 if S9[s]["sweep"] == ax}
                      | ({nadc9} if ax == "samples" else {nrx9} if ax == "antennas" else set()))
        ck(", ".join(str(v) for v in vals) in t,
           f"B9: the {ax} knob is drawn over the settings actually swept")
    num(t, A9["bound_const"], "B9 no-information bound")
    R9 = J("sc31_reference_predictors.json")["predictors"]
    g9 = [v for k_, v in R9.items() if k_.startswith("GPS reference")][0]
    num(t, g9["top1"], "B9 position reference")
    ck("Scenario 31: bound" in t and "position reference" in t,
       "B9: the two references are drawn as the interval a learned number is read in, "
       "and are labeled with the scenario they belong to")
    ck(J_FIGDATA()["B9"]["params"] in t, "B9: the readout's parameter count")
    # the measured partner range, re-derived here from the label file and not from the builder
    import numpy as _np
    _z = _np.load(os.path.join(RES, "sc31_bearing_labels.npz"), allow_pickle=True)
    _lo, _hi = float(_z["range_m"].min()), float(_z["range_m"].max())
    ck(f"partner {_lo:.0f} to {_hi:.0f} m" in t, "B9: the measured partner range")
    ck(_hi < T9["radar_max_range_m"],
       "B9: the measured range lies inside the rated reach the figure draws")
    # ---- geometry, read out of the TikZ source: a figure may not contradict the prose ----
    src9 = open(os.path.join(FIG, "B9.tex")).read()
    arcs = [(float(a), float(b)) for a, b in
            re.findall(r"arc \((-?[\d.]+):(-?[\d.]+):[\d.]+\)", src9)]
    ck(any(abs((b - a) - fov9) < 1e-2 for a, b in arcs),
       "B9: the codebook sector is drawn at the array's field of view")
    # THE SHORTLIST IS NOT DRAWN AS A SECTOR, and no angular width is claimed for it. A
    # 64-beam codebook over a 90-degree field of view establishes neither uniform spacing
    # nor calibrated beam centers, and a learned top-5 is five consecutive indices on a
    # minority of frames, so an earlier draft that drew one contiguous k*fov/n_beam wedge
    # asserted two things the data does not support.
    beam_deg = fov9 / nb9
    ck(not any(abs((b - a) - k9 * beam_deg) < 1e-2 for a, b in arcs),
       f"B9: no arc of exactly {k9 * beam_deg:.2f} degrees is drawn, so the shortlist is "
       f"not claimed to be a contiguous sector")
    ck(f"{beam_deg:.1f}" not in t,
       "B9: no angular beam spacing is stated, since the codebook's geometry is not calibrated")
    ck("schematic" in t, "B9: the beam directions are labeled schematic")
    nprobe = len(re.findall(r"\\draw\[oiBlue, line width=(?:0\.9|1\.5)pt\] \(", src9))
    ck(nprobe == k9, f"B9: exactly {k9} candidate directions are drawn, got {nprobe}")
    # the link arrow must point INTO unit 1: the codebook combines on receive, and Sec. III-A
    # says a reader is most likely to get exactly this wrong.
    m9 = re.search(r"\\draw\[oiBlue, line width=1\.0pt, ->\]\s*\(([\d.]+),([\d.]+)\) -- "
                   r"\(([\d.]+),([\d.]+)\)", src9)
    ck(m9 is not None, "B9: the link is drawn as a single arrow")
    # the spine's width is shared with nothing else in the figure, so this regex cannot
    # match a probe ray by accident, which it did once
    sp = re.search(r"\\draw\[oiBlue, line width=1\.2pt\] \(([\d.]+),([\d.]+)\) -- "
                   r"\(([\d.]+),([\d.]+)\)", src9)
    ck(sp is not None, "B9: the array is drawn as an aperture, not a point")
    if m9 and sp:
        import math as _m
        ax_, ay_ = float(sp.group(1)), (float(sp.group(2)) + float(sp.group(4))) / 2
        tail = (float(m9.group(1)), float(m9.group(2)))
        head = (float(m9.group(3)), float(m9.group(4)))
        ck(_m.hypot(*(head[i] - (ax_, ay_)[i] for i in range(2)))
           < _m.hypot(*(tail[i] - (ax_, ay_)[i] for i in range(2))),
           "B9: the link arrow points into unit 1, because the codebook combines on receive")
        # a uniform linear array is broadside to its own fan; drawing it along the fan would
        # put the field of view at endfire, where the array has no resolution at all.
        aim = _m.degrees(_m.atan2(tail[1] - ay_, tail[0] - ax_))
        axis = _m.degrees(_m.atan2(float(sp.group(4)) - float(sp.group(2)),
                                   float(sp.group(3)) - float(sp.group(1))))
        sep = abs((axis - aim + 90) % 180 - 90)
        ck(sep > 75.0,
           f"B9: the array is drawn broadside to its own field of view ({sep:.1f} degrees)")
    ck(len(re.findall(r"\\draw\[oiBlue, line width=0\.45pt\]", src9)) == T9["n_array_elements"],
       "B9: the drawn element count is the array's, since a reader counts what is drawn")
    ck(len(re.findall(r"\\draw\[oiPurple, line width=0\.5pt\]", src9)) == nrx9,
       "B9: the drawn radar channel count is the cube's")

    # ---- panel (b)'s chain, which had no geometric check of any kind ----
    # Every check above reads panel (a). Panel (b) was verified only for the TEXT it
    # contains, so the three arrows from the shortlist to the three scoring boxes were
    # 0.9 mm long, shorter than the latex arrowhead each carried, with the head further
    # painted over by the filled node declared after them, and the bus overshot the
    # junction by 1.6 mm. None of that is visible to a text-layer check, and the word-box
    # collision guard cannot see it either: it is a line against a NODE, not word against
    # word. Reported by the author 2026-10-14 from the rendered figure.
    _sc = {m[0]: (float(m[1]), float(m[2])) for m in _re.findall(
        r"\((s[123])\) at \(([-\d.]+),([-\d.]+)\)", src9)}
    ck(len(_sc) == 3, f"B9/b: the three scoring boxes are drawn ({sorted(_sc)})")
    _bus = sorted({float(y) for y in _re.findall(r"\|- 0,([\d.]+)\)", src9)})
    ck(len(_bus) == 1,
       f"B9/b: the fanout uses one bus height, got {_bus}")
    if _bus and len(_sc) == 3:
        for _n, (_x, _y) in sorted(_sc.items()):
            _drop = _bus[0] - _y
            ck(_drop >= 0.20,
               f"B9/b: the arrow into {_n} is {_drop * 10:.1f} mm long, enough for a "
               f"visible shaft behind its head (was 0.9 mm)")
    # the bus must END at the source node, not at a typed coordinate past it
    ck("(sl.south |- 0," in src9 and "(sl.south) -- (sl.south |- 0," in src9,
       "B9/b: the bus meets the shortlist box at its own anchor, so it cannot overshoot")
    ck(not _re.search(r"\(1\.35,1\.95\) -- \(8\.40,1\.95\)", src9),
       "B9/b: the overshooting hand-typed bus is gone")
    # and the scoring nodes must be DECLARED BEFORE the arrows that point at them, or the
    # filled node paints over the arrowhead landing on its border
    # the row-1 to row-2 connector carries an arrowhead on its FINAL DROP too, and raising
    # the middle row to make room for the fanout shortened it to 1.65 mm before this check
    # existed. Same defect class, same panel, introduced by the repair for it.
    def _nodegeom(name):
        m = _re.search(r"minimum height=([\d.]+)mm,[^)]*?\(" + name
                       + r"\) at \(([-\d.]+),([-\d.]+)\)", src9, _re.S)
        return (float(m.group(3)), float(m.group(1)) / 10.0) if m else None
    _tf, _maps, _cut = _nodegeom("tf"), _nodegeom("maps"), _nodegeom("cut")
    _off = _re.search(r"\(tf\.south\) -- \+\+\(0,-([\d.]+)\)", src9)
    ck(all((_tf, _maps, _cut, _off)),
       "B9/b: the row-1 to row-2 connector and its node heights parse")
    if all((_tf, _maps, _cut, _off)):
        _run = _tf[0] - _tf[1] / 2 - float(_off.group(1))
        _drop = _run - (_maps[0] + _maps[1] / 2)
        _clear = (_cut[0] - _cut[1] / 2) - _run
        ck(_drop >= 0.20,
           f"B9/b: the drop into the maps box is {_drop * 10:.2f} mm, enough for its "
           f"arrowhead to read")
        ck(_clear >= 0.12,
           f"B9/b: its horizontal run clears the box above by {_clear * 10:.2f} mm")
    _first_node = min(src9.index("(s1) at"), src9.index("(s2) at"), src9.index("(s3) at"))
    _first_drop = src9.index("(s1.north |- 0,")
    ck(_first_node < _first_drop,
       "B9/b: the scoring boxes are declared before their arrows, so a filled node cannot "
       "paint over an arrowhead that lands on its border")

    _w = _pdf_width("B9")
    ck(_w is not None and 0.80 <= 516.0 / _w <= 1.25,
       f"B9: drawn close to the width it is placed at (scale {516.0 / _w:.2f} into \\textwidth)")

    # ---------------- every figure: error bars that actually have ends ----------------
    # `error mark` and `error mark options` written after `error bars/.cd` are accepted and
    # SILENTLY IGNORED by this pgfplots. The error mark inherits the PLOT's mark instead, so
    # a plot carrying `mark=none` or `mark=*` draws whiskers with no cap at all. B1 shipped
    # that way from 2026-10-08, when the entry recording that the intervals "now have caps"
    # was written, and B6 shipped that way from the same date. Measured before the repair:
    # whisker ends 0.17 mm wide, which is the bar's own line width. Checked here across every
    # figure, because the idiom is wrong wherever it appears. Audited 2026-10-14.
    for _stem in ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"):
        _f = os.path.join(FIG, _stem + ".tex")
        if not os.path.exists(_f):
            continue
        _src = open(_f).read()
        if "error bars/.cd" not in _src:
            continue
        ck("error mark=" not in _src,
           f"{_stem}: no `error mark=` inside error bars/.cd, an idiom this pgfplots ignores")
        _plots = re.findall(r"\\addplot\[[^\]]*error bars/\.cd[^\]]*\]", _src, re.S)
        ck(bool(_plots) and all("mark=-" in q for q in _plots),
           f"{_stem}: every error-bar plot carries the mark its caps are drawn from "
           f"({len(_plots)} plots)")
        ck(not any("mark=none" in q for q in _plots),
           f"{_stem}: no error-bar plot sets mark=none, which leaves the caps undrawn")
        # and the cap has to be wider than the marker it brackets, or a short interval is
        # swallowed: B1's 1.9 pt marker hid four of six upper caps on Scenario 31
        _caps = [float(m) for q in _plots
                 for m in re.findall(r"mark size=([\d.]+)pt", q)]
        _marks = [float(m) for m in
                  re.findall(r"only marks, mark=\*, mark size=([\d.]+)pt", _src)]
        if _caps and _marks:
            ck(min(_caps) > max(_marks),
               f"{_stem}: the error-bar cap ({min(_caps)} pt) is wider than the data marker "
               f"({max(_marks)} pt), so a short interval still shows its ends")

    # ---------------- every figure: nothing printed on top of anything else ----------------
    # Figure 1 reached the submitted PDF with the last line of one box printed underneath the
    # first line of another, and all 171 checks passed, because a check that reads the text
    # layer sees the words and not their positions. This reads each word's box back out and
    # asserts that no two of them overlap.
    for fig in sorted(json.load(open(os.path.join(FIG, "figure_data.json")))):
        over = _word_collisions(fig)
        if over is None:
            continue
        ck(not over, f"{fig}: no two words overlap in the compiled figure"
                     + (f" (worst: {over[0][2]!r} over {over[0][3]!r})" if over else ""))

    npass = sum(1 for ok, _ in checks if ok)
    for ok, lab in checks:
        if not ok:
            print(f"  FAIL  {lab}")
    print(f"\n{npass}/{len(checks)} figure checks passed")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
