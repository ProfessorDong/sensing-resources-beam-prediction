#!/usr/bin/env python
"""Emit paper_b/numbers.tex: every load-bearing number in the manuscript, as a LaTeX macro.

The manuscript types no numbers. It uses \\PBxxx macros defined here, each read from a results
JSON, so a changed result propagates to the text instead of silently disagreeing with it. This
is the same rule the figures follow.

Usage: python experiments/make_paper_b_macros.py
"""
# The two chirp durations Table III prices, in microseconds. Every sample rate the
# manuscript quotes as a comparator is DERIVED from these and the sample count, so the
# paper makes no unsourced claim about what a particular front end provides.
TC_FAST_US, TC_SLOW_US = 2.56, 10.24
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(os.path.dirname(HERE), "paper_b", "numbers.tex")
_c = {}
L = []


def J(n):
    if n not in _c:
        _c[n] = json.load(open(os.path.join(RES, n)))
    return _c[n]


def M(name, val, fmt="{:.4f}"):
    """Define \\PB<name>. Names must be letters only (TeX restriction)."""
    assert name.isalpha(), name
    txt = val if isinstance(val, str) else fmt.format(val)
    L.append(rf"\newcommand{{\PB{name}}}{{{txt}}}")
    return val


A = "sc31_resource_ablation_G_max.json"
D = "sc31_resource_ablation_G.json"
T = "sc33_resource_ablation_G_max.json"
TD = "sc33_resource_ablation_G.json"
MM = "sc31_bearing_mediation_max.json"
MAM = "sc31_mediation_analysis_max.json"
P = "sc31_pooling_comparison.json"
G = "geometry_ceiling.json"
B = "isac_airtime_budget_max.json"
R = "sc33_replication_score_max.json"
U = "utilization_definitional_audit.json"
O = "sc31_route_overlap.json"
RO = "bearing_from_range_only.json"
SR = "sc33_mediation_restricted.json"
WORD = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour",
        125: "OneTwentyFive", 250: "TwoFifty"}

# ---- scenario-level constants -------------------------------------------------
M("BoundOne", J(A)["bound_const"])
M("BoundThree", J(T)["bound_const"])
M("TrainOne", J(A)["n_train"], "{:,}")
M("TestOne", J(A)["n_test"], "{:,}")
M("TrainThree", J(T)["n_train"], "{:,}")
M("TestThree", J(T)["n_test"], "{:,}")
M("Seeds", J(A)["seeds"], "{:d}")
M("Epochs", J(A)["epochs"], "{:d}")
M("MedSeeds", J(MM)["seeds"], "{:d}")

# ---- C1: dose-response --------------------------------------------------------
for tag, f in (("One", A), ("Three", T)):
    S = J(f)["settings"]
    for c in (8, 16, 32, 64, 125, 250):
        M(f"Ch{WORD[c]}{tag}", S[f"chirps={c}"]["top1_median"])
        M(f"Coll{WORD[c]}{tag}", S[f"chirps={c}"]["n_collapsed"], "{:d}")
    for k, nm in (("samples=64", "SampSixtyFour"), ("samples=128", "SampOneTwentyEight"),
                  ("rx=1", "RxOne"), ("rx=2", "RxTwo")):
        M(f"{nm}{tag}", S[k]["top1_median"])
    M(f"TopFiveBest{tag}", S["chirps=250"]["top5_mean"])
M("SlopeOne", J(A)["chirp_trend"]["slope_top1_per_log2_chirp"])
TS = "top_step_125_to_250.json"
for tag, k in (("One", "sc31"), ("Three", "sc33")):
    M(f"TopStep{tag}", J(TS)[k]["delta"])
    M(f"TopStepLo{tag}", J(TS)[k]["lo"])
    M(f"TopStepHi{tag}", J(TS)[k]["hi"])
    M(f"TopStepP{tag}", J(TS)[k]["p"], "{:.3f}")
M("GainBestOne", J(A)["settings"]["chirps=250"]["top1_median"] - J(A)["bound_const"])
M("GainBestThree", J(T)["settings"]["chirps=250"]["top1_median"] - J(T)["bound_const"])
M("GainRxOneOne", J(A)["settings"]["rx=1"]["top1_median"] - J(A)["bound_const"])

# ---- ceiling ------------------------------------------------------------------
for tag, k in (("One", "sc31"), ("Three", "sc33")):
    M(f"Ceil{tag}", J(G)[k]["ceiling_top1"])
    M(f"CeilFrac{tag}", 100 * J(G)[k]["radar_frac_of_ceiling"], "{:.1f}")
    # the winning family differs by scenario, so the descriptor is a macro too
    fam, par = J(G)[k]["best_family"], J(G)[k]["best_param"]
    desc = (rf"$k{{=}}{par}$ nearest neighbors" if fam == "knn"
            else rf"a {par}-bin bearing lookup")
    M(f"CeilDesc{tag}", desc, None)
M("CeilTopFiveOne", J(G)["sc31"]["ceiling_top5"])
M("CeilOneNN", [r for r in J(G)["sc31"]["candidates"]
                if r["family"] == "knn" and r["param"] == 1][0]["top1"])

# ---- R1's stronger baselines: non-learned, radar-driven ----
for tag, k in (("One", "sc31"), ("Three", "sc33")):
    B2 = J(f"{k}_stronger_baselines.json")["baselines"]
    M(f"PeakAz{tag}", B2["peak-az"]["top1"])
    M(f"PeakAzVsBound{tag}", B2["peak-az"]["vs_bound"], "{:+.4f}")
    M(f"Persist{tag}", B2["persistence"]["top1"])
    M(f"PersistTopFive{tag}", B2["persistence"]["top5"])
    M(f"PersistFrac{tag}", 100 * B2["persistence"]["frac_of_ceiling"], "{:.1f}")
    # The peak-to-bearing correlation must come from the variant the paper REPORTS, which is
    # the one with the first range bin excluded. The as-published argmax is a leakage artifact
    # emitting three distinct azimuths on Sc.31, so its correlation describes the artifact.
    _pk = J(f"{k}_reference_predictors.json")["predictors"][
        "radar-peak (first range bin excluded)"]["peak_azimuth_vs_bearing"]
    M(f"PeakCorr{tag}", _pk["pearson"], "{:+.4f}")
    M(f"PeakCorrP{tag}", _pk["pearson_p"], "{:.3f}")
    M(f"PeakAzCount{tag}", _pk["distinct_peak_azimuths"], "{:d}")

# ---- C2: airtime --------------------------------------------------------------
M("Tsec", J(B)["T_sec_us"], "{:g}")
M("NBeam", J(B)["n_beam"], "{:d}")
M("NAdc", J(B)["n_adc_samples"], "{:d}")
M("Exhaustive", J(B)["exhaustive_us"], "{:.0f}")
M("FramePeriod", J(B)["frame_period_us"] / 1000.0, "{:.1f}")
M("BestTotal", J(B)["expected_overhead"]["25.6"]["best_total_us"], "{:.0f}")
M("BestN", J(B)["expected_overhead"]["25.6"]["best_n_chirps"], "{:d}")
be = {r["n_chirps"]: r for r in J(B)["per_setting"]}
M("BreakEvenBest", be[250]["breakeven_adc_msps"], "{:.0f}")
M("BreakEvenEight", be[8]["breakeven_adc_msps"], "{:.1f}")
M("BreakEvenEightOpt", be[8]["breakeven_adc_msps_optimistic"], "{:.1f}")
und = [r["breakeven_adc_msps"] / r["breakeven_adc_msps_optimistic"] for r in J(B)["per_setting"]]
M("FallbackUnderLo", min(und), "{:.1f}")
M("FallbackUnderHi", max(und), "{:.1f}")
_nadc = J("alignment_cost_model.json")["n_adc"]
_adc_slow, _adc_fast = _nadc / TC_SLOW_US, _nadc / TC_FAST_US
assert _adc_slow == int(_adc_slow) and _adc_fast == int(_adc_fast), (_adc_slow, _adc_fast)
M("AdcSlow", _adc_slow, "{:.0f}"); M("AdcFast", _adc_fast, "{:.0f}")
M("BreakEvenRatioLo", be[250]["breakeven_adc_msps"] / 25, "{:.1f}")
M("BreakEvenRatioHi", be[250]["breakeven_adc_msps"] / 10, "{:.1f}")
M("GainEight", be[8]["vs_bound"], "{:+.4f}")

# ---- C3: estimation theory ----------------------------------------------------
S = J(MM)["settings"]
M("BearBound", J(MM)["bound_rmse_deg"], "{:.3f}")
M("BearBoundThree", J("sc33_bearing_mediation_max.json")["bound_rmse_deg"], "{:.3f}")
for c in (8, 250):
    M(f"Bear{WORD[c]}", S[f"chirps={c}"]["rmse_deg_median"], "{:.3f}")
M("BearRxOne", S["rx=1"]["rmse_deg_median"], "{:.3f}")
M("BearRxTwo", S["rx=2"]["rmse_deg_median"], "{:.3f}")
bb, b4, b1 = J(MM)["bound_rmse_deg"], S["chirps=250"]["rmse_deg_median"], S["rx=1"]["rmse_deg_median"]
M("RxOneShare", 100 * (bb - b1) / (bb - b4), "{:.0f}")
M("ApertureDeltaOne", J(A)["settings"]["chirps=250"]["top1_median"]
  - J(A)["settings"]["rx=1"]["top1_median"], "{:+.4f}")
M("ApertureShare", 100 * (J(A)["settings"]["chirps=250"]["top1_median"]
                          - J(A)["settings"]["rx=1"]["top1_median"])
  / (J(A)["settings"]["chirps=250"]["top1_median"] - J(A)["bound_const"]), "{:.1f}")
M("LinkTwoR", J(MAM)["link2_pearson_r"])
def sci(v, sig=1):
    """Format a small p-value as LaTeX math, e.g. 7.9\\times10^{-5}."""
    e = int(np.floor(np.log10(abs(v))))
    # single backslash: in a raw string "\\" is two characters, which LaTeX reads as a
    # line break followed by the literal word "times".
    return rf"{v/10**e:.{sig}f}\times10^{{{e}}}"


M("LinkTwoP", sci(J(MAM)["link2_pearson_p"]), None)
M("LinkTwoRho", J(MAM)["link2_spearman_r"])
M("PartialDirectR", J(MAM)["partial_logR_gain_given_rmse"][0])
M("PartialDirectP", J(MAM)["partial_logR_gain_given_rmse"][1], "{:.3f}")
M("PartialMedR", J(MAM)["partial_rmse_gain_given_logR"][0])
M("PartialMedP", J(MAM)["partial_rmse_gain_given_logR"][1], "{:.3f}")
fd = json.load(open(os.path.join(os.path.dirname(HERE), "paper_b_figures", "figure_data.json")))
M("ExpB", fd["B2"]["b"], "{:.3f}")
M("ExpBLo", fd["B2"]["b_lo"], "{:.3f}")
M("ExpBHi", fd["B2"]["b_hi"], "{:.3f}")
M("Floor", fd["B2"]["floor"], "{:.2f}")
# Degeneracy against airtime: four curves, all negative, but NOT all significant. The weakest
# is the one with the fewest degenerate runs to begin with, so it carries the least signal.
_b7 = fd["B7"]
_rhos = [_b7[k]["spearman"] for k in ("beam31", "beam33", "bear31", "bear33")]
M("DegRhoLo", max(_rhos), "{:+.2f}")
M("DegRhoHi", min(_rhos), "{:+.2f}")
M("DegNSig", sum(1 for k in ("beam31", "beam33", "bear31", "bear33")
                 if _b7[k]["p"] < 0.05), "{:d}")
M("DegWeakP", max(_b7[k]["p"] for k in ("beam31", "beam33", "bear31", "bear33")), "{:.2f}")
_b7keys = ("beam31", "beam33", "bear31", "bear33")
M("DegNWeak", sum(1 for k in _b7keys if _b7[k]["p"] >= 0.05), "{:d}")
# Holm over the four as one confirmatory family
_ps = sorted(_b7[k]["p"] for k in _b7keys)
M("DegNHolm", sum(1 for i, q in enumerate(_ps) if q * (len(_ps) - i) < 0.05), "{:d}")
M("DegWeakNzOne", _b7["beam31"]["n_nonzero"], "{:d}")
M("DegWeakNzTwo", _b7["bear31"]["n_nonzero"], "{:d}")
M("DegNSettings", len(_b7["beam31"]["points"]), "{:d}")
# ---- the network and training configuration the paper describes --------------------
# Derived by instantiating the class the experiment imports, and by parsing the training
# function it calls, so the manuscript cannot drift from the code again. Before 2026-10-08
# the paper said batch normalization, rectified-linear units and a single linear layer; the
# implementation uses GroupNorm, SiLU and two dense layers.
import re as _re
_net_src = open(os.path.join(HERE, "audit_readout_spatial.py")).read()
assert "nn.GroupNorm(8" in _net_src and "nn.SiLU()" in _net_src, "network changed"
import torch as _torch  # noqa: F401
sys.path.insert(0, HERE)
from audit_readout_spatial import ImgReadoutSpatial as _IRS
M("NetParams", sum(q.numel() for q in _IRS(2, 64).parameters() if q.requires_grad), "{:,d}")
_tr_src = open(os.path.join(HERE, "audit_collapse_fix.py")).read()
_wd = _re.search(r"AdamW\(.*?weight_decay=([0-9.e-]+)", _tr_src).group(1)
_bs = _re.search(r"batch_size=(\d+), shuffle=True", _tr_src).group(1)
assert "CosineAnnealingLR" in _tr_src, "scheduler changed"
M("Lr", 3e-4, "{:g}"); M("Wd", float(_wd), "{:g}"); M("Batch", int(_bs), "{:d}")
M("Epochs", J(A)["epochs"], "{:d}")
# The angular DISPLAY grid, parsed from the shipped chain rather than typed. The spatial
# transform is zero-padded from the cube's element count to this many bins, so the bearing
# axis is shown finer than the aperture resolves and Section III-A must not call the two
# the same thing. A chain that changed its transform length would otherwise leave the
# manuscript asserting the old one.
_ab_src = open(os.path.join(HERE, "exp_sc31_resource_ablation.py")).read()
_ang = int(_re.search(r"np\.fft\.fft\(rd,\s*(\d+),\s*axis=0\)", _ab_src).group(1))
_nrx = int(J("testbed_facts.json")["cube_shape"][0])
assert _ang > _nrx, \
    "the angle transform is no longer zero-padded; Section III-A's wording must change"
M("AngGrid", _ang, "{:d}")
M("RxMax", _nrx, "{:d}")
# ---- the persistence eligibility rule as IMPLEMENTED, and why that threshold ---------
_pm = _re.search(r"dt_ms <= (\d+)",
                 open(os.path.join(HERE, "audit_reference_predictors.py")).read()).group(1)
M("PersMaxGap", int(_pm), "{:d}")
M("FramePeriodHi", J(B)["frame_period_p5_p95_us"][1] / 1000.0, "{:.1f}")
# R orders the full-aperture settings only. Across all ten it inverts three times: rx=1 has
# R = 0 yet beats the shortest frame, and 32 chirps is worse than the two-element setting.
_M31 = J("sc31_bearing_mediation_max.json")["settings"]
_full = [k for k in _M31 if _M31[k]["n_rx"] == 4]
_ord = sorted(_M31, key=lambda k: _M31[k]["resource_product_R"])
_inv = sum(1 for i in range(1, len(_ord))
           if _M31[_ord[i]]["rmse_deg_median"] > _M31[_ord[i - 1]]["rmse_deg_median"])
M("NFullAperture", len(_full), "{:d}")
M("NOrderInversions", _inv, "{:d}")
assert _inv > 0, "if R really orders the sweep, the scoping text must be revisited"
# Scoping to the full-aperture settings does NOT rescue a literal ordering claim. Over all
# pairs of the eight, R and the measured error disagree once, and one pair has exactly equal R
# with errors differing by more than a seed standard deviation. The 2026-10-09 audit was right
# that "orders the eight settings" is false as stated; the manuscript now claims a decreasing
# trend over a resource coordinate instead, and these macros keep that wording required.
import itertools as _it
_fp = sorted(((_M31[k]["resource_product_R"], k, _M31[k]["rmse_deg_median"],
               _M31[k]["rmse_deg_sd"]) for k in _full), key=lambda r: r[0])
_finv = [(a, b) for a, b in _it.combinations(_fp, 2) if a[0] < b[0] and a[2] < b[2]]
_fties = [(a, b) for a, b in _it.combinations(_fp, 2) if a[0] == b[0]]
assert _finv and _fties, "the scoped ordering claim would need rewriting"
M("FullInversions", len(_finv), "{:d}")
M("FullInvLo", _finv[0][0][1].replace("=", "${=}$"), "{}")
M("FullInvHi", _finv[0][1][1].replace("=", "${=}$"), "{}")
M("FullInvErrLo", _finv[0][0][2], "{:.2f}")
M("FullInvErrHi", _finv[0][1][2], "{:.2f}")
M("FullTieGap", abs(_fties[0][0][2] - _fties[0][1][2]), "{:.2f}")
M("FullTieSd", max(_fties[0][0][3], _fties[0][1][3]), "{:.2f}")
assert abs(_fties[0][0][2] - _fties[0][1][2]) > max(_fties[0][0][3], _fties[0][1][3])
M("RangeOnlyOne", J(RO)["sc31"]["rmse_from_range_only_deg"], "{:.3f}")
M("RangeOnlyThree", J(RO)["sc33"]["rmse_from_range_only_deg"], "{:.3f}")
M("RangeCorrOne", J(RO)["sc31"]["pearson_range_bearing"], "{:+.4f}")
M("RouteFracOne", 100 * J(O)["frac_within_0p05sd"], "{:.0f}")
M("RouteFracTight", 100 * J(O)["frac_within_0p01sd"], "{:.0f}")

# ---- C4: controls -------------------------------------------------------------
M("PoolNFlip", J(P)["n_verdicts"] - J(P)["n_verdicts_unchanged"], "{:d}")
M("PoolNTotal", J(P)["n_verdicts"], "{:d}")
M("PoolNSet", len(J(P)["per_setting"]), "{:d}")
M("PoolMean", float(np.mean([r["delta"] for r in J(P)["per_setting"]])), "{:+.4f}")
M("PoolMaxP", max(r["p"] for r in J(P)["per_setting"]), "{:.4f}")
iso = {v["comparison"]: v for v in J(P)["verdicts"]}
for nm, key in (("IsoA", "chirps=125 -> samples=128"), ("IsoB", "chirps=64 -> samples=64")):
    v = iso[key]
    M(f"{nm}Dec", v["decimate"]["delta"]); M(f"{nm}DecP", v["decimate"]["p"], "{:.4f}")
    M(f"{nm}Max", v["max"]["delta"]); M(f"{nm}MaxP", v["max"]["p"], "{:.3f}")
sat = iso["chirps=125 -> chirps=250"]
M("SatDec", sat["decimate"]["delta"]); M("SatDecP", sat["decimate"]["p"], "{:.3f}")
M("SatMax", sat["max"]["delta"]); M("SatMaxP", sat["max"]["p"], "{:.4f}")
SD = J(TD)["settings"]
M("ThreeDecOneTwentyFive", SD["chirps=125"]["top1_median"])
M("ThreeDecTwoFifty", SD["chirps=250"]["top1_median"])
M("ThreeDecDrop", SD["chirps=250"]["top1_median"] - SD["chirps=125"]["top1_median"], "{:+.4f}")
M("UtilRatio", J(U)[-1]["ratio"], "{:.3f}")
M("UtilResid", J(U)[-1]["residual_sd"])
M("UtilR", J(U)[-1]["pearson_r"])
M("ScThreeLinkAll", J(SR)["link2_all"][0]); M("ScThreeLinkAllP", J(SR)["link2_all"][1], "{:.3f}")
M("ScThreeLinkOk", J(SR)["link2_working_only"][0])
M("ScThreeLinkOkP", J(SR)["link2_working_only"][1], "{:.2f}")
M("ScThreeNFail", J(SR)["n_instrument_failures"], "{:d}")
M("ScThreeCollRho", J(SR)["collapse_spearman"][0])
# EXACT, as Sec. V-B specifies: the analysis JSON stores scipy's asymptotic p (0.0003),
# which is an order of magnitude smaller than the enumeration gives at n = 6.
assert abs(_b7["bear33"]["spearman"] - J(SR)["collapse_spearman"][0]) < 1e-9
M("ScThreeCollP", _b7["bear33"]["p"], "{:.4f}")

# ---- Sc.33 replication --------------------------------------------------------
M("RepRoneGain", J(R)["criteria"]["R1"]["gain"])
M("RepRoneShort", 0.05 - J(R)["criteria"]["R1"]["gain"])
M("RepRtwoDelta", J(R)["criteria"]["R2"]["delta"])
M("RepRtwoLo", J(R)["criteria"]["R2"]["ci"][0])
M("RepRtwoHi", J(R)["criteria"]["R2"]["ci"][1])
M("RepVerdict", J(R)["verdict"].lower(), None) if False else None
M("RepSpearman", J(R)["dose_response"]["spearman"][0])
M("RepSpearmanP", J(R)["dose_response"]["spearman"][1], "{:.4f}")
M("RepFullDelta", J(R)["dose_response"]["full_range"]["delta"])
M("RepFullLo", J(R)["dose_response"]["full_range"]["ci"][0])
M("RepFullHi", J(R)["dose_response"]["full_range"]["ci"][1])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(
    "% Generated by experiments/make_paper_b_macros.py. Do not edit.\n"
    "% Every number in the manuscript is a macro defined here and read from a results JSON.\n"
    + "\n".join(sorted(set(L))) + "\n")
print(f"wrote {len(set(L))} macros to {OUT}")


# ---------------------------------------------------------------- tables
def _tables():
    _ST = J("static_shortlist.json")["scenarios"]
    """Emit paper_b/tables.tex. Same rule: no number is typed, all read from JSONs."""
    R31 = J("sc31_reference_predictors.json")
    R33 = J("sc33_reference_predictors.json")
    CM = J("alignment_cost_model.json")
    ST = J("corrected_statistics.json")
    # The dedicated regime is priced at two chirp durations. Derive the rates they imply and
    # check the slower one against \PBAdcHi, the fastest automotive rate the prose quotes.
    # The caption used to call them "bracketing automotive front ends"; neither is below the
    # range, so that was wrong and, as it happens, weaker than the truth.
    TC_FAST, TC_SLOW = TC_FAST_US, TC_SLOW_US
    ADC_MULT = (CM["n_adc"] / TC_FAST) / (CM["n_adc"] / TC_SLOW)
    assert ADC_MULT == int(ADC_MULT), ADC_MULT
    rows = [("chirps=8", "8 chirps"), ("chirps=16", "16"), ("chirps=32", "32"),
            ("chirps=64", "64"), ("chirps=125", "125"), ("chirps=250", "250 (full)"),
            ("samples=64", "64 samples"), ("samples=128", "128"),
            ("rx=1", "1 element"), ("rx=2", "2 elements")]
    sec = {"chirps": "Sensing airtime $\\Nchirp$", "samples": "Bandwidth $\\Nsamp$",
           "antennas": "Aperture $m$"}
    out = ["% Generated by experiments/make_paper_b_macros.py. Do not edit.",
           r"\begin{table}[!t]",
           r"\caption{Resource sweep and reference predictors. Learned top-1 is the median "
           r"over " + str(J(A)["seeds"]) + r" seeds, top-5 the mean over the same seeds; "
           r"``degen.'' counts runs within $0.005$ of the train-modal share or of the bound, "
           r"an accuracy proxy and not an inspection of outputs; a dagger marks a setting "
           r"whose bootstrap interval does not exclude the bound. References are fitted on "
           r"training rows and use no radar but the azimuth peak; the exceptions are the "
           r"bound, the best constant on test and, in the top-5 column, the best fixed five "
           r"beams there, and the previous-beam row, on its eligible subset. The power-aware "
           r"static row is the probe set of \eqref{eq:static}, chosen for power and scored "
           r"on beam identity over every test label, as the bound is.}",
           r"\label{tab:sweep}", r"\centering", r"\footnotesize",
           # 0.5pt off the default 6pt: renaming the column "deg." to "degen." in the
           # 2026-10-13 float audit widened the header by 0.93pt and overfilled the column.
           r"\setlength{\tabcolsep}{5.5pt}",
           r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
           r"& \multicolumn{3}{c}{Scenario 31} & \multicolumn{2}{c}{Scenario 33}\\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
           r"Setting & top-1 & top-5 & degen. & top-1 & degen.\\", r"\midrule"]
    DAG = r"$^{\dagger}$"
    miss31 = set(ST["clears_bound"]["31"]["not_clearing"])
    miss33 = set(ST["clears_bound"]["33"]["not_clearing"])
    assert ST["clears_bound"]["31"]["rule"].startswith("2.5th bootstrap percentile")
    assert miss31 == set() and miss33 == {"chirps=8", "chirps=16"}, (miss31, miss33)
    last = None
    for k, lab in rows:
        sw = J(A)["settings"][k]["sweep"]
        # The three sweep subheadings duplicated what the row labels already say, at three
        # lines of a table in a paper with no page left; a rule between groups does the job.
        if sw != last:
            if last is not None:
                out.append(r"\addlinespace[2pt]")
            last = sw
        a, t = J(A)["settings"][k], J(T)["settings"][k]
        # Which settings clear the bound is a recorded determination. Reading the point
        # estimate against the bound row is not the same test: on Scenario 33 the shortest
        # frame sits ABOVE the bound (0.1389 vs 0.1371) while its bootstrap interval does not
        # exclude it, so an unmarked table reads as if it cleared.
        dag = {"31": lambda kk: kk in miss31, "33": lambda kk: kk in miss33}
        out.append(rf"\quad {lab} & {a['top1_median']:.4f}{DAG if dag['31'](k) else ''} & "
                   rf"{a['top5_mean']:.4f} & {a['n_collapsed']} & "
                   rf"{t['top1_median']:.4f}{DAG if dag['33'](k) else ''} & "
                   rf"{t['n_collapsed']}\\")
    def rp(d, key, field):
        return d["predictors"][key][field]
    out += [r"\midrule",
            r"\multicolumn{6}{@{}l}{\emph{Reference predictors}}\\",
            rf"\quad Fixed training prior & {rp(R31,'train-prior top-5','top1'):.4f} & "
            rf"{rp(R31,'train-prior top-5','top5'):.4f} & & "
            rf"{rp(R33,'train-prior top-5','top1'):.4f} & \\",
            # The policy that gives up least received power is at or below chance on beam
            # identity. That is the paper's sharpest statement that the two currencies
            # differ, so it belongs beside the predictors it is read against.
            # SCORED ON EVERY TEST LABEL, the same mask the bound row below uses. Until
            # 2026-10-13 this row carried the finite-power subset while the bound carried
            # all labels, which put a fixed-probe policy ABOVE the best constant on
            # Scenario 31. That is arithmetically impossible on matched rows and was a
            # denominator mismatch, not a result. A static list needs powers to be FITTED
            # but not to be SCORED for exact-beam inclusion, so all labels are available.
            rf"\quad Power-aware static & "
            rf"{_ST['31']['all_label_accuracy']['1']['static']:.4f} & "
            rf"{_ST['31']['all_label_accuracy']['5']['static']:.4f} & & "
            rf"{_ST['33']['all_label_accuracy']['1']['static']:.4f} & \\",
            rf"\quad No-information bound & {J(A)['bound_const']:.4f} & "
            rf"{rp(R31,'hindsight-constant top-1','top5'):.4f} & & "
            rf"{J(T)['bound_const']:.4f} & \\",
            rf"\quad Radar azimuth peak & "
            rf"{rp(R31,'radar-peak (first range bin excluded)','top1'):.4f} & "
            rf"{rp(R31,'radar-peak (first range bin excluded)','top5'):.4f} & & "
            rf"{rp(R33,'radar-peak (first range bin excluded)','top1'):.4f} & \\",
            rf"\quad Previous beam & "
            rf"{rp(R31,'persistence (temporally eligible)','top1'):.4f} & "
            rf"{rp(R31,'persistence (temporally eligible)','top5'):.4f} & & "
            rf"{rp(R33,'persistence (temporally eligible)','top1'):.4f} & \\",
            rf"\quad Partner position & "
            rf"{rp(R31,'GPS reference (knn 20, val-selected)','top1'):.4f} & "
            rf"{rp(R31,'GPS reference (knn 20, val-selected)','top5'):.4f} & & "
            rf"{rp(R33,'GPS reference (knn 20, val-selected)','top1'):.4f} & \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ---- reproducibility / configuration
    out += [r"\begin{table}[!t]",
            r"\caption{Configuration of the two scenarios. Splits are contiguous blocks in "
            r"time with a gap; the optimization and validation blocks together form the "
            r"training region. No radar file and no recording sequence appears in both the "
            r"training region and the test block.}",
            r"\label{tab:config}", r"\centering", r"\footnotesize",
            r"\begin{tabular}{@{}l r r@{}}", r"\toprule",
            r"Quantity & Scenario 31 & Scenario 33\\", r"\midrule"]
    for lab, key, fmt in [("Frames, total", "n", "{:,}"),
                          ("Optimization frames", "n_fit", "{:,}"),
                          ("Validation frames", "n_val", "{:,}"),
                          ("Gap frames", "n_gap", "{:,}"),
                          ("Test frames", "n_test", "{:,}")]:
        out.append(rf"{lab} & {fmt.format(R31[key])} & {fmt.format(R33[key])}\\")
    out += [rf"Shared sequences, train and test & "
            rf"{len(R31['train_test_sequence_overlap'])} & "
            rf"{len(R33['train_test_sequence_overlap'])}\\",
            rf"Rows failing the temporal rule & {R31['persistence_ineligible_test_rows']} & "
            rf"{R33['persistence_ineligible_test_rows']}\\",
            rf"No-information bound & {J(A)['bound_const']:.4f} & {J(T)['bound_const']:.4f}\\",
            rf"Bearing bound, degrees & {J(MM)['bound_rmse_deg']:.3f} & "
            rf"{J('sc33_bearing_mediation_max.json')['bound_rmse_deg']:.3f}\\",
            rf"Seeds per setting, beam task & {J(A)['seeds']} & {J(T)['seeds']}\\",
            rf"Seeds per setting, bearing task & {J(MM)['seeds']} & "
            rf"{J('sc33_bearing_mediation_max.json')['seeds']}\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ---- alignment cost, by regime
    out += [r"\begin{table}[!t]",
            r"\caption{Expected alignment overhead in microseconds under the retained-shortlist "
            r"model of \eqref{eq:overhead}, with $k=5$ and $\Tsec = " + f"{CM['T_sec_us']:g}" +
            r"~\mu$s. The reuse column prices only the beam training, what a system pays "
            r"when the radar frame is already being collected; the dedicated column adds "
            r"$\Nchirp\Tobs$ at the full " + f"{CM['n_adc']}" + r"-sample frame for "
            r"$\Tobs = " + f"{TC_FAST:g}" + r"$ and $" + f"{TC_SLOW:g}" + r"~\mu$s, that is "
            + f"{CM['n_adc'] / TC_FAST:g}" + r" and " + f"{CM['n_adc'] / TC_SLOW:g}" +
            r"~Msps, in that order. A dash means the policy acquires no frame, so its "
            r"dedicated cost is its reuse cost, which Section~\ref{sec:cost} compares against "
            r"the prior. Position and previous-beam entries need information a radar-only "
            r"predictor does not have.}",
            r"\label{tab:cost}", r"\centering", r"\footnotesize",
            r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
            r"& \multicolumn{2}{c}{Scenario 31} & \multicolumn{2}{c}{Scenario 33}\\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"Alignment strategy & reuse & dedicated & reuse & dedicated\\", r"\midrule"]
    def ded(sc, nc, tc):
        g = CM["scenarios"][sc]["dedicated_latency_plus_airtime"][tc]
        return [r for r in g if r["n_chirps"] == nc][0]["total_us"]
    c1, c3 = CM["scenarios"]["31"], CM["scenarios"]["33"]
    out += [rf"Exhaustive sweep & {c1['airtime_exhaustive_us']:.1f} & -- & "
            rf"{c3['airtime_exhaustive_us']:.1f} & --\\",
            rf"Fixed training prior & {c1['airtime_prior_us']:.1f} & -- & "
            rf"{c3['airtime_prior_us']:.1f} & --\\",
            rf"Radar, 8 chirps & {c1['per_setting'][0]['airtime_retained_us']:.1f} & "
            rf"{ded('31',8,'2.56'):.0f}/{ded('31',8,'10.24'):.0f} & "
            rf"{c3['per_setting'][0]['airtime_retained_us']:.1f} & "
            rf"{ded('33',8,'2.56'):.0f}/{ded('33',8,'10.24'):.0f}\\",
            rf"Radar, 250 chirps & {c1['airtime_radar_full_us']:.1f} & "
            rf"{ded('31',250,'2.56'):.0f}/{ded('31',250,'10.24'):.0f} & "
            rf"{c3['airtime_radar_full_us']:.1f} & "
            rf"{ded('33',250,'2.56'):.0f}/{ded('33',250,'10.24'):.0f}\\",
            r"\midrule",
            rf"Previous beam & {c1['airtime_persistence_us']:.1f} & -- & "
            rf"{c3['airtime_persistence_us']:.1f} & --\\",
            rf"Partner position & {c1['airtime_gps_us']:.1f} & -- & "
            rf"{c3['airtime_gps_us']:.1f} & --\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    _und = [r["breakeven_adc_msps"] / r["breakeven_adc_msps_optimistic"]
            for r in J(B)["per_setting"]]
    iso = {v["comparison"]: v for v in J(P)["verdicts"]}
    out += [r"\begin{table}[!t]",
            r"\caption{Four candidate results of our own that did not survive their controls. "
            r"Each row states what was concluded before the control and after it. In the first "
          r"row a negative difference favors the full-bandwidth arm, which is the direction "
          r"the candidate result claimed.}",
            r"\label{tab:reversals}", r"\centering", r"\footnotesize",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\begin{tabular}{@{}p{0.29\columnwidth} p{0.31\columnwidth} "
            r"p{0.30\columnwidth}@{}}", r"\toprule",
            r"Candidate result & Before the control & After\\", r"\midrule"]
    a_, b_ = iso["chirps=125 -> samples=128"], iso["chirps=64 -> samples=64"]
    s_ = iso["chirps=125 -> chirps=250"]
    ep = ST["equal_payload"]
    out += [
        rf"Bandwidth beats airtime at equal payload & "
        rf"${a_['decimate']['delta']:+.4f}$ ($p={a_['decimate']['p']:.4f}$) and "
        rf"${b_['decimate']['delta']:+.4f}$ ($p={b_['decimate']['p']:.4f}$) & "
        rf"${ep['exact']['delta']:+.4f}$ ($p={ep['exact']['permutation_p']:.3f}$); the second "
        rf"pair reverses to ${ep['approximate']['delta']:+.4f}$ "
        rf"($p={ep['approximate']['permutation_p']:.3f}$, interval spans zero) on a "
        rf"{ep['approximate']['mismatch_pct']:.1f}\,\% payload mismatch\\",
        r"\addlinespace[2pt]",
        rf"Saturation beyond 125 chirps & ${s_['decimate']['delta']:+.4f}$ "
        rf"($p={s_['decimate']['p']:.3f}$), flat & ${s_['max']['delta']:+.4f}$ "
        rf"($p={s_['max']['p']:.4f}$), a rise\\",
        r"\addlinespace[2pt]",
        rf"Break-even sampled window, dedicated regime & "
        rf"solved with the fallback term of \eqref{{eq:overhead}} omitted & "
        rf"restoring it raises the required sample rate by "
        rf"{min(_und):.1f}$\times$ to {max(_und):.1f}$\times$\\",
        r"\addlinespace[2pt]",
        rf"Mediation on Scenario 33 & ${J(SR)['link2_all'][0]:.4f}$ "
        rf"($p={J(SR)['link2_all'][1]:.3f}$) & ${J(SR)['link2_working_only'][0]:.4f}$ "
        rf"($p={J(SR)['link2_working_only'][1]:.2f}$), instrument failures excluded\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    PB_ = os.path.join(os.path.dirname(HERE), "paper_b")
    p = os.path.join(PB_, "tables.tex")
    open(p, "w").write("\n".join(out))
    # ALSO one file per table, so the manuscript can input each near its first citation.
    # Inputting all four at one point let LaTeX place Table IV on page 6 while its only
    # discussion is on page 12, six pages later: a float can be placed at or after its
    # source position and never before, so the input point is the real constraint.
    # tables.tex stays the concatenation and remains the single artifact the verifier
    # reads, which then asserts that each split file appears in it verbatim.
    txt = "\n".join(out)
    blocks = ["\\begin{table}" + b for b in txt.split("\\begin{table}")[1:]]
    names = ["tab_sweep", "tab_config", "tab_cost", "tab_reversals"]
    assert len(blocks) == len(names), f"{len(blocks)} tables, expected {len(names)}"
    hdr = "% Generated by experiments/make_paper_b_macros.py. Do not edit.\n"
    for nm, blk in zip(names, blocks):
        open(os.path.join(PB_, nm + ".tex"), "w").write(hdr + blk.rstrip() + "\n")
    print(f"wrote tables to {p} and to {', '.join(n + '.tex' for n in names)}")


_tables()


# ---------------------------------------------------------------- audit corrections
def _audit_macros():
    """Macros introduced by the 2026-10-06 audit corrections."""
    L2 = []

    def M2(name, val, fmt="{:.4f}"):
        assert name.isalpha(), name
        txt = val if isinstance(val, str) else fmt.format(val)
        L2.append(rf"\newcommand{{\PB{name}}}{{{txt}}}")

    C = J("alignment_cost_model.json")
    TB = J("testbed_facts.json")
    M2("BandLink", TB["link_band_ghz"])
    M2("BandRadar", TB["radar_band_ghz"])
    M2("RadarRange", TB["radar_max_range_m"], "{:d}")
    M2("NElem", TB["n_array_elements"], "{:d}")
    M2("FoV", TB["array_fov_deg"], "{:d}")
    M2("RateHz", TB["modality_rate_hz"], "{:d}")
    M2("NRx", TB["cube_shape"][0], "{:d}")
    M2("NChirpMax", TB["cube_shape"][2], "{:d}")
    S = J("corrected_statistics.json")
    for tag, k in (("One", "31"), ("Three", "33")):
        c = C["scenarios"][k]
        R = J(f"sc{k}_reference_predictors.json")["predictors"]
        M2(f"PriorFive{tag}", c["p5_prior"])
        M2(f"PriorOne{tag}", R["train-prior top-1"]["top1"])
        M2(f"BoundMinusPrior{tag}",
           J(f"sc{k}_resource_ablation_G_max.json")["bound_const"]
           - R["train-prior top-1"]["top1"], "{:.4f}")
        M2(f"RadarFive{tag}", c["per_setting"][-1]["top5"])
        M2(f"PersFive{tag}", c["p5_persistence"])
        M2(f"GpsFive{tag}", c["p5_gps"])
        M2(f"AirExh{tag}", c["airtime_exhaustive_us"], "{:.1f}")
        M2(f"AirPrior{tag}", c["airtime_prior_us"], "{:.1f}")
        M2(f"AirRadar{tag}", c["airtime_radar_full_us"], "{:.1f}")
        M2(f"AirPers{tag}", c["airtime_persistence_us"], "{:.1f}")
        M2(f"AirGps{tag}", c["airtime_gps_us"], "{:.1f}")
        M2(f"ReuseVsPrior{tag}", c["reuse_budget_vs_prior_us"], "{:.1f}")
        M2(f"ReuseVsExh{tag}", c["reuse_budget_vs_exhaustive_us"], "{:.1f}")
        M2(f"DedBreakeven{tag}", c["dedicated_breakeven_T_chirp_vs_prior_us"], "{:.3f}")
        M2(f"DedBreakevenExh{tag}", c["dedicated_breakeven_T_chirp_vs_exhaustive_us"], "{:.3f}")
        M2(f"DedRate{tag}", C["n_adc"] / c["dedicated_breakeven_T_chirp_vs_prior_us"], "{:.0f}")
        M2(f"DedRateExh{tag}", C["n_adc"] / c["dedicated_breakeven_T_chirp_vs_exhaustive_us"], "{:.0f}")
        q = c["dedicated_breakeven_by_predictor_quality"]["rows"]
        M2(f"DedPerfEight{tag}", q[0]["adc_perfect_msps"], "{:.0f}")
        M2(f"DedPosEight{tag}", q[0]["adc_position_msps"], "{:.0f}")
        M2(f"DedPerfFull{tag}", q[-1]["adc_perfect_msps"], "{:.0f}")
        M2(f"DedPosFull{tag}", q[-1]["adc_position_msps"], "{:.0f}")
        M2(f"DedPerfTcEight{tag}", q[0]["breakeven_Tc_perfect_us"], "{:.1f}")
        # Break-even requires a chirp NO LONGER than its threshold, and the two thresholds
        # differ: a position-quality shortlist leaves less room than a perfect one. Quoting
        # one of them for both, in the wrong direction, was a defect.
        M2(f"DedPosTcEight{tag}", q[0]["breakeven_Tc_position_us"], "{:.1f}")
        assert q[0]["breakeven_Tc_position_us"] < q[0]["breakeven_Tc_perfect_us"]
        # Our own predictor's conditional break-even at each end of the airtime axis. The
        # full-frame figure is not the requirement of every design in the sweep.
        M2(f"DedCondEight{tag}", q[0]["adc_measured_msps"], "{:.0f}")
        M2(f"DedCondFull{tag}", q[-1]["adc_measured_msps"], "{:.0f}")
        assert abs(q[-1]["adc_measured_msps"]
                   - C["n_adc"] / c["dedicated_breakeven_T_chirp_vs_prior_us"]) < 1.0
        M2(f"DedRateMult{tag}", (C["n_adc"] / c["dedicated_breakeven_T_chirp_vs_prior_us"])
           / (C["n_adc"] / TC_FAST_US), "{:.1f}")
        M2(f"AirRadarRestart{tag}", c["per_setting"][-1]["airtime_restart_us"], "{:.1f}")
        M2(f"AirBestEight{tag}", c["per_setting"][0]["airtime_retained_us"], "{:.1f}")
        g1 = c["dedicated_latency_plus_airtime"]["2.56"]
        M2(f"DedFastEight{tag}", g1[0]["total_us"], "{:.0f}")
        M2(f"DedFastFull{tag}", g1[-1]["total_us"], "{:.0f}")
        g2 = c["dedicated_latency_plus_airtime"]["10.24"]
        M2(f"DedSlowEight{tag}", g2[0]["total_us"], "{:.0f}")
        M2(f"DedSlowFull{tag}", g2[-1]["total_us"], "{:.0f}")
        # corrected reference predictors
        g = [v for kk, v in R.items() if kk.startswith("GPS reference")][0]
        M2(f"GpsRef{tag}", g["top1"])
        M2(f"GpsRefTopFive{tag}", g["top5"])
        M2(f"PeakRaw{tag}", R["radar-peak (as published)"]["top1"])
        M2(f"PeakRawClasses{tag}", R["radar-peak (as published)"]["distinct_top1_classes"], "{:d}")
        M2(f"PeakFixed{tag}", R["radar-peak (first range bin excluded)"]["top1"])
        M2(f"PeakFixedClasses{tag}",
           R["radar-peak (first range bin excluded)"]["distinct_top1_classes"], "{:d}")
        M2(f"PersEligible{tag}", R["persistence (temporally eligible)"]["top1"])
        M2(f"PersIneligible{tag}", J(f"sc{k}_reference_predictors.json")
           ["persistence_ineligible_test_rows"], "{:d}")
        M2(f"NFit{tag}", J(f"sc{k}_reference_predictors.json")["n_fit"], "{:,}")
        M2(f"NVal{tag}", J(f"sc{k}_reference_predictors.json")["n_val"], "{:,}")
        # corrected statistics
        s = S[f"sc{k}"]
        M2(f"SlopeMed{tag}", s["slope_of_medians_per_doubling"])
        M2(f"RhoExact{tag}", s["spearman_exact_two_sided_p"], "{:.4f}")
        cb = S["clears_bound"][k]
        M2(f"NClear{tag}", cb["n_clearing"], "{:d}")
        M2(f"NSettings{tag}", cb["n_settings"], "{:d}")
        pk = S["peak_predictor_concentration"][k]
        M2(f"PeakFixedShare{tag}",
           100 * pk["radar-peak (first range bin excluded)"]["max_class_share"], "{:.0f}")
        M2(f"PeakRawShare{tag}",
           100 * pk["radar-peak (as published)"]["max_class_share"], "{:.0f}")
        M2(f"Rho{tag}", s["spearman_rho"])
        # the radar's share of the GPS comparator's excess accuracy
        A = J(f"sc{k}_resource_ablation_G_max.json")
        bound, best = A["bound_const"], A["settings"]["chirps=250"]["top1_median"]
        M2(f"ShareGps{tag}", 100 * (best - bound) / (g["top1"] - bound), "{:.1f}")
    pc = S["partial_correlations"]
    M2("PartDf", pc["df"], "{:d}")
    M2("PartDirectP", pc["partial_logR_gain_given_rmse"]["corrected_p"], "{:.3f}")
    M2("PartMedP", pc["partial_rmse_gain_given_logR"]["corrected_p"], "{:.3f}")
    e = S["exponent"]
    M2("ExpBoot", e["point_b_median_based"], "{:.3f}")
    M2("ExpBootLo", e["independent_per_setting"]["ci95"][0], "{:.3f}")
    M2("ExpBootHi", e["independent_per_setting"]["ci95"][1], "{:.3f}")
    M2("ExpJointLo", e["joint_seed"]["ci95"][0], "{:.3f}")
    M2("ExpJointHi", e["joint_seed"]["ci95"][1], "{:.3f}")
    ep = S["equal_payload"]
    M2("IsoExactDelta", ep["exact"]["delta"], "{:+.4f}")
    M2("IsoExactP", ep["exact"]["permutation_p"], "{:.3f}")
    M2("IsoApproxDelta", ep["approximate"]["delta"], "{:+.4f}")
    M2("IsoApproxP", ep["approximate"]["permutation_p"], "{:.4f}")
    M2("IsoApproxMismatch", ep["approximate"]["mismatch_pct"], "{:.1f}")
    # --- radar front-end validation (validate_radar_frontend.py)
    FE = J("radar_frontend_validation.json")
    M2("FeChecks", FE["n_checks"], "{:d}")
    M2("FeGrid", len(FE["synthetic_grid"]), "{:d}")
    M2("FeAzSlope", FE["axis_identity"]["azimuth_sweep"]["slope_bins_per_cycle"], "{:.1f}")
    M2("FeSnr", FE["noise"]["snr_db"], "{:g}")
    M2("FeTrials", FE["noise"]["trials"], "{:d}")
    M2("FeDopDrop", FE["decimation"]["doppler_bins_discarded"], "{:d}")
    M2("FeDopTotal", FE["decimation"]["doppler_bins"], "{:d}")
    M2("FeDopDropPct",
       100 * FE["decimation"]["doppler_bins_discarded"] / FE["decimation"]["doppler_bins"],
       "{:.0f}")
    for tag, k in (("One", "Scenario31"), ("Three", "Scenario33")):
        r = FE["real"][k]
        M2(f"FeKept{tag}", 100 * r["fraction_energy_kept"], "{:.1f}")
        M2(f"FeFirstBin{tag}", 100 * r["fraction_peak_in_first_range_bin"], "{:.1f}")
        M2(f"FeFrames{tag}", r["frames"], "{:d}")

    # --- leakage removal and the re-run sweep (audit_dc_removal.py, compare_sc31_dc.py)
    DA = J("dc_removal_audit.json")
    for tag, k in (("One", "31"), ("Three", "33")):
        m = DA["scenarios"][k]["modes"]
        M2(f"DcArtifactBefore{tag}", 100 * m["none"]["frac_argmax_pooled_bin0"], "{:.1f}")
        M2(f"DcArtifactAfter{tag}", 100 * m["mean"]["frac_argmax_pooled_bin0"], "{:.1f}")
        M2(f"DcEnergyBefore{tag}", 100 * m["none"]["energy_frac_first2_range_bins"], "{:.1f}")
        M2(f"DcEnergyAfter{tag}", 100 * m["mean"]["energy_frac_first2_range_bins"], "{:.1f}")
        M2(f"DcFrames{tag}", DA["scenarios"][k]["n_frames"], "{:d}")
    DC = J("sc31_dc_comparison.json")
    M2("DcNSettings", DC["n_settings"], "{:d}")
    M2("DcMeanDelta", DC["mean_delta"], "{:+.4f}")
    M2("DcNSig", DC["n_significant"], "{:d}")
    M2("DcNHigher", DC["n_higher"], "{:d}")
    M2("DcNVerdicts", DC["n_verdicts"], "{:d}")
    M2("DcNVerdictsSame", DC["n_verdicts_unchanged"], "{:d}")
    M2("DcNClear", DC["clears_bound"]["leakage_removed"]["n_clearing"], "{:d}")
    t = DC["chirp_trend_removed"]
    M2("DcRho", t["spearman_rho"], "{:+.4f}")
    M2("DcRhoP", t["spearman_exact_two_sided_p"], "{:.4f}")
    M2("DcSlope", t["slope_of_medians_per_doubling"], "{:+.4f}")
    M2("DcChEight", t["median"][0])
    M2("DcChTwoFifty", t["median"][-1])
    # a MAGNITUDE, because the prose calls it a decrease: printing "-0.0178" after
    # the word "decrease" states the change twice and the sign once too often
    M2("DcWorst", abs(min(r["delta"] for r in DC["per_setting"])), "{:.4f}")
    M2("DcPeakTopOne", DC["azimuth_peak_predictor"]["leakage removed"]["top1"])
    FD = J("radar_frontend_validation_dcmean.json")
    M2("DcFeChecks", FD["n_checks"], "{:d}")
    M2("DcFePass", FD["n_passed"], "{:d}")

    # --- beam coherence time (audit_beam_coherence.py) and the mediation joint test
    BC = J("beam_coherence.json")
    M2("BcHpbw", BC["hpbw_deg"], "{:.2f}")
    M2("BcTol", BC["tolerated_pointing_error_deg"], "{:.2f}")
    M2("BcZeta", BC["zeta"], "{:g}")
    for tag, k in (("One", "31"), ("Three", "33")):
        b = BC["scenarios"][k]
        M2(f"BcRate{tag}", b["los_rate_deg_per_s"]["median"], "{:.2f}")
        M2(f"BcRateHi{tag}", b["los_rate_deg_per_s"]["p95"], "{:.1f}")
        M2(f"BcTb{tag}", b["beam_coherence_median_ms"], "{:.0f}")
        M2(f"BcTbLo{tag}", 1000 * b["beam_coherence_s"]["p5"], "{:.0f}")
        M2(f"BcRatio{tag}", b["ratio_TB_over_frame_period_median"], "{:.1f}")
        M2(f"BcFracStale{tag}", 100 * b["frac_pairs_TB_below_frame_period"], "{:.1f}")
        M2(f"BcPairs{tag}", b["n_eligible_pairs"], "{:d}")
    JS = J("joint_significance.json")
    M2("JsN", JS["n_settings"], "{:d}")
    M2("JsAlphaR", JS["alpha_resource_to_mediator"]["r"], "{:+.4f}")
    M2("JsAlphaP", JS["alpha_resource_to_mediator"]["p"], "{:.4f}")
    M2("JsBetaR", JS["beta_mediator_to_outcome_given_resource"]["r"], "{:+.4f}")
    M2("JsBetaP", JS["beta_mediator_to_outcome_given_resource"]["p"], "{:.4f}")
    M2("JsMaxP", JS["max_of_the_two_p"], "{:.4f}")
    M2("JsAlphaDf", JS["alpha_resource_to_mediator"]["df"], "{:d}")
    M2("JsBetaDf", JS["beta_mediator_to_outcome_given_resource"]["df"], "{:d}")

    # ---- beam quality, decision age and recording-level spread (2026-10-09) ----
    # From beam_power_quality.json: the measured received power of all 64 beams, used for
    # EVALUATION ONLY, converts a top-k miss into the decibels the link actually gives up.
    Q = J("beam_power_quality.json")
    M2("QMarginLo", Q["margins_db"][0], "{:g}")
    M2("QKHi", max(Q["k_values"]), "{:d}")
    M2("QMarginHi", Q["margins_db"][1], "{:g}")
    for tag, k in (("One", "31"), ("Three", "33")):
        q = Q["scenarios"][k]
        M2(f"QNDrop{tag}", q["n_test_nonfinite_power"], "{:d}")
        # Persistence on the SAME rows the radar is scored on, which is the intersection of
        # the temporal eligibility rule and the finite-power mask. The paper used to compare
        # 0.3901 on 1,925 temporally eligible rows with a radar figure on the 1,905 rows that
        # are also power-valid, and then said the denominators matched.
        M2(f"QPersMatch{tag}",
           q["comparators"]["persistence"]["eligible_rows"]["1"]["top_k_inclusion"], "{:.4f}")
        M2(f"QNEvalElig{tag}", q["n_eval_eligible"], "{:,d}")
        M2(f"QNEval{tag}", q["n_eval_all"], "{:d}")
        M2(f"QNSeq{tag}", q["n_test_sequences"], "{:d}")
        full = q["learned"]["spatial|chirps=250"]
        short = q["learned"]["spatial|chirps=8"]
        b5, b1 = full["by_k"]["5"], full["by_k"]["1"]
        M2(f"QReg{tag}", b5["regret_mean_db"]["median"], "{:.3f}")
        M2(f"QRegPNinety{tag}", b5["regret_p90_db"]["median"], "{:.3f}")
        M2(f"QOut{tag}", 100 * b5["outage_3dB"]["median"], "{:.2f}")
        M2(f"QIncl{tag}", b5["inclusion"]["median"], "{:.4f}")
        M2(f"QInclLo{tag}", b5["inclusion"]["p2p5"], "{:.4f}")
        M2(f"QInclHi{tag}", b5["inclusion"]["p97p5"], "{:.4f}")
        M2(f"QInclElig{tag}", b5["inclusion_eligible"]["median"], "{:.4f}")
        M2(f"QTopOneElig{tag}", b1["inclusion_eligible"]["median"], "{:.4f}")
        M2(f"QTopOneAll{tag}", b1["inclusion"]["median"], "{:.4f}")
        M2(f"QRegShort{tag}", short["by_k"]["5"]["regret_mean_db"]["median"], "{:.3f}")
        M2(f"QOutShort{tag}", 100 * short["by_k"]["5"]["outage_3dB"]["median"], "{:.2f}")
        M2(f"QOvh{tag}", full["overhead_us"]["median"], "{:.1f}")
        M2(f"QOvhLo{tag}", full["overhead_us"]["p2p5"], "{:.1f}")
        M2(f"QOvhHi{tag}", full["overhead_us"]["p97p5"], "{:.1f}")
        M2(f"QRegKOne{tag}", full["by_k"]["1"]["regret_mean_db"]["median"], "{:.3f}")
        M2(f"QRegKSixteen{tag}", full["by_k"]["16"]["regret_mean_db"]["median"], "{:.3f}")
        cm = q["comparators"]
        M2(f"QRegPrior{tag}", cm["fixed training prior"]["all_rows"]["5"]["regret_mean_db"],
           "{:.3f}")
        M2(f"QOutPrior{tag}",
           100 * cm["fixed training prior"]["all_rows"]["5"]["outage_3dB"], "{:.2f}")
        M2(f"QRegPers{tag}", cm["persistence"]["eligible_rows"]["5"]["regret_mean_db"],
           "{:.3f}")
        M2(f"QRegGps{tag}", cm["position reference"]["all_rows"]["5"]["regret_mean_db"],
           "{:.3f}")
        M2(f"QOutPers{tag}",
           100 * cm["persistence"]["eligible_rows"]["5"]["outage_3dB"], "{:.2f}")
        M2(f"QOutGps{tag}",
           100 * cm["position reference"]["all_rows"]["5"]["outage_3dB"], "{:.2f}")
        # the regret ratio is the quantity top-k cannot express
        M2(f"QRegRatio{tag}",
           cm["fixed training prior"]["all_rows"]["5"]["regret_mean_db"]
           / b5["regret_mean_db"]["median"], "{:.1f}")
        M2(f"QOutRatio{tag}",
           cm["fixed training prior"]["all_rows"]["5"]["outage_3dB"]
           / b5["outage_3dB"]["median"], "{:.1f}")
        # decision age: the largest lag that survived the eligibility chain
        lags = sorted(q["delay"], key=int)
        d1, dL = q["delay"][lags[0]], q["delay"][lags[-1]]
        M2(f"QLagOneMs{tag}", d1["elapsed_ms_median"], "{:.0f}")
        M2(f"QLagMaxMs{tag}", dL["elapsed_ms_median"], "{:.0f}")
        M2(f"QLagMaxN{tag}", dL["n_eval"], "{:d}")
        M2(f"QDelayTopOne{tag}", dL["1"]["inclusion_median"], "{:.4f}")
        M2(f"QDelayTopFive{tag}", dL["5"]["inclusion_median"], "{:.4f}")
        M2(f"QDelayReg{tag}", dL["5"]["regret_mean_db_median"], "{:.3f}")
        M2(f"QDelayOut{tag}", 100 * dL["5"]["outage_3dB_median"], "{:.2f}")
        M2(f"QNTest{tag}", q["n_test"], "{:d}")
        M2(f"QRegMed{tag}", full["by_k"]["5"]["regret_median_db"]["median"], "{:.3f}")
        # the fresh/stale pair on IDENTICAL future frames, which is what isolates age
        _lg = sorted(q["delay"], key=int)[-1]
        _d = q["delay"][_lg]
        M2(f"QDelaySteps{tag}", _d["n_logged_steps"], "{:d}")
        M2(f"QDelayMsLo{tag}", _d["elapsed_ms_min"], "{:.0f}")
        M2(f"QDelayMsHi{tag}", _d["elapsed_ms_max"], "{:.0f}")
        M2(f"QDelayFreshIncl{tag}", _d["5"]["fresh_inclusion_median"], "{:.4f}")
        M2(f"QDelayFreshReg{tag}", _d["5"]["fresh_regret_mean_db_median"], "{:.3f}")
        M2(f"QDelayPairedReg{tag}", _d["5"]["paired_regret_increase_db_median"], "{:.3f}")
        M2(f"QDelayFreshOut{tag}", 100 * _d["5"]["fresh_outage_3dB_median"], "{:.2f}")
        # the dedicated margin at each end of the airtime axis, and its sensitivities
        _dm = q["dedicated_margin"]["settings"]
        for _lab, _nm in (("chirps=8", "Eight"), ("chirps=250", "Full")):
            _a = _dm[_lab]["all_test"]
            # The shortest frame's margin is sub-microsecond and the full frame's is in the
            # hundreds, so one format would print false precision on one of them.
            _fmt = "{:+.3f}" if abs(_a["mean_us"]) < 100 else "{:+.0f}"
            M2(f"QMargin{_nm}{tag}", _a["mean_us"], _fmt)
            M2(f"QMargin{_nm}Med{tag}", _a["median_us"], _fmt)
            M2(f"QMargin{_nm}NBeat{tag}", _a["n_seeds_cheaper"], "{:d}")
            M2(f"QMargin{_nm}BootLo{tag}", _a["seed_bootstrap_mean_us"][0], "{:+.2f}")
            M2(f"QMargin{_nm}BootHi{tag}", _a["seed_bootstrap_mean_us"][1], "{:+.2f}")
            if _a.get("breakeven_Tsec_us"):
                M2(f"QTsecBreakEven{_nm}{tag}", _a["breakeven_Tsec_us"], "{:.3f}")
        # the recording-level tail, which the pooled Scenario 33 reversal depends on
        _t = q["tail_by_recording"]
        M2(f"QTailPooled{tag}", _t["pooled_difference_pp"], "{:+.2f}")
        M2(f"QTailNSeq{tag}", len(_t["rows"]), "{:d}")
        _w = max(_t["rows"], key=lambda r: abs(r["difference_pp"]))
        M2(f"QTailSeq{tag}", _w["sequence"], "{:d}")
        M2(f"QTailSeqN{tag}", _w["n"], "{:d}")
        M2(f"QTailSeqDiff{tag}", _w["difference_pp"], "{:+.1f}")
        M2(f"QTailLoo{tag}",
           _t["leave_one_out_difference_pp"][str(_w["sequence"])], "{:+.2f}")
        M2(f"QTailNBelow{tag}",
           sum(1 for r in _t["rows"] if r["difference_pp"] <= 0), "{:d}")
        M2(f"QTailLooLo{tag}", _t["loo_min_pp"], "{:+.2f}")
        M2(f"QTailLooHi{tag}", _t["loo_max_pp"], "{:+.2f}")
        # the joint resource/shortlist frontier and the validation-selected point
        _f = q["frontier"]["selection"]["dedicated"]
        _s, _df = _f["selected"], _f["default"]
        M2(f"QFrSetting{tag}", _s["setting"].replace("=", "${=}$"), "{}")
        M2(f"QFrK{tag}", _s["k"], "{:d}")
        M2(f"QFrCost{tag}", _s["cost_dedicated_us"], "{:.1f}")
        M2(f"QFrReg{tag}", _s["test"]["mean_regret_db"], "{:.3f}")
        M2(f"QFrExc{tag}", 100 * _s["test"]["exceed_3dB"], "{:.2f}")
        M2(f"QFrDefCost{tag}", _df["cost_dedicated_us"], "{:.1f}")
        M2(f"QFrDefReg{tag}", _df["test"]["mean_regret_db"], "{:.3f}")
        M2(f"QFrDefExc{tag}", 100 * _df["test"]["exceed_3dB"], "{:.2f}")
        M2(f"QFrRatio{tag}", _f["cost_ratio_vs_default"], "{:.1f}")
        M2(f"QFrNPts{tag}", _f["n_candidates"], "{:d}")
        M2(f"QFrTarget{tag}", _f["validation_target_db"], "{:.3f}")
        _r = q["frontier"]["selection"]["reuse"]
        M2(f"QFrReuseSetting{tag}", _r["selected"]["setting"].replace("=", "${=}$"), "{}")
        M2(f"QFrReuseK{tag}", _r["selected"]["k"], "{:d}")
        M2(f"QFrReuseCost{tag}", _r["selected"]["cost_reuse_us"], "{:.1f}")
        # ---- the power-aware static probe set, and what it leaves the radar to buy ----
        # The fixed training prior ranks beams by how often each is the exact best beam,
        # which is the right static policy for an inclusion question and the wrong one for
        # a power objective. This block prices the static set the objective asks for.
        _S = J("static_shortlist.json")["scenarios"][k]
        _sd = _S["selection_dedicated"]["selected"]
        _sr = _S["selection_reuse"]["selected"]
        _pb = _S["paired_vs_selected_radar"]
        M2(f"StFit{tag}", _S["n_fit_finite"], "{:,d}")
        M2(f"StNCand{tag}", _S["selection_dedicated"]["n_candidates"], "{:d}")
        M2(f"StNK{tag}", len(_S["k_grid"]), "{:d}")
        _g6 = _S["selection_dedicated_sampled_grid"]
        M2(f"StSampledCost{tag}", _g6["cost_us"], "{:.0f}")
        M2(f"StSampledK{tag}", _g6["k"], "{:d}")
        M2(f"StSampledSaving{tag}", _g6["saving_us"], "{:.0f}")
        # the cheapest RADAR policy meeting the same target under the same expanded grid,
        # so the dedicated comparison is grid-matched. Quoting the six-length grid's radar
        # cost beside the sixteen-length grid's static cost would flatter the static side.
        _cr = _S["selection_dedicated"]["cheapest_by_family"]["radar"]
        M2(f"StRadarCost{tag}", _cr["cost_us"], "{:.2f}")
        M2(f"StRadarK{tag}", _cr["k"], "{:d}")
        M2(f"StRadarSetting{tag}", _cr["policy"].replace("radar, ", "").replace("=", "${=}$"), "{}")
        M2(f"StRadarReg{tag}", _cr["test"]["mean_regret_db"], "{:.3f}")
        M2(f"StRadarExc{tag}", 100 * _cr["test"]["exceed_3dB"], "{:.2f}")
        M2(f"StNMeet{tag}", _S["selection_dedicated"]["n_meeting_target"], "{:d}")
        M2(f"StK{tag}", _sd["k"], "{:d}")
        M2(f"StCost{tag}", _sd["cost_us"], "{:.0f}")
        M2(f"StReg{tag}", _sd["test"]["mean_regret_db"], "{:.3f}")
        M2(f"StExc{tag}", 100 * _sd["test"]["exceed_3dB"], "{:.2f}")
        M2(f"StIncl{tag}", _sd["test"]["inclusion"], "{:.3f}")
        M2(f"StUsesRadar{tag}",
           "a radar setting" if _S["selection_dedicated"]["uses_radar"]
           else "no radar at all", "{}")
        # the reuse regime, where the frame is already paid for
        M2(f"StReusePolicy{tag}", _sr["policy"].replace("=", "${=}$"), "{}")
        M2(f"StReuseKK{tag}", _sr["k"], "{:d}")
        M2(f"StReuseCostUs{tag}", _sr["cost_us"], "{:.0f}")
        M2(f"StReuseTie{tag}", _S["selection_reuse"]["n_tied_at_min_cost"], "{:d}")
        M2(f"StReuseReg{tag}", _sr["test"]["mean_regret_db"], "{:.3f}")
        M2(f"StReuseExc{tag}", 100 * _sr["test"]["exceed_3dB"], "{:.2f}")
        # the paired comparison at the radar's own shortlist length
        M2(f"StPairK{tag}", _pb["k"], "{:d}")
        M2(f"StPairStatic{tag}", _pb["static_mean_db"], "{:.3f}")
        M2(f"StPairRadar{tag}", _pb["radar_median_seed_mean_db"], "{:.3f}")
        M2(f"StPairDiff{tag}", _pb["difference_db"], "{:+.3f}")
        M2(f"StPairLo{tag}", _pb["ci95_whole_recording"][0], "{:+.3f}")
        M2(f"StPairHi{tag}", _pb["ci95_whole_recording"][1], "{:+.3f}")
        M2(f"StNRec{tag}", _pb["n_recordings"], "{:d}")
        M2(f"StNRecBetter{tag}", _pb["n_recordings_static_better"], "{:d}")
        # where the two policies cross, and what the radar is worth below that
        _mk = {m["k"]: m for m in _S["matched_k"]}
        M2(f"StCrossTail{tag}", _S["k_crossover_tail"], "{:d}")
        # The MEAN does not cross on Scenario 31 at any length evaluated, so there is no
        # crossover macro for it there and the prose must not imply one. Scenario 33's is
        # real. The 2026-10-13 audit caught the abstract compressing both into one verb.
        if _S["k_crossover_mean"] is not None:
            M2(f"StCrossMean{tag}", _S["k_crossover_mean"], "{:d}")
            # the margin at the crossing. Searching six lengths put Scenario 33's crossover
            # at k=5 when the first empirical crossing is k=4, and the margin there is
            # 0.005 dB: an empirical crossing, not an established transition, and the prose
            # has to say which it is. Caught 2026-10-14.
            M2(f"StCrossMeanMargin{tag}", _S["k_crossover_mean_margin_db"], "{:.3f}")
        # the radar candidates the frontier figure draws, which is every setting at every
        # length the selection searches and not the six stored frontier lengths
        M2(f"StNRadarPts{tag}", len(_S["radar_grid"]), "{:d}")
        # what the inner validation block is actually made of, since the selection
        # constraint is evaluated on it and nothing else
        _vr = _S["validation_recordings"]
        M2(f"StValNRec{tag}", _vr["n_recordings"], "{:d}")
        M2(f"StValTopShare{tag}", 100 * _vr["largest_share"], "{:.0f}")
        M2(f"StValShared{tag}", _vr["shared_with_optimization"], "{:d}")
        M2(f"StNKGrid{tag}", len(_S["k_grid"]), "{:d}")
        # the longest matched length, where the two are closest on the mean
        _kmax = max(int(m["k"]) for m in _S["matched_k"])
        _mx = {int(m["k"]): m for m in _S["matched_k"]}[_kmax]
        M2(f"StSixteenK{tag}", _kmax, "{:d}")
        M2(f"StSixteenRadar{tag}", _mx["radar_full_frame"]["mean_regret_db"], "{:.3f}")
        M2(f"StSixteenStatic{tag}", _mx["static"]["mean_regret_db"], "{:.3f}")
        # counts, not only rates: "no catastrophic frames" was false, and a count says so
        M2(f"StNFin{tag}", _S["test"]["5"]["n"], "{:,d}")
        M2(f"StFiveNExc{tag}",
           int(round(_S["test"]["5"]["exceed_3dB"] * _S["test"]["5"]["n"])), "{:d}")
        M2(f"StOneRadar{tag}", _mk[1]["radar_full_frame"]["mean_regret_db"], "{:.3f}")
        M2(f"StOneStatic{tag}", _mk[1]["static"]["mean_regret_db"], "{:.3f}")
        M2(f"StOneRadarExc{tag}", 100 * _mk[1]["radar_full_frame"]["exceed_3dB"], "{:.1f}")
        M2(f"StOneStaticExc{tag}", 100 * _mk[1]["static"]["exceed_3dB"], "{:.1f}")
        M2(f"StFiveRadar{tag}", _mk[5]["radar_full_frame"]["mean_regret_db"], "{:.3f}")
        M2(f"StFiveStatic{tag}", _mk[5]["static"]["mean_regret_db"], "{:.3f}")
        M2(f"StFiveRadarExc{tag}", 100 * _mk[5]["radar_full_frame"]["exceed_3dB"], "{:.2f}")
        M2(f"StFiveStaticExc{tag}", 100 * _mk[5]["static"]["exceed_3dB"], "{:.2f}")
        M2(f"StFiveIncl{tag}", _mk[5]["static"]["inclusion"], "{:.3f}")
        M2(f"StFiveRadarIncl{tag}", _mk[5]["radar_full_frame"]["inclusion"], "{:.3f}")
        # what the objective buys a static policy, holding the fitting rows fixed
        M2(f"StPriorSel{tag}", _S["prior_test"][str(_sd["k"])]["mean_regret_db"], "{:.3f}")
        # The policy that gives up least power scores at or below the no-information bound
        # on exact-beam accuracy. Inclusion at k=1 IS top-1 for a fixed probe set.
        # ON EVERY TEST LABEL, which is the mask the bound itself uses. Scoring the static
        # row on the finite-power subset while the bound used all labels made a constant
        # policy appear to beat the best constant on Scenario 31; that was a denominator
        # mismatch, caught by the 2026-10-13 audit, and these macros are the matched pair.
        _al = _S["all_label_accuracy"]
        M2(f"StTopOne{tag}", _al["1"]["static"], "{:.4f}")
        M2(f"StTopFive{tag}", _al["5"]["static"], "{:.4f}")
        M2(f"StNAllLabel{tag}", _al["1"]["n"], "{:,d}")
        M2(f"StBoundAll{tag}", _S["bound_all_labels"], "{:.4f}")
        # the finite-power values stay available, for the rows that accompany regret
        M2(f"StTopOneFin{tag}", _S["test"]["1"]["inclusion"], "{:.4f}")
        M2(f"StTopFiveFin{tag}", _S["test"]["5"]["inclusion"], "{:.4f}")
        assert _al["1"]["static"] <= _S["bound_all_labels"] + 1e-12, \
            "a fixed single beam cannot beat the best constant on matched rows"
        # The hindsight set is greedy, so it is one FEASIBLE set and not an optimum. A
        # single-beam replacement search exhibits strictly better sets of the same size at
        # most lengths. The claim is counted over the whole grid rather than read off the
        # selected length, since the search is itself a heuristic: failing to improve a set
        # is not evidence that the set is optimal, so only the improvements are evidence.
        _no = _S["hindsight_not_optimal"]
        _beat = {k_: v for k_, v in _no.items() if not v["greedy_is_optimal"]}
        M2(f"StSwapNBeat{tag}", len(_beat), "{:d}")
        M2(f"StSwapNK{tag}", len(_no), "{:d}")
        M2(f"StSwapMax{tag}", max(v["improvement_db"] for v in _beat.values()), "{:.3f}")
        M2(f"StSwapMaxK{tag}",
           int(max(_beat.items(), key=lambda t: t[1]["improvement_db"])[0]), "{:d}")
        assert len(_beat) > 0, (
            "the counterexample the paper relies on has vanished: the swap search no "
            "longer beats the hindsight greedy set at any length")
        _g = _S["learned_top5_geometry"]
        M2(f"StSpanLearned{tag}", _g["median_index_span"], "{:.0f}")
        M2(f"StSpanStatic{tag}", _g["static_index_span"], "{:d}")
        M2(f"StConsec{tag}", 100 * _g["frac_consecutive"], "{:.1f}")
        # and how much of the static set's strength is the repeated route
        M2(f"StGapMax{tag}", max(abs(v) for v in _S["transfer_gap_db"].values()), "{:.3f}")
        M2(f"StGapSel{tag}", _S["transfer_gap_db"][str(_sd["k"])], "{:.4f}")
        _sp = q["seed_spread"]
        M2(f"QSeedIqrW{tag}", _sp["top1_iqr_width"], "{:.4f}")
        ps = q["per_sequence"]
        M2(f"QSeqIqrW{tag}", ps["top1_iqr"][1] - ps["top1_iqr"][0], "{:.4f}")
        M2(f"QSeqN{tag}", ps["n_sequences_scored"], "{:d}")
        M2(f"QSeqMed{tag}", ps["top1_median"], "{:.4f}")
        M2(f"QSeqLo{tag}", ps["top1_min"], "{:.4f}")
        M2(f"QSeqHi{tag}", ps["top1_max"], "{:.4f}")
        M2(f"QSeqIqrLo{tag}", ps["top1_iqr"][0], "{:.4f}")
        M2(f"QSeqIqrHi{tag}", ps["top1_iqr"][1], "{:.4f}")
        # the alternative architecture, at the two ends of the airtime axis
        w = q["learned"].get("wide|chirps=250")
        if w:
            M2(f"QWideTopOne{tag}", w["top1_median"], "{:.4f}")
            M2(f"QWideIncl{tag}", w["by_k"]["5"]["inclusion"]["median"], "{:.4f}")
            M2(f"QWideReg{tag}", w["by_k"]["5"]["regret_mean_db"]["median"], "{:.3f}")
            M2(f"QWideOvh{tag}", w["overhead_us"]["median"], "{:.1f}")
            ws = q["learned"]["wide|chirps=8"]
            M2(f"QWideShortTopOne{tag}", ws["top1_median"], "{:.4f}")
            M2(f"QWideRxOne{tag}", q["learned"]["wide|rx=1"]["top1_median"], "{:.4f}")
    # The deterministic re-run against the published sweep. The published run used cuDNN
    # autotuning, so a seed there fixes the initialization and the shuffling but not the
    # kernel; re-running seed 42 at the full setting gave 0.2271 and then 0.2363. This
    # compares the two runs setting by setting, which is a replication rather than a
    # reproduction, and the deviation must stay inside the seed spread or the claim fails.
    import numpy as _np
    # `A` is a local name inside this function by the time we get here, so name the files.
    for tag, k in (("One", "31"), ("Three", "33")):
        _z = _np.load(os.path.join(RES, f"beam_shortlists_sc{k}.npz"))
        _pub = J(f"sc{k}_resource_ablation_G_max.json")["settings"]
        _dev, _sd = [], []
        for _lab, _v in _pub.items():
            _key = f"spatial|{_lab}|top1"
            if _key not in _z.files:
                continue
            _dev.append(abs(float(_np.median(_z[_key])) - _v["top1_median"]))
            _sd.append(_v["top1_sd"])
        assert len(_dev) == len(_pub), (len(_dev), len(_pub))
        assert max(_dev) < max(_sd), (max(_dev), max(_sd))
        M2(f"ReplMaxDev{tag}", max(_dev), "{:.4f}")
        M2(f"ReplNSettings{tag}", len(_dev), "{:d}")

    # the alternative network's capacity, parsed from its definition like the published one
    _sl_src = open(os.path.join(HERE, "exp_beam_shortlists.py")).read()
    assert "nn.BatchNorm2d(b)" in _sl_src and "nn.ReLU(inplace=True)" in _sl_src
    import importlib
    _sl = importlib.import_module("exp_beam_shortlists")
    M2("NetParamsWide",
       sum(q.numel() for q in _sl.ImgReadoutWide(2, 64).parameters() if q.requires_grad),
       "{:,d}")
    M2("NetCapacityRatio",
       sum(q.numel() for q in _sl.ImgReadoutWide(2, 64).parameters() if q.requires_grad)
       / sum(q.numel() for q in _sl.ImgReadoutSpatial(2, 64).parameters()
             if q.requires_grad), "{:.1f}")
    M2("QNArch", 2, "{:d}")

    out = os.path.join(os.path.dirname(HERE), "paper_b", "numbers_audit.tex")
    open(out, "w").write(
        "% Generated by experiments/make_paper_b_macros.py. Do not edit.\n"
        "% Macros introduced by the 2026-10-06 audit corrections.\n"
        + "\n".join(sorted(set(L2))) + "\n")
    print(f"wrote {len(set(L2))} audit macros to {out}")


_audit_macros()
