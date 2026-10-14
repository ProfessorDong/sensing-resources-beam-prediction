#!/usr/bin/env python
"""Re-check every load-bearing number quoted in CLAUDE.md and PAPER_B_PLAN.md.

Why this exists. On 2026-10-04 the Sc.31 numbers changed three times in one day: recipe G
replaced the 10-seed run, block-max replaced the decimating readout, and the geometry ceiling
replaced a loose 1-NN estimate. Each time, prose in two markdown files had to follow. The Paper A
discipline is that a literal in a figure is an ASSERTION against a results JSON; this applies the
same rule to the planning documents, so a stale number fails here instead of reaching a reviewer.

Each CHECK is (label, json file, accessor, expected). Edit the expected value ONLY after
confirming the JSON changed for a reason.

Usage: python verify_paper_b_claims.py
Exit code is non-zero if anything disagrees.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
_cache = {}


def J(name):
    if name not in _cache:
        p = os.path.join(RES, name)
        _cache[name] = json.load(open(p)) if os.path.exists(p) else None
    return _cache[name]


def med(f, k): return J(f)["settings"][k]["top1_median"]
def mean(f, k): return J(f)["settings"][k]["top1_mean"]
def sd(f, k): return J(f)["settings"][k]["top1_sd"]
def t5(f, k): return J(f)["settings"][k]["top5_mean"]
def coll(f, k): return J(f)["settings"][k]["n_collapsed"]
def rmse(f, k): return J(f)["settings"][k]["rmse_deg_median"]

A = "sc31_resource_ablation_G_max.json"
D = "sc31_resource_ablation_G.json"
T = "sc33_resource_ablation_G_max.json"
M = "sc31_bearing_mediation.json"
MA = "sc31_mediation_analysis.json"
MM = "sc31_bearing_mediation_max.json"
MAM = "sc31_mediation_analysis_max.json"
P = "sc31_pooling_comparison.json"
G = "geometry_ceiling.json"
B = "isac_airtime_budget_max.json"
R = "sc33_replication_score_max.json"
U = "utilization_definitional_audit.json"
TD = "sc33_resource_ablation_G.json"   # Sc.33 recipe G + DECIMATE, the isolation control
O = "sc31_route_overlap.json"

TB = "testbed_facts.json"
CM = "alignment_cost_model.json"
CS = "corrected_statistics.json"
R1 = "sc31_reference_predictors.json"
R3 = "sc33_reference_predictors.json"


def cm(sc): return J(CM)["scenarios"][sc]


BC = "beam_coherence.json"
JS = "joint_significance.json"
CS = "corrected_statistics.json"


def bc(sc): return J(BC)["scenarios"][sc]


DCA = "dc_removal_audit.json"
DCC = "sc31_dc_comparison.json"
FEDC = "radar_frontend_validation_dcmean.json"
RP1, RP3 = "sc31_reference_predictors.json", "sc33_reference_predictors.json"
PEAK_REPORTED = "radar-peak (first range bin excluded)"


BPQ = "beam_power_quality.json"


def bq(sc): return J(BPQ)["scenarios"][sc]


def bqf(sc, k, f, stat="median"):
    """The full-frame learned shortlist at length k. 2026-10-09."""
    return bq(sc)["learned"]["spatial|chirps=250"]["by_k"][str(k)][f][stat]


def bqc(sc, name, k, f, rows="all_rows"):
    return bq(sc)["comparators"][name][rows][str(k)][f]


def DA(sc, mode): return J(DCA)["scenarios"][sc]["modes"][mode]


def pk(sc):
    return J(f"sc{sc}_reference_predictors.json")["predictors"][PEAK_REPORTED][
        "peak_azimuth_vs_bearing"]


def qual(sc, i): return cm(sc)["dedicated_breakeven_by_predictor_quality"]["rows"][i]


def posref(sc):
    P = J(f"sc{sc}_reference_predictors.json")["predictors"]
    return [v for k, v in P.items() if k.startswith("GPS reference")][0]



FE = "radar_frontend_validation.json"

CHECKS = [
    # --- Sc.31 headline, pool=max ---
    ("Sc.31 bound", A, lambda: J(A)["bound_const"], 0.0901, 5e-5),
    ("Sc.31 pool tag", A, lambda: J(A)["pool"], "max", None),
    ("Sc.31 seeds", A, lambda: J(A)["seeds"], 15, None),
    ("Sc.31 chirps=8 median", A, lambda: med(A, "chirps=8"), 0.1120, 5e-5),
    ("Sc.31 chirps=250 median", A, lambda: med(A, "chirps=250"), 0.2357, 5e-5),
    ("Sc.31 chirps=250 top5", A, lambda: t5(A, "chirps=250"), 0.5877, 5e-5),
    ("Sc.31 chirps=125 median", A, lambda: med(A, "chirps=125"), 0.2098, 5e-5),
    ("Sc.31 rx=1 median", A, lambda: med(A, "rx=1"), 0.2026, 5e-5),
    ("Sc.31 rx=2 median", A, lambda: med(A, "rx=2"), 0.2154, 5e-5),
    ("Sc.31 samples=64 median", A, lambda: med(A, "samples=64"), 0.1914, 5e-5),
    ("Sc.31 samples=128 median", A, lambda: med(A, "samples=128"), 0.2159, 5e-5),
    ("Sc.31 chirps=8 collapsed", A, lambda: coll(A, "chirps=8"), 3, None),
    ("Sc.31 chirps=250 collapsed", A, lambda: coll(A, "chirps=250"), 0, None),
    ("Sc.31 rx=1 gain over bound", A,
     lambda: med(A, "rx=1") - J(A)["bound_const"], 0.1125, 1e-4),
    ("Sc.31 rx=4 gain over bound", A,
     lambda: med(A, "chirps=250") - J(A)["bound_const"], 0.1456, 1e-4),
    ("aperture share of the gain (rx=1->rx=4)", A,
     lambda: (med(A, "chirps=250") - med(A, "rx=1"))
             / (med(A, "chirps=250") - J(A)["bound_const"]), 0.227, 2e-3),
    # --- decimate, for the retractions ---
    ("Sc.31 decimate chirps=125 median", D, lambda: med(D, "chirps=125"), 0.1894, 5e-5),
    ("Sc.31 decimate chirps=250 median", D, lambda: med(D, "chirps=250"), 0.1864, 5e-5),
    ("Sc.31 decimate samples=128 median", D, lambda: med(D, "samples=128"), 0.1741, 5e-5),
    ("Sc.31 decimate chirps=64 median", D, lambda: med(D, "chirps=64"), 0.1665, 5e-5),
    ("Sc.31 decimate samples=64 median", D, lambda: med(D, "samples=64"), 0.1461, 5e-5),
    # --- pooling comparison ---
    ("pooling: verdicts unchanged", P, lambda: J(P)["n_verdicts_unchanged"], 6, None),
    ("pooling: verdicts total", P, lambda: J(P)["n_verdicts"], 10, None),
    ("pooling: all 10 settings improved", P,
     lambda: sum(1 for r in J(P)["per_setting"] if r["delta"] > 0 and r["p"] < 0.05), 10, None),
    # --- mediation (decimate-paired) ---
    ("bearing bound deg", M, lambda: J(M)["bound_rmse_deg"], 16.833, 5e-4),
    ("bearing seeds", M, lambda: J(M)["seeds"], 30, None),
    ("bearing chirps=8 RMSE", M, lambda: rmse(M, "chirps=8"), 16.304, 5e-4),
    ("bearing chirps=250 RMSE", M, lambda: rmse(M, "chirps=250"), 13.761, 5e-4),
    ("bearing rx=1 RMSE", M, lambda: rmse(M, "rx=1"), 15.223, 5e-4),
    ("rx=1 share of bearing-error reduction", M,
     lambda: (J(M)["bound_rmse_deg"] - rmse(M, "rx=1"))
             / (J(M)["bound_rmse_deg"] - rmse(M, "chirps=250")), 0.524, 2e-3),
    ("link 2 Pearson r", MA, lambda: J(MA)["link2_pearson_r"], -0.9105, 5e-4),
    ("link 2 Pearson p < 1e-3", MA, lambda: J(MA)["link2_pearson_p"] < 1e-3, True, None),
    # --- mediation, pool=max (the headline pair) ---
    ("max bearing bound deg", MM, lambda: J(MM)["bound_rmse_deg"], 16.833, 5e-4),
    ("max bearing seeds", MM, lambda: J(MM)["seeds"], 30, None),
    ("max bearing chirps=250 RMSE", MM, lambda: rmse(MM, "chirps=250"), 13.680, 5e-4),
    ("max bearing rx=1 RMSE", MM, lambda: rmse(MM, "rx=1"), 13.924, 5e-4),
    ("max bearing rx=2 RMSE", MM, lambda: rmse(MM, "rx=2"), 13.673, 5e-4),
    ("aperture is flat in bearing (rx=1 vs rx=4 within 0.3 deg)", MM,
     lambda: abs(rmse(MM, "rx=1") - rmse(MM, "chirps=250")) < 0.3, True, None),
    ("rx=1 share of bearing reduction (max)", MM,
     lambda: (J(MM)["bound_rmse_deg"] - rmse(MM, "rx=1"))
             / (J(MM)["bound_rmse_deg"] - rmse(MM, "chirps=250")), 0.923, 3e-3),
    ("max link 2 Pearson r", MAM, lambda: J(MAM)["link2_pearson_r"], -0.9335, 5e-4),
    ("max partial: direct path collapses", MAM,
     lambda: J(MAM)["partial_logR_gain_given_rmse"][1] > 0.05, True, None),
    ("max partial: mediator path survives", MAM,
     lambda: J(MAM)["partial_rmse_gain_given_logR"][1] < 0.05, True, None),
    # --- range alone predicts bearing (the constrained-route control) ---
    ("Sc.31 bearing from range only (deg)", "bearing_from_range_only.json",
     lambda: J("bearing_from_range_only.json")["sc31"]["rmse_from_range_only_deg"],
     6.866, 5e-4),
    ("Sc.33 bearing from range only (deg)", "bearing_from_range_only.json",
     lambda: J("bearing_from_range_only.json")["sc33"]["rmse_from_range_only_deg"],
     5.926, 5e-4),
    ("range-only beats the RADAR bearing estimate on Sc.31", "bearing_from_range_only.json",
     lambda: J("bearing_from_range_only.json")["sc31"]["rmse_from_range_only_deg"]
             < rmse(MM, "chirps=250"), True, None),
    # --- utilization is definitional ---
    ("utilization definitional verdict", U, lambda: J(U)[-1]["definitional"], True, None),
    ("utilization bracket/gain ratio (max)", U, lambda: J(U)[-1]["ratio"], 0.114, 2e-3),
    # --- geometry ceiling ---
    ("Sc.31 ceiling top-1", G, lambda: J(G)["sc31"]["ceiling_top1"], 0.5224, 5e-5),
    ("Sc.31 ceiling family", G, lambda: J(G)["sc31"]["best_family"], "knn", None),
    ("Sc.33 ceiling family", G, lambda: J(G)["sc33"]["best_family"], "bearing-lookup", None),
    ("ceiling is the max over candidates (sc31)", G,
     lambda: J(G)["sc31"]["ceiling_top1"] >= max(c["top1"] for c in J(G)["sc31"]["candidates"]),
     True, None),
    ("ceiling is the max over candidates (sc33)", G,
     lambda: J(G)["sc33"]["ceiling_top1"] >= max(c["top1"] for c in J(G)["sc33"]["candidates"]),
     True, None),
    ("Sc.31 radar share of ceiling", G,
     lambda: J(G)["sc31"]["radar_frac_of_ceiling"], 0.337, 2e-3),
    ("Sc.33 ceiling top-1", G, lambda: J(G)["sc33"]["ceiling_top1"], 0.4892, 5e-5),
    ("Sc.33 radar share of ceiling", G,
     lambda: J(G)["sc33"]["radar_frac_of_ceiling"], 0.130, 2e-3),
    # --- R1's stronger baselines: non-learned and radar-driven ---
    ("Sc.31 radar peak is BELOW the bound", "sc31_stronger_baselines.json",
     lambda: J("sc31_stronger_baselines.json")["baselines"]["peak-az"]["vs_bound"] < 0,
     True, None),
    ("Sc.33 radar peak is BELOW the bound", "sc33_stronger_baselines.json",
     lambda: J("sc33_stronger_baselines.json")["baselines"]["peak-az"]["vs_bound"] < 0,
     True, None),
    ("Sc.31 peak-az top-1", "sc31_stronger_baselines.json",
     lambda: J("sc31_stronger_baselines.json")["baselines"]["peak-az"]["top1"], 0.0672, 5e-5),
    ("Sc.31 persistence top-1", "sc31_stronger_baselines.json",
     lambda: J("sc31_stronger_baselines.json")["baselines"]["persistence"]["top1"],
     0.3936, 5e-5),
    ("persistence BEATS the best radar setting on Sc.31", "sc31_stronger_baselines.json",
     lambda: J("sc31_stronger_baselines.json")["baselines"]["persistence"]["top1"]
             > med(A, "chirps=250"), True, None),
    ("Sc.31 peak azimuth barely tracks true bearing", "sc31_stronger_baselines.json",
     lambda: abs(J("sc31_stronger_baselines.json")["peak_az_vs_gps_bearing"]["pearson"]) < 0.1,
     True, None),
    # --- route overlap ---
    ("Sc.31 route: frac within 0.05 sd", O, lambda: J(O)["frac_within_0p05sd"], 1.0, 1e-9),
    ("Sc.31 route: frac within 0.01 sd", O, lambda: J(O)["frac_within_0p01sd"], 0.6604, 5e-4),
    # --- airtime budget ---
    ("T_sec us", B, lambda: J(B)["T_sec_us"], 5.0, 1e-9),
    ("exhaustive us", B, lambda: J(B)["exhaustive_us"], 320.0, 1e-9),
    ("frame period ms", B, lambda: J(B)["frame_period_us"] / 1000.0, 92.1, 0.05),
    ("recoverable airtime fraction", B, lambda: J(B)["recoverable_airtime_frac"], 0.0, 1e-9),
    ("dedicated radar does not pay (25.6us)", B,
     lambda: J(B)["expected_overhead"]["25.6"]["pays_for_itself"], False, None),
    ("break-even at 250 chirps (honest, Msps)", B,
     lambda: [r for r in J(B)["per_setting"] if r["n_chirps"] == 250][0]["breakeven_adc_msps"],
     392.4, 0.2),
    ("break-even at 8 chirps (honest, Msps)", B,
     lambda: [r for r in J(B)["per_setting"] if r["n_chirps"] == 8][0]["breakeven_adc_msps"],
     21.2, 0.1),
    ("only the 8-chirp setting is ADC-reachable", B,
     lambda: [r["n_chirps"] for r in J(B)["per_setting"]
              if r["reachable_by_automotive_adc"]], [8], None),
    # --- Sc.33 replication ---
    ("Sc.33 bound", T, lambda: J(T)["bound_const"], 0.1371, 5e-5),
    ("Sc.33 chirps=250 median", T, lambda: med(T, "chirps=250"), 0.1828, 5e-5),
    ("Sc.33 chirps=8 median", T, lambda: med(T, "chirps=8"), 0.1389, 5e-5),
    ("Sc.33 R1 fails", R, lambda: J(R)["criteria"]["R1"]["pass"], False, None),
    ("Sc.33 R1 gain", R, lambda: J(R)["criteria"]["R1"]["gain"], 0.0457, 1e-4),
    ("Sc.33 R2 passes", R, lambda: J(R)["criteria"]["R2"]["pass"], True, None),
    ("Sc.33 R3 retired", R, lambda: J(R)["criteria"]["R3"]["retired"], True, None),
    ("Sc.33 verdict", R, lambda: J(R)["verdict"], "PARTIAL REPLICATION", None),
    ("Sc.33 collapse falls with resources", T,
     lambda: [coll(T, f"chirps={c}") for c in (8, 16, 32, 64, 125, 250)],
     [7, 8, 6, 1, 0, 0], None),
    # --- Sc.33 mediation: link 2 does NOT replicate ---
    ("Sc.33 bearing bound deg", "sc33_bearing_mediation_max.json",
     lambda: J("sc33_bearing_mediation_max.json")["bound_rmse_deg"], 19.452, 5e-4),
    ("Sc.33 instrument fails in 4 settings", "sc33_mediation_restricted.json",
     lambda: J("sc33_mediation_restricted.json")["n_instrument_failures"], 4, None),
    ("Sc.33 link2 over ALL settings (artifact)", "sc33_mediation_restricted.json",
     lambda: J("sc33_mediation_restricted.json")["link2_all"][0], -0.6381, 5e-4),
    ("Sc.33 link2 restricted r", "sc33_mediation_restricted.json",
     lambda: J("sc33_mediation_restricted.json")["link2_working_only"][0], -0.2137, 5e-4),
    ("Sc.33 link2 restricted is NOT significant", "sc33_mediation_restricted.json",
     lambda: J("sc33_mediation_restricted.json")["link2_working_only"][1] > 0.05, True, None),
    ("Sc.33 collapse-vs-chirps Spearman", "sc33_mediation_restricted.json",
     lambda: J("sc33_mediation_restricted.json")["collapse_spearman"][0], -0.9856, 5e-4),
    ("Sc.33 collapse counts", "sc33_mediation_restricted.json",
     lambda: [n for _, n in J("sc33_mediation_restricted.json")["collapse_vs_chirps"]],
     [30, 30, 24, 21, 10, 1], None),
    # --- Sc.33 isolation control: recipe G + decimate ---
    ("Sc.33 decimate pool tag", TD, lambda: J(TD)["pool"], "decimate", None),
    ("Sc.33 decimate recipe", TD, lambda: J(TD)["recipe"], "G", None),
    ("Sc.33 decimate same bound", TD, lambda: J(TD)["bound_const"], 0.1371, 5e-5),
    ("Sc.33 decimate chirps=64 median", TD, lambda: med(TD, "chirps=64"), 0.1631, 5e-5),
    ("Sc.33 decimate chirps=125 median", TD, lambda: med(TD, "chirps=125"), 0.1622, 5e-5),
    ("Sc.33 decimate chirps=250 median", TD, lambda: med(TD, "chirps=250"), 0.1461, 5e-5),
    ("Sc.33 decimate 125->250 FALLS", TD,
     lambda: med(TD, "chirps=250") - med(TD, "chirps=125") < 0, True, None),
    ("Sc.33 max 125->250 does not fall", T,
     lambda: med(T, "chirps=250") - med(T, "chirps=125") > 0, True, None),
    ("Sc.33 decimate chirps=8 collapsed", TD, lambda: coll(TD, "chirps=8"), 11, None),
    ("Sc.33 decimate sweep is NOT monotone", TD,
     lambda: all(med(TD, f"chirps={a}") <= med(TD, f"chirps={b}")
                 for a, b in zip((8, 16, 32, 64, 125), (16, 32, 64, 125, 250))), False, None),

    # --- 2026-10-06 audit corrections. Every literal here is quoted in CLAUDE.md's rewrite
    # section; a failure means the JSON moved and the prose must be fixed BEFORE this value.
    ("testbed: radar band", TB, lambda: J(TB)["radar_band_ghz"], "76--81", None),
    ("testbed: link band", TB, lambda: J(TB)["link_band_ghz"], "60", None),
    ("testbed: separate radios", TB, lambda: J(TB)["separate_radios"], True, None),
    ("testbed: array elements", TB, lambda: J(TB)["n_array_elements"], 16, None),
    ("testbed: codebook", TB, lambda: J(TB)["n_beams"], 64, None),
    ("cost: exhaustive Sc.31", CM, lambda: cm("31")["airtime_exhaustive_us"], 320.0, 5e-3),
    ("cost: fixed prior Sc.31", CM, lambda: cm("31")["airtime_prior_us"], 227.8, 5e-2),
    ("cost: radar full frame Sc.31", CM, lambda: cm("31")["airtime_radar_full_us"], 146.6, 5e-2),
    ("cost: persistence Sc.31", CM, lambda: cm("31")["airtime_persistence_us"], 64.5, 5e-2),
    ("cost: position Sc.31", CM, lambda: cm("31")["airtime_gps_us"], 51.1, 5e-2),
    ("cost: reuse saving vs prior Sc.31", CM,
     lambda: cm("31")["reuse_budget_vs_prior_us"], 81.2, 5e-2),
    ("cost: reuse saving vs exhaustive Sc.31", CM,
     lambda: cm("31")["reuse_budget_vs_exhaustive_us"], 173.4, 5e-2),
    ("cost: reuse saving vs prior Sc.33", CM,
     lambda: cm("33")["reuse_budget_vs_prior_us"], 24.8, 5e-2),
    ("cost: REUSE beats the prior, so 'never pays' is void", CM,
     lambda: cm("31")["airtime_radar_full_us"] < cm("31")["airtime_prior_us"], True, None),
    ("cost: dedicated break-even rate vs prior Sc.31", CM,
     lambda: J(CM)["n_adc"] / cm("31")["dedicated_breakeven_T_chirp_vs_prior_us"], 789, 1.0),
    ("cost: dedicated beats the prior at NO setting", CM,
     lambda: any(z["beats_prior"] for g in cm("31")["dedicated_latency_plus_airtime"].values()
                 for z in g), False, None),
    ("cost: perfect shortlist at 8 chirps is ADC-reachable", CM,
     lambda: qual("31", 0)["adc_perfect_msps"], 10, 1.0),
    ("cost: position-quality at 8 chirps is ADC-reachable", CM,
     lambda: qual("31", 0)["adc_position_msps"], 12, 1.0),
    ("cost: perfect shortlist at the full frame is not", CM,
     lambda: qual("31", -1)["adc_perfect_msps"], 316, 1.0),
    ("reference: position predictor Sc.31", R1, lambda: posref("31")["top1"], 0.5000, 5e-5),
    ("reference: position predictor Sc.31 top-5", R1, lambda: posref("31")["top5"], 0.9114, 5e-5),
    ("reference: position predictor Sc.33", R3, lambda: posref("33")["top1"], 0.4543, 5e-5),
    ("reference: position predictor is VALIDATION-selected, not test-selected", R1,
     lambda: "validation" in posref("31")["note"], True, None),
    ("reference: radar share of position, Sc.31", R1,
     lambda: 100 * (med(A, "chirps=250") - J(A)["bound_const"])
             / (posref("31")["top1"] - J(A)["bound_const"]), 35.5, 0.05),
    ("reference: radar share of position, Sc.33", R3,
     lambda: 100 * (med(T, "chirps=250") - J(T)["bound_const"])
             / (posref("33")["top1"] - J(T)["bound_const"]), 14.4, 0.05),
    ("reference: azimuth peak AS PUBLISHED emits 1 class on Sc.31", R1,
     lambda: J(R1)["predictors"]["radar-peak (as published)"]["distinct_top1_classes"], 1, None),
    ("reference: repaired azimuth peak Sc.31", R1,
     lambda: J(R1)["predictors"]["radar-peak (first range bin excluded)"]["top1"], 0.0565, 5e-5),
    ("reference: repaired azimuth peak is near-constant on Sc.33", R3,
     lambda: J(R3)["predictors"]["radar-peak (first range bin excluded)"]["max_class_share"],
     0.9803, 5e-4),
    ("reference: eligible persistence Sc.31", R1,
     lambda: J(R1)["predictors"]["persistence (temporally eligible)"]["top1"], 0.3901, 5e-5),
    ("reference: ineligible rows Sc.31", R1,
     lambda: J(R1)["persistence_ineligible_test_rows"], 39, None),
    ("split: optimization frames Sc.31", R1, lambda: J(R1)["n_fit"], 4418, None),
    ("split: validation frames Sc.31", R1, lambda: J(R1)["n_val"], 490, None),
    ("split: NO shared recording sequence Sc.31", R1,
     lambda: len(J(R1)["train_test_sequence_overlap"]), 0, None),
    ("stats: exact Spearman p Sc.31", CS,
     lambda: J(CS)["sc31"]["spearman_exact_two_sided_p"], 0.0028, 5e-5),
    ("stats: exact Spearman p Sc.33", CS,
     lambda: J(CS)["sc33"]["spearman_exact_two_sided_p"], 0.0167, 5e-5),
    ("stats: slope of MEDIANS Sc.31", CS,
     lambda: J(CS)["sc31"]["slope_of_medians_per_doubling"], 0.0227, 5e-5),
    ("stats: partial df", CS, lambda: J(CS)["partial_correlations"]["df"], 6, None),
    ("stats: direct-path partial p at the right df", CS,
     lambda: J(CS)["partial_correlations"]["partial_logR_gain_given_rmse"]["corrected_p"],
     0.206, 5e-4),
    ("stats: mediator-path partial p at the right df", CS,
     lambda: J(CS)["partial_correlations"]["partial_rmse_gain_given_logR"]["corrected_p"],
     0.016, 5e-4),
    ("stats: exponent point estimate", CS,
     lambda: J(CS)["exponent"]["point_b_median_based"], 0.403, 5e-4),
    ("stats: exponent CI, joint-seed low", CS,
     lambda: J(CS)["exponent"]["joint_seed"]["ci95"][0], 0.295, 2e-3),
    ("stats: exponent CI, joint-seed high", CS,
     lambda: J(CS)["exponent"]["joint_seed"]["ci95"][1], 0.548, 2e-3),
    ("stats: the exponent interval still excludes the CRB's b=1", CS,
     lambda: J(CS)["exponent"]["joint_seed"]["ci95"][1] < 1.0, True, None),
    ("stats: exact equal-payload pair", CS,
     lambda: J(CS)["equal_payload"]["exact"]["delta"], 0.0061, 5e-5),
    ("stats: the approximate pair's payload mismatch", CS,
     lambda: J(CS)["equal_payload"]["approximate"]["mismatch_pct"], 2.3, 0.05),
    ("stats: all 10 Sc.31 settings clear the bound", CS,
     lambda: J(CS)["clears_bound"]["31"]["n_clearing"], 10, None),
    ("stats: 8 of 10 Sc.33 settings clear the bound", CS,
     lambda: J(CS)["clears_bound"]["33"]["n_clearing"], 8, None),
    ("stats: the Sc.33 exceptions are the two shortest frames", CS,
     lambda: J(CS)["clears_bound"]["33"]["not_clearing"], ["chirps=8", "chirps=16"], None),


    # --- radar front-end validation (validate_radar_frontend.py). The manuscript's Sec. IX-A
    # asserts these; a drifted front end must fail here, not in review.
    ("front end: every check passed", FE,
     lambda: J(FE)["n_passed"] == J(FE)["n_checks"], True, None),
    ("front end: the transform under test is the production one", FE,
     lambda: "_from_cube" in J(FE)["transform"], True, None),
    ("front end: azimuth scale is the transform length", FE,
     lambda: J(FE)["axis_identity"]["azimuth_sweep"]["slope_bins_per_cycle"], 64.0, 0.05),
    ("front end: a broadside target lands on the centre azimuth bin", FE,
     lambda: J(FE)["axis_identity"]["azimuth_sweep"]["bin"][
         len(J(FE)["axis_identity"]["azimuth_sweep"]["bin"]) // 2], 32, None),
    ("front end: one element gives an EXACTLY flat azimuth axis", FE,
     lambda: J(FE)["aperture"]["rx=1"]["azimuth_profile_ptp"], 0.0, 0.0),
    ("front end: azimuthal concentration grows with aperture", FE,
     lambda: J(FE)["aperture"]["rx=4"]["concentration"]
             > J(FE)["aperture"]["rx=2"]["concentration"]
             > J(FE)["aperture"]["rx=1"]["concentration"], True, None),
    ("front end: target recovered in every noise trial", FE,
     lambda: J(FE)["noise"]["hits"] == J(FE)["noise"]["trials"], True, None),
    ("front end: decimation discards most Doppler bins", FE,
     lambda: J(FE)["decimation"]["doppler_bins_discarded"], 186, None),
    ("front end: most range energy is in the kept half, Sc.31", FE,
     lambda: 100 * J(FE)["real"]["Scenario31"]["fraction_energy_kept"], 73.2, 0.05),
    ("front end: most range energy is in the kept half, Sc.33", FE,
     lambda: 100 * J(FE)["real"]["Scenario33"]["fraction_energy_kept"], 86.1, 0.05),
    ("front end: the zero-range artifact dominates Sc.31", FE,
     lambda: 100 * J(FE)["real"]["Scenario31"]["fraction_peak_in_first_range_bin"],
     100.0, 0.05),
    ("front end: and does NOT dominate Sc.33", FE,
     lambda: 100 * J(FE)["real"]["Scenario33"]["fraction_peak_in_first_range_bin"], 4.3, 0.05),

    # --- leakage removal and the re-run sweep (audit_dc_removal.py, compare_sc31_dc.py,
    # validate_radar_frontend.py --dc mean). Section IX-A reports all of this. The point of
    # these checks is that the dose-response must NOT depend on the leakage term; if a future
    # change makes it depend on it, the verdict count below is what fails.
    ("leakage: the artifact is removed on Sc.31", DCA,
     lambda: 100 * DA("31", "mean")["frac_argmax_pooled_bin0"], 0.0, 0.0),
    ("leakage: the artifact is removed on Sc.33", DCA,
     lambda: 100 * DA("33", "mean")["frac_argmax_pooled_bin0"], 0.0, 0.0),
    ("leakage: it dominated Sc.31 before removal", DCA,
     lambda: 100 * DA("31", "none")["frac_argmax_pooled_bin0"], 99.8, 0.05),
    ("leakage: energy in the first two range bins, before", DCA,
     lambda: 100 * DA("31", "none")["energy_frac_first2_range_bins"], 45.7, 0.05),
    ("leakage: energy in the first two range bins, after", DCA,
     lambda: 100 * DA("31", "mean")["energy_frac_first2_range_bins"], 0.3, 0.05),
    ("leakage: a notch beyond bin 0 adds nothing", DCA,
     lambda: abs(DA("31", "notch4")["frac_argmax_pooled_bin0"]
                 - DA("31", "mean")["frac_argmax_pooled_bin0"]) < 1e-12, True, None),
    ("leakage: the repaired chain passes every synthetic check", FEDC,
     lambda: J(FEDC)["n_passed"] == J(FEDC)["n_checks"], True, None),
    ("leakage: the repaired chain was validated at 15 checks", FEDC,
     lambda: J(FEDC)["n_checks"], 15, None),
    ("leakage: the repaired run is the one factor it claims", DCC,
     lambda: J(DCC)["pool"] == "max" and J(DCC)["recipe"] == "G"
             and J(DCC)["seeds"]["leakage_removed"] == 15, True, None),
    ("leakage: EVERY step verdict is unchanged", DCC,
     lambda: J(DCC)["n_verdicts_unchanged"] == J(DCC)["n_verdicts"], True, None),
    ("leakage: 10 verdicts were re-tested", DCC,
     lambda: J(DCC)["n_verdicts"], 10, None),
    ("leakage: all 10 settings still clear the bound", DCC,
     lambda: J(DCC)["clears_bound"]["leakage_removed"]["n_clearing"], 10, None),
    ("leakage: the mean change is small and negative", DCC,
     lambda: J(DCC)["mean_delta"], -0.0055, 0.0002),
    ("leakage: significant at only one setting", DCC,
     lambda: J(DCC)["n_significant"], 1, None),
    ("leakage: the airtime axis is monotone on the repaired input", DCC,
     lambda: J(DCC)["chirp_trend_removed"]["monotone"], True, None),
    ("leakage: Spearman is +1 on the repaired input", DCC,
     lambda: J(DCC)["chirp_trend_removed"]["spearman_rho"], 1.0, 1e-9),
    ("leakage: exact p of that trend", DCC,
     lambda: J(DCC)["chirp_trend_removed"]["spearman_exact_two_sided_p"], 0.0028, 0.0002),
    ("leakage: full-frame median on the repaired input", DCC,
     lambda: J(DCC)["chirp_trend_removed"]["median"][-1], 0.2352, 0.0002),
    ("leakage: the azimuth peak STILL fails the bound", DCC,
     lambda: J(DCC)["azimuth_peak_predictor"]["leakage removed"]["clears_bound"], False, None),
    # The peak-to-bearing correlation the manuscript quotes must come from the variant it
    # reports, not from the leakage artifact. Both are stored; this pins which one is used.
    ("peak: the reported variant's bearing correlation, Sc.31", RP1,
     lambda: pk("31")["pearson"], 0.0400, 0.0002),
    ("peak: it is NOT significant on Sc.31", RP1,
     lambda: pk("31")["pearson_p"] > 0.05, True, None),
    ("peak: the reported variant's bearing correlation, Sc.33", RP3,
     lambda: pk("33")["pearson"], -0.0160, 0.0002),
    ("peak: the reported variant is not degenerate, Sc.31", RP1,
     lambda: pk("31")["distinct_peak_azimuths"], 47, None),

    # --- beam coherence time (audit_beam_coherence.py). Section VII asserts that the latency
    # term does not bind; if the LOS rotation rate or the array geometry ever changes, the
    # ratio below is what fails rather than the prose.
    ("coherence: the tolerated pointing error is the Gaussian half-power one", BC,
     lambda: J(BC)["tolerated_pointing_error_deg"], 3.173, 0.002),
    ("coherence: 16 elements give a 6.35 deg broadside beamwidth", BC,
     lambda: J(BC)["hpbw_deg"], 6.35, 0.01),
    ("coherence: T_B on Sc.31, ms", BC, lambda: bc("31")["beam_coherence_median_ms"],
     1089.3, 0.5),
    ("coherence: T_B on Sc.33, ms", BC, lambda: bc("33")["beam_coherence_median_ms"],
     1249.8, 0.5),
    ("coherence: T_B exceeds the refresh interval on Sc.31", BC,
     lambda: bc("31")["ratio_TB_over_frame_period_median"], 11.83, 0.02),
    ("coherence: and on Sc.33", BC,
     lambda: bc("33")["ratio_TB_over_frame_period_median"], 13.61, 0.02),
    ("coherence: no Sc.31 frame pair is stale within one interval", BC,
     lambda: 100 * bc("31")["frac_pairs_TB_below_frame_period"], 0.0, 0.05),
    ("coherence: the median LOS rotation rate, Sc.31 deg/s", BC,
     lambda: bc("31")["los_rate_deg_per_s"]["median"], 2.91, 0.01),
    ("coherence: the rate is measured, not inferred from a heading", BC,
     lambda: "no heading is assumed" in J(BC)["adaptation"], True, None),

    # --- the mediation, tested as MacKinnon et al. recommend (audit_joint_significance.py)
    ("mediation: the joint significance test PASSES", JS,
     lambda: J(JS)["joint_significance"], True, None),
    ("mediation: both paths significant, larger p", JS,
     lambda: J(JS)["max_of_the_two_p"], 0.0164, 0.0002),
    ("mediation: alpha path, resource to mediator", JS,
     lambda: J(JS)["alpha_resource_to_mediator"]["r"], -0.8037, 0.0002),
    ("mediation: beta path, mediator to outcome given resource", JS,
     lambda: J(JS)["beta_mediator_to_outcome_given_resource"]["r"], -0.8031, 0.0002),
    ("mediation: beta path uses df = n - 3", JS,
     lambda: J(JS)["beta_mediator_to_outcome_given_resource"]["df"], 6, None),
    ("mediation: the low-power causal-steps step is NOT significant", JS,
     lambda: J(JS)["baron_kenny_direct_path_given_mediator"]["p"] > 0.05, True, None),
    ("mediation: m = 1 is excluded because log R is undefined there", JS,
     lambda: J(JS)["excluded"], ["rx=1"], None),
    ("mediation: nine settings enter the test", JS, lambda: J(JS)["n_settings"], 9, None),
    # The partial correlation computed here must agree with the corrected-statistics file,
    # which derives it independently.
    ("mediation: beta path agrees with corrected_statistics.json", JS,
     lambda: abs(J(JS)["beta_mediator_to_outcome_given_resource"]["p"]
                 - J(CS)["partial_correlations"]["partial_rmse_gain_given_logR"]["corrected_p"])
             < 1e-6, True, None),
]



# ---- beam quality, decision age and recording-level spread (2026-10-09) ----------------
# A new results JSON with no verifier is the structural defect the October table audit
# found, so beam_power_quality.json is guarded here from the day it was written.
CHECKS += [
    ("Sc.31 full frame, mean power regret at k=5 (dB)",
     BPQ, lambda: bqf("31", 5, "regret_mean_db"), 0.225, 5e-4),
    ("Sc.31 full frame, 3 dB outage at k=5",
     BPQ, lambda: bqf("31", 5, "outage_3dB"), 0.00617, 5e-5),
    ("Sc.31 full frame, top-5 inclusion at k=5",
     BPQ, lambda: bqf("31", 5, "inclusion"), 0.5952, 5e-4),
    ("Sc.33 full frame, mean power regret at k=5 (dB)",
     BPQ, lambda: bqf("33", 5, "regret_mean_db"), 0.457, 5e-4),
    ("Sc.31 fixed prior, mean power regret at k=5 (dB)",
     BPQ, lambda: bqc("31", "fixed training prior", 5, "regret_mean_db"), 0.461, 5e-4),
    ("Sc.31 position reference, mean power regret at k=5 (dB)",
     BPQ, lambda: bqc("31", "position reference", 5, "regret_mean_db"), 0.005, 5e-4),
    # the radar beats the no-sensing prior on the mean loss on BOTH scenarios
    ("radar mean regret below the prior's on both scenarios",
     BPQ, lambda: all(bqf(sc, 5, "regret_mean_db")
                 < bqc(sc, "fixed training prior", 5, "regret_mean_db")
                 for sc in ("31", "33")), True, None),
    # but its 3 dB outage is HIGHER on Scenario 33, which is the finding top-k hid
    ("Sc.33 radar 3 dB outage exceeds the prior's",
     BPQ, lambda: bqf("33", 5, "outage_3dB")
             > bqc("33", "fixed training prior", 5, "outage_3dB"), True, None),
    ("Sc.31 radar 3 dB outage is below the prior's",
     BPQ, lambda: bqf("31", 5, "outage_3dB")
             < bqc("31", "fixed training prior", 5, "outage_3dB"), True, None),
    # mean regret cannot rise with k: a longer shortlist contains a shorter one
    ("mean regret is non-increasing in k on both scenarios",
     BPQ, lambda: all(bqf(sc, a, "regret_mean_db") >= bqf(sc, b, "regret_mean_db")
                 for sc in ("31", "33")
                 for a, b in zip((1, 2, 3, 5, 8, 16), (2, 3, 5, 8, 16))), True, None),
    # matched denominators: the radar's score barely moves on persistence's eligible rows
    ("Sc.31 radar top-5 moves under 0.005 on the eligible subset",
     BPQ, lambda: abs(bqf("31", 5, "inclusion")
                 - bqf("31", 5, "inclusion_eligible")) < 0.005, True, None),
    # a decision ages: top-5 falls at every lag, and the ideal pointing time does not cover it
    ("Sc.31 delayed top-5 falls monotonically with lag",
     BPQ, lambda: all(bq("31")["delay"][a]["5"]["inclusion_median"]
                 >= bq("31")["delay"][b]["5"]["inclusion_median"]
                 for a, b in zip(("1", "2", "3"), ("2", "3", "5"))), True, None),
    ("Sc.31 top-5 at the largest lag", BPQ,
     lambda: bq("31")["delay"]["5"]["5"]["inclusion_median"], 0.4053, 5e-4),
    ("the largest lag is well inside the ideal pointing time", BPQ,
     lambda: bq("31")["delay"]["5"]["elapsed_ms_median"]
             < 0.5 * J(BC)["scenarios"]["31"]["beam_coherence_median_ms"], True, None),
    ("Sc.31 per-sequence top-1 spans at least 0.2", BPQ,
     lambda: bq("31")["per_sequence"]["top1_max"]
             - bq("31")["per_sequence"]["top1_min"] > 0.2, True, None),
    ("Sc.31 has 15 test sequences and Sc.33 has 5", BPQ,
     lambda: (bq("31")["n_test_sequences"], bq("33")["n_test_sequences"]), (15, 5), None),
    # the alternative architecture preserves the resource ordering on both scenarios
    ("the wide readout preserves the airtime ordering on both scenarios", BPQ,
     lambda: all(bq(sc)["learned"][f"wide|chirps={a}"]["top1_median"]
                 < bq(sc)["learned"][f"wide|chirps={b}"]["top1_median"]
                 for sc in ("31", "33") for a, b in ((8, 64), (64, 250))), True, None),
    ("the wide readout is better at the full frame on Sc.31 and not on Sc.33", BPQ,
     lambda: (bq("31")["learned"]["wide|chirps=250"]["by_k"]["5"]["inclusion"]["median"]
              > bqf("31", 5, "inclusion"),
              bq("33")["learned"]["wide|chirps=250"]["by_k"]["5"]["inclusion"]["median"]
              > bqf("33", 5, "inclusion")), (True, False), None),
]


def main():
    npass = nfail = nskip = 0
    for label, f, acc, exp, tol in CHECKS:
        if J(f) is None:
            print(f"  SKIP  {label:<46} (missing {f})"); nskip += 1; continue
        try:
            got = acc()
        except Exception as e:
            print(f"  FAIL  {label:<46} accessor raised {type(e).__name__}: {e}")
            nfail += 1; continue
        ok = (abs(got - exp) <= tol) if tol is not None else (got == exp)
        if ok:
            npass += 1
        else:
            nfail += 1
            print(f"  FAIL  {label:<46} expected {exp!r}, got {got!r}  [{f}]")
    print(f"\n{npass} passed, {nfail} failed, {nskip} skipped, {len(CHECKS)} checks")
    if nfail:
        print("A failure means a results JSON changed. Update the prose in CLAUDE.md and")
        print("PAPER_B_PLAN.md and the expected value here, in that order, and say why.")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
