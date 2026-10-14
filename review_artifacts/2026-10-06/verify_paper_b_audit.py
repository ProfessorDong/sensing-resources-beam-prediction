"""Read-only checks supporting PAPER_B_AUDIT_2026-10-06.md.

Run with /home/dong/anaconda3/envs/w26/bin/python. Prints JSON; does not
train models or overwrite any research result. Optional --bootstrap 2000
recomputes illustrative, estimator-consistent exponent intervals.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/results"


def read(name):
    return json.loads((RES / name).read_text())


def check_scenario(sc):
    csv_path = ROOT / f"datasets/deepsense6g/Scenario{sc}/scenario{sc}_dev.csv"
    rows = list(csv.DictReader(csv_path.open()))
    n = len(rows)
    nt, gap = int(.7 * n), int(.02 * n)
    start, nv = nt + gap, nt // 10
    y = np.array([int(r["unit1_beam"]) - 1 for r in rows])
    seq = np.array([int(r["seq_index"]) for r in rows])
    idx = np.array([int(r["index"]) for r in rows])
    paths = [r["unit1_radar"] for r in rows]
    ct, ce = np.bincount(y[:nt], minlength=64), np.bincount(y[start:], minlength=64)
    p5_prior = float(np.isin(y[start:], np.argsort(-ct)[:5]).mean())
    lab = np.load(RES / f"sc{sc}_bearing_labels.npz")
    assert np.array_equal(lab["beam"], y)
    b = lab["bearing_deg"].astype(float)
    times = []
    for row in rows:
        hms, frac = row["time_stamp"].split("-")
        h, m, s = map(int, hms.split(":"))
        times.append((h * 3600 + m * 60 + s) * 1e6 + int(frac))
    dt_ms = np.diff(times) / 1000
    # An explicit illustrative temporal validity rule, not the original protocol.
    bad = ((seq[start:] != seq[start - 1:-1])
           | (idx[start:] != idx[start - 1:-1] + 1)
           | (dt_ms[start - 1:] <= 0) | (dt_ms[start - 1:] > 200))
    result = {
        "n": n, "optimization_frames": nt - nv, "validation_frames": nv,
        "gap_frames": gap, "test_frames": n - start,
        "train_test_radar_path_overlap": len(set(paths[:nt]) & set(paths[start:])),
        "train_test_sequence_overlap": sorted(map(int, set(seq[:nt]) & set(seq[start:]))),
        "train_sequences": sorted(map(int, set(seq[:nt]))),
        "test_sequences": sorted(map(int, set(seq[start:]))),
        "previous_rows_failing_explicit_temporal_rule": int(bad.sum()),
        "persistence_top1_on_valid_subset": float((y[start:] == y[start - 1:-1])[~bad].mean()),
        "train_prior_top1": float((y[start:] == ct.argmax()).mean()),
        "train_prior_top5": p5_prior,
        "test_oracle_constant_top5": float(np.isin(y[start:], np.argsort(-ce)[:5]).mean()),
        "bearing_train_mean_test_RMSE_deg": float(np.sqrt(np.mean((b[start:] - b[:nt].mean()) ** 2))),
        "bearing_test_optimal_constant_RMSE_deg": float(b[start:].std()),
        "same_sequence_median_row_interval_ms": float(np.median(dt_ms[seq[1:] == seq[:-1]])),
    }
    d = read(f"sc{sc}_resource_ablation_G_max.json")
    chirps = [8, 16, 32, 64, 125, 250]
    medians = [d["settings"][f"chirps={c}"]["top1_median"] for c in chirps]
    for setting in d["settings"].values():
        assert np.isclose(np.median(setting["top1_seeds"]), setting["top1_median"])
    rho = stats.spearmanr(chirps, medians).statistic
    null = [stats.spearmanr(chirps, p).statistic for p in itertools.permutations(medians)]
    result.update({
        "chirp_median_slope_per_doubling": float(np.polyfit(np.log2(chirps), medians, 1)[0]),
        "stored_mean_slope_per_doubling": d["chirp_trend"]["slope_top1_per_log2_chirp"],
        "spearman_rho": float(rho),
        "spearman_asymptotic_p": float(stats.spearmanr(chirps, medians).pvalue),
        "spearman_exact_two_sided_p": float(np.mean(np.abs(null) >= abs(rho) - 1e-12)),
    })
    overhead = {}
    for c in chirps:
        p5 = d["settings"][f"chirps={c}"]["top5_mean"]
        reuse = 25 + 320 * (1 - p5)
        overhead[str(c)] = {
            "top5_mean": p5,
            "paper_model_sunk_sensing_overhead_us": reuse,
            "paper_model_savings_vs_exhaustive_us": 320 - reuse,
            "paper_model_max_incremental_cost_vs_train_prior_us": 320 * (p5 - p5_prior),
            "paper_model_total_at_10p24_us_per_chirp": c * 10.24 + reuse,
        }
    result["airtime_algebra_not_hardware_validation"] = overhead
    result["train_prior_paper_model_overhead_us"] = 25 + 320 * (1 - p5_prior)
    return result


def fim_check():
    m, ns, nc = 4, 16, 8
    n, k, l = np.meshgrid(np.arange(m), np.arange(ns), np.arange(nc), indexing="ij")
    ph = np.exp(2j * np.pi * (.11 * n + .23 * k + .07 * l))
    derivatives = [(2j * np.pi * z * ph).ravel() for z in (n, k, l)]
    derivatives += [ph.ravel(), (1j * ph).ravel()]
    J = np.array([[2 * np.real(np.vdot(a, b)) for b in derivatives] for a in derivatives])
    schur = J[:3, :3] - J[:3, 3:] @ np.linalg.inv(J[3:, 3:]) @ J[3:, :3]
    expected = 8 * np.pi ** 2 * ns * nc * m * (m * m - 1) / 12
    assert np.isclose(schur[0, 0], expected)
    return {
        "raw_frequency_block_nu_fR_cross_term": float(J[0, 1]),
        "amplitude_eliminated_frequency_block_max_off_diagonal":
            float(np.max(np.abs(schur - np.diag(np.diag(schur))))),
        "spatial_information_matches_formula": True,
        "full_five_parameter_sigma_nu": float(np.sqrt(np.linalg.inv(J)[0, 0])),
    }


def exponent_check(nboot):
    d = read("sc31_bearing_mediation_max.json")["settings"]
    settings = [v for v in d.values() if v["n_rx"] == 4]
    R = np.array([v["resource_product_R"] for v in settings], float)
    x = R / R.max()
    values = np.array([v["rmse_deg_seeds"] for v in settings])
    fun = lambda z, a, b, c: a * z ** (-b) + c

    def fit(msq, init=None):
        return optimize.curve_fit(fun, x, msq,
            p0=init if init is not None else [msq.max(), 1., msq.min() / 2],
            bounds=([0, .05, 0], [np.inf, 6., msq.max()]), maxfev=200000)[0]

    point = fit(np.median(values, axis=1) ** 2)
    mean_point = fit(np.mean(values, axis=1) ** 2)
    out = {"median_based_point_b": float(point[1]), "mean_based_point_b": float(mean_point[1]),
           "source_bootstrap_uses_mean_while_point_uses_median": True}
    if nboot:
        rng = np.random.default_rng(20261006)
        for paired in (False, True):
            estimates = []
            for _ in range(nboot):
                ids = rng.integers(values.shape[1], size=(values.shape[1],)
                                   if paired else values.shape)
                sample = values[:, ids] if paired else np.take_along_axis(values, ids, axis=1)
                try:
                    estimates.append(fit(np.median(sample, axis=1) ** 2, point)[1])
                except (ValueError, RuntimeError):
                    pass
            out["joint_seed_median_bootstrap" if paired else "independent_median_bootstrap"] = {
                "resamples": nboot, "converged": len(estimates),
                "percentile95": np.percentile(estimates, [2.5, 97.5]).tolist(),
            }
    return out


def peak_check(sc):
    """Recompute the top-1 peak lookup without modifying the large cache."""
    cache = np.load(RES / f"cache_sc{sc}_ablation_max.npz")
    X, y = cache["4|256|250"], cache["beams"]
    n = len(y)
    nt, start = int(.7 * n), int(.7 * n) + int(.02 * n)
    flat = X[:, 0].reshape(n, -1).argmax(1)
    rr, az = np.unravel_index(flat, (64, 64))
    default = collections.Counter(y[:nt].tolist()).most_common(1)[0][0]
    table = []
    for k in range(64):
        vals = y[:nt][az[:nt] == k].tolist()
        table.append(collections.Counter(vals).most_common(1)[0][0] if vals else default)
    pred = np.array(table)[az[start:]]
    return {
        "all_frames_first_pooled_range_bin_peak_count": int((rr == 0).sum()),
        "all_frames": n,
        "test_first_pooled_range_bin_peak_count": int((rr[start:] == 0).sum()),
        "test_prediction_counts": {str(k): v for k, v in collections.Counter(pred.tolist()).items()},
        "test_top1": float((pred == y[start:]).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--peak-audit", action="store_true")
    args = ap.parse_args()
    out = {"review_date": "2026-10-06", "scenarios": {str(s): check_scenario(s) for s in (31, 33)}}
    m = read("sc31_mediation_analysis_max.json")
    out["partial_correlations"] = {}
    for key in ("partial_logR_gain_given_rmse", "partial_rmse_gain_given_logR"):
        r, old_p = m[key]
        df = 9 - 1 - 2
        out["partial_correlations"][key] = {
            "r": r, "reported_p": old_p, "correct_reference_df": df,
            "correct_reference_p": float(2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r*r)), df)),
        }
    out["nominal_equal_payload_pairs"] = [[256 * 125, 128 * 250], [256 * 64, 64 * 250]]
    out["fim_check"] = fim_check()
    out["exponent_check"] = exponent_check(args.bootstrap)
    if args.peak_audit:
        out["peak_baseline_checks"] = {str(s): peak_check(s) for s in (31, 33)}
    text = (ROOT / "paper_b/PaperB.tex").read_text()
    macros = dict(re.findall(r"\\newcommand\{\\(PB\w+)\}\{([^{}]*)\}",
                            (ROOT / "paper_b/numbers.tex").read_text()))
    abstract = text.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    for key, value in macros.items():
        abstract = abstract.replace("\\" + key, value)
    out["approximate_abstract_word_count"] = len(re.sub(r"\\[A-Za-z]+", "", abstract).split())
    files = [ROOT / "paper_b" / f for f in ("PaperB.tex", "PaperB.pdf", "numbers.tex", "tables.tex")]
    files += [ROOT / "experiments" / f for f in (
        "make_geometry_ceiling.py", "audit_isac_airtime_budget.py", "exp_sc31_resource_ablation.py",
        "analyze_sc31_mediation.py", "make_paper_b_figures.py", "audit_collapse_fix.py")]
    files += [RES / f"sc{s}_resource_ablation_G_max.json" for s in (31, 33)]
    out["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
