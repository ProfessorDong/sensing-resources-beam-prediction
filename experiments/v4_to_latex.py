"""
Convert experiments/results/v4/v4_results.json into LaTeX-ready TikZ
coordinate strings and table snippets.

Usage:  python experiments/v4_to_latex.py
Outputs: experiments/results/v4/latex_snippets.txt
"""
import json, os, sys, math

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results/v4/v4_results.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "results/v4/latex_snippets.txt")

DATASETS = [
    ("DeepSense_ISAC", "ISAC indoor"),
    ("DeepSense_V2I",  "V2I outdoor"),
    ("RADIal",         "RADIal"),
]
METHODS = ["S2S-Diff", "U-Net", "CVAE", "pix2pix"]
COLORS = {
    "S2S-Diff": "okabeBlue",
    "U-Net":    "okabeOrange",
    "CVAE":     "okabeGreen",
    "pix2pix":  "okabePurple",
}
MARKS = {"S2S-Diff": "*", "U-Net": "square*", "CVAE": "triangle*", "pix2pix": "diamond*"}


def fmt_num(x, prec=3):
    if abs(x) >= 100: return f"{x:.1f}"
    if abs(x) >= 10:  return f"{x:.2f}"
    return f"{x:.{prec}f}"


def coords_str(xs, ys):
    return " ".join(f"({x},{y:.6f})" for x, y in zip(xs, ys))


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: {RESULTS_PATH} not found")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        R = json.load(f)

    out = []
    out.append("% v4 -> LaTeX snippets, generated automatically.")
    out.append("% Drop-in coordinates and table cells for the manuscript.")
    out.append("")

    # ----- (1) Reliability diagrams -----
    out.append("% =========== (1) RELIABILITY DIAGRAM COORDINATES ===========")
    for ds_key, ds_label in DATASETS:
        ds = R.get(ds_key)
        if not ds or "reliability" not in ds:
            continue
        out.append(f"%% {ds_label}")
        for method in METHODS:
            mdata = ds["reliability"].get(method)
            if not mdata: continue
            xs = mdata["predicted_means"]
            ys = mdata["empirical_freqs"]
            out.append(f"%% {method}:")
            out.append(f"\\addplot[color={COLORS[method]}, mark={MARKS[method]}, mark size=1.4pt] coordinates {{{coords_str(xs, ys)}}};")
        out.append("")

    # ----- (2) DPS row for Table III -----
    out.append("% =========== (2) DPS ROW (Table III) ===========")
    out.append("% Format: IoU / F1 / NLL / Brier / ECE / Cham / RMSE")
    for ds_key, ds_label in DATASETS:
        ds = R.get(ds_key)
        if not ds or "dps" not in ds:
            continue
        d = ds["dps"]
        cells = [fmt_num(d["IoU"]), fmt_num(d["F1"]),
                 fmt_num(d["NLL"]), fmt_num(d["Brier"]),
                 fmt_num(d["ECE"]), fmt_num(d["Chamfer"], prec=2),
                 fmt_num(d["RMSE"])]
        lat = fmt_num(d.get("Latency_ms", 0), prec=1)
        out.append(f"% {ds_label} DPS row (latency={lat}ms):")
        out.append(" & ".join([" & S2S-Diff w/ DPS"] + cells) + r" \\")
    out.append("")

    # ----- (3) Multi-seed S2S-Diff mean ± std -----
    out.append("% =========== (3) MULTI-SEED S2S-DIFF mean ± std ===========")
    for ds_key, ds_label in DATASETS:
        ds = R.get(ds_key)
        if not ds or "multi_seed_s2s" not in ds:
            continue
        s = ds["multi_seed_s2s"]["summary"]
        out.append(f"% {ds_label}:")
        for k in ["IoU", "F1", "NLL", "Brier", "ECE"]:
            mu = s[f"{k}_mean"]; sd = s[f"{k}_std"]
            out.append(f"%   {k}: {fmt_num(mu)} \\pm {fmt_num(sd)}")
        out.append("")

    # ----- (4) Posterior coverage table -----
    out.append("% =========== (4) POSTERIOR COVERAGE ===========")
    out.append("% nominal alpha=0.10/0.05/0.01  -> empirical hits")
    for ds_key, ds_label in DATASETS:
        ds = R.get(ds_key)
        if not ds or "coverage" not in ds:
            continue
        c = ds["coverage"]
        out.append(f"% {ds_label}: alpha=0.10 -> {c.get('alpha=0.10', 0):.3f},  alpha=0.05 -> {c.get('alpha=0.05', 0):.3f},  alpha=0.01 -> {c.get('alpha=0.01', 0):.3f}")
    out.append("")

    # ----- (5) Sample count S ablation -----
    out.append("% =========== (5) SAMPLE COUNT S ABLATION ===========")
    for ds_key, ds_label in DATASETS:
        ds = R.get(ds_key)
        if not ds or "sample_count_ablation" not in ds:
            continue
        sca = ds["sample_count_ablation"]
        ss = sorted(int(k) for k in sca.keys())
        for metric in ["IoU", "ECE", "NLL"]:
            ys = [sca[str(s)][metric] for s in ss]
            out.append(f"%% {ds_label} -- {metric}:")
            out.append(f"\\addplot[color={COLORS['S2S-Diff']}, mark={MARKS['S2S-Diff']}] coordinates {{{coords_str(ss, ys)}}};")
        out.append("")

    # ----- (6) SNR sweep -----
    out.append("% =========== (6) SNR ROBUSTNESS SWEEP ===========")
    for ds_key, ds_label in DATASETS:
        ds = R.get(ds_key)
        if not ds or "snr_sweep" not in ds:
            continue
        out.append(f"%% {ds_label}:")
        snrs = sorted(int(k) for k in ds["snr_sweep"].keys())
        for method in METHODS:
            for metric in ["IoU", "ECE"]:
                ys = [ds["snr_sweep"][str(s)][method][metric] for s in snrs]
                out.append(f"%%   {method} -- {metric}:")
                out.append(f"\\addplot[color={COLORS[method]}, mark={MARKS[method]}] coordinates {{{coords_str(snrs, ys)}}};")
        out.append("")

    # ----- summary stats useful for prose -----
    out.append("% =========== (X) PROSE-USEFUL SUMMARIES ===========")
    for ds_key, ds_label in DATASETS:
        ds = R.get(ds_key)
        if not ds: continue
        # SNR=10 vs SNR=-5 calibration retention
        if "snr_sweep" in ds:
            snrs = sorted(int(k) for k in ds["snr_sweep"].keys())
            out.append(f"%% {ds_label}:")
            for method in METHODS:
                if "S2S-Diff" not in method and "U-Net" not in method:
                    continue
                hi = ds["snr_sweep"][str(snrs[-1])][method]  # highest SNR (best)
                lo = ds["snr_sweep"][str(snrs[0])][method]   # lowest SNR (worst)
                out.append(f"%%   {method}: IoU {fmt_num(hi['IoU'])}->{fmt_num(lo['IoU'])} ({fmt_num(snrs[-1])}->{fmt_num(snrs[0])} dB)")
                out.append(f"%%   {method}: ECE {fmt_num(hi['ECE'])}->{fmt_num(lo['ECE'])}")
        # Sample count knee
        if "sample_count_ablation" in ds:
            sca = ds["sample_count_ablation"]
            ss = sorted(int(k) for k in sca.keys())
            out.append(f"%%   sample count ECE: " +
                       ", ".join(f"S={s}->{fmt_num(sca[str(s)]['ECE'])}" for s in ss))

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(out))
    print(f"Wrote {OUT_PATH}  ({len(out)} lines)")


if __name__ == "__main__":
    main()
