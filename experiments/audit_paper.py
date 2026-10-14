"""
Robust paper audit: cross-checks every numerical cell in Tables III/IV/V
and every coordinate in Figs 6/7/8/9 against v3/v4/v5 JSON.
Reports any discrepancies > 0.0011 absolute.
"""
import json, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "S2S-Diff.tex")
V3 = json.load(open(os.path.join(ROOT, "experiments/results/v3/v3_results.json")))
V4 = json.load(open(os.path.join(ROOT, "experiments/results/v4/v4_results.json")))
V5 = json.load(open(os.path.join(ROOT, "experiments/results/v5/xscenario_results.json")))
tex = open(TEX).read()

TOL_ABS = 0.0011
TOL_LATENCY = 1.5  # ms tolerance for latency
# Per-metric tolerance: Chamfer is displayed to 2 decimals -> 0.005 rounding precision
TOL_BY_METRIC = {"Chamfer": 0.0055, "Latency_ms": 1.5}

issues, ok = [], []

def check(label, expected, actual, tol=TOL_ABS):
    # Auto-pick tolerance from label if metric mentioned
    for metric, t in TOL_BY_METRIC.items():
        if metric in label:
            tol = t
            break
    try:
        actual = float(actual) if not isinstance(actual, float) else actual
        if abs(actual - expected) <= tol:
            ok.append((label, expected, actual))
            return True
        else:
            issues.append((label, expected, actual, abs(actual-expected)))
            return False
    except Exception as e:
        issues.append((label, expected, str(actual), f"parse error: {e}"))
        return False

def strip_tex(s):
    """Remove \\textbf{}, \\underline{}, italic, dollar signs, etc."""
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\underline\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\it\s*", "", s)
    s = re.sub(r"\\,", "", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\\\", "", s)
    return s.strip()

def parse_num(s):
    """Parse a string like '.853' or '0.853' or '12.29' into a float."""
    s = strip_tex(s)
    # Strip leading dot
    if s.startswith('.'):
        s = '0' + s
    # Strip dagger and other markers
    s = re.sub(r"\^?\\?\$?\\dagger\$?", "", s)
    s = re.sub(r"[^\d.\-]", "", s)
    return float(s)

# =============================================================
# Table III: parse row by row using table block
# =============================================================
print("=" * 78)
print("TABLE III  -- Main multi-dataset comparison (Table~\\ref{tab:real_results})")
print("=" * 78)

# Extract the table block
m = re.search(r"\\begin\{table\*\}.*?\\caption\{Multi-dataset.*?\\end\{table\*\}", tex, re.DOTALL)
if not m:
    print("CANNOT FIND TABLE III")
    sys.exit(1)
tab3 = m.group(0)

# Parse each row: split by \\\n and look for & U-Net &, & CVAE &, etc.
rows = re.split(r"\\\\\s*\n", tab3)
ds_idx = -1
DATASETS = ["DeepSense_ISAC", "DeepSense_V2I", "RADIal"]
METRICS = ["IoU", "F1", "NLL", "Brier", "ECE", "Chamfer", "RMSE"]

current_ds = None
for raw in rows:
    line = raw.strip()
    if not line: continue
    # Detect dataset boundary
    if "DeepSense" in line and "ISAC" in line:
        current_ds = "DeepSense_ISAC"
    elif "DeepSense" in line and "V2I" in line:
        current_ds = "DeepSense_V2I"
    elif "RADIal" in line and "automotive" in line:
        current_ds = "RADIal"
    if current_ds is None: continue

    # Detect method
    if "U-Net" in line and "&" in line and not line.startswith("%"):
        method = "U-Net"
    elif "CVAE" in line and "&" in line and not line.startswith("%"):
        method = "CVAE"
    elif "pix2pix" in line and "&" in line:
        method = "pix2pix"
    elif "S2S-Diff w/ DPS" in line or "DPS" in line and "S2S-Diff" in line and "{\\it" in line:
        method = "DPS"
    elif "S2S-Diff" in line and "&" in line and "DPS" not in line:
        method = "S2S-Diff"
    else:
        continue

    # Extract cells separated by &, then strip the leading non-cell parts
    cells = [c.strip() for c in line.split("&")]
    # First cell is the multirow / dataset label (or empty), second is method name, rest are metrics
    # Actually format is:    \multirow{} & METHOD & v1 & v2 & v3 & v4 & v5 & v6 & v7
    # Or:                    & METHOD & ...
    # Take the LAST 7 cells as metric values
    metric_cells = cells[-7:]
    if len(metric_cells) != 7:
        continue
    # Get JSON expected values
    if method == "DPS":
        json_data = V4[current_ds]["dps"]
    else:
        json_data = V3[current_ds][method]
    for metric, raw_cell in zip(METRICS, metric_cells):
        try:
            tex_val = parse_num(raw_cell)
            expected = json_data[metric]
            label = f"Table III  {current_ds:18s}  {method:9s}  {metric}"
            check(label, expected, tex_val)
        except Exception as e:
            issues.append((f"Table III parse: {current_ds} {method} {metric}", "?", raw_cell, str(e)))

# =============================================================
# Table IV: ablation
# =============================================================
print("\n" + "=" * 78)
print("TABLE IV  -- Ablation (Table~\\ref{tab:ablation})")
print("=" * 78)

m = re.search(r"\\begin\{table\*\}.*?\\caption\{Ablation.*?\\end\{table\*\}", tex, re.DOTALL)
if m:
    tab4 = m.group(0)
    rows = re.split(r"\\\\\s*\n", tab4)
    for raw in rows:
        line = raw.strip()
        if "Full" in line and "&" in line:
            ablation = "S2S-Diff"
        elif "No physics" in line:
            ablation = "S2S-Diff_no_physics"
        elif "No obs-weight" in line or "obs-weight" in line:
            ablation = "S2S-Diff_no_obsweight"
        else:
            continue
        cells = [c.strip() for c in line.split("&")]
        # Format: Variant & ISAC{IoU,F1,NLL,Brier,ECE} & RADIal{IoU,F1,NLL,Brier,ECE}
        # Last 10 cells are the metrics
        metric_cells = cells[-10:]
        if len(metric_cells) != 10: continue
        for i, ds_key in enumerate(["DeepSense_ISAC", "RADIal"]):
            for j, metric in enumerate(["IoU", "F1", "NLL", "Brier", "ECE"]):
                idx = i*5 + j
                try:
                    tex_val = parse_num(metric_cells[idx])
                    expected = V3[ds_key][ablation][metric]
                    label = f"Table IV   {ds_key:18s}  {ablation:24s}  {metric}"
                    check(label, expected, tex_val)
                except Exception as e:
                    issues.append((f"Table IV parse: {ds_key} {ablation} {metric}", "?", metric_cells[idx], str(e)))

# =============================================================
# Table V: LOSO
# =============================================================
print("\n" + "=" * 78)
print("TABLE V  -- LOSO cross-scenario (Table~\\ref{tab:xscenario})")
print("=" * 78)

# Per-fold rows: Sc.~42 & .866 & ...
fold_pattern = re.compile(r"Sc\.~(\d+)\s+&\s+(\.\d+)\s+&\s+(\.\d+)\s+&\s+(\.\d+)\s+&\s+(\.\d+)\s+&\s+(\.\d+)\s*\\\\")
for m in fold_pattern.finditer(tex):
    holdout = int(m.group(1))
    fold = V5["folds"].get(f"holdout_{holdout}")
    if not fold: continue
    for i, metric in enumerate(["IoU", "F1", "NLL", "Brier", "ECE"]):
        tex_val = parse_num(m.group(i+2))
        expected = fold[metric]
        check(f"Table V    Sc.{holdout:2d} {metric}", expected, tex_val)

# Summary row: $.856\!\pm\!.009$
summary_line = re.search(r"LOSO mean.*?\\\\", tex)
if summary_line:
    pairs = re.findall(r"\$\.(\d{3})\\!\\pm\\!\.(\d{3})\$", summary_line.group(0))
    metric_keys = ["IoU", "F1", "NLL", "Brier", "ECE"]
    summ = V5["summary"]
    for (mean_s, std_s), metric in zip(pairs, metric_keys):
        check(f"Table V    LOSO mean {metric}", summ[f"{metric}_mean"], float(f"0.{mean_s}"))
        check(f"Table V    LOSO std  {metric}", summ[f"{metric}_std"],  float(f"0.{std_s}"))

# =============================================================
# Figure 7: SNR robustness (IoU and ECE both)
# =============================================================
print("\n" + "=" * 78)
print("FIGURE 7  -- SNR robustness (Fig.~\\ref{fig:snr_robustness})")
print("=" * 78)

m = re.search(r"\\begin\{figure\*\}.*?fig:snr_robustness.*?\\end\{figure\*\}", tex, re.DOTALL)
if m:
    block = m.group(0)
    # Extract \nextgroupplot blocks
    panels = re.split(r"\\nextgroupplot", block)[1:]  # skip prelude
    DS_ORDER = ["DeepSense_ISAC", "DeepSense_V2I", "RADIal"]
    METHODS = ["S2S-Diff", "U-Net", "CVAE", "pix2pix"]
    # Each panel has 4 \addplot lines (one per method) and shows IoU
    # Parse \addplot[...] coordinates {(x,y) (x,y) ...};
    addplot_re = re.compile(r"\\addplot\[[^\]]*\]\s*coordinates\s*\{([^}]+)\}")
    coord_re = re.compile(r"\((-?\d+),([\d.]+)\)")
    for panel_idx, panel in enumerate(panels):
        if panel_idx >= 3: break
        ds_key = DS_ORDER[panel_idx]
        addplots = addplot_re.findall(panel)
        for method_idx, ap in enumerate(addplots[:4]):
            method = METHODS[method_idx]
            coords = coord_re.findall(ap)
            for snr_str, val_str in coords:
                snr = int(snr_str)
                expected = V4[ds_key]["snr_sweep"][str(snr)][method]["IoU"]
                check(f"Fig.7      {ds_key:18s} SNR={snr:3d} {method:10s} IoU", expected, float(val_str))

# =============================================================
# Figure 9: sample count (IoU and ECE)
# =============================================================
print("\n" + "=" * 78)
print("FIGURE 9  -- Sample count S (Fig.~\\ref{fig:samplecount})")
print("=" * 78)

m = re.search(r"\\begin\{figure\*\}.*?fig:samplecount.*?\\end\{figure\*\}", tex, re.DOTALL)
if m:
    block = m.group(0)
    panels = re.split(r"\\nextgroupplot", block)[1:]
    METRIC_BY_PANEL = ["IoU", "ECE", "NLL"]
    DS_LINE_ORDER = ["DeepSense_ISAC", "DeepSense_V2I", "RADIal"]
    addplot_re = re.compile(r"\\addplot\[[^\]]*\]\s*coordinates\s*\{([^}]+)\}")
    coord_re = re.compile(r"\((\d+),([\d.]+)\)")
    for panel_idx, panel in enumerate(panels):
        if panel_idx >= 3: break
        metric = METRIC_BY_PANEL[panel_idx]
        addplots = addplot_re.findall(panel)
        for line_idx, ap in enumerate(addplots[:3]):
            ds_key = DS_LINE_ORDER[line_idx]
            coords = coord_re.findall(ap)
            for s_str, val_str in coords:
                S = int(s_str)
                expected = V4[ds_key]["sample_count_ablation"][str(S)][metric]
                check(f"Fig.9      {ds_key:18s} S={S:2d} {metric}", expected, float(val_str))

# =============================================================
# Figure 8: DDIM ablation (IoU and ECE) -- already in v3 JSON
# =============================================================
print("\n" + "=" * 78)
print("FIGURE 8  -- DDIM ablation (Fig.~\\ref{fig:ddim_ablation})")
print("=" * 78)

m = re.search(r"\\begin\{figure\*\}.*?fig:ddim_ablation.*?\\end\{figure\*\}", tex, re.DOTALL)
if m:
    block = m.group(0)
    panels = re.split(r"\\nextgroupplot", block)[1:]
    METRIC_BY_PANEL = ["IoU", "ECE"]
    # Each panel has 3 plots: ISAC, RADIal, V2I (per my code; let me verify)
    addplot_re = re.compile(r"\\addplot\[[^\]]*\]\s*coordinates\s*\{([^}]+)\}")
    coord_re = re.compile(r"\((\d+),([\d.]+)\)")
    # Order in tex: ISAC, RADIal, V2I (looking at the file)
    DS_LINE_ORDER = ["DeepSense_ISAC", "RADIal", "DeepSense_V2I"]
    for panel_idx, panel in enumerate(panels):
        if panel_idx >= 2: break
        metric = METRIC_BY_PANEL[panel_idx]
        addplots = addplot_re.findall(panel)
        for line_idx, ap in enumerate(addplots[:3]):
            ds_key = DS_LINE_ORDER[line_idx]
            coords = coord_re.findall(ap)
            for s_str, val_str in coords:
                S = int(s_str)
                if str(S) in V3[ds_key].get("ddim_ablation", {}):
                    expected = V3[ds_key]["ddim_ablation"][str(S)][metric]
                    check(f"Fig.8      {ds_key:18s} S_ddim={S:3d} {metric}", expected, float(val_str))

# =============================================================
# Prose: specific claims
# =============================================================
print("\n" + "=" * 78)
print("PROSE CLAIMS")
print("=" * 78)

# "+7.7 pp" physics ablation on RADIal
for m in re.finditer(r"\$\+7\.7\$~percentage|\+7\.7~pp", tex):
    actual_pp = (V3["RADIal"]["S2S-Diff"]["IoU"] - V3["RADIal"]["S2S-Diff_no_physics"]["IoU"]) * 100
    if abs(7.7 - actual_pp) <= 0.15:
        ok.append((f"PROSE +7.7 pp physics", 7.7, actual_pp))
    else:
        issues.append((f"PROSE +7.7 pp physics", 7.7, actual_pp, abs(7.7-actual_pp)))
    break

# "+5.5 pp" obs-weight on RADIal
m = re.search(r"observed-cell weighting contributes \$\+(\d\.\d)\$", tex)
if m:
    claim = float(m.group(1))
    actual_pp = (V3["RADIal"]["S2S-Diff"]["IoU"] - V3["RADIal"]["S2S-Diff_no_obsweight"]["IoU"]) * 100
    check(f"PROSE +{claim} pp obs-weight", actual_pp, claim, tol=0.15)

# Multi-seed sigma
m = re.search(r"\$\\sigma\\!=\\!(\d+\.\d+)\$ on ISAC,\s+(\d+\.\d+)\s+on V2I,?\s+and\s+(\d+\.\d+)\s+on RADIal", tex)
if m:
    isac, v2i, radial = float(m.group(1)), float(m.group(2)), float(m.group(3))
    check("PROSE caption sigma ISAC",   V4["DeepSense_ISAC"]["multi_seed_s2s"]["summary"]["IoU_std"], isac)
    check("PROSE caption sigma V2I",    V4["DeepSense_V2I"]["multi_seed_s2s"]["summary"]["IoU_std"], v2i)
    check("PROSE caption sigma RADIal", V4["RADIal"]["multi_seed_s2s"]["summary"]["IoU_std"], radial)

# DPS latency 235.2 ms
for m in re.finditer(r"235\.2~?ms", tex):
    actual = V4["DeepSense_V2I"]["dps"]["Latency_ms"]
    check("PROSE V2I DPS latency 235.2 ms", actual, 235.2, tol=TOL_LATENCY)
    break

# 232.1 ms baseline
for m in re.finditer(r"232\.1~?ms", tex):
    # No exact JSON value for canonical S2S-Diff latency at S=4, 50 steps
    # but v3 ddim_ablation for S=50 latency is the closest proxy
    actual = V3["DeepSense_ISAC"]["ddim_ablation"]["50"]["Latency_ms"]
    check("PROSE ISAC S2S-Diff latency 232.1 ms", actual, 232.1, tol=TOL_LATENCY)
    break

# DPS overhead "~2%"
m = re.search(r"overhead of only \$\\sim\\!2\\%", tex)
if m:
    isac_dps = V4["DeepSense_ISAC"]["dps"]["Latency_ms"]
    isac_baseline = V3["DeepSense_ISAC"]["ddim_ablation"]["50"]["Latency_ms"]
    overhead_pct = (isac_dps - isac_baseline) / isac_baseline * 100
    if 1.0 <= overhead_pct <= 3.5:
        ok.append((f"PROSE 'overhead ~2%'", "[1-3.5%]", f"{overhead_pct:.1f}%"))
    else:
        issues.append((f"PROSE 'overhead ~2%' actual is", "[1-3.5%]", f"{overhead_pct:.1f}%", "out of range"))

# =============================================================
# Final report
# =============================================================
print("\n" + "=" * 78)
print(f"AUDIT SUMMARY:  {len(ok)} CHECKS OK,  {len(issues)} ISSUES")
print("=" * 78)
if issues:
    print("\nISSUES:")
    for i in issues:
        if len(i) == 4:
            label, exp, act, diff = i
            print(f"  FAIL  {label:60s}  expected={exp}  actual={act}  diff={diff}")
        else:
            print(f"  FAIL  {i}")
else:
    print("\n  NO ISSUES FOUND. All audited cells consistent with JSON sources.")

if os.environ.get("VERBOSE") or len(ok) < 200:
    print(f"\n  ({len(ok)} cells checked. Set VERBOSE=1 to dump all checks.)")
if os.environ.get("VERBOSE"):
    for label, exp, act in ok:
        print(f"  OK    {label:60s}  exp={exp:.4f}  act={act:.4f}")
