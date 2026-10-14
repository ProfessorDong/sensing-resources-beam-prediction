"""Check that every number RENDERED in the Paper A figures matches its source JSON.

`make_paper_a_figures.py` asserts on the way in. This asserts on the way out: it extracts
the text layer from each compiled PDF and confirms the values actually printed there are the
ones in `results/`. That catches a class of error the input assertions cannot -- a stale PDF,
a hand-edit to a `.tex`, or a format string that silently drops a digit.

Run after `make_paper_a_figures.py`.  Exit code 1 if anything fails.
"""
import os, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "paper_a_figures")
RES = os.path.join(HERE, "results")


def text(pdf):
    p = os.path.join(FIG, pdf)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run make_paper_a_figures.py first")
    return subprocess.run(["pdftotext", "-layout", p, "-"],
                          capture_output=True, text=True).stdout


def main():
    fd = json.load(open(os.path.join(FIG, "figure_data.json")))
    t1, t3, t5 = (text("F1_one_line_baselines.pdf"), text("F3_taxonomy.pdf"),
                  text("F5_reference_degeneracy.pdf"))
    t2, t4 = text("F2_four_way_battery.pdf"), text("F4_metric_dominated.pdf")
    t6, t7, t8 = (text("F6_representation_blocked.pdf"), text("F7_healthy_benchmark.pdf"),
                  text("F8_range_stratified.pdf"))
    c = []
    for r in fd["F1"]["rows"]:
        c.append((f"F1 margin {r['benchmark']}", f"{r['margin']:+.4f}" in t1))
    F3 = fd["F3"]
    for k, fmt in (("ps_const", "{:.4f}"), ("ps_model", "{:.4f}"), ("it_thresh", "{:.4f}"),
                   ("it_model", "{:.4f}"), ("rd_bound", "{:.4f}"), ("rd_floor", "{:.4f}"),
                   ("rb_gps", "{:+.4f}"), ("vod_util", "{:+.4f}"), ("vod_slope", "{:+.4f}"),
                   ("hbeam_instr", "{:.4f}"), ("md_ece_blind", "{:.4f}"),
                   ("md_ece_real", "{:.4f}")):
        c.append((f"F3 {k}", fmt.format(F3[k]) in t3))
    F5 = fd["F5"]
    for i, v in enumerate(F5["gt_gain_floor"]):
        c.append((f"F5 gain[{i}]", f"{v:+.4f}" in t5))
    c += [("F5 bound", f"{F5['bound']:.4f}" in t5),
          ("F5 floor", f"{F5['floor']:.4f}" in t5),
          ("F5 exact count", f"{F5['bound_count']:,}" in t5),
          ("F5 n_test", f"{F5['n_test']:,}" in t5),
          ("F5 label classes", str(F5["label_n_distinct"]) in t5),
          ("F1 carries no transcribed published value", "0.3866" not in t1),
          ("F3 prints no '-0.0000'", "-0.0000" not in t3)]

    # the VoD per-ring baselines must still reproduce the published aggregates
    vodr = json.load(open(os.path.join(RES, "vod_range_baselines.json")))
    c.append(("VoD ring baselines reproduce the published aggregate within the "
              "documented pooling gap", vodr["aggregate_cross_check"]["rel_diff"] < 0.03))

    tb = json.load(open(os.path.join(RES, "trivial_baselines.json")))
    mu = json.load(open(os.path.join(RES, "measurement_utilization.json")))
    est = json.load(open(os.path.join(RES, "f1_estimator_audit.json")))
    estc = json.load(open(os.path.join(RES, "f1_estimator_audit_cached.json")))
    rad = json.load(open(os.path.join(RES, "f1_radial_rerun.json")))
    by = {r["benchmark"]: r for r in fd["F1"]["rows"]}
    # Figure 1 must use ONE IoU estimator. The 2026-09-28 audit found it was mixing
    # per-frame (every model number) with pooled (most baselines); these checks make the
    # mixing impossible to reintroduce silently.
    for k, src in (("DeepSense_ISAC", est), ("DeepSense_V2I", est),
                   ("DeepSense_Sc36", estc), ("View_of_Delft_r1", estc)):
        for b in ("constant", "threshold"):
            c.append((f"F1 {k} {b} uses the per-frame estimator",
                      abs(by[k][b] - src[k][b]["per_frame"]) < 1e-12))
    c += [("source ISAC model is the utilization battery",
           abs(by["DeepSense_ISAC"]["model"] - mu["DeepSense_ISAC"]["real_Y"]["IoU"]) < 1e-12),
          ("F1 RADIal model is the 2026-09-28 rerun",
           abs(by["RADIal"]["model"] - rad["model"]["IoU"]) < 1e-12),
          ("F1 RADIal threshold is the 2026-09-28 rerun",
           abs(by["RADIal"]["threshold"] - rad["threshold"]["IoU"]) < 1e-12),
          ("F1 records that the August RADIal tensors do not reproduce",
           rad["fingerprint"]["reproduces_august"] is False),
          ("F1 seed band comes from the one replicated run",
           abs(fd["F1"]["seed_sd"] - 0.00666733610757523) < 1e-12),
          ("F1 still has exactly one win",
           sum(1 for r in fd["F1"]["rows"] if r["model_wins"]) == 1)]

    # ---- F2
    F2 = fd["F2"]
    for k, v in F2["battery"].items():
        c.append((f"F2 real IoU {k}", f"{v['real']:.4f}" in t2))
    for k, v in F2["decomp"].items():
        c.append((f"F2 frame-specific % {k}",
                  f"{100*v['frac_frame_specific']:.0f}\\%" in t2 or
                  f"{100*v['frac_frame_specific']:.0f}%" in t2))
    # ---- F4
    F4 = fd["F4"]
    c += [("F4 width ratio", f"{F4['isac_gap']['width']['ratio']:.2f}" in t4),
          ("F4 mass ratio", f"{F4['isac_gap']['mass']['ratio']:.2f}" in t4),
          ("F4 base rate", f"{F4['radial_base_rate']:.4f}" in t4),
          ("F4 ratio span lo", f"{F4['radial_ratio_worst']:.1f}" in t4),
          ("F4 ratio span hi", f"{F4['radial_ratio_best']:.0f}" in t4)]
    for m, v in F4["radial"].items():
        c.append((f"F4 mean_pred {m}", f"{v['mean_pred']:.3f}" in t4))
    # ---- F6
    spa = json.load(open(os.path.join(RES, "sc36_spatial_readout_audit.json")))
    for t, v in fd["F6"].items():
        c.append((f"F6 bound {t}", f"{v['bound']:.4f}" in t6))
        # the image path must be validated, and the verdict must not depend on the readout
        c.append((f"F6 {t} instrument printed", f"{v['instrument_gain']:+.4f}" in t6))
        c.append((f"F6 {t} instrument clears the bound", v["instrument_valid"] is True))
        c.append((f"F6 {t} instrument matches the audit",
                  abs(v["instrument_gain"] - max(
                      spa["tasks"][t]["results"]["gps_blob_INSTRUMENT/pooled"]["vs_bound"],
                      spa["tasks"][t]["results"]["gps_blob_INSTRUMENT/spatial"]["vs_bound"]))
                  < 1e-9))
        for arm in ("gt_scene", "recon", "radar"):
            for ro in ("pooled", "spatial"):
                c.append((f"F6 {t} {arm}/{ro} stays below the bound",
                          v["spatial_readout"][f"{arm}/{ro}"] <= 0.05))
        c.append((f"F6 {t} mode 4 survives in the audit",
                  spa["tasks"][t]["mode4_survives"] is True))
        for r in v["rows"]:
            c.append((f"F6 {t} {r['name']}", f"{r['gain']:+.4f}" in t6))
    # ---- F7
    F7 = fd["F7"]
    c += [("F7 slope", f"{F7['slope']:+.4f}" in t7),
          ("F7 margin r1", f"{F7['margin'][0]:+.4f}" in t7),
          ("F7 margin r5", f"{F7['margin'][2]:+.4f}" in t7)]
    for pt in F7["pts"]:
        c.append((f"F7 density {pt:.0f}", f"{pt:.0f}" in t7))
    # ---- F8
    F8 = fd["F8"]
    c += [("F8 Sc.36 aggregate constant", f"{F8['sc36']['agg_constant']:.4f}" in t8),
          ("F8 Sc.36 aggregate model", f"{F8['sc36']['agg_radar']:.4f}" in t8),
          ("F8 VoD aggregate threshold", f"{F8['vod']['agg_threshold']:.4f}" in t8),
          ("F8 VoD aggregate model", f"{F8['vod']['agg_radar']:.4f}" in t8),
          ("F8 Sc.36 crossover is far", F8["sc36"]["beats_constant"] == [False, False, True, True]),
          ("F8 VoD crossover is near", F8["vod"]["beats_threshold"] == [True, True, False, False])]

    bad = [n for n, ok in c if not ok]
    for n, ok in c:
        print(f"  {'PASS' if ok else '*** FAIL'}  {n}")
    print(f"\n{len(c)-len(bad)}/{len(c)} rendered values verified against source JSONs")
    if bad:
        print("FAILURES: " + ", ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
