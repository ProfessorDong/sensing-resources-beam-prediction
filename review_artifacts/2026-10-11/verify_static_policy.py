#!/usr/bin/env python
"""Independent re-derivation of the power-aware static policy and everything the paper says about it.

Written against the raw CSVs, the published per-beam power files and the saved shortlist
rankings. It imports no module from experiments/, so agreement with
`audit_static_shortlist.py` is a reproduction and not a shared bug, and it compares its own
numbers against the manuscript's macros rather than against that script's JSON.

Read-only. Exit code is the result.
"""
import collections, csv, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "datasets", "deepsense6g")
RES = os.path.join(ROOT, "experiments", "results")
PB = os.path.join(ROOT, "paper_b")
NBEAM, KS, TSEC = 64, (1, 2, 3, 5, 8, 16), 5.0
KS_ALL = tuple(range(1, 17))   # the selection searches every length (2026-10-13)
checks = []


def ck(cond, label):
    checks.append((bool(cond), label))


def macros():
    out, txt = {}, ""
    for f in ("numbers.tex", "numbers_audit.tex"):
        txt += open(os.path.join(PB, f)).read()
    for m in re.finditer(r"\\newcommand\{\\(PB\w+)\}\{", txt):
        i, d = m.end(), 1
        while i < len(txt) and d:
            d += (txt[i] == "{") - (txt[i] == "}")
            i += 1
        out[m.group(1)] = txt[m.end():i - 1]
    return out


M = macros()


def load(sc):
    rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}",
                                                 f"scenario{sc}_dev.csv"))))
    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    seq = np.array([int(r["seq_index"]) for r in rows])
    P = np.array([np.loadtxt(os.path.join(DS, f"Scenario{sc}",
                                          r["unit1_pwr_60ghz"].lstrip("./"))) for r in rows])
    return y, seq, P, np.isfinite(P).all(1)


def greedy_direct(P, n):
    """The objective as written in the paper, with no log identity and no shortcut."""
    out = []
    for _ in range(n):
        cand = [b for b in range(NBEAM) if b not in out]
        r = [float(np.mean(10.0 * np.log10(P.max(1) / P[:, out + [b]].max(1))))
             for b in cand]
        out.append(cand[int(np.argmin(r))])
    return out


def main():
    for sc, tag in ((31, "One"), (33, "Three")):
        y, seq, P, fin = load(sc)
        z = np.load(os.path.join(RES, f"beam_shortlists_sc{sc}.npz"))
        n, ntr, nte = len(y), int(z["n_train"]), int(z["n_test"])
        nval = max(1, ntr // 10)
        val = np.arange(ntr - nval, ntr)
        te = np.arange(n - nte, n)
        ff = np.arange(0, ntr - nval)[fin[np.arange(0, ntr - nval)]]
        fv = val[fin[val]]
        ft = te[fin[te]]
        ck(len(ff) == int(M[f"PBStFit{tag}"].replace(",", "")),
           f"Sc.{sc}: the fitting rows are the finite-power optimization rows")
        order = greedy_direct(P[ff], 16)

        def sc_(rows, k):
            L = 10.0 * np.log10(P[rows].max(1) / P[rows][:, order[:k]].max(1))
            return L.mean(), (L > 3.0).mean(), (L <= 1e-12).mean()

        ksel = int(M[f"PBStK{tag}"])
        mr, ex, inc = sc_(ft, ksel)
        ck(abs(mr - float(M[f"PBStReg{tag}"])) < 5e-4,
           f"Sc.{sc}: the selected static policy's test mean regret, {mr:.4f} dB")
        ck(abs(100 * ex - float(M[f"PBStExc{tag}"])) < 5e-3,
           f"Sc.{sc}: its exceedance, {100 * ex:.4f}%")
        # top-1 ON EVERY TEST LABEL, which is the mask the bound uses. Scoring it on the
        # finite-power subset, as this check did until 2026-10-13, compares two populations
        # and let a fixed single beam appear to beat the best constant.
        i1 = float(np.isin(y[te], order[:1]).mean())
        ck(abs(i1 - float(M[f"PBStTopOne{tag}"])) < 5e-5,
           f"Sc.{sc}: the static policy's top-1 on all {len(te)} labels, {i1:.4f}")
        A = json.load(open(os.path.join(
            RES, f"sc{sc}_resource_ablation_G_max.json")))
        bound_all = float(collections.Counter(
            y[te].tolist()).most_common(1)[0][1]) / len(te)
        ck(abs(bound_all - A["bound_const"]) < 5e-4,
           f"Sc.{sc}: the all-label bound re-derived here, {bound_all:.4f}")
        ck(i1 <= bound_all + 1e-12,
           f"Sc.{sc}: on matched labels the static top-1 cannot exceed the best constant")
        # the selection: among every policy needing no side information, the cheapest that
        # meets the validation target. Rebuilt here from the saved rankings, not read.
        Q = json.load(open(os.path.join(RES, "beam_power_quality.json")))
        F = Q["scenarios"][str(sc)]["frontier"]
        tgt = F["selection"]["dedicated"]["validation_target_db"]
        # EVERY shortlist length from 1 to 16, for radar and static alike. The saved
        # rankings carry all sixteen, so restricting either side to the six the figures
        # quote samples the candidate set instead of searching it.
        best = None
        for k in KS_ALL:
            v = 10.0 * np.log10(P[fv].max(1) / P[fv][:, order[:k]].max(1)).mean()
            if v <= tgt + 1e-12 and (best is None or TSEC * k < best[1]):
                best = ("static", TSEC * k, k)
        frame_us = {p_["setting"]: p_["cost_dedicated_us"] - TSEC * p_["k"]
                    for p_ in F["points"]}
        for setting, fr in sorted(frame_us.items()):
            slv = z[f"spatial|{setting}|val_shortlist"]
            for k in KS_ALL:
                v = float(np.median([float(np.mean(10.0 * np.log10(
                    P[fv].max(1) / np.take_along_axis(
                        P[fv], slv[si][fin[val]][:, :k].astype(np.int64), 1).max(1))))
                    for si in range(slv.shape[0])]))
                c = fr + TSEC * k
                if v <= tgt + 1e-12 and (best is None or c < best[1]):
                    best = ("radar", c, k)
        ck(best is not None and best[0] == "static" and best[2] == ksel,
           f"Sc.{sc}: the cheapest admissible dedicated policy is the static set at "
           f"k={ksel}, not a radar setting (got {best})")
        ck(abs(best[1] - float(M[f"PBStCost{tag}"])) < 0.5,
           f"Sc.{sc}: its modeled cost, {best[1]:.1f} us")
        # the crossover the prose describes: radar ahead at k=1, behind in the tail by k=5
        full = {p_["k"]: p_ for p_ in F["points"] if p_["setting"] == "chirps=250"}
        m1, e1, _ = sc_(ft, 1)
        m5, e5, i5 = sc_(ft, 5)
        ck(full[1]["test"]["mean_regret_db"] < m1,
           f"Sc.{sc}: the radar leads the static set at one probe")
        ck(e5 < full[5]["test"]["exceed_3dB"],
           f"Sc.{sc}: the static set has the lighter tail by five probes")
        ck(i5 < full[5]["test"]["inclusion"],
           f"Sc.{sc}: and holds the best beam less often while doing it")
        # the route diagnostic
        hind = greedy_direct(P[ft], ksel)
        Lh = 10.0 * np.log10(P[ft].max(1) / P[ft][:, hind[:ksel]].max(1)).mean()
        ck(abs((mr - Lh) - float(M[f"PBStGapSel{tag}"])) < 5e-4,
           f"Sc.{sc}: the gap to a test-fitted static set, {mr - Lh:.4f} dB")
        # persistence and the radar must be scored on the same rows where the prose says so
        pers = Q["scenarios"][str(sc)]["comparators"]["persistence"]["eligible_rows"]["1"]
        ck(abs(pers["top_k_inclusion"] - float(M[f"PBQPersMatch{tag}"])) < 5e-5,
           f"Sc.{sc}: persistence on the matched rows")
        ck(pers["n_eval"] == Q["scenarios"][str(sc)]["n_eval_eligible"],
           f"Sc.{sc}: those rows are the eligible power-valid ones")

    npass = sum(1 for ok, _ in checks if ok)
    for ok, lab in checks:
        if not ok:
            print(f"  FAIL  {lab}")
    print(f"\n{npass}/{len(checks)} independent checks passed")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
