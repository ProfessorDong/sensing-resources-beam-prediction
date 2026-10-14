"""Independent, read-only re-derivation of the beam-quality numbers added 2026-10-09.

Written against the saved shortlists and the raw CSV/power files, deliberately NOT importing
audit_beam_power_regret.py, so that a shared bug in the analysis cannot hide here. Prints
"ok" or "FAIL" per quantity and exits non-zero on any failure.
"""
import csv, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "results")
DS = os.path.join(ROOT, "datasets", "deepsense6g")
Q = json.load(open(os.path.join(RES, "beam_power_quality.json")))
MAC = open(os.path.join(ROOT, "paper_b", "numbers_audit.tex")).read()
fails = []


def ck(cond, msg, got=None, want=None):
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}" + ("" if cond else f"   got {got} want {want}"))
    if not cond:
        fails.append(msg)


def macro(name):
    import re
    m = re.search(r"\\newcommand\{\\PB" + name + r"\}\{([^}]*)\}", MAC)
    assert m, name
    return m.group(1)


for sc in ("31", "33"):
    print(f"\n=== Scenario {sc} ===")
    tag = "One" if sc == "31" else "Three"
    rows = list(csv.DictReader(open(os.path.join(DS, f"Scenario{sc}", f"scenario{sc}_dev.csv"))))
    n = len(rows)
    ntr = int(0.70 * n)
    start = ntr + int(0.02 * n)
    y = np.array([int(float(r["unit1_beam"])) for r in rows]) - 1
    seq = np.array([int(r["seq_index"]) for r in rows])
    idx = np.array([int(r["index"]) for r in rows])
    t = []
    for r in rows:
        hms, frac = r["time_stamp"].split("-")
        h, m, s_ = (int(x) for x in hms.split(":"))
        t.append((h * 3600 + m * 60 + s_) * 1e6 + int(frac))
    dt = np.diff(np.array(t, float)) / 1000.0
    P = np.array([np.loadtxt(os.path.join(DS, f"Scenario{sc}",
                                          r["unit1_pwr_60ghz"].lstrip("./"))) for r in rows])
    fin = np.isfinite(P).all(1)
    rec = Q["scenarios"][sc]

    # 1. the label is the argmax of the power vector, which is what licenses the metric
    ck(bool((P[fin].argmax(1) == y[fin]).all()), "every label is the argmax of its power vector")
    ck(rec["n_test_nonfinite_power"] == int((~fin[start:]).sum()),
       "count of dropped non-finite test frames",
       rec["n_test_nonfinite_power"], int((~fin[start:]).sum()))
    ck(rec["n_eval_all"] == int(fin[start:].sum()), "evaluated frame count")
    ck(int(macro(f"QNDrop{tag}")) == int((~fin[start:]).sum()), "macro QNDrop")
    ck(int(macro(f"QNEval{tag}")) == int(fin[start:].sum()), "macro QNEval")
    ck(rec["n_test_sequences"] == len(set(seq[start:].tolist())), "test sequence count")
    ck(int(macro(f"QNSeq{tag}")) == len(set(seq[start:].tolist())), "macro QNSeq")

    m = fin[start:]
    Pt, yt = P[start:], y[start:]

    def regret(sl):
        best = Pt[m].max(1)
        got = np.take_along_axis(Pt[m], sl[m].astype(np.int64), 1).max(1)
        return 10 * np.log10(best / got)

    # 2. the fixed prior needs no model, so it is fully re-derivable here
    import collections
    prior = [b for b, _ in collections.Counter(y[:ntr].tolist()).most_common(5)]
    sl = np.tile(np.array(prior), (len(yt), 1))
    L = regret(sl)
    want = rec["comparators"]["fixed training prior"]["all_rows"]["5"]
    ck(abs(L.mean() - want["regret_mean_db"]) < 1e-9, "fixed-prior mean regret",
       L.mean(), want["regret_mean_db"])
    ck(abs((L > 3.0).mean() - want["outage_3dB"]) < 1e-12, "fixed-prior 3 dB outage")
    ck(abs(float((sl[m][:, 0] == yt[m]).mean() if False else (sl[m] == yt[m][:, None]).any(1).mean())
           - want["top_k_inclusion"]) < 1e-12, "fixed-prior top-5 inclusion")
    ck(abs(float(L.mean()) - float(macro(f"QRegPrior{tag}"))) < 5e-4, "macro QRegPrior",
       L.mean(), macro(f"QRegPrior{tag}"))

    # 3. persistence on its own eligible rows, also model-free
    elig = np.zeros(n, bool)
    elig[1:] = ((seq[1:] == seq[:-1]) & (idx[1:] == idx[:-1] + 1) & (dt > 0) & (dt <= 200))
    prev = np.concatenate([[y[0]], y[:-1]])
    offs = [0, -1, 1, -2, 2]
    psl = np.stack([np.clip(prev + d, 0, 63) for d in offs], 1)[start:]
    me = m & elig[start:]
    best = Pt[me].max(1)
    got = np.take_along_axis(Pt[me], psl[me].astype(np.int64), 1).max(1)
    Lp = 10 * np.log10(best / got)
    wp = rec["comparators"]["persistence"]["eligible_rows"]["5"]
    ck(abs(Lp.mean() - wp["regret_mean_db"]) < 1e-9, "persistence mean regret on eligible rows")
    ck(abs(float(Lp.mean()) - float(macro(f"QRegPers{tag}"))) < 5e-4, "macro QRegPers")

    # 4. the learned full-frame shortlist, from the SAVED predictions
    z = np.load(os.path.join(RES, f"beam_shortlists_sc{sc}.npz"))
    S = z["spatial|chirps=250|shortlist"]
    ck(np.array_equal(z["beams_all"], y), "shortlist file labels match the CSV")
    ck(np.array_equal(z["test_row_index"], np.arange(start, n)), "shortlist test rows match")
    regs = np.array([regret(S[i][:, :5]).mean() for i in range(S.shape[0])])
    out3 = np.array([(regret(S[i][:, :5]) > 3.0).mean() for i in range(S.shape[0])])
    inc = np.array([(S[i][m][:, :5] == yt[m][:, None]).any(1).mean() for i in range(S.shape[0])])
    ck(abs(np.median(regs) - float(macro(f"QReg{tag}"))) < 5e-4, "macro QReg",
       np.median(regs), macro(f"QReg{tag}"))
    ck(abs(100 * np.median(out3) - float(macro(f"QOut{tag}"))) < 5e-3, "macro QOut",
       100 * np.median(out3), macro(f"QOut{tag}"))
    ck(abs(np.median(inc) - float(macro(f"QIncl{tag}"))) < 5e-5, "macro QIncl")
    # regret must fall monotonically in k, since a longer shortlist contains a shorter one
    mk = [np.median([regret(S[i][:, :k]).mean() for i in range(S.shape[0])])
          for k in (1, 2, 3, 5, 8, 16)]
    ck(all(a >= b for a, b in zip(mk, mk[1:])), "mean regret is non-increasing in k", mk)
    ck(abs(mk[0] - float(macro(f"QRegKOne{tag}"))) < 5e-4, "macro QRegKOne")
    ck(abs(mk[-1] - float(macro(f"QRegKSixteen{tag}"))) < 5e-4, "macro QRegKSixteen")

    # 5. expected overhead is linear in the top-5 rate, so re-derive it from inclusion
    p5 = np.array([(S[i][m][:, :5] == yt[m][:, None]).any(1).mean() for i in range(S.shape[0])])
    ov = 5 * 5.0 + (1 - p5) * (64 - 5) * 5.0
    ck(abs(np.median(ov) - float(macro(f"QOvh{tag}"))) < 0.05, "macro QOvh",
       np.median(ov), macro(f"QOvh{tag}"))

    # 6. the delay evaluation: rebuild the eligibility chain independently
    lag = max(int(k) for k in rec["delay"])
    ok = np.ones(len(yt), bool)
    el = np.zeros(len(yt))
    for j in range(len(yt)):
        i0 = start + j
        if i0 + lag >= n:
            ok[j] = False; continue
        tot = 0.0
        for q in range(lag):
            a = i0 + q
            if not (seq[a + 1] == seq[a] and idx[a + 1] == idx[a] + 1 and 0 < dt[a] <= 200):
                ok[j] = False; break
            tot += dt[a]
        el[j] = tot
    valid = ok & fin[start:] & np.r_[fin[start + lag:], np.zeros(lag, bool)]
    d = rec["delay"][str(lag)]
    ck(int(valid.sum()) == d["n_eval"], "delay eligible count", int(valid.sum()), d["n_eval"])
    ck(abs(np.median(el[valid]) - d["elapsed_ms_median"]) < 1e-9, "delay elapsed median")
    jj = np.arange(len(yt)) + lag
    t1 = [float((S[i][valid][:, 0] == yt[jj[valid]]).mean()) for i in range(S.shape[0])]
    t5 = [float((S[i][valid][:, :5] == yt[jj[valid]][:, None]).any(1).mean())
          for i in range(S.shape[0])]
    ck(abs(np.median(t1) - d["1"]["inclusion_median"]) < 1e-12, "delayed top-1")
    ck(abs(np.median(t5) - d["5"]["inclusion_median"]) < 1e-12, "delayed top-5")
    ck(abs(np.median(t1) - float(macro(f"QDelayTopOne{tag}"))) < 5e-5, "macro QDelayTopOne")
    ck(abs(np.median(t5) - float(macro(f"QDelayTopFive{tag}"))) < 5e-5, "macro QDelayTopFive")
    # the delayed decision must be no better than the undelayed one
    ck(np.median(t5) < np.median(inc), "a delayed shortlist is worse than a fresh one",
       np.median(t5), np.median(inc))

    # 7. the per-sequence spread, and that it is wider than the seed interval
    sq = seq[start:]
    per = []
    for g in sorted(set(sq.tolist())):
        mm = (sq == g) & m
        if mm.sum() < 20:
            continue
        per.append(np.median([float((S[i][mm][:, 0] == yt[mm]).mean())
                              for i in range(S.shape[0])]))
    per = np.array(per)
    ps = rec["per_sequence"]
    ck(len(per) == ps["n_sequences_scored"], "sequences scored", len(per), ps["n_sequences_scored"])
    ck(abs(per.min() - ps["top1_min"]) < 1e-12 and abs(per.max() - ps["top1_max"]) < 1e-12,
       "per-sequence range")
    t1s = np.array([float((S[i][m][:, 0] == yt[m]).mean()) for i in range(S.shape[0])])
    seed_w = np.percentile(t1s, 97.5) - np.percentile(t1s, 2.5)
    iqr_w = np.percentile(per, 75) - np.percentile(per, 25)
    # The manuscript makes this comparison for Scenario 31 only, and says why: Scenario 33
    # has 5 test sequences, which cannot support an interquartile range. Asserting it there
    # too would be stricter than the paper and would fail at 1.5x, so the ratio is printed
    # for Scenario 33 and asserted only where it is claimed.
    print(f"        recording IQR / seed interval = {iqr_w / seed_w:.2f}"
          f"  (IQR {iqr_w:.4f}, seed {seed_w:.4f}, {len(per)} sequences)")
    if sc == "31":
        ck(iqr_w > 2 * seed_w, "Sc.31 recording spread exceeds twice the seed interval",
           round(iqr_w, 4), round(seed_w, 4))
    else:
        ck(len(per) <= 5, "Sc.33 has too few test sequences for this comparison", len(per))

    # 8. the deterministic re-run must agree with the published sweep inside the seed spread
    pub = json.load(open(os.path.join(RES, f"sc{sc}_resource_ablation_G_max.json")))["settings"]
    dev = max(abs(float(np.median(z[f"spatial|{k}|top1"])) - v["top1_median"])
              for k, v in pub.items())
    ck(abs(dev - float(macro(f"ReplMaxDev{tag}"))) < 5e-5, "macro ReplMaxDev", dev)
    ck(dev < max(v["top1_sd"] for v in pub.values()),
       "the replication deviation is inside the seed spread")

print(f"\n{'FAILED: ' + '; '.join(fails) if fails else 'all independent checks passed'}")
sys.exit(1 if fails else 0)
