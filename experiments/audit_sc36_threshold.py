"""One-line threshold baseline for Sc.36, on the EXACT frames the models are evaluated on.

WHY THIS EXISTS
---------------
Figure F1 of Paper A compares each model against the best ONE-LINE baseline on the same
test frames. Four of the five benchmarks already have that:

  DeepSense ISAC / V2I / RADIal  `trivial_baselines.json`  (same loaders and splits as
                                 `measurement_utilization.json`, verified by inspection)
  View-of-Delft                  `vod_sparsity.json`       ("recomputed on the exact eval
                                 subset used here")

Sc.36 does not. Its threshold baseline exists only from the 120-frame degeneracy screen
(`verify_DeepSense_Sc36.json`: channel 2, tau 0.52, test IoU 0.2800), and that subset is
not the 1,000-frame systematic sample the arms were scored on -- the same mismatch that
made the screen's constant read 0.3410 against the true 0.3919. So recompute it here.

PROTOCOL (matches every other baseline in this project)
  * channel and tau are fitted on TRAIN only, by grid search over tau
  * the fitted rule is applied unchanged to the eval frames
  * eval frames are reproduced exactly: systematic stride over the whole test block,
    `step = len(test)/eval_limit`, as `exp_sc36_hbeam.py` does

Output: results/sc36_threshold_baseline.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from exp_sc36_hbeam import load_cache

EVAL_LIMIT = 1000        # matches results/sc36_hbeam.json config
TAUS = np.arange(0.02, 1.00, 0.01)


def iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else 0.0


def fit_threshold(tr_meas, tr_occ):
    """Best (channel, tau) by IoU on TRAIN. No test data is touched."""
    best = (-1.0, None, None)
    for c in range(tr_meas.shape[1]):
        ch = tr_meas[:, c]
        for t in TAUS:
            s = iou(ch > t, tr_occ)
            if s > best[0]:
                best = (s, c, float(t))
    return best


def main(cache="sc36_stride1.npz", tag=""):
    tr, te, meta = load_cache(1, name=cache)
    tr_s, tr_m, _ = tr
    te_s, te_m, _ = te
    n_full = len(te_s)
    step = n_full / EVAL_LIMIT
    sel = (np.arange(EVAL_LIMIT) * step).astype(int)
    te_s, te_m = te_s[sel], te_m[sel]
    print(f"{cache}: train {len(tr_s)}  test {n_full} -> eval {len(te_s)} "
          f"(systematic stride ~{step:.1f})")

    tr_occ, te_occ = tr_s[:, 0] > 0.5, te_s[:, 0] > 0.5
    train_iou, ch, tau = fit_threshold(tr_m, tr_occ)
    test_iou = iou(te_m[:, ch] > tau, te_occ)
    const_train_mean = float(tr_occ.mean())
    # the constant arm, for cross-checking against the stored 0.3919
    const_iou = iou(np.full(te_occ.shape, const_train_mean) > 0.5, te_occ) \
        if const_train_mean > 0.5 else iou(np.full(te_occ.shape, const_train_mean) > 0.5, te_occ)
    print(f"  threshold fitted on train: channel {ch}, tau {tau:.2f}, train IoU {train_iou:.4f}")
    print(f"  -> test IoU on the matched eval frames: {test_iou:.4f}")
    print(f"  (screen value on its own 120-frame subset was 0.2800, channel 2, tau 0.52)")
    print(f"  train occupancy mean {const_train_mean:.4f}")

    out = {"cache": cache, "eval_limit": EVAL_LIMIT, "n_eval": int(len(te_s)),
           "n_test_full": int(n_full), "eval_stride": float(step),
           "threshold": {"channel": int(ch), "tau": tau,
                         "train_IoU": float(train_iou), "test_IoU": float(test_iou)},
           "train_occupancy_mean": const_train_mean,
           "screen_value_for_comparison": {"channel": 2, "tau": 0.52, "test_IoU": 0.2800425236128716,
                                           "note": "120-frame screen subset, NOT matched"},
           "protocol": "channel and tau fitted on train by IoU grid search; applied unchanged "
                       "to the systematically-strided eval frames"}
    p = os.path.join(RESULTS_DIR, f"sc36_threshold_baseline{tag}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"Saved: {p}")
    return out


if __name__ == "__main__":
    main()
