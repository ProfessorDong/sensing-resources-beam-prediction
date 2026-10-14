# What Sensing Resources Buy a Communication Link in Radar-Aided Beam Prediction

Code, result records and verification suite for

> L. Dong, "What Sensing Resources Buy a Communication Link in Radar-Aided Beam
> Prediction," submitted to *IEEE Transactions on Wireless Communications*, 2026.

Radar-aided beam prediction is usually reported as an accuracy figure, which says nothing
about how much sensing was spent, how much beam training it displaces, or what the link
gives up when the prediction is wrong. On the raw FMCW cube of two public DeepSense 6G
scenarios this study sweeps the three resources an ISAC system trades, **sensing airtime,
bandwidth and receive aperture**; reads accuracy against reference predictors on both sides;
prices it in microseconds of the alignment it replaces; and scores it in the received power
the recording measured for all 64 beams.

**No number in the paper is typed by hand.** Every value is a macro generated from the result
files by `experiments/make_paper_b_macros.py`, and a verification suite reads the values back
out of the rendered PDF and the compiled figures and checks them against their sources.

The manuscript sources are **not** in this repository.

---

## Quick start

```bash
python -m pip install -r requirements.txt
# see data/README.md first: DeepSense 6G Scenarios 31 and 33 are not redistributed here

python experiments/verify_paper_b_claims.py        # 216 checks against the result JSONs
python experiments/verify_crb_derivation.py        # the bearing CRB against a numerical FIM
```

Those two run on the tracked result files alone and need no dataset download.

---

## What produces what

| Script | Output |
|---|---|
| `exp_sc31_resource_ablation.py` | the airtime, bandwidth and aperture sweep |
| `exp_beam_shortlists.py` | per-frame ordered top-16 shortlists, every seed and setting |
| `audit_beam_power_regret.py` | power regret `L_k` and the exceedance `Pr[L_k > 3 dB]` |
| `audit_static_shortlist.py` | the power-aware static probe set, the 192-policy selection |
| `audit_alignment_cost.py` | alignment overhead in the three cost regimes |
| `exp_sc31_bearing_mediation.py` | the bearing regressor behind the CRB section |
| `audit_corrected_statistics.py` | exact rank-correlation p values, bound-clearing intervals |
| `validate_radar_frontend.py` | 15 front-end checks: are the maps what their axes claim |
| `audit_dc_removal.py`, `compare_sc31_dc.py` | the transmit-leakage repair and its re-run |
| `make_paper_b_figures.py` | the six placed figures as TikZ source |
| `make_paper_b_macros.py` | every `\PB...` macro and the four generated tables |

## Verification

Eight suites. The first four read the paper's own artifacts; the last four are independent
re-derivations that import no analysis module from `experiments/` and rebuild the quantities
from the raw CSVs and power files.

```bash
python experiments/verify_paper_b_claims.py                     # 216
python experiments/verify_paper_b_figures.py                    # 497
# (the manuscript verifier is not in this repository; see above)
python experiments/verify_crb_derivation.py
python review_artifacts/2026-10-09/verify_quality.py            #  68
python review_artifacts/2026-10-10/verify_revision.py           #  53
python review_artifacts/2026-10-11/verify_static_policy.py      #  28
python review_artifacts/2026-10-14/check_paired_and_crossover.py   # 86
```

`verify_paper_b_figures.py` reads the compiled figure PDFs, so it needs a figure build. The
other six run against the tracked result files alone. The manuscript verifier is not here: it
reads the paper, the rendered PDF and the cover letter, none of which are published.

Each script under `review_artifacts/` writes its results beside itself when run. Those outputs
are not versioned, so that a reader compares against a fresh run rather than a stored one.

## What the study found

* Scored in received power rather than exact-beam accuracy, a shortlist of five from the full
  frame misses the best beam in two frames of five yet gives up a mean of **0.225 dB**,
  against **0.461 dB** for a prior that uses no sensing.
* A **static probe set chosen for the power objective** is a far stronger comparator than a
  prior ranked by beam identity. The radar leads at one probe; by five the tail has crossed
  on both scenarios.
* Minimizing modeled cost at a declared quality target selects **no radar** when the link must
  acquire the frame, and a radar setting only where the frame is already paid for.
* The bearing Cramer-Rao bound supplies a resource coordinate along which the measured error
  broadly decreases, **without** ordering the eight full-aperture settings or predicting the
  rate: the fitted exponent is about 0.40 against the bound's 1.
* Four candidate results of the study's own did not survive their controls and are reported
  as reversals rather than removed.

## Reading the result files

`experiments/results/*.json` are the records the paper is generated from. The ones the
manuscript reads most are `sc31_resource_ablation_G_max.json`, `beam_power_quality.json`,
`static_shortlist.json`, `alignment_cost_model.json`, `corrected_statistics.json`,
`sc31_bearing_mediation_max.json` and `testbed_facts.json`, the last holding the hardware
facts with their provenance so no figure hardcodes them.

## A note on references in the code

A few docstrings mention `CLAUDE.md` and `PAPER_B_PLAN.md`. Those are the project's internal
engineering record and plan, kept outside this repository: they hold the development history,
including the claims that were retracted along the way. Nothing in the verification suite
reads them; `verify_paper_b_claims.py` re-checks numbers that are *quoted* in them against the
result JSONs, and it runs correctly without them.

## License

MIT, see `LICENSE`. The DeepSense 6G recordings are third-party and carry their own terms.
