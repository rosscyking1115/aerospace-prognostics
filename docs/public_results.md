# Public Results Summary

This page is the short, public-facing result ledger for the project. It
summarizes the tracked benchmark evidence without asking a reviewer to read the
full research log first.

The numbers below are useful engineering checkpoints, not certification claims.
Raw telemetry and generated artifacts stay out of Git; the detailed reproduction
commands and longer interpretation live in the linked phase documents.

## Headline Readout

- Current deployable RUL model: validation-selected C-MAPSS HGB policy.
- Current strongest FD001 deep model: calibrated Transformer with asymmetric
  late-error loss and mini-batch monotonic regularization.
- Current spacecraft anomaly status: baseline and alert-policy layer across
  SMAP/MSL, plus real event-wise detection baselines on the ESA-ADB
  Mission1 and Mission2 lightweight subsets, comparing a fixed and a
  validation-selected threshold (conservative baselines, not leaderboard claims).
- Current production evidence: FastAPI serving, Docker smoke checks, SBOM,
  release bundle, provenance, model card, validation, benchmark, read-only demo
  image, and Streamlit operations console.

## C-MAPSS RUL Results

The project uses NASA C-MAPSS official test RUL files for reported RUL
checkpoints.

### Selection hygiene (what was and was not chosen on validation)

Stated precisely, because "validation-selected" does not cover the whole policy:

- **Chosen on train-side validation.** The validation split holds out whole
  *units* and truncates their histories to a 30-cycle horizon
  (`make_cmapss_temporal_validation_split`), so no engine appears in both halves
  and the validation task mirrors the official test task. The feature policy
  (`engineered` vs `regime_engineered`), the HGB parameter set, and the sensor
  filter were all scored on that split. Feature standardisation, the regime
  clusterer, and the NASA-shift calibration are all fit on training/validation
  rows only, never on official test rows.
- **Chosen on the official test set.** The per-subset rolling window
  (`CMAPSS_ENGINEERED_DEFAULT_WINDOWS` = FD001:10, FD002:3, FD003:5, FD004:3) is
  the argmax of the rolling-window sweep in
  [phase1_cmapss_baseline_results.md](phase1_cmapss_baseline_results.md), and
  that sweep is scored on official test. There is no validation-side window
  sweep in the codebase. The window is therefore a test-selected
  hyperparameter that the otherwise validation-selected policy inherits.
- **The official test set has been read more than once.** The Phase 1 note
  records four successive test-scored comparisons (raw-cycle, engineered,
  window sweep, regime-aware), and the FD001 deep table below ranks five
  candidates by official-test RMSE/NASA score.

The practical consequence: treat the headline FD001 numbers as **mildly
optimistic**, in the way any repeatedly-consulted benchmark leaderboard is.
They are honest reproducible checkpoints, not a clean single-shot held-out
estimate. Closing this properly means sweeping the window on validation and
re-reporting; that is tracked as open work rather than quietly ignored.

### Naive Floor (What Any Real Model Must Beat)

Before reading any model number, read the floor. These are constant predictors
that ignore the sensors entirely — `train_median` emits the central capped
training RUL for every test unit, `rul_cap` emits the ceiling. Reproduce with:

```powershell
uv run aerospace-prognostics cmapss-naive-baseline --data-dir data/raw/cmapss --subset FD001
```

| Subset | Naive `train_median` RMSE | Naive `train_median` NASA | Naive `rul_cap` RMSE | Naive `rul_cap` NASA |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 49.819876 | 166570.542613 | 64.615323 | 1502475.412851 |
| FD002 | 58.034805 | 582920.773435 | 69.367746 | 5163062.801809 |
| FD003 | 63.142537 | 1138771.635164 | 64.666065 | 1390904.754962 |
| FD004 | 65.033645 | 2986938.875795 | 66.716269 | 4030492.673107 |

Against this floor the HGB policy is doing real work, not reproducing the label
distribution: on FD001 it cuts RMSE from `49.82` to `13.01` (3.8x) and the NASA
score from `166570` to `253` (657x). The NASA gap is far larger than the RMSE
gap because the score punishes late predictions exponentially, and a constant
is late on roughly half the fleet. Every subset clears the floor by a wide
margin, so the reported numbers reflect learned degradation signal.

### Current HGB Policy

| Subset | Feature Policy | HGB Policy | Official Test RMSE | Official Test NASA Score |
| --- | --- | --- | ---: | ---: |
| FD001 | regime engineered | default | 13.012889 | 253.465322 |
| FD002 | engineered | slow regularized | 27.568929 | 8697.396161 |
| FD003 | regime engineered | slow regularized | 14.394433 | 358.507217 |
| FD004 | regime engineered | default | 29.215870 | 7106.739881 |

This is the current classical policy baseline. It improves the aggregate NASA
score from `16597.477390` to `16416.108581` versus the earlier
validation-selected feature baseline.

### FD001 Deep-Model Comparison

| Rank | Candidate | Official Test RMSE | Official Test NASA Score | Readout |
| ---: | --- | ---: | ---: | --- |
| 1 | HGB policy | 13.012889 | 253.465322 | Current deployable leader |
| 2 | Transformer, asymmetric late loss + mini-batch monotonic penalty + validation-fitted NASA-shift calibration | 14.246672 | 271.486206 | Best deep FD001 row so far |
| 3 | Transformer, asymmetric late loss + NASA-shift calibration | 14.267129 | 272.578820 | Strong deep diagnostic |
| 4 | Transformer, predicted-bin NASA-shift calibration | 14.460923 | 278.840518 | Calibration improves NASA score but not enough |
| 5 | Raw 40-epoch Transformer | 14.339589 | 299.978246 | Strong sequence baseline, still behind HGB |

The deep track is working end to end: sequence exports, CNN, residual CNN,
LSTM/BiLSTM, TCN, Transformer, validation-selection diagnostics, calibration,
asymmetric losses, and monotonic penalties are all exercised through CLI and
workflow paths. The honest result is that HGB still wins FD001 today, while the
deep track has a credible path through NASA-aware losses, inference-safe
calibration, and better monotonic/health-index constraints.

## SMAP/MSL Spacecraft Anomaly Results

Track B establishes spacecraft telemetry anomaly plumbing before making larger
claims. It supports Telemanom/Kaggle SMAP/MSL ingestion, channel export,
classical baselines, compact LSTM forecasting, threshold policies, comparison
reports, and manifest verification.

| Checkpoint | Result |
| --- | --- |
| All-channel classical sweep | 81 unique channels, 243 classical rows |
| Classical winners | robust z-score 41, PCA reconstruction 21, Isolation Forest 19 |
| Default robust z-score | mean point-wise F1 0.165343, mean false-alarm rate 0.187988 |
| Best global robust threshold in first sweep | threshold 5.0, mean F1 0.168552, mean false-alarm rate 0.155975 |
| Spacecraft-family alert policy | SMAP threshold 5, MSL threshold 10 under a 0.15 mean false-alarm budget |
| Comparison-ready robust policy | wins 15 of 81 channels, mean F1 0.160525, false-alarm rate 0.134247, point-adjusted F1 0.541768 |

The anomaly result is intentionally conservative. Point-adjusted F1 is tracked
because it is common in the literature, but point-wise precision/recall and
false-alarm rate remain the primary readout because point adjustment can make
weak detectors look stronger on long labelled intervals.

## ESA-ADB Results

The anomaly track's forward direction is ESA-ADB: an under-used benchmark,
carried through its real protocol at the detection tier, with no comparison set.

**What that phrasing is doing.** Earlier wording here called ESA-ADB "fresher
and less-saturated" than SMAP/MSL. That is a claim about the literature, and
nothing in this repository evidences it — there is no citation count, no
leaderboard survey, and no reproduction of a published ESA-ADB result to sit
beside these numbers. The only comparison below is one threshold policy against
another, both from the same baseline. "Under-used" is what the work supports;
anything stronger would need evidence this repository does not hold. Recorded
rather than quietly reworded, because the correction narrows a published claim.

Both benchmark missions run on real telemetry with
a robust z-score baseline fit on nominal training points only, scored event-wise
on the chronological test window (communication gaps excluded). Two threshold
policies are compared: a fixed `5.0` cutoff and a validation-selected cutoff
chosen on the last three months of training.

| Mission | Policy | Chosen τ | False alarms | Precision | Recall | F0.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mission1 | fixed | 5 | 0 | 1.000 | 0.415 | 0.780 |
| Mission1 | validation | 5 (fallback) | 0 | 1.000 | 0.415 | 0.780 |
| Mission2 | fixed | 5 | 10139 | 0.373 | 1.000 | 0.426 |
| Mission2 | validation | 20 | 2 | 0.999 | 0.986 | 0.997 |

The honest reading, not cherry-picked:

- The same fixed threshold is precise on Mission1 but over-alarms badly on
  Mission2's noisier 11 channels, so a hardcoded threshold is not portable.
- A naive validation selection (pure argmax F0.5) rescued Mission2 (`0.426` →
  `0.997`) but overfit Mission1: its validation window has only ~4 events with no
  false alarms at any threshold, so it chose the grid-edge τ=3 that then produced
  1282 test false alarms (F0.5 `0.780` → `0.506`). Longer validation windows did
  not help — the signal is absent from Mission1's training half.
- The robust selector now falls back to the conservative default when the
  validation window is too sparse to discriminate, and otherwise takes the most
  conservative near-best threshold. Mission1 falls back to τ=5 (`0.780`) and
  Mission2 selects τ=20 (`0.997`). The takeaway — trust validation selection only
  when the window is informative — is the finding worth keeping.
- Mission2's near-perfect numbers are lenient, not SOTA: its events are mostly
  long "Rare Event" subsequences, and event-wise detection counts an event as
  caught if any sample in its interval fires.

This is protocol-shaped event-wise detection evidence, not a full ESA-ADB
leaderboard claim: only the detection top of the official metric hierarchy is
computed, and the official zero-order-hold resampling is not yet applied. The
**recall** figures use ESA-ADB's event-wise definition exactly; the **precision**
figures use `telemeval`'s run-based definition, which diverges from ESA-ADB's
TNR-corrected `EW_*` in ways documented and pinned by divergence tests upstream.
Read them as telemeval event-wise precision, not as `EW_precision`.
Details and reproduction:
[phase3_esa_adb_intake.md](phase3_esa_adb_intake.md).

### Evaluation Correction (Recorded For Honesty)

The first Mission1 run reported recall `0.236842`. Auditing the evaluation
before publishing showed the cause: the scorer was measuring detections on the
chronological test window only, but the event denominator still included events
that fall entirely in the training half — events for which there are no
test-window predictions and which can therefore never be detected. Those
training-only events were counted as missed, understating recall.

Restricting the event set to events that overlap the test window — the correct
protocol, since a model is not penalised for data it was never asked to score —
raises the honest recall to `0.415385` and the F0.5 to `0.780347`. The corrected
numbers are the ones reported above; the pre-correction `0.24` is kept here only
to document the fix. The correction *raises* the reported number, so it is
recorded explicitly rather than quietly adopted.

## Deployment Evidence

The project already proves more than model training:

- packaged C-MAPSS model artifact with stable artifact identity;
- artifact inspection, validation, benchmark, model card, promotion report,
  SBOM, release bundle, and provenance;
- FastAPI service with health, readiness, schema, metrics, drift summaries,
  API-key authentication, rate limiting, and mounted-model smoke tests;
- Docker serving image with healthcheck, OCI labels, dependency-surface checks,
  and CI manifest generation;
- Docker Compose local product stack for API, console, and SQLite app state;
- read-only Streamlit demo image with baked-in quickstart evidence;
- operations console for fleet triage, model registry, prediction history,
  evidence downloads, outcomes, operator decisions, and fleet priority policy.

## Limitations

- These are benchmark and engineering-reference results, not aircraft or
  spacecraft certification evidence.
- The deployable model currently covers C-MAPSS FD001-style RUL inference.
- The strongest deep model is still behind the HGB policy on FD001.
- SMAP/MSL anomaly work is a baseline and alert-policy layer, not a reproduced
  Telemanom or ESA-ADB leaderboard claim.
- Generated model binaries, raw telemetry, SQLite app state, and release outputs
  are intentionally excluded from Git.
- The repository code is MIT licensed; dataset licenses are recorded separately
  (for example, ESA-ADB data is `CC BY 3.0 IGO`).

## Conformal Prediction Intervals

Distribution-free RUL intervals with unit-grouped calibration, added after the
calibration work above. The full account, including two negative results, is in
[writeup.md](writeup.md) §6; the numbers are read from git-tracked artifacts under
`artifacts/conformal/`.

| Subset | Nominal | Calibration units | Mean coverage over 10 splits | Mean width | Splits at or above nominal |
| --- | ---: | ---: | ---: | ---: | ---: |
| FD001 | 0.90 | 30 | 0.948000 | 56.329239 | 8 / 10 |
| FD002 | 0.99 | 99 | 0.916988 | 105.222252 | 0 / 10 |

Coverage is always reported with width, because an infinite interval covers
everything. Two results are worth more than the passing row:

- **FD001 cannot support a distribution-free 99% guarantee at the unit level.** The
  finite-sample rank needs 99 exchangeable calibration engines and FD001 has 100
  training engines in total. The pipeline returns an infinite interval rather than a
  plausible number. Row-pooled calibration — treating each of 3,780 cycles as an
  independent draw — manufactures a finite `44.99`-cycle radius at `0.99` coverage on
  the same data. That apparent success is the leak, not a result.
- **FD002's `0.907336` is marginal coverage that describes no engine.** Zero of ten
  splits reach the nominal `0.99`. Split by whether the truth is inside the range
  the model can express: `0.985149` over 202 engines below the 125-cycle training
  cap, `0.631579` over the 57 above it. Conformal prediction guarantees marginal,
  not conditional, coverage, and cannot repair a systematically biased centre. A
  marginal coverage figure quoted without a subgroup breakdown is not safe to act
  on.

## Source Notes

- The connected account of the whole project:
  [writeup.md](writeup.md)
- C-MAPSS classical baseline details:
  [phase1_cmapss_baseline_results.md](phase1_cmapss_baseline_results.md)
- C-MAPSS deep baseline details:
  [phase2_cmapss_deep_baselines.md](phase2_cmapss_deep_baselines.md)
- SMAP/MSL anomaly details:
  [phase2_spacecraft_anomaly_baselines.md](phase2_spacecraft_anomaly_baselines.md)
- Deployment evidence:
  [deployment.md](deployment.md)
