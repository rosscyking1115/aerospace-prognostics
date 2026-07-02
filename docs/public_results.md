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
  SMAP/MSL, with ESA-ADB reserved for serious future benchmark claims.
- Current production evidence: FastAPI serving, Docker smoke checks, SBOM,
  release bundle, provenance, model card, validation, benchmark, read-only demo
  image, and Streamlit operations console.

## C-MAPSS RUL Results

The project uses NASA C-MAPSS official test RUL files for reported RUL
checkpoints. Model selection is kept on train-side validation where possible,
then the official test table is used once for the reported checkpoint.

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
- Public launch still needs a hosted-demo URL, final public license decision,
  and launch copy before the private repository should be made public.

## Source Notes

- C-MAPSS classical baseline details:
  [phase1_cmapss_baseline_results.md](phase1_cmapss_baseline_results.md)
- C-MAPSS deep baseline details:
  [phase2_cmapss_deep_baselines.md](phase2_cmapss_deep_baselines.md)
- SMAP/MSL anomaly details:
  [phase2_spacecraft_anomaly_baselines.md](phase2_spacecraft_anomaly_baselines.md)
- Deployment evidence:
  [deployment.md](deployment.md)
