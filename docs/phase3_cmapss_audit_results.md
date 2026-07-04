# Phase 3 C-MAPSS Audit Results

This note records the first Phase 3 audit run against the current best FD001
deep candidate:

- model: mini-batch monotonic asymmetric-loss Transformer with validation-fitted
  predicted-bin NASA-shift calibration;
- artifact directory:
  `artifacts/phase2_fd001_transformer_h32_40e_asymmetric_monotonic_w0p1`;
- prior leaderboard result: RMSE `14.246672`, NASA score `271.486206`.

## Command

```powershell
uv run aerospace-prognostics cmapss-phase3-audit `
  --calibration-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_monotonic_w0p1/results/cmapss_deep_validation_selection_predictions.csv `
  --predictions-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_monotonic_w0p1/results/cmapss_deep_predictions.csv `
  --calibrated-predictions-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_monotonic_w0p1/results/cmapss_deep_predictions_predicted_bin_nasa_shift_s500_calibrated.csv `
  --output-json artifacts/phase2_fd001_transformer_h32_40e_asymmetric_monotonic_w0p1/results/cmapss_phase3_audit.json `
  --output-markdown artifacts/phase2_fd001_transformer_h32_40e_asymmetric_monotonic_w0p1/results/cmapss_phase3_audit.md
```

## Result Summary

The audit fit a `0.90` validation absolute-residual interval radius of
`25.308556` RUL cycles and evaluated it on 100 FD001 official-test units.

| Metric | Value |
|---|---:|
| Official-test coverage | `0.900000` |
| Mean interval width | `48.665198` |
| Late-prediction rows | `51` |
| Late-prediction coverage | `0.960784` |
| Uncovered late predictions | `2` |
| Unit failure notes | `10` |
| Predicted-bin interval coverage | `0.880000` |
| Predicted-bin mean interval width | `43.826050` |
| Predicted-bin global-floor coverage | `0.900000` |
| Predicted-bin global-floor mean interval width | `50.679584` |
| Tail-fallback interval coverage | `0.940000` |
| Tail-fallback mean interval width | `51.534539` |
| Tail-fallback late-prediction failures | `1` |
| Tail sweep rows | `9` |
| Lowest-width sweep candidate covering units `16` and `67` | threshold `76`, confidence `0.99` |
| Raw monotonicity violation rate | `0.000000` |
| Calibrated monotonicity violation rate | `0.000000` |

Coverage by actual-RUL bin shows the main weakness:

| Actual RUL bin | Rows | Coverage | Mean abs error | Late rate |
|---|---:|---:|---:|---:|
| `0-30` | 25 | `1.000000` | `3.445250` | `0.680000` |
| `31-60` | 14 | `1.000000` | `6.486341` | `0.642857` |
| `61-90` | 15 | `0.800000` | `15.653236` | `0.666667` |
| `91-120` | 33 | `0.939394` | `10.894987` | `0.454545` |
| `121+` | 13 | `0.615385` | `20.973107` | `0.000000` |

The top uncovered cases are mostly early high-RUL compression errors. The two
uncovered late cases are unit `67` and unit `16`, which remain important because
late RUL overestimation is the higher-risk PHM failure mode.

## Predicted-Bin Interval Comparison

The follow-up comparison fit interval radii by validation predicted-RUL bin,
then evaluated the same official-test rows:

| Strategy | Coverage | Mean interval width | Uncovered late predictions |
|---|---:|---:|---:|
| Global validation residual radius | `0.900000` | `48.665198` | `2` |
| Predicted-bin validation residual radius | `0.880000` | `43.826050` | `2` |
| Predicted-bin radius with global floor | `0.900000` | `50.679584` | `2` |

Fitted predicted-bin radii:

| Predicted RUL bin | Validation rows | Radius |
|---|---:|---:|
| `all` | 3109 | `25.308556` |
| `0-30` | 61 | `13.360367` |
| `31-60` | 515 | `18.449928` |
| `61-90` | 661 | `32.023178` |
| `91-120` | 1872 | `24.426781` |

The validation predictions did not populate a separate `121+` predicted-RUL
bin, so high predicted-RUL cases fall back to the global radius. The simple
predicted-bin strategy narrows intervals overall but does not reduce uncovered
late cases and slightly worsens total coverage.

The global-floor check keeps the predicted-bin strategy from shrinking any bin
below the global validation radius. It restores total coverage to `0.900000`,
but it widens intervals beyond the global baseline and still leaves the same two
uncovered late predictions. That makes it a useful guardrail candidate, not a
complete calibration improvement.

## Tail Fallback Experiment

The tail-fallback experiment keeps the global validation interval for all rows
by default, then widens rows with predicted RUL at or above `91` cycles using a
validation-fitted `0.95` absolute-residual radius from predicted-tail rows.

| Strategy | Coverage | Mean interval width | Uncovered late predictions |
|---|---:|---:|---:|
| Global validation residual radius | `0.900000` | `48.665198` | `2` |
| Predicted-tail fallback radius | `0.940000` | `51.534539` | `1` |

Tail fallback calibration:

| Tail threshold | Tail confidence | Tail validation rows | Global radius | Tail radius |
|---:|---:|---:|---:|---:|
| `91.000000` | `0.950000` | 1840 | `25.308556` | `28.361046` |

The experiment is inference-safe because it uses predicted RUL, not actual RUL.
It covers late-overestimate unit `16`, but late-overestimate unit `67` remains
uncovered. Mean interval width increases by `2.869341` cycles while median
interval width stays unchanged at `50.617111`.

## Tail Fallback Sweep

The threshold/confidence sweep quantifies the interval-width cost of covering
both global late-overestimate misses:

| Tail threshold | Tail confidence | Tail rows | Radius | Coverage | Mean width | Late failures | Recovered late units | Remaining late units |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| `61` | `0.95` | 2505 | `31.251747` | `0.950000` | `56.034755` | `1` | `16` | `67` |
| `61` | `0.975` | 2505 | `35.974571` | `0.970000` | `61.891057` | `1` | `16` | `67` |
| `61` | `0.99` | 2505 | `41.494377` | `0.990000` | `68.735616` | `0` | `16,67` |  |
| `76` | `0.95` | 2194 | `30.879509` | `0.950000` | `54.570408` | `1` | `16` | `67` |
| `76` | `0.975` | 2194 | `35.644310` | `0.960000` | `59.621097` | `1` | `16` | `67` |
| `76` | `0.99` | 2194 | `40.452194` | `0.970000` | `64.717455` | `0` | `16,67` |  |
| `91` | `0.95` | 1840 | `28.361046` | `0.940000` | `51.534539` | `1` | `16` | `67` |
| `91` | `0.975` | 1840 | `31.251747` | `0.950000` | `54.251798` | `1` | `16` | `67` |
| `91` | `0.99` | 1840 | `37.414322` | `0.960000` | `60.044618` | `1` | `16` | `67` |

The lowest-width candidate that covers both late-overestimate units is threshold
`76` with tail confidence `0.99`. It removes the late failures, but mean interval
width increases by `16.052257` cycles over the global baseline, so it should be
treated as an aggressive safety candidate rather than the current default.

## Unit Failure Notes

The audit now records ranked unit-level notes for rows missed by any interval
strategy. In the current FD001 run, the top 10 failure notes are missed by all
three interval strategies: global interval, predicted-bin interval, and
predicted-bin interval with global floor.

Before the tail fallback, the two highest-risk misses were late overestimates:

| Unit | Actual RUL bin | Predicted RUL bin | Failure | Note |
|---:|---|---|---|---|
| `67` | `61-90` | `91-120` | `late_uncovered` | late overestimate; uncovered by all three strategies |
| `16` | `61-90` | `91-120` | `late_uncovered` | late overestimate; uncovered by all three strategies |

With the tail fallback, unit `16` becomes covered, while unit `67` remains
uncovered.

The remaining top notes are early underestimates, mostly cases where actual RUL
is `91-120` or `121+` but the model compresses the prediction into `31-60`,
`61-90`, or `91-120`. This reinforces that the main weakness is tail
calibration/compression, not monotonicity.

## Interpretation

The first Phase 3 audit does not justify adding a stronger constrained loss yet.
The current best deep candidate already has no official-test monotonicity
violations in the final per-unit rows available for this audit, and the raw vs
calibrated comparison does not change that. The urgent weakness is interval and
tail calibration:

- high-RUL units, especially `121+`, are under-covered and compressed early;
- mid-RUL `61-90` units are also under the nominal `0.90` coverage target;
- late-risk coverage is better than overall coverage, but two uncovered late
  cases remain.
- predicted-bin interval calibration is not enough because validation
  predictions do not provide a separate `121+` predicted-RUL radius and the
  narrower low/mid predicted-bin radii reduce overall coverage.
- the global-floor variant is safer than raw predicted-bin intervals but does
  not reduce high-risk late failures, so unit-level failure analysis is still
  needed before changing the deployed uncertainty policy.
- pre-tail unit-level notes showed both late-overestimate failures uncovered
  under the global, predicted-bin, and global-floor policies.
- the tail fallback is the first interval experiment that improves overall
  coverage and reduces late-overestimate misses, but it does not eliminate the
  highest-risk unit `67` miss.
- the sweep shows unit `67` can be covered only by much wider `0.99`
  tail-confidence candidates in this grid; the lowest-width option is
  threshold `76`, confidence `0.99`, with mean width `64.717455`.

Next work should freeze a Phase 3 C-MAPSS recommendation: keep global intervals
as the deployable baseline, list `91/0.95` as the balanced experimental tail
fallback, and list `76/0.99` as the aggressive safety candidate that covers both
late-overestimate misses at substantial width cost.
