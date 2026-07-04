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
| Predicted-bin interval coverage | `0.880000` |
| Predicted-bin mean interval width | `43.826050` |
| Predicted-bin global-floor coverage | `0.900000` |
| Predicted-bin global-floor mean interval width | `50.679584` |
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

Next work should improve calibration and tail diagnostics before more training
losses: add unit-level failure notes for the uncovered official-test cases,
then decide whether calibration should remain global, adopt a conservative
floor, or use another tail-specific fallback.
