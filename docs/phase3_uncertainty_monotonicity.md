# Phase 3 Uncertainty And Monotonicity

This is the near-term Phase 3 research board while the repository remains
private. The goal is auditable PHM evidence, not novelty for its own sake.

## Baseline We Already Have

- The deployable C-MAPSS HGB artifact exposes train-residual absolute-quantile
  RUL intervals for API, dashboard, and model-card evidence.
- Phase 2 deep-model runs emit validation-selection and official-test
  prediction CSVs, error diagnostics, RUL-bin diagnostics, unit diagnostics, and
  monotonicity diagnostics.
- Deep-model calibration experiments already support validation-fitted affine,
  predicted-bin residual, and predicted-bin NASA-shift corrections.

## Current Phase 3 Slice

Run the focused C-MAPSS audit after a Phase 2 deep run or calibration run:

```powershell
uv run aerospace-prognostics cmapss-phase3-audit `
  --calibration-csv artifacts/PHASE2_RUN/results/cmapss_deep_validation_selection_predictions.csv `
  --predictions-csv artifacts/PHASE2_RUN/results/cmapss_deep_predictions.csv `
  --calibrated-predictions-csv artifacts/PHASE2_RUN/results/cmapss_deep_predictions_predicted_bin_nasa_shift_s500_calibrated.csv `
  --output-json artifacts/PHASE2_RUN/results/cmapss_phase3_audit.json `
  --output-markdown artifacts/PHASE2_RUN/results/cmapss_phase3_audit.md
```

The command fits symmetric RUL intervals only from validation-selection
absolute residuals, then evaluates official-test coverage. It reports:

- empirical coverage against the requested confidence;
- mean and median interval width;
- actual-RUL-bin coverage;
- late-prediction coverage and uncovered late-risk cases;
- top uncovered interval failures;
- raw monotonicity diagnostics;
- optional raw-vs-calibrated monotonicity comparison.

## Decision Rules

- Treat interval evidence as operational triage evidence, not certification.
- If coverage or late-risk coverage misses the target, improve calibration or
  selection before claiming stronger uncertainty behavior.
- If monotonicity violations are material, inspect affected unit trajectories
  before adding a constrained loss.
- Add new constrained losses only after the audit points to a specific failure
  mode that the existing diagnostics and calibration cannot explain.

## Next Build Slices

1. Done: run the audit on the current best Phase 2
   Transformer/asymmetric-loss artifacts and record the evidence summary in
   [docs/phase3_cmapss_audit_results.md](phase3_cmapss_audit_results.md).
2. Done: compare validation-fitted predicted-bin interval widths against the
   global validation absolute-residual interval. The simple predicted-bin
   strategy narrowed intervals but reduced coverage, so it is diagnostic only.
3. Done: add a high-RUL-aware global-floor check for predicted-bin intervals.
   It restores nominal coverage but widens intervals and does not reduce
   uncovered late predictions.
4. Done: add unit-level failure notes for the uncovered official-test cases.
   The two late-overestimate misses remain uncovered under the pre-tail global,
   predicted-bin, and global-floor strategies.
5. Done: add a stronger inference-safe predicted-tail fallback experiment. It
   improves coverage to `0.94` and covers one of the two late-overestimate
   misses, but unit `67` remains uncovered.
6. Done: run a small threshold/confidence sweep. The lowest-width candidate
   covering both late-overestimate units is threshold `76`, confidence `0.99`,
   with mean interval width `64.717455`.
7. Done: freeze the Phase 3 C-MAPSS recommendation in
   [docs/phase3_cmapss_recommendation.md](phase3_cmapss_recommendation.md).
   The global interval remains the deployable baseline, `91/0.95` is the
   balanced experimental tail fallback, and `76/0.99` is the aggressive safety
   candidate.
8. Start ESA-ADB as a separate protocol intake only after the C-MAPSS Phase 3
   evidence loop is stable.
