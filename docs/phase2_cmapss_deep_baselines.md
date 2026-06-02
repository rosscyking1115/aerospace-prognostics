# Phase 2 C-MAPSS Deep Baselines

This note records the first Phase 2 sequence-model checkpoint. Phase 1 ended with reproducible C-MAPSS sequence exports, so Phase 2 starts by proving that a PyTorch model can train from those tensors and report the same RMSE and NASA asymmetric RUL score used by the classical baseline.

## Track A Start

| Field | Value |
|---|---|
| Dataset | NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set |
| Input artifacts | `artifacts/sequences/cmapss/<subset>/{train,validation,validation_selection,test}_sequences.npz` |
| Sequence export command | `uv run aerospace-prognostics cmapss-export-sequences --data-dir data/raw/cmapss --output-dir artifacts/sequences/cmapss --window-size 30 --stride 1` |
| Deep model baselines | Compact PyTorch 1D-CNN; residual CNN; LSTM/BiLSTM; TCN; Transformer encoder |
| Window size | 30 cycles |
| Features | 24 standardized operating-setting and sensor channels |
| Optimizer | Adam |
| Loss | MSE by default; optional `nasa_surrogate`, MSE+NASA blended losses, and asymmetric late-weighted MSE losses |
| Random seed | 42 |
| Metrics | RMSE and NASA asymmetric RUL score |
| Checkpoint policies | `validation_nasa` or `final` |

The first model is intentionally compact. It is a plumbing baseline for sequence training, validation tracking, checkpoint selection, CLI execution, result serialization, and repeatable tests, not the final deep-learning architecture.

Sequence exports now separate two validation roles:

- `validation_sequences.npz` keeps one final observed window per held-out validation unit, mirroring the official C-MAPSS test setup.
- `validation_selection_sequences.npz` contains rolling windows from the same held-out validation units and is used for checkpoint selection. This gives early stopping more signal while preserving the final-window evaluation artifact for reporting.

## Command

CNN baseline:

```powershell
uv run aerospace-prognostics cmapss-cnn-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --batch-size 256 --hidden-channels 32 --learning-rate 0.001 --checkpoint-policy final --output-json artifacts/results/cmapss_cnn_fd001_final_baseline.json --output-csv artifacts/results/cmapss_cnn_fd001_final_baseline.csv --history-json artifacts/results/cmapss_cnn_fd001_final_history.json
```

Validation-selected checkpoint run:

```powershell
uv run aerospace-prognostics cmapss-cnn-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --batch-size 256 --hidden-channels 32 --learning-rate 0.001 --checkpoint-policy validation_nasa --output-json artifacts/results/cmapss_cnn_fd001_baseline.json --output-csv artifacts/results/cmapss_cnn_fd001_baseline.csv --history-json artifacts/results/cmapss_cnn_fd001_history.json
```

LSTM/BiLSTM baseline:

```powershell
uv run aerospace-prognostics cmapss-lstm-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --batch-size 256 --hidden-size 64 --num-layers 1 --bidirectional --checkpoint-policy validation_nasa --output-json artifacts/results/cmapss_bilstm_fd001_baseline.json --output-csv artifacts/results/cmapss_bilstm_fd001_baseline.csv --history-json artifacts/results/cmapss_bilstm_fd001_history.json
```

TCN baseline:

```powershell
uv run aerospace-prognostics cmapss-tcn-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --batch-size 256 --hidden-channels 64 --num-levels 3 --kernel-size 3 --checkpoint-policy validation_nasa --output-json artifacts/results/cmapss_tcn_fd001_baseline.json --output-csv artifacts/results/cmapss_tcn_fd001_baseline.csv --history-json artifacts/results/cmapss_tcn_fd001_history.json
```

The shared comparison workflow can now exercise a stronger TCN variant with channel-wise LayerNorm, weight-normalized causal convolutions, and mean temporal pooling:

```powershell
uv run aerospace-prognostics cmapss-deep-baseline-compare --sequence-dir artifacts/sequences/cmapss --subsets FD001 --models tcn --epochs 50 --batch-size 256 --hidden-sizes 32 64 --learning-rates 0.001 0.0003 --tcn-levels 4 --tcn-normalization layer_norm --tcn-weight-norm --tcn-pooling mean --checkpoint-policy validation_nasa --output-json artifacts/results/cmapss_tcn_enhanced_fd001.json --output-csv artifacts/results/cmapss_tcn_enhanced_fd001.csv
```

Transformer encoder baseline:

```powershell
uv run aerospace-prognostics cmapss-transformer-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --batch-size 256 --d-model 64 --num-heads 4 --num-layers 2 --dim-feedforward 128 --checkpoint-policy validation_nasa --output-json artifacts/results/cmapss_transformer_fd001_baseline.json --output-csv artifacts/results/cmapss_transformer_fd001_baseline.csv --history-json artifacts/results/cmapss_transformer_fd001_history.json
```

Compact architecture and learning-rate comparison:

```powershell
uv run aerospace-prognostics cmapss-deep-baseline-compare --sequence-dir artifacts/sequences/cmapss --subsets FD001 --models cnn rescnn bilstm tcn transformer --epochs 50 --batch-size 256 --hidden-sizes 32 64 --learning-rates 0.001 0.0003 --checkpoint-policy validation_nasa --output-json artifacts/results/cmapss_deep_compare_fd001.json --output-csv artifacts/results/cmapss_deep_compare_fd001.csv
```

Ranked Phase 1 versus Phase 2 report:

```powershell
uv run aerospace-prognostics cmapss-compare-rul-results --baseline-csv artifacts/results/cmapss_hgb_policy_baseline.csv --candidate-csv artifacts/results/cmapss_deep_compare_fd001.csv --output-csv artifacts/results/cmapss_phase2_model_comparison.csv --output-markdown artifacts/results/cmapss_phase2_model_comparison.md
```

Reproducible Phase 2 workflow:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2 --subsets FD001 --models cnn rescnn bilstm tcn transformer --epochs 50 --hidden-sizes 32 64 --learning-rates 0.001 0.0003
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2/phase2_run_manifest.json --output-markdown artifacts/phase2/phase2_manifest_audit.md
```

All deep baseline commands and the workflow accept `--training-loss mse`, `--training-loss nasa_surrogate`, `--training-loss mse_nasa_blend_w0p001`, `--training-loss mse_nasa_blend_w0p0001`, `--training-loss asymmetric_mse_late_w1p5`, `--training-loss asymmetric_mse_late_w2`, `--training-loss asymmetric_mse_late_w3`, `--training-loss target_weighted_mse_high_w2`, or `--training-loss target_weighted_mse_mid_high_w1p5`. The pure surrogate keeps training differentiable while matching the NASA RUL score's asymmetric shape more closely: late predictions are penalized with the harsher denominator used by the official metric, and early predictions use the gentler denominator. The blended losses add a small NASA-surrogate penalty to MSE so the optimization stays near the stable RUL regression scale while still nudging the model toward the asymmetric metric. The asymmetric MSE variants keep ordinary squared-error scale but multiply only late-prediction squared errors by 1.5, 2, or 3, giving us a steadier alternative to the exponential NASA surrogate. The target-weighted MSE variants keep squared-error scale but emphasize either high-RUL targets or the known mid/high-RUL diagnostic bands from validation-selection errors. Non-default runs include the loss name in model names and record the training loss in the Phase 2 manifest and audit report. The comparison command and workflow also accept `--models rescnn`; in that shared sweep interface, `--tcn-levels` controls the residual CNN block count. For `--models tcn`, the shared sweep path also records `--tcn-normalization`, `--tcn-weight-norm`, and `--tcn-pooling` so enhanced temporal-convolution experiments are auditable from the run manifest.

The workflow writes `phase2_summary.md` and `phase2_run_manifest.json` under the artifact directory. It also writes `results/cmapss_deep_predictions.csv`, a per-unit official-test diagnostics table with actual RUL, predicted RUL, signed error, absolute error, end cycle, and early/late error split for every deep-model candidate. The companion `results/cmapss_deep_prediction_diagnostics.csv`, `results/cmapss_deep_prediction_rul_bins.csv`, `results/cmapss_deep_prediction_monotonicity.csv`, and markdown report summarize mean error, mean/max absolute error, late-prediction rate, actual-RUL-bin calibration, temporal monotonicity, and the highest-error units. The same diagnostics are now emitted for rolling validation-selection windows under `results/cmapss_deep_validation_selection_predictions.csv`, `results/cmapss_deep_validation_selection_prediction_diagnostics.csv`, `results/cmapss_deep_validation_selection_prediction_rul_bins.csv`, `results/cmapss_deep_validation_selection_prediction_monotonicity.csv`, and `results/cmapss_deep_validation_selection_prediction_diagnostics.md`, so calibration, tail-error, and temporal-consistency ideas can be judged on train-only validation behavior before the official test table is touched. The manifest records run parameters, artifact paths, SHA-256/size checksums for the model outputs, prediction diagnostics, and sequence bundles, Python/platform/dependency versions, and Git commit state so a Phase 2 C-MAPSS run can be audited or reproduced from one bundle. The verify command checks manifest structure, referenced artifact existence, artifact checksums, and CSV row counts; with `--output-markdown`, it also writes a compact audit report.

Real FD001 workflow smoke run:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_smoke --subsets FD001 --models cnn tcn --epochs 3 --batch-size 256 --hidden-sizes 16 --learning-rates 0.001 --tcn-levels 1 --validation-horizon 30 --checkpoint-policy validation_nasa
```

This is a reproducibility and orchestration check, not a quality benchmark. It confirms that the workflow can regenerate sequence tensors, train multiple deep candidates, run the Phase 1 HGB policy baseline, and emit the ranked comparison bundle from the real FD001 files in one command.

First FD001 benchmark-shaped workflow run:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_benchmark_20e --subsets FD001 --models cnn bilstm tcn transformer --epochs 20 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --tcn-levels 2 --validation-horizon 30 --checkpoint-policy validation_nasa
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_benchmark_20e/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_benchmark_20e/phase2_manifest_audit.md
```

This run trained all four Phase 2 model families for 20 epochs on real FD001 sequence tensors, produced four deep results and five comparison rows, and verified the run manifest with `status=ok` across 13 checked artifacts. The Phase 1 HGB policy remains the best row, but the Transformer checkpoint is now close enough to be a useful deep-learning baseline rather than only a plumbing test.

Focused Transformer/TCN follow-up:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_tcn_40e --subsets FD001 --models transformer tcn --epochs 40 --batch-size 256 --hidden-sizes 32 64 --learning-rates 0.001 0.0003 --tcn-levels 3 --validation-horizon 30 --checkpoint-policy validation_nasa
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_tcn_40e/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_tcn_40e/phase2_manifest_audit.md
```

This sweep produced eight deep results and verified the run manifest with `status=ok` across 13 checked artifacts. The best deep row improved from NASA score 352.015482 in the 20-epoch run to 299.978246, narrowing the gap to the Phase 1 HGB policy but not beating it yet.

Enhanced TCN architecture check:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_tcn_h32_40e_layernorm_wn_mean --subsets FD001 --models tcn --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --tcn-levels 4 --tcn-normalization layer_norm --tcn-weight-norm --tcn-pooling mean --validation-horizon 30 --checkpoint-policy validation_nasa
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_tcn_h32_40e_layernorm_wn_mean/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_tcn_h32_40e_layernorm_wn_mean/phase2_manifest_audit.md
```

This run verified with `status=ok` across 21 checked artifacts, but it is not a promotion path. The stronger TCN variant selected epoch 14 and scored official-test RMSE 21.389947 with NASA score 1936.346453, far behind both the 40-epoch Transformer and the Phase 1 HGB policy. The result is still useful: it rules out the full LayerNorm + weight normalization + mean-pooling bundle as a simple fix, while preserving those knobs for narrower ablations such as last-step pooling with normalization only.

Narrow TCN ablations using the same FD001 sequence export showed that the pooling choice was the largest issue. Keeping last-step pooling and adding only LayerNorm improved the full enhanced bundle substantially, but still did not approach the best Transformer or HGB policy:

| TCN Variant | Selected Epoch | Official Test RMSE | Official Test NASA Score |
|---|---:|---:|---:|
| `l4_layer_norm_last` | 22 | 16.333360 | 388.493948 |
| `l4_layer_norm_weight_norm_last` | 22 | 17.254959 | 459.037869 |
| `l4_plain_last` | 34 | 17.918531 | 476.481255 |
| `l4_layer_norm_weight_norm_mean` | 14 | 21.389947 | 1936.346453 |

The next TCN-only experiment should keep `--tcn-pooling last`, avoid weight norm by default, and try LayerNorm with smaller architecture changes such as `--tcn-levels 3` or hidden size 64. The broader Track A priority remains the Transformer path because the best calibrated Transformer still leads the deep-model board.

Longer single-configuration Transformer diagnostic:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_80e --subsets FD001 --models transformer --epochs 80 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_h32_80e/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_h32_80e/phase2_manifest_audit.md
```

This run verified with `status=ok` across 13 checked artifacts. It selected the same epoch 39 checkpoint as the 40-epoch focused sweep and reproduced the same official-test score, so simply extending this configuration to 80 epochs does not close the remaining HGB gap.

Prediction-diagnostics refresh for the best FD001 Transformer:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_diagnostics --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_h32_40e_diagnostics/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_h32_40e_diagnostics/phase2_manifest_audit.md
```

This run reproduced the focused-sweep Transformer result exactly: selected epoch 39, official-test RMSE 14.339589, NASA score 299.978246, versus the Phase 1 HGB policy RMSE 13.012889 and NASA score 253.465322. It also verified the expanded diagnostics manifest with `status=ok` across 16 checked artifacts.

The prediction diagnostics report shows that the remaining FD001 error is not a pure late-prediction issue. Across 100 official-test units, the Transformer has mean error -2.515066, mean absolute error 10.759874, max absolute error 41.532448, late-prediction rate 0.43, and early-prediction rate 0.57. The highest-error row is unit 67, where actual RUL is 77 and predicted RUL is 118.532448, a late error of 41.532448 cycles. The next major misses are early underestimates on units 45, 93, 73, 11, 96, 12, 89, and 25. That points the next model-quality work toward tail calibration and unit-level failure-mode analysis, not just globally pushing predictions earlier or later.

RUL-bin diagnostics refresh:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_rul_bins --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_h32_40e_rul_bins/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_h32_40e_rul_bins/phase2_manifest_audit.md
```

The expanded manifest verified with `status=ok` across 17 checked artifacts. The RUL-bin report shows the model is strongest near end-of-life and weakest at the mid/high-RUL tails:

| Actual RUL Bin | Units | Mean Error | Mean Abs Error | Max Abs Error | Late Rate | Early Rate |
|---|---:|---:|---:|---:|---:|---:|
| 0-30 | 25 | -1.265065 | 3.968502 | 9.769899 | 0.400000 | 0.600000 |
| 31-60 | 14 | -0.342602 | 6.148713 | 13.652969 | 0.500000 | 0.500000 |
| 61-90 | 15 | 6.563425 | 17.094106 | 41.532448 | 0.733333 | 0.266667 |
| 91-120 | 33 | -1.542672 | 11.262260 | 40.162529 | 0.454545 | 0.545455 |
| 121+ | 13 | -20.202055 | 20.202055 | 32.714767 | 0.000000 | 1.000000 |

This narrows the next experiment: preserve the strong low-RUL behavior, reduce the dangerous late bias in the 61-90 bin, and address the high-RUL compression that pushes every 121+ unit early. Candidate fixes include RUL-range-weighted validation diagnostics, target transformation or loss weighting by actual-RUL bin, and calibration layers on top of the sequence model outputs.

Validation-selection diagnostics refresh:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/phase2_manifest_audit.md
```

The validation-diagnostics manifest verified with `status=ok` across 21 checked artifacts. The official-test score is unchanged, selected epoch 39 with RMSE 14.339589 and NASA score 299.978246, while the validation-selection prediction table adds 3,109 train-only diagnostic rows.

| Split | Rows | Mean Error | Mean Abs Error | Max Abs Error | Late Rate | Early Rate |
|---|---:|---:|---:|---:|---:|---:|
| Official test | 100 | -2.515066 | 10.759874 | 41.532448 | 0.430000 | 0.570000 |
| Validation selection | 3109 | -4.223712 | 11.774581 | 55.148552 | 0.297523 | 0.702477 |

Validation-selection bins confirm that the official-test tail issues are not random test-only noise:

| Actual RUL Bin | Validation Rows | Mean Error | Mean Abs Error | Max Abs Error | Late Rate | Early Rate |
|---|---:|---:|---:|---:|---:|---:|
| 0-30 | 20 | -0.804168 | 6.982858 | 19.718727 | 0.400000 | 0.600000 |
| 31-60 | 600 | 1.782091 | 9.811895 | 55.148552 | 0.485000 | 0.515000 |
| 61-90 | 600 | 3.545005 | 13.864291 | 48.582710 | 0.558333 | 0.441667 |
| 91-120 | 587 | -4.102196 | 14.238667 | 50.725517 | 0.495741 | 0.504259 |
| 121+ | 1302 | -10.678729 | 10.678729 | 53.914551 | 0.000000 | 1.000000 |

The strongest validation-backed diagnosis is now high-RUL compression: every 121+ validation-selection window and every 121+ official-test unit is predicted early. The 61-90 range is still the late-risk band, but validation shows a weaker late skew than the official test set. The next modelling slice should therefore use validation-selection diagnostics to fit or select a correction before official-test re-evaluation, with a priority on high-RUL target compression and mid-RUL late-risk control.

Validation-fitted affine calibration:

```powershell
uv run aerospace-prognostics cmapss-calibrate-deep-predictions --calibration-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_validation_selection_predictions.csv --predictions-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_predictions.csv --output-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_predictions_affine_calibrated.csv --output-calibration-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_affine_calibration.csv --output-diagnostics-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_affine_calibrated_diagnostics.csv --output-rul-bins-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_affine_calibrated_rul_bins.csv --output-markdown artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_affine_calibrated_diagnostics.md
```

The affine calibration was fit only on validation-selection rows for the matching subset/model pair: 3,109 rows, intercept 7.601563, slope 0.963080, mean validation actual RUL 95.715021, and mean raw predicted RUL 91.491309. It slightly improved official-test RMSE from 14.339589 to 14.271605, but worsened the NASA asymmetric score from 299.978246 to 345.146822 because it shifted the model late overall: mean error moved from -2.515066 to 2.391162, late-prediction rate from 0.43 to 0.64, and mean absolute error from 10.759874 to 11.185774. This is a useful diagnostic baseline, not a promotion candidate. It confirms that the next correction should be asymmetric and tail-aware rather than a single global affine shift, especially because validation targets are capped while official-test RUL can expose higher-RUL behavior.

Validation-fitted predicted-bin residual calibration:

```powershell
uv run aerospace-prognostics cmapss-calibrate-deep-predictions --method predicted_bin_residual --calibration-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_validation_selection_predictions.csv --predictions-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_predictions.csv --output-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_predictions_predicted_bin_residual_calibrated.csv --output-calibration-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_predicted_bin_residual_calibration.csv --output-diagnostics-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_predicted_bin_residual_calibrated_diagnostics.csv --output-rul-bins-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_predicted_bin_residual_calibrated_rul_bins.csv --output-markdown artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_predicted_bin_residual_calibrated_diagnostics.md
```

The predicted-bin residual method is inference-safe because bins are assigned from raw predicted RUL, not actual RUL. With the default shrinkage strength of 100, it fit six correction rows: global plus raw-predicted bins 0-30, 31-60, 61-90, 91-120, and 121+. Every bin learned a positive correction because validation-selection predictions are early-biased in each raw-prediction range. Official-test RMSE improved from 14.339589 to 14.224681, but NASA score still worsened from 299.978246 to 341.716670; mean error moved to 1.450547 and late-prediction rate rose to 0.59. A quick shrinkage sweep found the same tradeoff: stronger shrinkage reduces the NASA penalty but approaches the raw model. This keeps residual-bin calibration as a diagnostic tool, while the promotion path should be NASA-aware model selection or training loss.

Validation-fitted predicted-bin NASA-shift calibration:

```powershell
uv run aerospace-prognostics cmapss-calibrate-deep-predictions --method predicted_bin_nasa_shift --shrinkage-strength 200 --nasa-shift-max 10 --nasa-shift-step 1 --calibration-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_validation_selection_predictions.csv --predictions-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_predictions.csv --output-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_predictions_predicted_bin_nasa_shift_s200_calibrated.csv --output-calibration-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s200_calibration.csv --output-diagnostics-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s200_calibrated_diagnostics.csv --output-rul-bins-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s200_calibrated_rul_bins.csv --output-markdown artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s200_calibrated_diagnostics.md
```

This method keeps the inference-safe raw-predicted-RUL bins from predicted-bin residual calibration, but chooses each bin's additive shift by minimizing validation-selection NASA score rather than mean residual error. A compact shrinkage sweep over strengths 0, 25, 50, 100, 200, and 500 found `--shrinkage-strength 200` as the best official-test NASA point for the raw MSE Transformer; the shift-search maximum did not change the selected shifts for the tested 10/20/30-cycle bounds. The selected run fit six correction rows: global plus raw-predicted bins 0-30, 31-60, 61-90, 91-120, and 121+. Official-test NASA score improved from 299.978246 to 278.840518, while RMSE worsened modestly from 14.339589 to 14.460923. The tradeoff is visible in the diagnostics: mean absolute error moved from 10.759874 to 10.725938, late-prediction rate from 0.43 to 0.47, and max absolute error improved from 41.532448 to 39.392414. This is not yet enough to beat the Phase 1 HGB policy score of 253.465322, but it is the first validation-fitted calibration that improves the official-test NASA metric instead of only RMSE.

NASA-aware loss smoke checks:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_10e_mse_reference --subsets FD001 --models transformer --epochs 10 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --training-loss mse
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_10e_mse_nasa_blend_w0p001 --subsets FD001 --models transformer --epochs 10 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --training-loss mse_nasa_blend_w0p001
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_10e_mse_nasa_blend_w0p0001 --subsets FD001 --models transformer --epochs 10 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --training-loss mse_nasa_blend_w0p0001
```

These short runs are loss-path checks, not promotion candidates. All three manifests verified with `status=ok` across 21 checked artifacts. At a 10-epoch budget the blended losses stay effectively on the MSE trajectory, while the pure NASA surrogate undertrains badly:

| Training Loss | Epochs | Official Test RMSE | Official Test NASA Score | Mean Error | Late Rate |
|---|---:|---:|---:|---:|---:|
| `mse` | 10 | 45.047094 | 10736.524843 | -32.541825 | 0.190000 |
| `mse_nasa_blend_w0p0001` | 10 | 45.049205 | 10738.845270 | -32.543126 | 0.190000 |
| `mse_nasa_blend_w0p001` | 10 | 45.068130 | 10759.665550 | -32.554764 | 0.190000 |
| `nasa_surrogate` | 10 | 61.077437 | 52942.409835 | -45.432663 | 0.240000 |

The takeaway is useful even though the scores are weak: pure NASA-surrogate optimization is too steep as a standalone objective, while the blended variants are safe knobs for longer Transformer sweeps. Any promotion-quality comparison should use the known 40-epoch Transformer budget or longer; 10 epochs mostly measures undertraining.

40-epoch blended-loss Transformer check:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_mse_nasa_blend_w0p0001 --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --training-loss mse_nasa_blend_w0p0001
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_mse_nasa_blend_w0p001 --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --training-loss mse_nasa_blend_w0p001
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_h32_40e_mse_nasa_blend_w0p0001/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_h32_40e_mse_nasa_blend_w0p0001/phase2_manifest_audit.md
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_h32_40e_mse_nasa_blend_w0p001/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_h32_40e_mse_nasa_blend_w0p001/phase2_manifest_audit.md
```

Both manifests verified with `status=ok` across 21 checked artifacts. The blended losses stayed on the same selected checkpoint pattern as MSE, selecting epoch 39, and produced only tiny official-test changes:

| Training Loss | Selected Epoch | Validation RMSE | Validation NASA Score | Official Test RMSE | Official Test NASA Score | Mean Error | Late Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mse` | 39 | 15.026410 | 11200.658784 | 14.339589 | 299.978246 | -2.515066 | 0.430000 |
| `mse_nasa_blend_w0p0001` | 39 | 15.026353 | 11200.184598 | 14.339547 | 299.974989 | -2.515430 | 0.430000 |
| `mse_nasa_blend_w0p001` | 39 | 15.025851 | 11195.969932 | 14.339186 | 299.946394 | -2.518681 | 0.430000 |

The `w0p001` blend is the best of this small loss-shaping check, but the improvement is marginal: NASA score moves by about -0.031853 versus the MSE Transformer and remains well behind the Phase 1 HGB policy score of 253.465322. That means loss blending is safe to carry into later sweeps, but it is not by itself the missing model-quality step. The next stronger candidate should be architecture or constraint work: residual temporal blocks, monotonic degradation penalties, or target/health-index shaping.

40-epoch asymmetric late-weighted MSE Transformer check:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5 --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --training-loss asymmetric_mse_late_w1p5
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w2 --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --training-loss asymmetric_mse_late_w2
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w3 --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --training-loss asymmetric_mse_late_w3
```

All three manifests verified with `status=ok` across 21 checked artifacts. The late-weighted MSE family gives a steadier NASA-aware training knob than the pure exponential surrogate. The gentlest tested weight is best: `asymmetric_mse_late_w1p5` improves official-test RMSE from 14.339589 to 14.154401 and NASA score from 299.978246 to 279.487610 versus the raw MSE Transformer. It is the best uncalibrated deep-model result so far and is much stronger than the tiny MSE+NASA blends, but it still does not beat the tuned predicted-bin NASA-shift calibrated score of 278.840518. The `w3` run overcorrects toward early predictions, dropping late rate to 0.33 but worsening RMSE and giving back most of the NASA gain.

| Training Loss | Selected Epoch | Validation RMSE | Validation NASA Score | Official Test RMSE | Official Test NASA Score | Mean Error | Late Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `asymmetric_mse_late_w1p5` | 38 | 15.035633 | 10366.646450 | 14.154401 | 279.487610 | -2.310351 | 0.520000 |
| `asymmetric_mse_late_w2` | 38 | 15.531929 | 10395.852405 | 14.511629 | 280.140792 | -3.781685 | 0.430000 |
| `asymmetric_mse_late_w3` | 38 | 16.459933 | 10966.431567 | 15.201202 | 298.970218 | -5.745096 | 0.330000 |

Asymmetric late-weighted MSE plus NASA-shift calibration:

```powershell
uv run aerospace-prognostics cmapss-calibrate-deep-predictions --method predicted_bin_nasa_shift --shrinkage-strength 500 --nasa-shift-max 10 --nasa-shift-step 1 --calibration-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_deep_validation_selection_predictions.csv --predictions-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_deep_predictions.csv --output-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_deep_predictions_predicted_bin_nasa_shift_s500_calibrated.csv --output-calibration-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s500_calibration.csv --output-diagnostics-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s500_calibrated_diagnostics.csv --output-rul-bins-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s500_calibrated_rul_bins.csv --output-markdown artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s500_calibrated_diagnostics.md
```

Combining the best uncalibrated deep model with validation-fitted predicted-bin NASA-shift calibration produced the best deep FD001 NASA score so far. A compact shrinkage sweep over 0, 25, 50, 100, 200, and 500 found `--shrinkage-strength 500` best: NASA score improved from the uncalibrated asymmetric-loss model's 279.487610 to 272.578820, while RMSE moved from 14.154401 to 14.267129. The calibrated diagnostics show mean error -2.733377, mean absolute error 10.569140, max absolute error 39.383894, and late rate 0.500000. This still trails the Phase 1 HGB policy NASA score of 253.465322, but it confirms the current deep promotion path: pair a NASA-aware training objective with validation-only, inference-safe calibration rather than relying on either alone.

| Shrinkage Strength | Official Test RMSE | Official Test NASA Score |
|---:|---:|---:|
| 500 | 14.267129 | 272.578820 |
| 200 | 14.294852 | 272.855933 |
| 100 | 14.330953 | 274.358891 |
| 50 | 14.391875 | 277.255074 |
| 25 | 14.469072 | 280.724865 |
| 0 | 14.660178 | 288.942564 |

Target-weighted MSE check:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5 --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --training-loss target_weighted_mse_mid_high_w1p5
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5/phase2_manifest_audit.md
uv run aerospace-prognostics cmapss-calibrate-deep-predictions --method predicted_bin_nasa_shift --shrinkage-strength 500 --nasa-shift-max 10 --nasa-shift-step 1 --calibration-csv artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5/results/cmapss_deep_validation_selection_predictions.csv --predictions-csv artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5/results/cmapss_deep_predictions.csv --output-csv artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5/results/cmapss_deep_predictions_predicted_bin_nasa_shift_s500_calibrated.csv --output-calibration-csv artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s500_calibration.csv --output-diagnostics-csv artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s500_calibrated_diagnostics.csv --output-rul-bins-csv artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s500_calibrated_rul_bins.csv --output-markdown artifacts/phase2_fd001_transformer_h32_40e_target_weighted_mid_high_w1p5/results/cmapss_deep_prediction_predicted_bin_nasa_shift_s500_calibrated_diagnostics.md
```

The mid/high-RUL target-weighted MSE check verified with `status=ok` across 21 checked artifacts. It slightly improved RMSE relative to the raw MSE Transformer but worsened the asymmetric score: official-test RMSE 14.292994 and NASA score 303.568376, with mean error -1.808940 and late rate 0.430000. Applying the same predicted-bin NASA-shift calibration with shrinkage 500 improved the weighted model to RMSE 14.267305 and NASA score 281.424365, with mean error -2.334318, MAE 10.586762, max absolute error 39.791366, and late rate 0.470000. That is useful but still behind the asymmetric-loss plus NASA-shift result of NASA 272.578820. Target weighting is therefore a diagnostic lever for RMSE/tail pressure, not the current promotion path.

FD001 deep leaderboard refresh:

```powershell
uv run aerospace-prognostics cmapss-compare-rul-results --baseline-csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_hgb_policy_baseline.csv --candidate-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_compare.csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_deep_compare.csv --prediction-csv artifacts/phase2_fd001_transformer_h32_40e_validation_diagnostics/results/cmapss_deep_predictions_predicted_bin_nasa_shift_s200_calibrated.csv artifacts/phase2_fd001_transformer_h32_40e_asymmetric_mse_late_w1p5/results/cmapss_deep_predictions_predicted_bin_nasa_shift_s500_calibrated.csv --prediction-model-suffixes predicted_bin_nasa_shift_s200 predicted_bin_nasa_shift_s500 --prediction-label phase2_calibrated --output-csv artifacts/phase2_fd001_deep_leaderboard/cmapss_fd001_deep_leaderboard.csv --output-markdown artifacts/phase2_fd001_deep_leaderboard/cmapss_fd001_deep_leaderboard.md
```

The comparison command now accepts prediction CSVs in addition to result tables, summarizing each prediction file into RMSE and NASA score before ranking. That keeps calibrated candidates in the same leaderboard without hand-built intermediate result CSVs. The refreshed FD001 leaderboard ranks HGB first, then the asymmetric-loss plus NASA-shift combo, then raw-MSE NASA-shift calibration, then the uncalibrated asymmetric and raw-MSE Transformer runs:

| Rank | Phase | Model Family | Official Test RMSE | Official Test NASA Score | NASA Delta vs HGB |
|---:|---|---|---:|---:|---:|
| 1 | `phase1_hgb_policy` | HGB policy | 13.012889 | 253.465322 | 0.000000 |
| 2 | `phase2_calibrated` | Asymmetric Transformer + NASA-shift s500 | 14.267129 | 272.578820 | 19.113497 |
| 3 | `phase2_calibrated` | MSE Transformer + NASA-shift s200 | 14.460923 | 278.840518 | 25.375195 |
| 4 | `phase2_deep` | Asymmetric Transformer | 14.154401 | 279.487610 | 26.022287 |
| 5 | `phase2_deep` | MSE Transformer | 14.339589 | 299.978246 | 46.512924 |

Residual CNN architecture smoke check:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_rescnn_h32_20e_smoke --subsets FD001 --models rescnn --epochs 20 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --tcn-levels 3 --validation-horizon 30 --checkpoint-policy validation_nasa
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_rescnn_h32_20e_smoke/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_rescnn_h32_20e_smoke/phase2_manifest_audit.md
```

The residual CNN adds same-length temporal convolution blocks with skip connections behind the shared Phase 2 sequence-training path. The manifest verified with `status=ok` across 21 checked artifacts. This first 20-epoch FD001 smoke run selected epoch 7 and landed between the simple CNN and the Transformer: official-test RMSE 21.711146 and NASA score 2003.639663. It is a useful architecture candidate for future sweeps, but not a promotion candidate yet; the 40-epoch Transformer remains the strongest deep model so far, and the Phase 1 HGB policy remains the overall FD001 leader.

Transformer RUL-cap sensitivity check:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_rulcap100 --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --rul-cap 100
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2_fd001_transformer_h32_40e_rulcap150 --subsets FD001 --models transformer --epochs 40 --batch-size 256 --hidden-sizes 32 --learning-rates 0.001 --validation-horizon 30 --checkpoint-policy validation_nasa --rul-cap 150
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_h32_40e_rulcap100/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_h32_40e_rulcap100/phase2_manifest_audit.md
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2_fd001_transformer_h32_40e_rulcap150/phase2_run_manifest.json --output-markdown artifacts/phase2_fd001_transformer_h32_40e_rulcap150/phase2_manifest_audit.md
```

Both manifests verified with `status=ok` across 21 checked artifacts. The target cap is a real modelling lever, but this small sweep did not beat the standard cap-125 Transformer. Cap 100 made the model safer against late predictions but compressed the high-RUL tail too hard; cap 150 reduced that tail compression but shifted the model late enough to hurt the asymmetric NASA score.

| RUL Cap | Selected Epoch | Validation RMSE | Validation NASA Score | Official Test RMSE | Official Test NASA Score | Mean Error | Late Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 31 | 9.650666 | 4431.123146 | 17.736904 | 397.619796 | -8.647011 | 0.340000 |
| 125 | 39 | 15.026410 | 11200.658784 | 14.339589 | 299.978246 | -2.515066 | 0.430000 |
| 150 | 35 | 20.662572 | 26347.826368 | 16.157872 | 626.009164 | 2.417123 | 0.530000 |

The result keeps cap 125 as the current default for FD001. More importantly, it rules out a simple "raise the cap to fix high-RUL compression" story: cap 150 improves the 121+ bin bias but creates too much mid-RUL and overall late risk, while cap 100 protects NASA score better than cap 150 at the cost of severe high-RUL underprediction. Future target-shaping work should be local or asymmetric rather than a single global cap change.

## FD001 First Result

| Checkpoint | Model | Selected Epoch | Validation RMSE | Validation NASA Score | Official Test RMSE | Official Test NASA Score |
|---|---|---:|---:|---:|---:|---:|
| Phase 1 reference | `hist_gradient_boosting_regime_engineered_w10_r6_default` | n/a | n/a | n/a | 13.012889 | 253.465322 |
| Phase 2 final-epoch smoke baseline | `cnn_1d_w30_e50_c32_final_e50` | 50 | 28.228949 | 1154.680228 | 22.755049 | 3854.711070 |
| Phase 2 validation-selected smoke baseline | `cnn_1d_w30_e50_c32_best_e2` | 2 | 14.486795 | 49.644294 | 45.900152 | 20542.376235 |
| Phase 2 workflow smoke baseline | `compare_tcn_h16_lr0p001_tcn_w30_e3_c16_l1_k3_best_e3` | 3 | 53.316125 | 1403204.206645 | 46.968692 | 19115.956643 |
| Phase 2 workflow smoke baseline | `compare_cnn_h16_lr0p001_cnn_1d_w30_e3_c16_best_e3` | 3 | 53.269599 | 1570720.650364 | 46.877603 | 21879.546393 |
| Phase 2 benchmark-shaped baseline | `compare_transformer_h32_lr0p001_transformer_w30_e20_d32_h4_l1_ff64_best_e20` | 20 | 19.251686 | 15309.424315 | 16.173196 | 352.015482 |
| Phase 2 focused sweep | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_best_e39` | 39 | 15.026410 | 11200.658784 | 14.339589 | 299.978246 |
| Phase 2 longer diagnostic | `compare_transformer_h32_lr0p001_transformer_w30_e80_d32_h4_l1_ff64_best_e39` | 39 | 15.026410 | 11200.658784 | 14.339589 | 299.978246 |
| Phase 2 prediction diagnostic refresh | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_best_e39` | 39 | 15.026410 | 11200.658784 | 14.339589 | 299.978246 |
| Phase 2 validation-affine diagnostic | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_best_e39_affine_calibrated` | 39 | 15.026410 | 11200.658784 | 14.271605 | 345.146822 |
| Phase 2 predicted-bin residual diagnostic | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_best_e39_predicted_bin_residual` | 39 | 15.026410 | 11200.658784 | 14.224681 | 341.716670 |
| Phase 2 predicted-bin NASA-shift diagnostic | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_best_e39_predicted_bin_nasa_shift_s200` | 39 | 15.026410 | 11200.658784 | 14.460923 | 278.840518 |
| Phase 2 40-epoch blended-loss check | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_loss_mse_nasa_blend_w0p001_best_e39` | 39 | 15.025851 | 11195.969932 | 14.339186 | 299.946394 |
| Phase 2 40-epoch blended-loss check | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_loss_mse_nasa_blend_w0p0001_best_e39` | 39 | 15.026353 | 11200.184598 | 14.339547 | 299.974989 |
| Phase 2 40-epoch asymmetric-loss check | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_loss_asymmetric_mse_late_w1p5_best_e38` | 38 | 15.035633 | 10366.646450 | 14.154401 | 279.487610 |
| Phase 2 40-epoch asymmetric-loss check | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_loss_asymmetric_mse_late_w2_best_e38` | 38 | 15.531929 | 10395.852405 | 14.511629 | 280.140792 |
| Phase 2 40-epoch asymmetric-loss check | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_loss_asymmetric_mse_late_w3_best_e38` | 38 | 16.459933 | 10966.431567 | 15.201202 | 298.970218 |
| Phase 2 asymmetric-loss NASA-shift diagnostic | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_loss_asymmetric_mse_late_w1p5_best_e38_predicted_bin_nasa_shift_s500` | 38 | 15.035633 | 10366.646450 | 14.267129 | 272.578820 |
| Phase 2 RUL-cap sensitivity | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_best_e31` | 31 | 9.650666 | 4431.123146 | 17.736904 | 397.619796 |
| Phase 2 RUL-cap sensitivity | `compare_transformer_h32_lr0p001_transformer_w30_e40_d32_h4_l1_ff64_best_e35` | 35 | 20.662572 | 26347.826368 | 16.157872 | 626.009164 |
| Phase 2 residual CNN smoke | `compare_rescnn_h32_lr0p001_rescnn_w30_e20_c32_b3_k3_best_e7` | 7 | 25.695627 | 52381.612522 | 21.711146 | 2003.639663 |
| Phase 2 10-epoch loss smoke | `compare_transformer_h32_lr0p001_transformer_w30_e10_d32_h4_l1_ff64_best_e10` | 10 | 54.524260 | 421709.482281 | 45.047094 | 10736.524843 |
| Phase 2 10-epoch loss smoke | `compare_transformer_h32_lr0p001_transformer_w30_e10_d32_h4_l1_ff64_loss_mse_nasa_blend_w0p0001_best_e10` | 10 | 54.526576 | 421800.208048 | 45.049205 | 10738.845270 |
| Phase 2 10-epoch loss smoke | `compare_transformer_h32_lr0p001_transformer_w30_e10_d32_h4_l1_ff64_loss_mse_nasa_blend_w0p001_best_e10` | 10 | 54.547339 | 422614.178776 | 45.068130 | 10759.665550 |
| Phase 2 10-epoch loss smoke | `compare_transformer_h32_lr0p001_transformer_w30_e10_d32_h4_l1_ff64_loss_nasa_surrogate_best_e10` | 10 | 72.427759 | 2080357.379217 | 61.077437 | 52942.409835 |
| Phase 2 focused sweep | `compare_transformer_h64_lr0p001_transformer_w30_e40_d64_h4_l1_ff128_best_e22` | 22 | 15.341944 | 11914.289576 | 14.759614 | 328.647072 |
| Phase 2 focused sweep | `compare_tcn_h64_lr0p0003_tcn_w30_e40_c64_l3_k3_best_e34` | 34 | 21.604288 | 26895.387798 | 17.417167 | 439.757150 |
| Phase 2 benchmark-shaped baseline | `compare_tcn_h32_lr0p001_tcn_w30_e20_c32_l2_k3_best_e13` | 13 | 23.086388 | 34936.741283 | 19.065325 | 807.884690 |
| Phase 2 benchmark-shaped baseline | `compare_cnn_h32_lr0p001_cnn_1d_w30_e20_c32_best_e13` | 13 | 26.932023 | 55135.426955 | 25.045474 | 3858.156520 |
| Phase 2 benchmark-shaped baseline | `compare_bilstm_h32_lr0p001_bilstm_w30_e20_h32_l1_best_e20` | 20 | 45.309112 | 168914.199343 | 37.189642 | 4237.429679 |

Both CNN results are worse than the Phase 1 HGB policy on FD001. That is useful, not alarming: it confirms the training and scoring path works while showing that this first architecture is underfit and poorly calibrated for the official test split.

The validation-selected run is especially important. It chooses epoch 2 because the original train-only validation split used only one final window per held-out unit, so a tiny validation signal looked strong while the official test score collapsed. Phase 2 now exports rolling validation-selection windows to reduce that failure mode before relying on early stopping as a model-selection signal.

The 20-epoch benchmark, 40-epoch focused sweep, enhanced-TCN check, 80-epoch diagnostic, prediction-diagnostics refresh, RUL-bin refresh, validation-selection diagnostics refresh, calibration checks, asymmetric-loss checks, and target-weighted loss check show the next Track A modelling problem clearly: deep sequence models are working end to end, but the classical HGB policy is still stronger on FD001. The Transformer is the strongest deep family so far, especially with hidden size 32 and learning rate 0.001. The lower 0.0003 learning rate undertrained badly at hidden size 32, while hidden size 64 improved but still trailed the smaller 0.001 run. Extending the best configuration to 80 epochs did not help because validation selection still stopped at epoch 39. The full enhanced-TCN bundle underperformed badly; narrower ablations show last-step pooling plus LayerNorm is much better than mean pooling or weight norm, but still not competitive with the Transformer. Validation-fitted NASA-shift calibration and asymmetric late-weighted MSE both improve NASA behavior, and their combination is the current best deep result. Target-weighted MSE improved RMSE pressure but did not beat the asymmetric NASA score. The immediate follow-up should move from "train longer" or "shift everything" to richer NASA-aware architecture diagnostics: tail-sensitive validation, regularization, residual temporal blocks, monotonic or health-index constraints, and unit-level error analysis before expanding the full grid to FD002-FD004.

## Current Interpretation

The Phase 2 pipeline is now real: exported sequence windows feed torch models, CNN, residual CNN, LSTM/BiLSTM, TCN, and Transformer baselines train from the CLI/workflow path, rolling validation-selection windows drive checkpoint choice, validation final-window artifacts remain available for reporting, official-test and validation-selection predictions are scored with the project metrics and emitted for per-window diagnostics, JSON/CSV outputs use the same result container as the classical baselines, optional history JSON records per-epoch training loss plus validation metrics, the comparison command can produce a single architecture/learning-rate sweep table, the reporting command ranks Phase 2 candidates against the Phase 1 HGB policy baseline, and `phase2-cmapss` ties the full Track A workflow together. The first real FD001 workflow smoke run produced one sequence export, two deep results, and three comparison rows.

The next deep-learning work should focus on model quality rather than plumbing:

- Continue FD001 with focused Transformer architecture, regularization, and validation diagnostics before expanding to FD002-FD004.
- Keep RUL cap 125 as the current FD001 default; cap 100 and cap 150 both worsened the best Transformer NASA score, so future target shaping should be local or asymmetric rather than a single global cap move.
- Use the workflow parameters for wider/deeper CNNs, residual or TCN-style blocks, attention heads, regularization, and learning-rate sweeps, but avoid promoting the full LayerNorm + weight-normalized + mean-pooled TCN bundle unless narrower ablations recover the score.
- Use `cmapss_deep_prediction_diagnostics.md`, `cmapss_deep_predictions.csv`, and the validation-selection diagnostic artifacts to inspect late-prediction clusters, high-absolute-error units, and validation-vs-test mismatch before adding broader grids.
- Use validation-selection errors by unit, end cycle, and RUL range as the first decision surface for calibration and tail-sensitive modelling before rechecking official-test behavior; affine and predicted-bin residual checks improved RMSE but failed the NASA-score tradeoff, asymmetric late-weighted MSE improved the best uncalibrated deep NASA score to 279.487610, and pairing that model with predicted-bin NASA-shift calibration improved the best deep NASA score to 272.578820.
- Use the ranked report command for every Phase 2 run bundle and calibrated prediction CSV, especially when checking the known FD003 validation mismatch.
- Keep all sensors for now; Phase 1 sensor-filter validation showed EDA filtering harms FD002 and FD004.
