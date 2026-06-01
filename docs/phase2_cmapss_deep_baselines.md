# Phase 2 C-MAPSS Deep Baselines

This note records the first Phase 2 sequence-model checkpoint. Phase 1 ended with reproducible C-MAPSS sequence exports, so Phase 2 starts by proving that a PyTorch model can train from those tensors and report the same RMSE and NASA asymmetric RUL score used by the classical baseline.

## Track A Start

| Field | Value |
|---|---|
| Dataset | NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set |
| Input artifacts | `artifacts/sequences/cmapss/<subset>/{train,validation,validation_selection,test}_sequences.npz` |
| Sequence export command | `uv run aerospace-prognostics cmapss-export-sequences --data-dir data/raw/cmapss --output-dir artifacts/sequences/cmapss --window-size 30 --stride 1` |
| Deep model baselines | Compact PyTorch 1D-CNN; LSTM/BiLSTM; TCN; Transformer encoder |
| Window size | 30 cycles |
| Features | 24 standardized operating-setting and sensor channels |
| Optimizer | Adam |
| Loss | MSE |
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

Transformer encoder baseline:

```powershell
uv run aerospace-prognostics cmapss-transformer-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --batch-size 256 --d-model 64 --num-heads 4 --num-layers 2 --dim-feedforward 128 --checkpoint-policy validation_nasa --output-json artifacts/results/cmapss_transformer_fd001_baseline.json --output-csv artifacts/results/cmapss_transformer_fd001_baseline.csv --history-json artifacts/results/cmapss_transformer_fd001_history.json
```

Compact architecture and learning-rate comparison:

```powershell
uv run aerospace-prognostics cmapss-deep-baseline-compare --sequence-dir artifacts/sequences/cmapss --subsets FD001 --models cnn bilstm tcn transformer --epochs 50 --batch-size 256 --hidden-sizes 32 64 --learning-rates 0.001 0.0003 --checkpoint-policy validation_nasa --output-json artifacts/results/cmapss_deep_compare_fd001.json --output-csv artifacts/results/cmapss_deep_compare_fd001.csv
```

Ranked Phase 1 versus Phase 2 report:

```powershell
uv run aerospace-prognostics cmapss-compare-rul-results --baseline-csv artifacts/results/cmapss_hgb_policy_baseline.csv --candidate-csv artifacts/results/cmapss_deep_compare_fd001.csv --output-csv artifacts/results/cmapss_phase2_model_comparison.csv --output-markdown artifacts/results/cmapss_phase2_model_comparison.md
```

Reproducible Phase 2 workflow:

```powershell
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2 --subsets FD001 --models cnn bilstm tcn transformer --epochs 50 --hidden-sizes 32 64 --learning-rates 0.001 0.0003
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2/phase2_run_manifest.json --output-markdown artifacts/phase2/phase2_manifest_audit.md
```

The workflow writes `phase2_summary.md` and `phase2_run_manifest.json` under the artifact directory. It also writes `results/cmapss_deep_predictions.csv`, a per-unit official-test diagnostics table with actual RUL, predicted RUL, signed error, absolute error, end cycle, and early/late error split for every deep-model candidate. The companion `results/cmapss_deep_prediction_diagnostics.csv`, `results/cmapss_deep_prediction_rul_bins.csv`, and markdown report summarize mean error, mean/max absolute error, late-prediction rate, actual-RUL-bin calibration, and the highest-error units. The same diagnostics are now emitted for rolling validation-selection windows under `results/cmapss_deep_validation_selection_predictions.csv`, `results/cmapss_deep_validation_selection_prediction_diagnostics.csv`, `results/cmapss_deep_validation_selection_prediction_rul_bins.csv`, and `results/cmapss_deep_validation_selection_prediction_diagnostics.md`, so calibration and tail-error ideas can be judged on train-only validation behavior before the official test table is touched. The manifest records run parameters, artifact paths, SHA-256/size checksums for the model outputs, prediction diagnostics, and sequence bundles, Python/platform/dependency versions, and Git commit state so a Phase 2 C-MAPSS run can be audited or reproduced from one bundle. The verify command checks manifest structure, referenced artifact existence, artifact checksums, and CSV row counts; with `--output-markdown`, it also writes a compact audit report.

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
| Phase 2 focused sweep | `compare_transformer_h64_lr0p001_transformer_w30_e40_d64_h4_l1_ff128_best_e22` | 22 | 15.341944 | 11914.289576 | 14.759614 | 328.647072 |
| Phase 2 focused sweep | `compare_tcn_h64_lr0p0003_tcn_w30_e40_c64_l3_k3_best_e34` | 34 | 21.604288 | 26895.387798 | 17.417167 | 439.757150 |
| Phase 2 benchmark-shaped baseline | `compare_tcn_h32_lr0p001_tcn_w30_e20_c32_l2_k3_best_e13` | 13 | 23.086388 | 34936.741283 | 19.065325 | 807.884690 |
| Phase 2 benchmark-shaped baseline | `compare_cnn_h32_lr0p001_cnn_1d_w30_e20_c32_best_e13` | 13 | 26.932023 | 55135.426955 | 25.045474 | 3858.156520 |
| Phase 2 benchmark-shaped baseline | `compare_bilstm_h32_lr0p001_bilstm_w30_e20_h32_l1_best_e20` | 20 | 45.309112 | 168914.199343 | 37.189642 | 4237.429679 |

Both CNN results are worse than the Phase 1 HGB policy on FD001. That is useful, not alarming: it confirms the training and scoring path works while showing that this first architecture is underfit and poorly calibrated for the official test split.

The validation-selected run is especially important. It chooses epoch 2 because the original train-only validation split used only one final window per held-out unit, so a tiny validation signal looked strong while the official test score collapsed. Phase 2 now exports rolling validation-selection windows to reduce that failure mode before relying on early stopping as a model-selection signal.

The 20-epoch benchmark, 40-epoch focused sweep, 80-epoch diagnostic, prediction-diagnostics refresh, RUL-bin refresh, validation-selection diagnostics refresh, and affine calibration check show the next Track A modelling problem clearly: deep sequence models are working end to end, but the classical HGB policy is still stronger on FD001. The Transformer is the strongest deep family so far, especially with hidden size 32 and learning rate 0.001. The lower 0.0003 learning rate undertrained badly at hidden size 32, while hidden size 64 improved but still trailed the smaller 0.001 run. Extending the best configuration to 80 epochs did not help because validation selection still stopped at epoch 39. A global validation-fitted affine correction improves RMSE slightly but worsens NASA score, so the immediate follow-up should move from "train longer" or "shift everything" to asymmetric calibration and architecture diagnostics: target/loss shaping, tail-sensitive validation, regularization, residual temporal blocks, and unit-level error analysis before expanding the full grid to FD002-FD004.

## Current Interpretation

The Phase 2 pipeline is now real: exported sequence windows feed torch models, CNN, LSTM/BiLSTM, TCN, and Transformer baselines train from the CLI, rolling validation-selection windows drive checkpoint choice, validation final-window artifacts remain available for reporting, official-test and validation-selection predictions are scored with the project metrics and emitted for per-window diagnostics, JSON/CSV outputs use the same result container as the classical baselines, optional history JSON records per-epoch training loss plus validation metrics, the comparison command can produce a single architecture/learning-rate sweep table, the reporting command ranks Phase 2 candidates against the Phase 1 HGB policy baseline, and `phase2-cmapss` ties the full Track A workflow together. The first real FD001 workflow smoke run produced one sequence export, two deep results, and three comparison rows.

The next deep-learning work should focus on model quality rather than plumbing:

- Continue FD001 with focused Transformer architecture, regularization, and validation diagnostics before expanding to FD002-FD004.
- Use the workflow parameters for wider/deeper CNNs, residual or TCN-style blocks, attention heads, regularization, and learning-rate sweeps.
- Use `cmapss_deep_prediction_diagnostics.md`, `cmapss_deep_predictions.csv`, and the validation-selection diagnostic artifacts to inspect late-prediction clusters, high-absolute-error units, and validation-vs-test mismatch before adding broader grids.
- Use validation-selection errors by unit, end cycle, and RUL range as the first decision surface for calibration and tail-sensitive modelling before rechecking official-test behavior; the first global affine check improved RMSE but failed the NASA-score tradeoff.
- Use the ranked report command for every Phase 2 run bundle, especially when checking the known FD003 validation mismatch.
- Keep all sensors for now; Phase 1 sensor-filter validation showed EDA filtering harms FD002 and FD004.
