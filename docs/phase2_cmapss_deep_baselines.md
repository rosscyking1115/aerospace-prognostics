# Phase 2 C-MAPSS Deep Baselines

This note records the first Phase 2 sequence-model checkpoint. Phase 1 ended with reproducible C-MAPSS sequence exports, so Phase 2 starts by proving that a PyTorch model can train from those tensors and report the same RMSE and NASA asymmetric RUL score used by the classical baseline.

## Track A Start

| Field | Value |
|---|---|
| Dataset | NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set |
| Input artifacts | `artifacts/sequences/cmapss/<subset>/{train,validation,test}_sequences.npz` |
| Sequence export command | `uv run aerospace-prognostics cmapss-export-sequences --data-dir data/raw/cmapss --output-dir artifacts/sequences/cmapss --window-size 30 --stride 1` |
| First deep model | Compact PyTorch 1D-CNN |
| Window size | 30 cycles |
| Features | 24 standardized operating-setting and sensor channels |
| Optimizer | Adam |
| Loss | MSE |
| Random seed | 42 |
| Metrics | RMSE and NASA asymmetric RUL score |
| Checkpoint policies | `validation_nasa` or `final` |

The first model is intentionally compact. It is a plumbing baseline for sequence training, validation tracking, checkpoint selection, CLI execution, result serialization, and repeatable tests, not the final deep-learning architecture.

## Command

```powershell
uv run aerospace-prognostics cmapss-cnn-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --batch-size 256 --hidden-channels 32 --learning-rate 0.001 --checkpoint-policy final --output-json artifacts/results/cmapss_cnn_fd001_final_baseline.json --output-csv artifacts/results/cmapss_cnn_fd001_final_baseline.csv --history-json artifacts/results/cmapss_cnn_fd001_final_history.json
```

Validation-selected checkpoint run:

```powershell
uv run aerospace-prognostics cmapss-cnn-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --batch-size 256 --hidden-channels 32 --learning-rate 0.001 --checkpoint-policy validation_nasa --output-json artifacts/results/cmapss_cnn_fd001_baseline.json --output-csv artifacts/results/cmapss_cnn_fd001_baseline.csv --history-json artifacts/results/cmapss_cnn_fd001_history.json
```

## FD001 First Result

| Checkpoint | Model | Selected Epoch | Validation RMSE | Validation NASA Score | Official Test RMSE | Official Test NASA Score |
|---|---|---:|---:|---:|---:|---:|
| Phase 1 reference | `hist_gradient_boosting_regime_engineered_w10_r6_default` | n/a | n/a | n/a | 13.012889 | 253.465322 |
| Phase 2 final-epoch smoke baseline | `cnn_1d_w30_e50_c32_final_e50` | 50 | 28.228949 | 1154.680228 | 22.755049 | 3854.711070 |
| Phase 2 validation-selected smoke baseline | `cnn_1d_w30_e50_c32_best_e2` | 2 | 14.486795 | 49.644294 | 45.900152 | 20542.376235 |

Both CNN results are worse than the Phase 1 HGB policy on FD001. That is useful, not alarming: it confirms the training and scoring path works while showing that this first architecture is underfit and poorly calibrated for the official test split.

The validation-selected run is especially important. It chooses epoch 2 because the small train-only validation split looks strong there, but that checkpoint performs badly on the official test set. Phase 2 should therefore improve validation design before relying on early stopping as a model-selection signal.

## Current Interpretation

The Phase 2 pipeline is now real: exported sequence windows feed a torch model, the model trains from the CLI, validation and official-test predictions are scored with the project metrics, JSON/CSV outputs use the same result container as the classical baselines, and optional history JSON records per-epoch training loss plus validation metrics.

The next deep-learning work should focus on model quality rather than plumbing:

- Improve the validation split before using early stopping as the default selection signal.
- Run FD001 architecture checks with wider/deeper CNNs, residual or TCN-style blocks, and learning-rate sweeps.
- Add LSTM/BiLSTM and TCN baselines against the same sequence exports.
- Compare all Phase 2 runs against the Phase 1 HGB policy table, especially the known FD003 validation mismatch.
- Keep all sensors for now; Phase 1 sensor-filter validation showed EDA filtering harms FD002 and FD004.
