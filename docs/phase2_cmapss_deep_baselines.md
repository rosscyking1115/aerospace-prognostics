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

## FD001 First Result

| Checkpoint | Model | Selected Epoch | Validation RMSE | Validation NASA Score | Official Test RMSE | Official Test NASA Score |
|---|---|---:|---:|---:|---:|---:|
| Phase 1 reference | `hist_gradient_boosting_regime_engineered_w10_r6_default` | n/a | n/a | n/a | 13.012889 | 253.465322 |
| Phase 2 final-epoch smoke baseline | `cnn_1d_w30_e50_c32_final_e50` | 50 | 28.228949 | 1154.680228 | 22.755049 | 3854.711070 |
| Phase 2 validation-selected smoke baseline | `cnn_1d_w30_e50_c32_best_e2` | 2 | 14.486795 | 49.644294 | 45.900152 | 20542.376235 |

Both CNN results are worse than the Phase 1 HGB policy on FD001. That is useful, not alarming: it confirms the training and scoring path works while showing that this first architecture is underfit and poorly calibrated for the official test split.

The validation-selected run is especially important. It chooses epoch 2 because the original train-only validation split used only one final window per held-out unit, so a tiny validation signal looked strong while the official test score collapsed. Phase 2 now exports rolling validation-selection windows to reduce that failure mode before relying on early stopping as a model-selection signal.

## Current Interpretation

The Phase 2 pipeline is now real: exported sequence windows feed torch models, CNN, LSTM/BiLSTM, TCN, and Transformer baselines train from the CLI, rolling validation-selection windows drive checkpoint choice, validation final-window artifacts remain available for reporting, official-test predictions are scored with the project metrics, JSON/CSV outputs use the same result container as the classical baselines, optional history JSON records per-epoch training loss plus validation metrics, the comparison command can produce a single architecture/learning-rate sweep table, and the reporting command ranks Phase 2 candidates against the Phase 1 HGB policy baseline.

The next deep-learning work should focus on model quality rather than plumbing:

- Run FD001 architecture checks with wider/deeper CNNs, residual or TCN-style blocks, attention heads, and learning-rate sweeps.
- Run the compact comparison command on real FD001 first, then expand to FD002-FD004 after checking runtime.
- Use the ranked report command for every Phase 2 run bundle, especially when checking the known FD003 validation mismatch.
- Keep all sensors for now; Phase 1 sensor-filter validation showed EDA filtering harms FD002 and FD004.
