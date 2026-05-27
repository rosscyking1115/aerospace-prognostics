# Phase 1 C-MAPSS Baseline Results

This note records the first real-data Phase 1 checkpoint. The raw NASA C-MAPSS files and generated artifacts are intentionally ignored by Git; this tracked summary preserves the reproducible headline results, data provenance, and interpretation.

## Run Configuration

| Field | Value |
|---|---|
| Dataset | NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set |
| Source URL | `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` |
| Local raw path | `data/raw/cmapss` |
| Workflow command | `uv run aerospace-prognostics phase1-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase1` |
| Model | `hist_gradient_boosting` |
| RUL cap | 125 cycles |
| Random seed | 42 |
| Standardization | Train-fitted feature standardization enabled |
| Verification | `cmapss-verify` returned `status=ok`, `files=12` |

## Dataset Shape

| Subset | Train Rows | Train Units | Test Rows | Test Units | Test RUL Values |
|---|---:|---:|---:|---:|---:|
| FD001 | 20,631 | 100 | 13,096 | 100 | 100 |
| FD002 | 53,759 | 260 | 33,991 | 259 | 259 |
| FD003 | 24,720 | 100 | 16,596 | 100 | 100 |
| FD004 | 61,249 | 249 | 41,214 | 248 | 248 |

## Raw-Cycle Baseline Metrics

| Subset | Operating Conditions | Fault Modes | RMSE | NASA Score |
|---|---:|---:|---:|---:|
| FD001 | 1 | 1 | 17.884910 | 912.259264 |
| FD002 | 6 | 1 | 29.395171 | 11542.260250 |
| FD003 | 1 | 2 | 21.751756 | 2265.368672 |
| FD004 | 6 | 2 | 30.072980 | 7712.617966 |

## Engineered Baseline Metrics

This second baseline adds observed cycle count, rolling means, rolling ranges, rolling slopes, and deltas from each unit's initial observed sensor values. It uses the same RUL cap, model family, seed, standardization approach, and evaluation metrics.

Command:

```powershell
uv run aerospace-prognostics cmapss-engineered-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_engineered_baseline.json --output-csv artifacts/results/cmapss_engineered_baseline.csv
```

| Subset | Model | RMSE | RMSE Change | NASA Score | NASA Score Change |
|---|---|---:|---:|---:|---:|
| FD001 | `hist_gradient_boosting_engineered_w5` | 13.887256 | -3.997654 | 296.420886 | -615.838378 |
| FD002 | `hist_gradient_boosting_engineered_w5` | 28.018580 | -1.376591 | 9709.164019 | -1833.096231 |
| FD003 | `hist_gradient_boosting_engineered_w5` | 14.248166 | -7.503590 | 347.862382 | -1917.506290 |
| FD004 | `hist_gradient_boosting_engineered_w5` | 29.453081 | -0.619899 | 7465.390723 | -247.227243 |

## Rolling-Window Sweep

The engineered baseline was swept across rolling windows of 3, 5, and 10 cycles. The command was:

```powershell
uv run aerospace-prognostics cmapss-engineered-window-sweep --data-dir data/raw/cmapss --rolling-windows 3 5 10 --output-json artifacts/results/cmapss_engineered_window_sweep.json --output-csv artifacts/results/cmapss_engineered_window_sweep.csv
```

| Subset | Window | RMSE | NASA Score |
|---|---:|---:|---:|
| FD001 | 3 | 14.158640 | 308.857426 |
| FD001 | 5 | 13.887256 | 296.420886 |
| FD001 | 10 | 13.394005 | 269.370224 |
| FD002 | 3 | 27.608489 | 8877.027954 |
| FD002 | 5 | 28.018580 | 9709.164019 |
| FD002 | 10 | 28.328163 | 9334.471466 |
| FD003 | 3 | 14.123918 | 357.204866 |
| FD003 | 5 | 14.248166 | 347.862382 |
| FD003 | 10 | 14.706469 | 391.700453 |
| FD004 | 3 | 29.367205 | 7131.745341 |
| FD004 | 5 | 29.453081 | 7465.390723 |
| FD004 | 10 | 29.653369 | 8912.270080 |

Best by NASA score:

| Subset | Best Window | RMSE | NASA Score |
|---|---:|---:|---:|
| FD001 | 10 | 13.394005 | 269.370224 |
| FD002 | 3 | 27.608489 | 8877.027954 |
| FD003 | 5 | 14.248166 | 347.862382 |
| FD004 | 3 | 29.367205 | 7131.745341 |

Shorter windows perform better on the multi-regime FD002 and FD004 subsets, while FD001 benefits from the longer 10-cycle summary. That suggests the next improvement should not force one feature horizon across all subsets.

The per-subset window checkpoint is available as:

```powershell
uv run aerospace-prognostics cmapss-engineered-best-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_engineered_best_baseline.json --output-csv artifacts/results/cmapss_engineered_best_baseline.csv
```

Its current output is:

| Subset | Selected Window | RMSE | NASA Score |
|---|---:|---:|---:|
| FD001 | 10 | 13.394005 | 269.370224 |
| FD002 | 3 | 27.608489 | 8877.027954 |
| FD003 | 5 | 14.248166 | 347.862382 |
| FD004 | 3 | 29.367205 | 7131.745341 |

Compared with the first raw-cycle baseline, the selected-window engineered checkpoint improves every subset on both RMSE and NASA score. It is now the Phase 1 classical baseline to beat.

## Regime-Aware Feature Check

The multi-regime subsets mix operating-condition effects with degradation effects, so the next check adds train-fitted operating-regime context. It clusters operational settings, adds one-hot regime indicators, and subtracts each train-fitted regime's sensor mean to create regime-residual sensor features. No test rows are used to fit the regime scaler, clusterer, or residual means.

Command:

```powershell
uv run aerospace-prognostics cmapss-regime-engineered-best-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_regime_engineered_best_baseline.json --output-csv artifacts/results/cmapss_regime_engineered_best_baseline.csv
```

| Subset | Selected-Window RMSE | Regime-Aware RMSE | Selected-Window NASA Score | Regime-Aware NASA Score | NASA Score Change |
|---|---:|---:|---:|---:|---:|
| FD001 | 13.394005 | 13.012889 | 269.370224 | 253.465322 | -15.904902 |
| FD002 | 27.608489 | 27.310525 | 8877.027954 | 9264.345504 | 387.317550 |
| FD003 | 14.248166 | 14.512316 | 347.862382 | 360.244233 | 12.381851 |
| FD004 | 29.367205 | 29.215870 | 7131.745341 | 7106.739881 | -24.005460 |

This is a useful experiment, but not a clean replacement for the selected-window engineered baseline. It improves NASA score on FD001 and FD004, slightly improves FD002 RMSE while worsening the asymmetric NASA score, and worsens FD003. The current Phase 1 default remains the selected-window engineered baseline until regime-aware adoption is selected per subset by validation data or improved with a stronger modelling strategy.

## Temporal Validation Candidate Check

To avoid tuning feature decisions directly against the official test RUL files, the next checkpoint adds a deterministic train-only validation split. It holds out 20% of training units, truncates each held-out unit 30 cycles before failure, trains on the remaining full run-to-failure units, and scores the candidate models against the known remaining cycles.

Command:

```powershell
uv run aerospace-prognostics cmapss-validate-feature-candidates --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_feature_candidates.json --output-csv artifacts/results/cmapss_validation_feature_candidates.csv
```

| Subset | Candidate | Validation RMSE | Validation NASA Score |
|---|---|---:|---:|
| FD001 | `hist_gradient_boosting_engineered_w10` | 7.785278 | 20.707446 |
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6` | 7.523112 | 19.265706 |
| FD002 | `hist_gradient_boosting_engineered_w3` | 17.313590 | 389.055355 |
| FD002 | `hist_gradient_boosting_regime_engineered_w3_r6` | 17.259898 | 333.644731 |
| FD003 | `hist_gradient_boosting_engineered_w5` | 6.922679 | 18.011404 |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6` | 6.301167 | 15.833362 |
| FD004 | `hist_gradient_boosting_engineered_w3` | 15.238235 | 261.455072 |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6` | 13.574715 | 174.416271 |

Selected by validation NASA score:

| Subset | Validation-Selected Candidate |
|---|---|
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6` |
| FD002 | `hist_gradient_boosting_regime_engineered_w3_r6` |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6` |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6` |

This validation split favours regime-aware features for every subset. However, the official test comparison above is mixed for FD002 and FD003, so this is evidence that regime context is promising, not proof that one split is sufficient for final model selection. The repeated-validation checkpoint below is the stronger train-only selection signal.

## Repeated Temporal Validation

The one-shot validation split was expanded across two random unit-holdout seeds and two truncation horizons. Each candidate is therefore scored on four train-only validation runs per subset.

Command:

```powershell
uv run aerospace-prognostics cmapss-validate-feature-candidates-repeated --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_feature_candidates_repeated.json --output-csv artifacts/results/cmapss_validation_feature_candidates_repeated.csv
```

Configuration:

| Field | Value |
|---|---|
| Validation fraction | 20% of train units |
| Random states | 11, 42 |
| Truncation horizons | 20, 30 cycles before failure |
| Runs per candidate | 4 |

| Subset | Candidate | Wins by NASA | Mean RMSE | Mean NASA Score |
|---|---|---:|---:|---:|
| FD001 | `hist_gradient_boosting_engineered_w10` | 1 | 7.312349 | 20.458500 |
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6` | 3 | 7.087184 | 18.494492 |
| FD002 | `hist_gradient_boosting_engineered_w3` | 1 | 14.039413 | 217.108673 |
| FD002 | `hist_gradient_boosting_regime_engineered_w3_r6` | 3 | 14.154992 | 226.452779 |
| FD003 | `hist_gradient_boosting_engineered_w5` | 2 | 6.869311 | 18.146537 |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6` | 2 | 6.605450 | 16.933487 |
| FD004 | `hist_gradient_boosting_engineered_w3` | 1 | 13.877963 | 224.599241 |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6` | 3 | 12.273893 | 204.179565 |

Selected by mean validation NASA score:

| Subset | Repeated-Validation-Selected Candidate |
|---|---|
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6` |
| FD002 | `hist_gradient_boosting_engineered_w3` |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6` |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6` |

This is now the most defensible train-only model-selection signal in Phase 1. FD002 is the cautionary case: regime-aware features win more individual validation runs, but their mean NASA score is worse because the asymmetric penalty is sensitive to larger late-prediction errors. The validation-selected policy should therefore be per-subset rather than globally enabling regime-aware features.

## Validation-Selected Official Test Baseline

The repeated-validation policy was then evaluated once on the official C-MAPSS test RUL files. This keeps model selection train-only, then uses the official test target table only for the reported checkpoint.

Command:

```powershell
uv run aerospace-prognostics cmapss-validation-selected-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_selected_baseline.json --output-csv artifacts/results/cmapss_validation_selected_baseline.csv
```

Policy:

| Subset | Selected Feature Candidate |
|---|---|
| FD001 | `regime_engineered` |
| FD002 | `engineered` |
| FD003 | `regime_engineered` |
| FD004 | `regime_engineered` |

Official test results:

| Subset | Model | RMSE | NASA Score |
|---|---|---:|---:|
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6` | 13.012889 | 253.465322 |
| FD002 | `hist_gradient_boosting_engineered_w3` | 27.608489 | 8877.027954 |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6` | 14.512316 | 360.244233 |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6` | 29.215870 | 7106.739881 |

Compared with the selected-window engineered checkpoint, the validation-selected policy improves FD001 and FD004, keeps FD002 unchanged, and worsens FD003. Aggregate NASA score still improves slightly, from 16626.005901 to 16597.477390, but the FD003 mismatch shows that the validation design is not yet a perfect proxy for the official test distribution. This is the current honest Phase 1 classical baseline policy, while the best observed test-only per-subset mix remains a diagnostic reference rather than a model-selection procedure.

## Compact HGB Hyperparameter Check

The validation-selected feature policy was then checked against a small HistGradientBoosting parameter grid. This is intentionally compact: the goal is to test whether the fixed baseline settings are obviously leaving classical-model performance on the table without turning Phase 1 into a large tuning exercise.

Command:

```powershell
uv run aerospace-prognostics cmapss-validate-hgb-grid --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_hgb_grid.json --output-csv artifacts/results/cmapss_validation_hgb_grid.csv
```

Grid:

| Label | Learning Rate | Max Iterations | L2 Regularization | Max Leaf Nodes |
|---|---:|---:|---:|---:|
| `default` | 0.05 | 200 | 0.01 | default |
| `slow_regularized` | 0.03 | 350 | 0.05 | default |
| `shallow_fast` | 0.08 | 160 | 0.02 | 15 |

Validation results:

| Subset | Candidate | Validation RMSE | Validation NASA Score |
|---|---|---:|---:|
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6_default` | 7.523112 | 19.265706 |
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6_slow_regularized` | 7.759949 | 20.168248 |
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6_shallow_fast` | 7.866713 | 22.616798 |
| FD002 | `hist_gradient_boosting_engineered_w3_default` | 17.313590 | 389.055355 |
| FD002 | `hist_gradient_boosting_engineered_w3_slow_regularized` | 16.966055 | 362.389803 |
| FD002 | `hist_gradient_boosting_engineered_w3_shallow_fast` | 19.839673 | 733.031141 |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6_default` | 6.301167 | 15.833362 |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6_slow_regularized` | 5.983843 | 14.666385 |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6_shallow_fast` | 6.441664 | 16.104216 |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6_default` | 13.574715 | 174.416271 |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6_slow_regularized` | 13.715958 | 189.839433 |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6_shallow_fast` | 13.360296 | 175.446008 |

Selected by validation NASA score:

| Subset | Validation-Selected HGB Candidate |
|---|---|
| FD001 | `default` |
| FD002 | `slow_regularized` |
| FD003 | `slow_regularized` |
| FD004 | `default` |

The slower, more regularized candidate helps FD002 and FD003 on validation, while the original default remains strongest for FD001 and FD004. The next checkpoint should evaluate this per-subset HGB policy on the official test table and verify whether it resolves or worsens the FD003 validation mismatch.

## HGB Policy Official Test Baseline

The compact HGB policy was then evaluated once on the official C-MAPSS test RUL files, using the repeated-validation feature policy and the validation-selected HGB parameter label for each subset.

Command:

```powershell
uv run aerospace-prognostics cmapss-hgb-policy-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_hgb_policy_baseline.json --output-csv artifacts/results/cmapss_hgb_policy_baseline.csv
```

Policy:

| Subset | Feature Candidate | HGB Candidate |
|---|---|---|
| FD001 | `regime_engineered` | `default` |
| FD002 | `engineered` | `slow_regularized` |
| FD003 | `regime_engineered` | `slow_regularized` |
| FD004 | `regime_engineered` | `default` |

Official test results:

| Subset | Model | RMSE | NASA Score |
|---|---|---:|---:|
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6_default` | 13.012889 | 253.465322 |
| FD002 | `hist_gradient_boosting_engineered_w3_slow_regularized` | 27.568929 | 8697.396161 |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6_slow_regularized` | 14.394433 | 358.507217 |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6_default` | 29.215870 | 7106.739881 |

Compared with the validation-selected feature baseline, the HGB policy improves FD002 and FD003 while leaving FD001 and FD004 unchanged. Aggregate NASA score improves from 16597.477390 to 16416.108581. The FD003 gap versus the selected-window engineered baseline remains, but the compact parameter policy reduces it without using official test data for model selection.

## EDA Sensor-Filter Validation Check

The next check tested whether the EDA near-constant, drift, and RUL-correlation summaries could safely reduce the sensor set before sequence modelling. The filter was fitted on the training side of the temporal validation split only. It removes near-flat channels and keeps sensors with either absolute RUL correlation of at least 0.05 or standardized early-to-late drift of at least 0.2.

Command:

```powershell
uv run aerospace-prognostics cmapss-validate-sensor-filters --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_sensor_filters.json --output-csv artifacts/results/cmapss_validation_sensor_filters.csv
```

Validation results:

| Subset | Candidate | Validation RMSE | Validation NASA Score |
|---|---|---:|---:|
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6_default` | 7.523112 | 19.265706 |
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6_default_eda_filtered` | 7.523112 | 19.265706 |
| FD002 | `hist_gradient_boosting_engineered_w3_slow_regularized` | 16.966055 | 362.389803 |
| FD002 | `hist_gradient_boosting_engineered_w3_slow_regularized_eda_filtered` | 32.257909 | 2764.565916 |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6_slow_regularized` | 5.983843 | 14.666385 |
| FD003 | `hist_gradient_boosting_regime_engineered_w5_r6_slow_regularized_eda_filtered` | 5.983843 | 14.666385 |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6_default` | 13.574715 | 174.416271 |
| FD004 | `hist_gradient_boosting_regime_engineered_w3_r6_default_eda_filtered` | 21.319421 | 1052.531875 |

Selected by validation NASA score:

| Subset | Sensor Policy |
|---|---|
| FD001 | `all_sensors` |
| FD002 | `all_sensors` |
| FD003 | `all_sensors` |
| FD004 | `all_sensors` |

This is a useful negative result. EDA filtering is neutral where it keeps the same effective signal, but it removes information the multi-regime FD002 and FD004 baselines need. The Phase 1 classical policy therefore keeps all sensors, while the EDA summaries remain useful for interpretation and for later model diagnostics rather than hard feature removal.

## Reproducible Phase 1 Workflow

The `phase1-cmapss` workflow now reproduces the full Phase 1 checkpoint in one command: manifest verification, per-subset EDA reports, the raw-cycle sanity baseline, the current validation-selected HGB policy baseline, the sensor-filter validation check, and a markdown summary.

Command:

```powershell
uv run aerospace-prognostics phase1-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase1
```

The Phase 2 bridge is the sequence export command. It writes compressed NumPy bundles for train windows, train-only validation final windows, and official-test final windows, plus metadata recording the split and feature configuration.

```powershell
uv run aerospace-prognostics cmapss-export-sequences --data-dir data/raw/cmapss --output-dir artifacts/sequences/cmapss --window-size 30 --stride 1
```

Sequence export output:

| Subset | Train Windows | Validation Windows | Test Windows | Features | Window Size |
|---|---:|---:|---:|---:|---:|
| FD001 | 14,022 | 20 | 100 | 24 | 30 |
| FD002 | 37,427 | 52 | 259 | 24 | 30 |
| FD003 | 17,618 | 20 | 100 | 24 | 30 |
| FD004 | 43,114 | 50 | 248 | 24 | 30 |

## Phase 1 Conclusion

The final Phase 1 classical policy is the validation-selected feature policy plus validation-selected HGB parameter policy, using all sensors. It is not claimed as a leaderboard model; it is the honest baseline that Phase 2 sequence models must beat.

Remaining known limitation: FD003 still shows a validation mismatch. The train-only validation policy prefers regime-aware features, while the selected-window engineered official-test checkpoint remains slightly stronger on FD003. This should be treated as a validation-design risk when comparing Phase 2 models.

## Provenance Checksums

| File | SHA-256 |
|---|---|
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |
| `train_FD002.txt` | `dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f` |
| `test_FD002.txt` | `de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02` |
| `RUL_FD002.txt` | `c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b` |
| `train_FD003.txt` | `2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2` |
| `test_FD003.txt` | `299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790` |
| `RUL_FD003.txt` | `df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb` |
| `train_FD004.txt` | `27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218` |
| `test_FD004.txt` | `1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5` |
| `RUL_FD004.txt` | `196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1` |

## Interpretation

FD001 is the cleanest first sanity check: one operating condition and one fault mode. The baseline performs best there, which is expected.

FD002 and FD004 are harder because operating-regime effects and degradation effects are mixed. Their higher RMSE and NASA scores confirm that the next baseline should be regime-aware rather than treating every row as directly comparable after one global standardization step.

FD003 has two fault modes under a single operating condition. Its score sits between FD001 and the multi-regime subsets, which matches the expected difficulty ordering.

The raw-cycle model is a deliberately conservative classical baseline. It is useful because it proves the end-to-end pipeline works on real NASA files: download, extraction, checksum manifest, EDA, feature table generation, train-fitted standardization, baseline training, and RMSE/NASA scoring.

The engineered baseline is a better Phase 1 checkpoint. The large gains on FD001 and FD003 suggest recent sensor dynamics and deltas from early observations capture useful degradation signal. FD002 and FD004 still remain difficult, which is consistent with their multiple operating regimes.

## Next Baseline Upgrade

The next modelling step moves into Phase 2 sequence models:

- 1D-CNN over exported C-MAPSS windows.
- LSTM/BiLSTM and optional TCN baselines.
- Transformer-style sequence model.
- Consistent comparison against the Phase 1 validation-selected HGB policy baseline.
- Continued attention to the FD003 validation mismatch.

Those changes should be reported as a third table beside the two checkpoints above so progress stays measurable.
