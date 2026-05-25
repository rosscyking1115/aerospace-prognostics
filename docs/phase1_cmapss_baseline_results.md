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

The next Phase 1 modelling step should improve the engineered baseline with:

- Regime-aware normalization or cluster-specific feature summaries for FD002 and FD004.
- Hyperparameter search over gradient-boosting settings.
- Window-size comparison for rolling features.
- Sensor filtering using the EDA near-constant and drift summaries.

Those changes should be reported as a third table beside the two checkpoints above so progress stays measurable.
