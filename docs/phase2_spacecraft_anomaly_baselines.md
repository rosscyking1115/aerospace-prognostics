# Phase 2 Spacecraft Anomaly Baselines

This note records the Track B kickoff for spacecraft telemetry anomaly detection. The first implementation is deliberately simple: dataset-agnostic classical baselines that can run on labelled telemetry CSVs before the project has a dedicated SMAP/MSL or ESA-ADB loader.

The baseline is useful before the SMAP/MSL and ESA-ADB loaders exist because it establishes the project contract for anomaly work:

- labelled telemetry CSVs can be scored from the CLI
- outputs include both point-wise and point-adjusted metrics
- prediction artifacts are written as JSON and row-level CSV files
- feature names, fitted parameters, thresholds, and model settings are recorded for auditability

## Baselines

The current classical comparison command runs:

- robust z-score/MAD thresholds
- PCA reconstruction error thresholds
- Isolation Forest

These are intentionally modest baselines. They are useful because spacecraft anomaly papers can overstate deep-model gains when simple robust statistics or reconstruction errors are not checked under the same metric policy.

## Command

Download Telemanom SMAP/MSL raw arrays and labels:

```powershell
uv run aerospace-prognostics smap-msl-download --output-dir data/raw/smap_msl --archive-path data/raw/downloads/smap_msl_telemanom.zip
```

The downloader can use the legacy public Telemanom `data.zip` URL and the `labeled_anomalies.csv` file from the Telemanom repository. Telemanom's current README points users to the Kaggle-hosted `patrickfleith/nasa-anomaly-detection-dataset-smap-msl` archive; if the legacy S3 URL is unavailable or returns HTTP 403, download that archive to `data/raw/downloads/smap_msl_telemanom.zip`, then rerun the command. Existing local archives are imported without another download, and the extracted arrays are written under `data/train` and `data/test` inside the requested output directory.

Telemanom raw-data summary and channel export:

```powershell
uv run aerospace-prognostics smap-msl-summary --data-dir data/raw/smap_msl
uv run aerospace-prognostics smap-msl-summary --data-dir data/raw/smap_msl --channel-id P-1
uv run aerospace-prognostics smap-msl-export-channel-csv --data-dir data/raw/smap_msl --channel-id P-1 --output-dir artifacts/smap_msl_channels --metadata-json artifacts/smap_msl_channels/P-1/export.json
```

The loader supports the Telemanom/Kaggle layout with `labeled_anomalies.csv` at the dataset root and arrays under either `train/` and `test/` or `data/train/` and `data/test/`. Each channel array is expected to have shape `(n_timesteps, n_inputs)`. Test labels are generated from inclusive anomaly intervals in `anomaly_sequences`.

Deterministic channel selection for broader benchmark sweeps:

```powershell
uv run aerospace-prognostics smap-msl-select-channels --data-dir data/raw/smap_msl --count 20 --strategy balanced --output-json artifacts/phase2_smap_msl_channel_selection/balanced_20_channels.json --output-csv artifacts/phase2_smap_msl_channel_selection/balanced_20_channels.csv
```

The default `balanced` strategy round-robins through spacecraft groups in label order, which avoids treating the first label rows as a representative benchmark by accident. Use `label_order` when the exact Telemanom label order is the intended sample.

Single robust baseline:

```powershell
uv run aerospace-prognostics telemetry-robust-zscore-baseline --train-csv artifacts/smap_msl_channels/P-1/train.csv --test-csv artifacts/smap_msl_channels/P-1/test.csv --label-column label --feature-columns feature_0 feature_1 --threshold 3.5 --output-json artifacts/results/smap_msl_p1_robust_zscore.json --predictions-csv artifacts/results/smap_msl_p1_robust_zscore_predictions.csv
```

Classical comparison:

```powershell
uv run aerospace-prognostics telemetry-classical-anomaly-baselines --train-csv artifacts/smap_msl_channels/P-1/train.csv --test-csv artifacts/smap_msl_channels/P-1/test.csv --label-column label --feature-columns feature_0 feature_1 --output-json artifacts/results/smap_msl_p1_classical_baselines.json --output-csv artifacts/results/smap_msl_p1_classical_baselines.csv --predictions-csv artifacts/results/smap_msl_p1_classical_predictions.csv
```

Direct multi-channel SMAP/MSL comparison:

```powershell
uv run aerospace-prognostics smap-msl-classical-baselines --data-dir data/raw/smap_msl --max-channels 5 --output-json artifacts/results/smap_msl_classical_baselines_sample.json --output-csv artifacts/results/smap_msl_classical_baselines_sample.csv
```

First forecasting-based SMAP/MSL baseline:

```powershell
uv run aerospace-prognostics smap-msl-lstm-forecast-baseline --data-dir data/raw/smap_msl --max-channels 5 --window-size 30 --epochs 10 --output-json artifacts/results/smap_msl_lstm_forecast_sample.json --output-csv artifacts/results/smap_msl_lstm_forecast_sample.csv
uv run aerospace-prognostics smap-msl-lstm-forecast-baseline --data-dir data/raw/smap_msl --max-channels 5 --window-size 30 --epochs 10 --threshold-method dynamic --output-json artifacts/results/smap_msl_lstm_dynamic_sample.json --output-csv artifacts/results/smap_msl_lstm_dynamic_sample.csv
```

This command trains a small LSTM on nominal train windows for each selected channel and predicts the next telemetry vector. The default threshold uses a robust train-error scale; `--threshold-method dynamic` applies a compact Telemanom-style nonparametric dynamic threshold over smoothed test forecast errors, including z-search, local windows, error buffering, and sequence pruning. It is a bridge toward the Hundman et al. forecasting baseline, not yet a byte-for-byte port of the original Keras implementation.

Rank the classical and forecasting outputs in one report:

```powershell
uv run aerospace-prognostics smap-msl-compare-anomaly-results --result-csv artifacts/results/smap_msl_classical_baselines_sample.csv artifacts/results/smap_msl_lstm_forecast_sample.csv artifacts/results/smap_msl_lstm_dynamic_sample.csv --source-labels classical lstm_robust lstm_dynamic --output-csv artifacts/results/smap_msl_anomaly_model_comparison.csv --output-markdown artifacts/results/smap_msl_anomaly_model_comparison.md
```

The comparison report ranks rows per channel by point-wise F1, then point-adjusted F1, then false-alarm and miss rates. Point-wise F1 stays first because point-adjustment can over-reward weak detectors on long labelled intervals.

One-command Track B bundle:

```powershell
uv run aerospace-prognostics phase2-smap-msl --data-dir data/raw/smap_msl --artifact-dir artifacts/phase2_smap_msl --max-channels 5 --window-size 30 --epochs 10
```

This writes classical, LSTM robust-threshold, LSTM dynamic-threshold, ranked comparison, and `phase2_smap_msl_summary.md` artifacts under the requested artifact directory.

## First Real SMAP/MSL Sample Run

The Kaggle SMAP/MSL archive imported successfully through `smap-msl-download` and produced 164 `.npy` arrays, covering 82 channels. The raw summary command reported 105 anomaly sequences across 55 SMAP and 27 MSL channels.

The first bounded workflow run used the first five channels, 10 LSTM epochs, 30-step forecast windows, three classical baselines, robust-threshold LSTM forecasts, and dynamic-threshold LSTM forecasts:

```powershell
uv run aerospace-prognostics phase2-smap-msl --data-dir data/raw/smap_msl --artifact-dir artifacts/phase2_smap_msl --max-channels 5 --window-size 30 --epochs 10 --batch-size 64
```

It generated 15 classical runs, 5 robust-threshold LSTM runs, 5 dynamic-threshold LSTM runs, and 25 ranked comparison rows.

| Channel | Spacecraft | Best Source | Best Model | Point-wise F1 | Point-adjusted F1 | False Alarm Rate |
|---|---|---|---|---:|---:|---:|
| E-1 | SMAP | lstm_dynamic | `lstm_forecast_dynamic_threshold` | 0.685279 | 0.958794 | 0.001249 |
| E-2 | SMAP | lstm_robust | `lstm_forecast_robust_threshold` | 0.274818 | 0.624944 | 0.235212 |
| E-3 | SMAP | lstm_robust | `lstm_forecast_robust_threshold` | 0.103143 | 0.811261 | 0.293483 |
| P-1 | SMAP | lstm_robust | `lstm_forecast_robust_threshold` | 0.115776 | 0.404525 | 0.285143 |
| S-1 | SMAP | classical | `robust_zscore` | 0.222222 | 0.311978 | 0.287084 |

The result is a useful warning sign, not a finished benchmark. E-1 shows the dynamic threshold can produce a strong, low-false-alarm detector on some channels, but several other winners have high false-alarm rates. The point-adjusted scores are often much larger than point-wise F1, including for detectors with very low point-wise recall, so point-wise precision/recall and false-alarm rate remain the primary readout.

To avoid relying only on the first five label rows, a second sample used an explicit mixed SMAP/MSL channel set:

```powershell
uv run aerospace-prognostics phase2-smap-msl --data-dir data/raw/smap_msl --artifact-dir artifacts/phase2_smap_msl_mixed_sample --channels P-1 S-1 E-1 E-2 E-3 M-1 M-2 P-10 F-7 C-1 --window-size 30 --epochs 10 --batch-size 64
```

That run produced 30 classical runs, 10 robust-threshold LSTM runs, 10 dynamic-threshold LSTM runs, and 50 ranked comparison rows. When `--channels` is supplied, the workflow now honors the full explicit list unless `--max-channels` is also provided.

| Channel | Spacecraft | Best Source | Best Model | Point-wise F1 | Point-adjusted F1 | False Alarm Rate |
|---|---|---|---|---:|---:|---:|
| C-1 | MSL | lstm_dynamic | `lstm_forecast_dynamic_threshold` | 0.531746 | 0.531746 | 0.124488 |
| E-1 | SMAP | lstm_dynamic | `lstm_forecast_dynamic_threshold` | 0.685279 | 0.958794 | 0.001249 |
| E-2 | SMAP | lstm_robust | `lstm_forecast_robust_threshold` | 0.274818 | 0.624944 | 0.235212 |
| E-3 | SMAP | lstm_robust | `lstm_forecast_robust_threshold` | 0.103143 | 0.811261 | 0.293483 |
| F-7 | MSL | lstm_dynamic | `lstm_forecast_dynamic_threshold` | 0.302352 | 0.419187 | 0.072339 |
| M-1 | MSL | lstm_robust | `lstm_forecast_robust_threshold` | 0.397296 | 0.850858 | 0.352113 |
| M-2 | MSL | classical | `isolation_forest` | 0.746284 | 0.965313 | 0.072183 |
| P-1 | SMAP | lstm_robust | `lstm_forecast_robust_threshold` | 0.115776 | 0.404525 | 0.285143 |
| P-10 | MSL | lstm_dynamic | `lstm_forecast_dynamic_threshold` | 0.182578 | 0.182578 | 0.196515 |
| S-1 | SMAP | classical | `robust_zscore` | 0.222222 | 0.311978 | 0.287084 |

The mixed sample reinforces the need for multiple baseline families. MSL channel M-2 is best served by Isolation Forest, while E-1 and C-1 favor dynamic LSTM thresholding. There is no single method that looks uniformly reliable yet. The regenerated workflow summary now records aggregate winner counts and average metrics by source/model: dynamic-threshold LSTM wins 4 of 10 channels, robust-threshold LSTM wins 4 of 10, and classical methods win 2 of 10. Across all rows in this sample, robust z-score has the highest mean point-wise F1 among the classical methods, while Isolation Forest has the lowest mean false-alarm rate.

The next bounded sweep used the deterministic balanced 20-channel selection:

```powershell
uv run aerospace-prognostics smap-msl-select-channels --data-dir data/raw/smap_msl --count 20 --strategy balanced --output-json artifacts/phase2_smap_msl_channel_selection/balanced_20_channels.json --output-csv artifacts/phase2_smap_msl_channel_selection/balanced_20_channels.csv
uv run aerospace-prognostics phase2-smap-msl --data-dir data/raw/smap_msl --artifact-dir artifacts/phase2_smap_msl_balanced_20 --channels M-6 P-1 M-1 S-1 M-2 E-1 S-2 E-2 P-10 E-3 T-4 E-4 T-5 E-5 F-7 E-6 M-3 E-7 M-4 E-8 --window-size 30 --epochs 10 --batch-size 64
```

That run produced 60 classical runs, 20 robust-threshold LSTM runs, 20 dynamic-threshold LSTM runs, and 100 ranked comparison rows. Winner counts were dynamic-threshold LSTM 8 of 20, robust-threshold LSTM 7 of 20, Isolation Forest 2 of 20, and robust z-score 3 of 20. The highest individual point-wise F1s came from M-6 dynamic LSTM (0.863962), E-7 dynamic LSTM (0.801712), and M-2 Isolation Forest (0.746284). The broad sample still shows several high false-alarm winners, so the conservative readout remains point-wise F1 plus false-alarm rate, not point-adjusted F1 alone.

If `--feature-columns` is omitted, the command uses all numeric columns from the train CSV except the label column. For SMAP/MSL channel exports, pass `feature_0 feature_1 ...` explicitly when you want to exclude the numeric `timestep` column.

## Metric Note

The command reports standard point-wise precision, recall, F1, false-alarm rate, and miss rate. It also reports point-adjusted F1, where detecting any point in a true anomaly segment marks the whole segment as detected.

Point-adjustment is included because it is common in time-series anomaly papers, but it can make weak detectors look stronger than they are. The point-wise metrics remain the primary conservative readout for this project until an official benchmark evaluator is available. For ESA-ADB, the next step is to use the benchmark's own hierarchical evaluation pipeline rather than inventing a project-specific substitute.

## Current Status

This is not yet a reproduced SMAP/MSL Telemanom LSTM baseline. It is the Track B baseline layer: raw SMAP/MSL loading, direct multi-channel classical runs, LSTM forecasting with robust and dynamic thresholding, robust statistics, PCA reconstruction, Isolation Forest, metric handling, artifact formats, model comparison reporting, workflow orchestration, and CLI execution. The next Track B steps are:

- scale the balanced SMAP/MSL benchmark beyond the current 20-channel sweep or tune thresholds per spacecraft family
- compare the compact dynamic threshold against Telemanom's original predictions/threshold outputs on downloaded SMAP/MSL data
- move serious benchmark claims to ESA-ADB with its official evaluation tools
