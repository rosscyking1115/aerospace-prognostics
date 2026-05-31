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

The downloader can use the legacy public Telemanom `data.zip` URL and the `labeled_anomalies.csv` file from the Telemanom repository. Telemanom's current README points users to the Kaggle-hosted `patrickfleith/nasa-anomaly-detection-dataset-smap-msl` archive; if the legacy S3 URL is unavailable, download that archive to `data/raw/downloads/smap_msl_telemanom.zip`, then rerun the command. Existing local archives are imported without another download, and the extracted arrays are written under `data/train` and `data/test` inside the requested output directory.

Telemanom raw-data summary and channel export:

```powershell
uv run aerospace-prognostics smap-msl-summary --data-dir data/raw/smap_msl
uv run aerospace-prognostics smap-msl-summary --data-dir data/raw/smap_msl --channel-id P-1
uv run aerospace-prognostics smap-msl-export-channel-csv --data-dir data/raw/smap_msl --channel-id P-1 --output-dir artifacts/smap_msl_channels --metadata-json artifacts/smap_msl_channels/P-1/export.json
```

The loader supports the Telemanom/Kaggle layout with `labeled_anomalies.csv` at the dataset root and arrays under either `train/` and `test/` or `data/train/` and `data/test/`. Each channel array is expected to have shape `(n_timesteps, n_inputs)`. Test labels are generated from inclusive anomaly intervals in `anomaly_sequences`.

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

If `--feature-columns` is omitted, the command uses all numeric columns from the train CSV except the label column. For SMAP/MSL channel exports, pass `feature_0 feature_1 ...` explicitly when you want to exclude the numeric `timestep` column.

## Metric Note

The command reports standard point-wise precision, recall, F1, false-alarm rate, and miss rate. It also reports point-adjusted F1, where detecting any point in a true anomaly segment marks the whole segment as detected.

Point-adjustment is included because it is common in time-series anomaly papers, but it can make weak detectors look stronger than they are. The point-wise metrics remain the primary conservative readout for this project until an official benchmark evaluator is available. For ESA-ADB, the next step is to use the benchmark's own hierarchical evaluation pipeline rather than inventing a project-specific substitute.

## Current Status

This is not yet a reproduced SMAP/MSL Telemanom LSTM baseline. It is the Track B baseline layer: raw SMAP/MSL loading, direct multi-channel classical runs, LSTM forecasting with robust and dynamic thresholding, robust statistics, PCA reconstruction, Isolation Forest, metric handling, artifact formats, and CLI execution. The next Track B steps are:

- run the direct multi-channel classical and LSTM-forecast comparisons on real SMAP/MSL channels and record an initial table
- compare the compact dynamic threshold against Telemanom's original predictions/threshold outputs on downloaded SMAP/MSL data
- move serious benchmark claims to ESA-ADB with its official evaluation tools
