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

```powershell
uv run aerospace-prognostics telemetry-robust-zscore-baseline --train-csv data/raw/smap_msl/train.csv --test-csv data/raw/smap_msl/test.csv --label-column label --threshold 3.5 --output-json artifacts/results/smap_msl_robust_zscore.json --predictions-csv artifacts/results/smap_msl_robust_zscore_predictions.csv
```

Classical comparison:

```powershell
uv run aerospace-prognostics telemetry-classical-anomaly-baselines --train-csv data/raw/smap_msl/train.csv --test-csv data/raw/smap_msl/test.csv --label-column label --output-json artifacts/results/smap_msl_classical_baselines.json --output-csv artifacts/results/smap_msl_classical_baselines.csv --predictions-csv artifacts/results/smap_msl_classical_predictions.csv
```

If `--feature-columns` is omitted, the command uses all numeric columns from the train CSV except the label column. For prepared SMAP/MSL or ESA-ADB extracts, pass explicit feature columns when metadata or time/index fields are numeric but should not be model inputs.

## Metric Note

The command reports standard point-wise precision, recall, F1, false-alarm rate, and miss rate. It also reports point-adjusted F1, where detecting any point in a true anomaly segment marks the whole segment as detected.

Point-adjustment is included because it is common in time-series anomaly papers, but it can make weak detectors look stronger than they are. The point-wise metrics remain the primary conservative readout for this project until an official benchmark evaluator is available. For ESA-ADB, the next step is to use the benchmark's own hierarchical evaluation pipeline rather than inventing a project-specific substitute.

## Current Status

This is not yet a reproduced SMAP/MSL Telemanom LSTM baseline. It is the Track B plumbing layer: robust statistics, PCA reconstruction, Isolation Forest, metric handling, artifact formats, and CLI execution. The next Track B steps are:

- add SMAP/MSL download or import instructions and a loader that preserves channel labels and anomaly intervals
- reproduce the forecasting-plus-dynamic-threshold LSTM baseline on the prepared SMAP/MSL split
- move serious benchmark claims to ESA-ADB with its official evaluation tools
