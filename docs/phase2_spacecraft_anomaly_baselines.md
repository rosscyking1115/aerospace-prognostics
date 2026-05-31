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
uv run aerospace-prognostics smap-msl-classical-baselines --data-dir data/raw/smap_msl --output-json artifacts/results/smap_msl_classical_all.json --output-csv artifacts/results/smap_msl_classical_all.csv
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
uv run aerospace-prognostics smap-msl-compare-anomaly-results --result-csv artifacts/results/smap_msl_classical_all.csv --source-labels classical_all --output-csv artifacts/results/smap_msl_classical_all_comparison.csv --output-markdown artifacts/results/smap_msl_classical_all_comparison.md
```

The comparison report ranks rows per channel by point-wise F1, then point-adjusted F1, then false-alarm and miss rates. It also writes aggregate winner counts and average metrics by source/model. Point-wise F1 stays first because point-adjustment can over-reward weak detectors on long labelled intervals.

One-command Track B bundle:

```powershell
uv run aerospace-prognostics phase2-smap-msl --data-dir data/raw/smap_msl --artifact-dir artifacts/phase2_smap_msl --max-channels 5 --window-size 30 --epochs 10 --robust-policy-false-alarm-budget 0.15
uv run aerospace-prognostics phase2-smap-msl-verify-manifest --manifest artifacts/phase2_smap_msl/phase2_smap_msl_run_manifest.json
```

This writes classical, LSTM robust-threshold, LSTM dynamic-threshold, optional robust-threshold policy, ranked comparison, `phase2_smap_msl_summary.md`, and `phase2_smap_msl_run_manifest.json` artifacts under the requested artifact directory. The run manifest records the selected channels, model and threshold parameters, artifact paths, row counts, SHA-256/size checksums, Python/platform versions, core dependency versions, and Git commit state for reproducibility. The verify command checks that the manifest is structurally valid, referenced artifacts exist, artifact checksums still match, and CSV row counts match the recorded counts. When `--robust-policy-false-alarm-budget` is supplied, the workflow also writes threshold sweep, operating-point, and comparison-ready policy artifacts, then includes the policy rows in the ranked comparison.

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

An all-channel classical-only sweep now covers the full unique-channel baseline floor:

```powershell
uv run aerospace-prognostics smap-msl-classical-baselines --data-dir data/raw/smap_msl --output-json artifacts/results/smap_msl_classical_all.json --output-csv artifacts/results/smap_msl_classical_all.csv
uv run aerospace-prognostics smap-msl-compare-anomaly-results --result-csv artifacts/results/smap_msl_classical_all.csv --source-labels classical_all --output-csv artifacts/results/smap_msl_classical_all_comparison.csv --output-markdown artifacts/results/smap_msl_classical_all_comparison.md
uv run aerospace-prognostics smap-msl-robust-threshold-sweep --data-dir data/raw/smap_msl --thresholds 3.5 5 7 10 15 --false-alarm-budget 0.15 --selection-group spacecraft --output-json artifacts/results/smap_msl_robust_threshold_sweep.json --output-csv artifacts/results/smap_msl_robust_threshold_sweep.csv --aggregate-json artifacts/results/smap_msl_robust_threshold_sweep_aggregate.json --aggregate-csv artifacts/results/smap_msl_robust_threshold_sweep_aggregate.csv --operating-point-json artifacts/results/smap_msl_robust_threshold_operating_points.json --operating-point-csv artifacts/results/smap_msl_robust_threshold_operating_points.csv --policy-json artifacts/results/smap_msl_robust_threshold_policy.json --policy-csv artifacts/results/smap_msl_robust_threshold_policy.csv
uv run aerospace-prognostics smap-msl-compare-anomaly-results --result-csv artifacts/results/smap_msl_classical_all.csv artifacts/results/smap_msl_robust_threshold_policy.csv --source-labels classical_all robust_policy_0p15 --output-csv artifacts/results/smap_msl_classical_all_with_policy_comparison.csv --output-markdown artifacts/results/smap_msl_classical_all_with_policy_comparison.md
```

The raw labels contain 82 rows but 81 unique channel IDs; `P-2` appears twice in the source labels. Benchmark selection and SMAP/MSL experiment execution now deduplicate by channel ID while preserving first-seen label order. The all-channel classical sweep generated 243 rows across 81 unique channels. Winner counts were robust z-score 41 of 81, PCA reconstruction 21 of 81, and Isolation Forest 19 of 81. Across all 81 rows per method, robust z-score had the highest mean point-wise F1 (0.165343) but also the highest mean false-alarm rate (0.187988), while PCA reconstruction and Isolation Forest had lower mean false-alarm rates at 0.059153 and 0.065929. This makes per-family threshold tuning the next sensible Track B improvement.

The robust z-score threshold sweep tested thresholds 3.5, 5, 7, 10, and 15 across all 81 unique channels. Threshold 5.0 is the best aggregate operating point in this first sweep: mean F1 rises slightly from 0.165343 to 0.168552 while mean false-alarm rate drops from 0.187988 to 0.155975. Higher thresholds continue to reduce false alarms, reaching 0.096831 at threshold 15, but mean F1 falls to 0.124858 as recall is lost. Per-channel wins remain mixed, so a single global robust threshold is not enough for a polished detector; the next threshold step should select operating points by spacecraft/channel family or optimize against a constrained false-alarm budget.

The sweep command can now select auditable operating points with `--false-alarm-budget`. With `--selection-group spacecraft`, it chooses the highest-F1 feasible threshold per SMAP/MSL family; if no threshold satisfies the budget for a group, the output marks that group as infeasible and reports the lowest false-alarm option. This keeps threshold tuning honest: the chosen policy records the constraint, threshold, feasibility flag, channel count, mean F1, point-adjusted F1, false-alarm rate, and miss rate.

Using a 0.15 mean false-alarm budget, both spacecraft families have feasible robust-threshold operating points in the current grid. MSL selects threshold 10 across 27 channels, with mean F1 0.170348 and mean false-alarm rate 0.147653. SMAP selects threshold 5 across 54 channels, with mean F1 0.155613 and mean false-alarm rate 0.127544. This is still a coarse family-level policy, but it is more deployment-shaped than a single global threshold because it names the alert budget and records whether each family actually satisfies it.

The selected policy can also be written as a comparison-ready result CSV. Compared against the all-channel classical table, the `robust_policy_0p15` rows win 15 of 81 channels. Its mean F1 is 0.160525 versus 0.165343 for the default robust z-score, while mean false-alarm rate improves from 0.187988 to 0.134247 and mean point-adjusted F1 rises from 0.465692 to 0.541768. That is a credible production-style tradeoff for an alerting baseline: fewer false alarms and better interval coverage, with a small point-wise F1 cost that remains visible in the report.

If `--feature-columns` is omitted, the command uses all numeric columns from the train CSV except the label column. For SMAP/MSL channel exports, pass `feature_0 feature_1 ...` explicitly when you want to exclude the numeric `timestep` column.

## Metric Note

The command reports standard point-wise precision, recall, F1, false-alarm rate, and miss rate. It also reports point-adjusted F1, where detecting any point in a true anomaly segment marks the whole segment as detected.

Point-adjustment is included because it is common in time-series anomaly papers, but it can make weak detectors look stronger than they are. The point-wise metrics remain the primary conservative readout for this project until an official benchmark evaluator is available. For ESA-ADB, the next step is to use the benchmark's own hierarchical evaluation pipeline rather than inventing a project-specific substitute.

## Current Status

This is not yet a reproduced SMAP/MSL Telemanom LSTM baseline. It is the Track B baseline layer: raw SMAP/MSL loading, direct multi-channel classical runs, LSTM forecasting with robust and dynamic thresholding, robust statistics, PCA reconstruction, Isolation Forest, metric handling, artifact formats, model comparison reporting, workflow orchestration, and CLI execution. The next Track B steps are:

- tune classical and LSTM thresholds per spacecraft family to reduce false alarms without hiding point-wise recall losses
- compare the compact dynamic threshold against Telemanom's original predictions/threshold outputs on downloaded SMAP/MSL data
- move serious benchmark claims to ESA-ADB with its official evaluation tools
