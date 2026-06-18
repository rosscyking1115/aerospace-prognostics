# Aerospace Prognostics

Portfolio research project for aerospace Prognostics & Health Management (PHM): turbofan Remaining Useful Life prediction, spacecraft telemetry anomaly detection, calibrated uncertainty, and deployment-aware engineering practice.

The working plan is tracked in [Aerospace_Prognostics_Project_Plan.md](Aerospace_Prognostics_Project_Plan.md).

## Current Slice

- NASA C-MAPSS evaluation metrics: RMSE and asymmetric RUL score.
- Piecewise-linear RUL cap helper for standard C-MAPSS target generation.
- C-MAPSS text-file loading, training-target generation, and baseline feature-table helpers.
- Train-fitted feature standardisation utilities to avoid test leakage.
- Sliding-window helpers for later CNN/LSTM/Transformer sequence models.
- Dataset checksum utilities for reproducible local data handling.
- C-MAPSS EDA summaries for sensor drift, flat sensors, correlations, and operating-regime clusters.
- C-MAPSS manifest and verification commands for dataset provenance.
- Phase 1 C-MAPSS workflow command for provenance, EDA, baseline results, and a markdown run summary.
- Phase 2 C-MAPSS workflow command for sequence exports, deep-model sweeps, official-test and validation-selection prediction diagnostics, ranked reporting, and a markdown run summary.
- `aerospace-prognostics cmapss-summary` CLI for sanity-checking local C-MAPSS files.
- `aerospace-prognostics cmapss-baseline` CLI for a first-pass scikit-learn gradient-boosting RUL baseline.
- `aerospace-prognostics cmapss-engineered-baseline-all` CLI for rolling/delta feature baselines.
- `aerospace-prognostics cmapss-regime-engineered-best-baseline-all` CLI for an experimental operating-regime-aware feature baseline.
- `aerospace-prognostics cmapss-validation-selected-baseline-all` CLI for the current repeated-validation-selected classical baseline policy.
- `aerospace-prognostics cmapss-hgb-policy-baseline-all` CLI for the current validation-selected feature and HGB parameter policy.
- `aerospace-prognostics cmapss-validate-feature-candidates` CLI for unit-held-out temporal validation before official test scoring.
- `aerospace-prognostics cmapss-validate-feature-candidates-repeated` CLI for repeated validation across seeds and truncation horizons.
- `aerospace-prognostics cmapss-validate-hgb-grid` CLI for compact HistGradientBoosting hyperparameter checks.
- `aerospace-prognostics cmapss-validate-sensor-filters` CLI for checking EDA-informed sensor filtering before adopting it.
- `aerospace-prognostics cmapss-export-sequences` CLI for Phase 2 train, validation-selection, validation, and test sequence tensor exports.
- `aerospace-prognostics cmapss-cnn-baseline` CLI for the first Phase 2 PyTorch 1D-CNN sequence baseline.
- `aerospace-prognostics cmapss-lstm-baseline` CLI for Phase 2 LSTM/BiLSTM sequence baselines.
- `aerospace-prognostics cmapss-tcn-baseline` CLI for Phase 2 temporal convolution sequence baselines.
- `aerospace-prognostics cmapss-transformer-baseline` CLI for Phase 2 attention-based sequence baselines.
- `aerospace-prognostics cmapss-deep-baseline-compare` CLI for compact Phase 2 architecture and learning-rate sweeps, including residual CNN candidates and enhanced TCN knobs.
- `aerospace-prognostics cmapss-calibrate-deep-predictions` CLI for validation-fitted affine, predicted-bin residual, and tunable NASA-shift calibration diagnostics.
- `aerospace-prognostics cmapss-compare-rul-results` CLI for Phase 1 vs Phase 2 ranked model-comparison reports, including prediction-derived calibrated candidates.
- `aerospace-prognostics telemetry-robust-zscore-baseline` CLI for the Phase 2 spacecraft anomaly-detection baseline plumbing.
- `aerospace-prognostics telemetry-classical-anomaly-baselines` CLI for robust z-score, PCA reconstruction, and Isolation Forest anomaly baseline comparisons.
- `aerospace-prognostics smap-msl-download`, `smap-msl-summary`, and `smap-msl-export-channel-csv` CLIs for Telemanom SMAP/MSL raw `.npy` arrays and label metadata.
- `aerospace-prognostics smap-msl-classical-baselines` CLI for direct multi-channel SMAP/MSL classical anomaly baseline runs.
- `aerospace-prognostics smap-msl-lstm-forecast-baseline` CLI for a first forecasting-based SMAP/MSL anomaly baseline.
- `aerospace-prognostics smap-msl-compare-anomaly-results` CLI for ranked Track B anomaly comparison reports.
- `aerospace-prognostics phase2-smap-msl` CLI for a one-command Track B anomaly baseline bundle.
- `aerospace-prognostics phase2-smap-msl-verify-manifest` CLI for checking the Phase 2 SMAP/MSL artifact bundle against its run manifest and writing an optional markdown audit report.
- `aerospace-prognostics phase2-cmapss` run manifests with artifact checksums, runtime provenance, Git state, verifier/audit reports, and optional NASA-surrogate, asymmetric, blended, target-weighted, mini-batch monotonic, and unit-batch monotonic training losses for Track A experiment bundles.
- Phase 2 C-MAPSS deep prediction diagnostics for official-test and validation-selection per-window actual RUL, predicted RUL, signed error, absolute error, early/late error split, aggregate error summaries, RUL-bin calibration views, monotonicity checks, per-unit high-error trajectory reports, and validation-fitted calibration checks.
- `aerospace-prognostics cmapss-package-hgb-policy`, `cmapss-inspect-artifact`, `cmapss-predict-artifact`, `cmapss-validate-artifact`, `cmapss-benchmark-artifact`, `cmapss-promotion-report`, and `serve-api` CLIs for the production-ML deployment track.
- FastAPI serving observability with `/health` liveness, `/ready` readiness plus artifact identity, `/schema` inference-contract discovery, request IDs, latency headers, JSON request logs, and `/metrics` request, prediction, and drift counters/gauges.
- Serving-time telemetry drift summaries and prediction distribution monitoring.
- Docker serving image with non-root runtime, liveness healthcheck, OCI traceability labels, runtime dependency-surface checks, mounted-model smoke checks, and private GHCR publishing for release tags.
- Docker Compose local product stack for API, Streamlit console, mounted quickstart model artifacts, and SQLite app state.
- Candidate artifact promotion metadata with stable IDs and rollback guidance.
- Deployment-candidate model cards with intended use, metrics, inference contract, limitations, monitoring, and rollback notes.
- Artifact latency benchmark reports with optional p95 promotion gates.
- Release-candidate evidence bundles with file SHA-256 digests, promotion gates, SBOM, model card, artifact identity checks, dashboard artifacts, and serving-image manifest linkage.
- In-toto/SLSA-style release provenance statements that bind release bundles to Git commit and CI workflow metadata.
- Dashboard-ready fleet payloads that combine calibrated prediction intervals, priority ranks, attention reasons, and promotion/release evidence for fleet triage surfaces.
- Standalone static fleet dashboard HTML rendering for release/demo artifacts.
- No-download `quickstart-cmapss-demo` CLI that generates a full fixture-based deployment evidence bundle.
- Streamlit local operations console with fleet triage, model registry, release-gate report cards, artifact evidence, API-backed or local batch prediction, SQLite-backed prediction history, operator decisions, audit events, and API readiness status.
- `aerospace-prognostics app-init-db` CLI for initializing and seeding the local app database from quickstart artifacts.
- Promotion evidence reports that combine validation, benchmark, model-card, and SBOM gates.
- Optional API-key authentication and serving rate limits for protected inference endpoints.
- `aerospace-prognostics generate-sbom` CLI for lockfile-derived CycloneDX-style dependency inventory generation.
- CI scaffold with linting, tests, and dependency audit.

## Phase 1 Artifacts

- Domain notes: [docs/phase1_turbofan_notes.md](docs/phase1_turbofan_notes.md)
- Real C-MAPSS baseline results: [docs/phase1_cmapss_baseline_results.md](docs/phase1_cmapss_baseline_results.md)
- EDA notebook scaffold: [notebooks/01_cmapss_phase1_eda.ipynb](notebooks/01_cmapss_phase1_eda.ipynb)

## Phase 2 Artifacts

- Sequence-model baseline notes: [docs/phase2_cmapss_deep_baselines.md](docs/phase2_cmapss_deep_baselines.md)
- Spacecraft anomaly baseline notes: [docs/phase2_spacecraft_anomaly_baselines.md](docs/phase2_spacecraft_anomaly_baselines.md)

## Deployment Artifacts

- Production ML deployment notes: [docs/deployment.md](docs/deployment.md)
- Local product deployment: [docs/local_deployment.md](docs/local_deployment.md)
- Repository public-launch strategy: [docs/repo_launch_strategy.md](docs/repo_launch_strategy.md)
- Product roadmap: [docs/product_roadmap.md](docs/product_roadmap.md)

## Developer Quickstart

```powershell
uv sync --dev
uv run pytest
uv run ruff check .
```

`uv sync --dev` installs the deep-learning dependencies used by Phase 2 PyTorch experiments. The default runtime dependency set is intentionally lighter so the serving image can run the classical deployment artifact without installing training-only libraries.

For a no-download deployment demo, run the tiny fixture quickstart:

```powershell
uv run aerospace-prognostics quickstart-cmapss-demo
uv run aerospace-prognostics app-init-db
uv run streamlit run src/aerospace_prognostics/app/streamlit_app.py
```

These commands write dashboard, release bundle, provenance, artifact-inspection, model-card, validation, benchmark, and SBOM artifacts under `artifacts/quickstart_cmapss`, seed a local SQLite app database under `artifacts/app`, then open an interactive local operations console with fleet triage, prediction, evidence, and run-history views. See [docs/quickstart.md](docs/quickstart.md).

To run the local product stack with both the API and console:

```powershell
uv run aerospace-prognostics quickstart-cmapss-demo
docker compose up --build
```

Then open `http://127.0.0.1:8501` for the console or `http://127.0.0.1:8000/ready` for API readiness. The console System tab shows API health/readiness, mounted model identity, workspace state, and database counts. See [docs/local_deployment.md](docs/local_deployment.md).

Raw telemetry and trained artifacts are intentionally ignored by Git. Keep datasets under `data/` or another documented local path, and record source URLs/checksums when adding download scripts.

Once C-MAPSS files are available locally:

```powershell
uv run aerospace-prognostics cmapss-download --output-dir data/raw/cmapss --archive-path data/raw/downloads/cmapss_nasa.zip
uv run aerospace-prognostics cmapss-summary --data-dir data/raw/cmapss --subset FD001
uv run aerospace-prognostics cmapss-manifest --data-dir data/raw/cmapss --output-json artifacts/data/cmapss_manifest.json
uv run aerospace-prognostics cmapss-verify --data-dir data/raw/cmapss --manifest artifacts/data/cmapss_manifest.json
uv run aerospace-prognostics cmapss-eda --data-dir data/raw/cmapss --subset FD001 --output-json artifacts/eda/fd001.json
uv run aerospace-prognostics cmapss-baseline --data-dir data/raw/cmapss --subset FD001
uv run aerospace-prognostics cmapss-baseline --data-dir data/raw/cmapss --subset FD001 --standardize --output-json artifacts/results/fd001_baseline.json
uv run aerospace-prognostics cmapss-baseline-all --data-dir data/raw/cmapss --standardize --output-json artifacts/results/cmapss_baseline.json --output-csv artifacts/results/cmapss_baseline.csv
uv run aerospace-prognostics cmapss-engineered-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_engineered_baseline.json --output-csv artifacts/results/cmapss_engineered_baseline.csv
uv run aerospace-prognostics cmapss-engineered-window-sweep --data-dir data/raw/cmapss --rolling-windows 3 5 10 --output-json artifacts/results/cmapss_engineered_window_sweep.json --output-csv artifacts/results/cmapss_engineered_window_sweep.csv
uv run aerospace-prognostics cmapss-engineered-best-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_engineered_best_baseline.json --output-csv artifacts/results/cmapss_engineered_best_baseline.csv
uv run aerospace-prognostics cmapss-regime-engineered-best-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_regime_engineered_best_baseline.json --output-csv artifacts/results/cmapss_regime_engineered_best_baseline.csv
uv run aerospace-prognostics cmapss-validation-selected-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_selected_baseline.json --output-csv artifacts/results/cmapss_validation_selected_baseline.csv
uv run aerospace-prognostics cmapss-hgb-policy-baseline-all --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_hgb_policy_baseline.json --output-csv artifacts/results/cmapss_hgb_policy_baseline.csv
uv run aerospace-prognostics cmapss-validate-feature-candidates --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_feature_candidates.json --output-csv artifacts/results/cmapss_validation_feature_candidates.csv
uv run aerospace-prognostics cmapss-validate-feature-candidates-repeated --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_feature_candidates_repeated.json --output-csv artifacts/results/cmapss_validation_feature_candidates_repeated.csv
uv run aerospace-prognostics cmapss-validate-hgb-grid --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_hgb_grid.json --output-csv artifacts/results/cmapss_validation_hgb_grid.csv
uv run aerospace-prognostics cmapss-validate-sensor-filters --data-dir data/raw/cmapss --output-json artifacts/results/cmapss_validation_sensor_filters.json --output-csv artifacts/results/cmapss_validation_sensor_filters.csv
uv run aerospace-prognostics cmapss-export-sequences --data-dir data/raw/cmapss --output-dir artifacts/sequences/cmapss --window-size 30 --stride 1
uv run aerospace-prognostics cmapss-cnn-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --checkpoint-policy final --output-json artifacts/results/cmapss_cnn_fd001_final_baseline.json --output-csv artifacts/results/cmapss_cnn_fd001_final_baseline.csv --history-json artifacts/results/cmapss_cnn_fd001_final_history.json
uv run aerospace-prognostics cmapss-lstm-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --hidden-size 64 --bidirectional --output-json artifacts/results/cmapss_bilstm_fd001_baseline.json --output-csv artifacts/results/cmapss_bilstm_fd001_baseline.csv --history-json artifacts/results/cmapss_bilstm_fd001_history.json
uv run aerospace-prognostics cmapss-tcn-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --hidden-channels 64 --num-levels 3 --output-json artifacts/results/cmapss_tcn_fd001_baseline.json --output-csv artifacts/results/cmapss_tcn_fd001_baseline.csv --history-json artifacts/results/cmapss_tcn_fd001_history.json
uv run aerospace-prognostics cmapss-transformer-baseline --sequence-dir artifacts/sequences/cmapss --subsets FD001 --epochs 50 --d-model 64 --num-heads 4 --num-layers 2 --dim-feedforward 128 --output-json artifacts/results/cmapss_transformer_fd001_baseline.json --output-csv artifacts/results/cmapss_transformer_fd001_baseline.csv --history-json artifacts/results/cmapss_transformer_fd001_history.json
uv run aerospace-prognostics cmapss-deep-baseline-compare --sequence-dir artifacts/sequences/cmapss --subsets FD001 --models cnn rescnn bilstm tcn transformer --epochs 50 --hidden-sizes 32 64 --learning-rates 0.001 0.0003 --output-json artifacts/results/cmapss_deep_compare_fd001.json --output-csv artifacts/results/cmapss_deep_compare_fd001.csv
uv run aerospace-prognostics cmapss-deep-baseline-compare --sequence-dir artifacts/sequences/cmapss --subsets FD001 --models tcn --epochs 50 --hidden-sizes 32 64 --learning-rates 0.001 0.0003 --tcn-levels 4 --tcn-normalization layer_norm --tcn-weight-norm --tcn-pooling mean --output-json artifacts/results/cmapss_tcn_enhanced_fd001.json --output-csv artifacts/results/cmapss_tcn_enhanced_fd001.csv
uv run aerospace-prognostics cmapss-compare-rul-results --baseline-csv artifacts/results/cmapss_hgb_policy_baseline.csv --candidate-csv artifacts/results/cmapss_deep_compare_fd001.csv --output-csv artifacts/results/cmapss_phase2_model_comparison.csv --output-markdown artifacts/results/cmapss_phase2_model_comparison.md
uv run aerospace-prognostics smap-msl-download --output-dir data/raw/smap_msl --archive-path data/raw/downloads/smap_msl_telemanom.zip
uv run aerospace-prognostics smap-msl-summary --data-dir data/raw/smap_msl
uv run aerospace-prognostics smap-msl-export-channel-csv --data-dir data/raw/smap_msl --channel-id P-1 --output-dir artifacts/smap_msl_channels --metadata-json artifacts/smap_msl_channels/P-1/export.json
uv run aerospace-prognostics telemetry-robust-zscore-baseline --train-csv artifacts/smap_msl_channels/P-1/train.csv --test-csv artifacts/smap_msl_channels/P-1/test.csv --label-column label --feature-columns feature_0 feature_1 --threshold 3.5 --output-json artifacts/results/smap_msl_p1_robust_zscore.json --predictions-csv artifacts/results/smap_msl_p1_robust_zscore_predictions.csv
uv run aerospace-prognostics telemetry-classical-anomaly-baselines --train-csv artifacts/smap_msl_channels/P-1/train.csv --test-csv artifacts/smap_msl_channels/P-1/test.csv --label-column label --feature-columns feature_0 feature_1 --output-json artifacts/results/smap_msl_p1_classical_baselines.json --output-csv artifacts/results/smap_msl_p1_classical_baselines.csv --predictions-csv artifacts/results/smap_msl_p1_classical_predictions.csv
uv run aerospace-prognostics smap-msl-classical-baselines --data-dir data/raw/smap_msl --max-channels 5 --output-json artifacts/results/smap_msl_classical_baselines_sample.json --output-csv artifacts/results/smap_msl_classical_baselines_sample.csv
uv run aerospace-prognostics smap-msl-lstm-forecast-baseline --data-dir data/raw/smap_msl --max-channels 5 --window-size 30 --epochs 10 --output-json artifacts/results/smap_msl_lstm_forecast_sample.json --output-csv artifacts/results/smap_msl_lstm_forecast_sample.csv
uv run aerospace-prognostics smap-msl-lstm-forecast-baseline --data-dir data/raw/smap_msl --max-channels 5 --window-size 30 --epochs 10 --threshold-method dynamic --output-json artifacts/results/smap_msl_lstm_dynamic_sample.json --output-csv artifacts/results/smap_msl_lstm_dynamic_sample.csv
uv run aerospace-prognostics smap-msl-compare-anomaly-results --result-csv artifacts/results/smap_msl_classical_baselines_sample.csv artifacts/results/smap_msl_lstm_forecast_sample.csv artifacts/results/smap_msl_lstm_dynamic_sample.csv --source-labels classical lstm_robust lstm_dynamic --output-csv artifacts/results/smap_msl_anomaly_model_comparison.csv --output-markdown artifacts/results/smap_msl_anomaly_model_comparison.md
uv run aerospace-prognostics phase2-smap-msl --data-dir data/raw/smap_msl --artifact-dir artifacts/phase2_smap_msl --max-channels 5 --window-size 30 --epochs 10
uv run aerospace-prognostics phase2-smap-msl-verify-manifest --manifest artifacts/phase2_smap_msl/phase2_smap_msl_run_manifest.json --output-markdown artifacts/phase2_smap_msl/phase2_smap_msl_manifest_audit.md
uv run aerospace-prognostics quickstart-cmapss-demo
uv run aerospace-prognostics app-init-db
uv run aerospace-prognostics cmapss-package-hgb-policy --data-dir data/raw/cmapss --subset FD001 --output-path artifacts/models/cmapss_fd001_hgb_policy.joblib --metadata-json artifacts/models/cmapss_fd001_hgb_policy_metadata.json --model-card-markdown artifacts/models/cmapss_fd001_hgb_policy_model_card.md
uv run aerospace-prognostics cmapss-inspect-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --output-json artifacts/models/cmapss_fd001_hgb_policy_inspection.json
uv run aerospace-prognostics cmapss-predict-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --input-csv artifacts/examples/fd001_telemetry.csv --output-json artifacts/predictions/fd001_predictions.json
uv run aerospace-prognostics cmapss-validate-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --metadata-json artifacts/models/cmapss_fd001_hgb_policy_metadata.json --input-csv artifacts/examples/fd001_telemetry.csv --output-json artifacts/models/cmapss_fd001_hgb_policy_validation.json
uv run aerospace-prognostics cmapss-benchmark-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --input-csv artifacts/examples/fd001_telemetry.csv --max-p95-latency-ms 100 --output-json artifacts/models/cmapss_fd001_hgb_policy_benchmark.json
uv run aerospace-prognostics cmapss-promotion-report --validation-json artifacts/models/cmapss_fd001_hgb_policy_validation.json --benchmark-json artifacts/models/cmapss_fd001_hgb_policy_benchmark.json --model-card-markdown artifacts/models/cmapss_fd001_hgb_policy_model_card.md --sbom-json artifacts/sbom/cyclonedx.json --output-json artifacts/models/cmapss_fd001_hgb_policy_promotion.json --output-markdown artifacts/models/cmapss_fd001_hgb_policy_promotion.md
uv run aerospace-prognostics dashboard-fleet-payload --prediction-json artifacts/predictions/fd001_predictions.json --promotion-json artifacts/models/cmapss_fd001_hgb_policy_promotion.json --output-json artifacts/dashboard/fleet_payload.json
uv run aerospace-prognostics dashboard-render-html --payload-json artifacts/dashboard/fleet_payload.json --output-html artifacts/dashboard/fleet_dashboard.html
uv run aerospace-prognostics cmapss-release-bundle --release-name cmapss-fd001-candidate --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --metadata-json artifacts/models/cmapss_fd001_hgb_policy_metadata.json --model-card-markdown artifacts/models/cmapss_fd001_hgb_policy_model_card.md --promotion-json artifacts/models/cmapss_fd001_hgb_policy_promotion.json --sbom-json artifacts/sbom/cyclonedx.json --dashboard-payload-json artifacts/dashboard/fleet_payload.json --dashboard-html artifacts/dashboard/fleet_dashboard.html --output-json artifacts/release/cmapss_fd001_release_bundle.json --output-markdown artifacts/release/cmapss_fd001_release_bundle.md
uv run aerospace-prognostics serve-api --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --host 127.0.0.1 --port 8000
uv run aerospace-prognostics phase1-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase1
uv run aerospace-prognostics phase2-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase2 --subsets FD001 --models cnn rescnn bilstm tcn transformer --epochs 50 --hidden-sizes 32 64 --learning-rates 0.001 0.0003 --training-loss mse
uv run aerospace-prognostics phase2-cmapss-verify-manifest --manifest artifacts/phase2/phase2_run_manifest.json --output-markdown artifacts/phase2/phase2_manifest_audit.md
```

Telemanom's current README points users to the Kaggle-hosted SMAP/MSL archive. If the legacy public S3 archive is unavailable, download `patrickfleith/nasa-anomaly-detection-dataset-smap-msl` to `data/raw/downloads/smap_msl_telemanom.zip`, then rerun `smap-msl-download`; the command will import that local archive without downloading it again.
