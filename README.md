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
- `aerospace-prognostics cmapss-summary` CLI for sanity-checking local C-MAPSS files.
- `aerospace-prognostics cmapss-baseline` CLI for a first-pass scikit-learn gradient-boosting RUL baseline.
- `aerospace-prognostics cmapss-engineered-baseline-all` CLI for rolling/delta feature baselines.
- `aerospace-prognostics cmapss-regime-engineered-best-baseline-all` CLI for an experimental operating-regime-aware feature baseline.
- CI scaffold with linting, tests, and dependency audit.

## Phase 1 Artifacts

- Domain notes: [docs/phase1_turbofan_notes.md](docs/phase1_turbofan_notes.md)
- Real C-MAPSS baseline results: [docs/phase1_cmapss_baseline_results.md](docs/phase1_cmapss_baseline_results.md)
- EDA notebook scaffold: [notebooks/01_cmapss_phase1_eda.ipynb](notebooks/01_cmapss_phase1_eda.ipynb)

## Developer Quickstart

```powershell
uv sync --dev
uv run pytest
uv run ruff check .
```

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
uv run aerospace-prognostics phase1-cmapss --data-dir data/raw/cmapss --artifact-dir artifacts/phase1
```
