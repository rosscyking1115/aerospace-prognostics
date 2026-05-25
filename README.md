# Aerospace Prognostics

Portfolio research project for aerospace Prognostics & Health Management (PHM): turbofan Remaining Useful Life prediction, spacecraft telemetry anomaly detection, calibrated uncertainty, and deployment-aware engineering practice.

The working plan is tracked in [Aerospace_Prognostics_Project_Plan.md](Aerospace_Prognostics_Project_Plan.md).

## Current Slice

- NASA C-MAPSS evaluation metrics: RMSE and asymmetric RUL score.
- Piecewise-linear RUL cap helper for standard C-MAPSS target generation.
- C-MAPSS text-file loading, training-target generation, and baseline feature-table helpers.
- Dataset checksum utilities for reproducible local data handling.
- `aerospace-prognostics cmapss-summary` CLI for sanity-checking local C-MAPSS files.
- `aerospace-prognostics cmapss-baseline` CLI for a first-pass scikit-learn gradient-boosting RUL baseline.
- CI scaffold with linting, tests, and dependency audit.

## Developer Quickstart

```powershell
uv sync --dev
uv run pytest
uv run ruff check .
```

Raw telemetry and trained artifacts are intentionally ignored by Git. Keep datasets under `data/` or another documented local path, and record source URLs/checksums when adding download scripts.

Once C-MAPSS files are available locally:

```powershell
uv run aerospace-prognostics cmapss-summary --data-dir data/raw/cmapss --subset FD001
uv run aerospace-prognostics cmapss-baseline --data-dir data/raw/cmapss --subset FD001
```
