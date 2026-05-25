# Aerospace Prognostics

Portfolio research project for aerospace Prognostics & Health Management (PHM): turbofan Remaining Useful Life prediction, spacecraft telemetry anomaly detection, calibrated uncertainty, and deployment-aware engineering practice.

The working plan is tracked in [Aerospace_Prognostics_Project_Plan.md](Aerospace_Prognostics_Project_Plan.md).

## Current Slice

- NASA C-MAPSS evaluation metrics: RMSE and asymmetric RUL score.
- Piecewise-linear RUL cap helper for standard C-MAPSS target generation.
- C-MAPSS text-file loading and training-target generation helpers.
- CI scaffold with linting, tests, and dependency audit.

## Developer Quickstart

```powershell
uv sync --dev
uv run pytest
uv run ruff check .
```

Raw telemetry and trained artifacts are intentionally ignored by Git. Keep datasets under `data/` or another documented local path, and record source URLs/checksums when adding download scripts.
