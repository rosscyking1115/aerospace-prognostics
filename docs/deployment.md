# Production ML Deployment Track

This track turns the research pipeline into a deployable ML system. The first production slice packages the current validation-selected HGB policy baseline, serves it behind a FastAPI contract, and keeps raw data plus generated model artifacts out of Git.

## Supported Flow

```powershell
uv run aerospace-prognostics cmapss-package-hgb-policy --data-dir data/raw/cmapss --subset FD001 --output-path artifacts/models/cmapss_fd001_hgb_policy.joblib --metadata-json artifacts/models/cmapss_fd001_hgb_policy_metadata.json
uv run aerospace-prognostics cmapss-predict-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --input-csv artifacts/examples/fd001_telemetry.csv --output-json artifacts/predictions/fd001_predictions.json
uv run aerospace-prognostics serve-api --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --host 127.0.0.1 --port 8000
```

## Container Serving

The serving image does not bundle a model artifact. A clean container starts and reports `missing_model` from `GET /health`, which lets CI validate the image without committing generated artifacts.

Run the image with an explicit model mount and environment variable when serving predictions:

```powershell
docker run --rm -p 8000:8000 `
  -v ${PWD}/artifacts/models:/models `
  -e AEROSPACE_PROGNOSTICS_MODEL_PATH=/models/cmapss_fd001_hgb_policy.joblib `
  aerospace-prognostics:ci
```

The API exposes:

- `GET /health`
- `GET /version`
- `GET /metrics`
- `POST /predict`

`POST /predict` accepts raw C-MAPSS telemetry rows with the canonical columns: `unit_number`, `time_in_cycles`, three operating settings, and 21 sensor values. The service groups rows by unit and returns one capped RUL prediction per unit using each unit's latest observed cycle.

Every API request receives `x-request-id` and `x-process-time-ms` response headers. If the caller sends `x-request-id`, the service preserves it; otherwise it generates one. Requests are logged as one-line JSON records with method, route, status code, request ID, and latency. `GET /metrics` exposes lightweight Prometheus-style counters and latency summaries for local container smoke checks and basic deployment monitoring.

## Production Readiness Notes

The artifact contains the trained scikit-learn model, feature policy, rolling-window configuration, operating-regime transformer when needed, train-fitted standardizer, input schema, feature schema, and run metadata. This makes inference reproducible without refitting preprocessing at request time.

Prediction outputs are bounded to `[0, rul_cap]` at inference time. The first real FD001 package smoke test used the validation-selected HGB policy artifact:

| Subset | Model | Feature Policy | Output Bounds | Official Test RMSE | Official Test NASA Score |
|---|---|---|---|---:|---:|
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6_default` | `regime_engineered` | `[0, 125]` | 12.990750 | 252.462482 |

Current scope:

- Local model artifact packaging with `joblib`.
- Batch inference from CSV.
- FastAPI inference surface with request validation and health/version endpoints.
- Structured JSON request logs, request IDs, latency headers, and scrapeable serving metrics.
- Tests for artifact round-trip and API prediction behavior.
- Dockerfile scaffold for containerized serving.
- CI Docker image build check.
- CI container startup smoke test against `GET /health`.

Next hardening steps:

- Add drift summaries for incoming telemetry and prediction distributions.
- Add model promotion metadata and rollback docs.
- Add authentication/rate limiting before any public deployment.
