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
  -e AEROSPACE_PROGNOSTICS_API_KEY=<set-a-secret-value> `
  -e AEROSPACE_PROGNOSTICS_RATE_LIMIT_PER_MINUTE=60 `
  aerospace-prognostics:ci
```

The API exposes:

- `GET /health`
- `GET /version`
- `GET /metrics`
- `POST /predict`

`POST /predict` accepts raw C-MAPSS telemetry rows with the canonical columns: `unit_number`, `time_in_cycles`, three operating settings, and 21 sensor values. The service groups rows by unit and returns one capped RUL prediction per unit using each unit's latest observed cycle.

Every API request receives `x-request-id` and `x-process-time-ms` response headers. If the caller sends `x-request-id`, the service preserves it; otherwise it generates one. Requests are logged as one-line JSON records with method, route, status code, request ID, and latency. `GET /metrics` exposes lightweight Prometheus-style counters and latency summaries for local container smoke checks and basic deployment monitoring.

Prediction responses include a `monitoring` block. It compares request telemetry means against train-fit artifact reference statistics with standardized mean-shift scores, lists columns above the drift threshold, and summarizes the request's prediction distribution. The same compact monitoring summary is emitted as structured JSON logs for downstream alerting.

## Authentication And Rate Limiting

`GET /health` is intentionally unauthenticated so container orchestrators can check readiness without a secret. `GET /version`, `GET /metrics`, and `POST /predict` enforce optional API-key authentication when `AEROSPACE_PROGNOSTICS_API_KEY` is set. Clients may send the key with either `x-api-key: <key>` or `Authorization: Bearer <key>`.

Set `AEROSPACE_PROGNOSTICS_RATE_LIMIT_PER_MINUTE` to a positive integer to enable an in-memory fixed-window rate limit for protected endpoints. Requests are bucketed by authenticated API key when authentication is enabled, otherwise by client host. The default value is `0`, which disables rate limiting for local development and CI smoke checks.

For public deployment, run the API behind a managed gateway, load balancer, or ingress layer that provides TLS termination, secret rotation, centralized rate limiting, request size limits, and audit logs. The built-in controls are a local safety layer and testable serving contract, not a replacement for platform perimeter controls.

## Promotion And Rollback

Every packaged artifact includes a `promotion` metadata block with:

- Stable `artifact_id` derived from model identity, selected policy, dataset shape, and official-test metrics.
- `stage=candidate` until a human or release process marks it as promoted outside the binary artifact.
- Official-test RMSE and NASA score, train/test row and unit counts, feature policy, HGB policy, RUL cap, and random seed.
- Rollback instructions stating that rollback is a pointer/env change to the previous promoted artifact, not a retraining event.

Promotion procedure:

1. Build a candidate artifact and metadata JSON with `cmapss-package-hgb-policy`.
2. Confirm the metadata JSON matches the intended subset, model policy, metrics, and `artifact_id`.
3. Run local or CI smoke checks against the serving container with the candidate mounted through `AEROSPACE_PROGNOSTICS_MODEL_PATH`.
4. Promote by updating the deployment artifact pointer, secret, mounted path, or model registry entry to the candidate artifact.
5. Keep the previously promoted artifact available until the new model has passed telemetry drift, prediction-distribution, latency, and error-rate checks.

Rollback procedure:

1. Restore the previous promoted artifact pointer or container environment value.
2. Restart or redeploy the serving process.
3. Confirm `GET /version` reports the prior artifact metadata and `GET /health` returns `ok`.
4. Preserve request logs and monitoring summaries from the failed candidate for post-incident review.

## Production Readiness Notes

The artifact contains the trained scikit-learn model, feature policy, rolling-window configuration, operating-regime transformer when needed, train-fitted standardizer, input schema, feature schema, train-fit telemetry reference statistics, promotion metadata, and run metadata. This makes inference reproducible without refitting preprocessing at request time.

Prediction outputs are bounded to `[0, rul_cap]` at inference time. The first real FD001 package smoke test used the validation-selected HGB policy artifact:

| Subset | Model | Feature Policy | Output Bounds | Official Test RMSE | Official Test NASA Score |
|---|---|---|---|---:|---:|
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6_default` | `regime_engineered` | `[0, 125]` | 12.990750 | 252.462482 |

Current scope:

- Local model artifact packaging with `joblib`.
- Batch inference from CSV.
- FastAPI inference surface with request validation and health/version endpoints.
- Structured JSON request logs, request IDs, latency headers, and scrapeable serving metrics.
- Request telemetry drift summaries and prediction-distribution monitoring.
- Promotion metadata with stable artifact IDs and rollback runbook.
- Optional API-key authentication and per-client serving rate limits.
- Tests for artifact round-trip and API prediction behavior.
- Dockerfile scaffold for containerized serving.
- CI Docker image build check.
- CI container startup smoke test against `GET /health`.
