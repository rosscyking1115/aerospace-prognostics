# Production ML Deployment Track

This track turns the research pipeline into a deployable ML system. The first production slice packages the current validation-selected HGB policy baseline, serves it behind a FastAPI contract, and keeps raw data plus generated model artifacts out of Git.

## Supported Flow

```powershell
uv run aerospace-prognostics cmapss-package-hgb-policy --data-dir data/raw/cmapss --subset FD001 --output-path artifacts/models/cmapss_fd001_hgb_policy.joblib --metadata-json artifacts/models/cmapss_fd001_hgb_policy_metadata.json --model-card-markdown artifacts/models/cmapss_fd001_hgb_policy_model_card.md
uv run aerospace-prognostics cmapss-predict-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --input-csv artifacts/examples/fd001_telemetry.csv --output-json artifacts/predictions/fd001_predictions.json
uv run aerospace-prognostics cmapss-validate-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --metadata-json artifacts/models/cmapss_fd001_hgb_policy_metadata.json --input-csv artifacts/examples/fd001_telemetry.csv --output-json artifacts/models/cmapss_fd001_hgb_policy_validation.json
uv run aerospace-prognostics cmapss-benchmark-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --input-csv artifacts/examples/fd001_telemetry.csv --max-p95-latency-ms 100 --output-json artifacts/models/cmapss_fd001_hgb_policy_benchmark.json
uv run aerospace-prognostics generate-sbom --lockfile uv.lock --output-json artifacts/sbom/cyclonedx.json
uv run aerospace-prognostics serve-api --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --host 127.0.0.1 --port 8000
```

Run `cmapss-validate-artifact` before promotion. It verifies that the joblib artifact exists, loads with a supported schema version, carries required promotion metadata, optionally matches the exported metadata JSON, and optionally produces at least one prediction from a telemetry CSV. The command exits non-zero when any validation check fails.

Run `cmapss-benchmark-artifact` with a representative telemetry CSV before promotion. It records artifact size, input rows, prediction count, and repeated inference latency summaries, and it exits non-zero if `--max-p95-latency-ms` is supplied and the p95 latency exceeds that budget.

The packaging command can also write a markdown model card. The card summarizes intended use, official-test metrics, feature policy, inference contract, serving monitoring, limitations, promotion gate, and rollback strategy so candidate-review evidence is readable without opening the binary artifact.

## Container Serving

The serving image does not bundle a model artifact. A clean container starts and reports `missing_model` from `GET /health`, which lets CI validate the image without committing generated artifacts. `GET /ready` returns HTTP 503 until a model artifact is loaded, so orchestrators can distinguish a live container from one that is ready for prediction traffic. When ready, the endpoint returns a minimal artifact identity with schema version, dataset, subset, model name, artifact ID, and stage; full metadata remains behind authenticated `GET /version`.

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
- `GET /ready`
- `GET /version`
- `GET /schema`
- `GET /metrics`
- `POST /predict`

`POST /predict` accepts raw C-MAPSS telemetry rows with the canonical columns: `unit_number`, `time_in_cycles`, three operating settings, and 21 sensor values. The service groups rows by unit and returns one capped RUL prediction per unit using each unit's latest observed cycle.

`GET /schema` returns the loaded artifact's concrete inference contract: required telemetry columns, row limits, grouping behavior, prediction fields, output RUL bounds, and monitoring block names. This makes the deployed model contract discoverable by clients and smoke-test jobs without exposing the binary artifact.

Every API request receives `x-request-id` and `x-process-time-ms` response headers. If the caller sends `x-request-id`, the service preserves it; otherwise it generates one. Requests are logged as one-line JSON records with method, route, status code, request ID, and latency. `GET /metrics` exposes lightweight Prometheus-style counters and latency summaries for local container smoke checks and basic deployment monitoring.

Prediction responses include a `monitoring` block. It compares request telemetry means against train-fit artifact reference statistics with standardized mean-shift scores, lists columns above the drift threshold, and summarizes the request's prediction distribution. The same compact monitoring summary is emitted as structured JSON logs for downstream alerting.

## Authentication And Rate Limiting

`GET /health` and `GET /ready` are intentionally unauthenticated so container orchestrators can check liveness and readiness without a secret. `GET /version`, `GET /schema`, `GET /metrics`, and `POST /predict` enforce optional API-key authentication when `AEROSPACE_PROGNOSTICS_API_KEY` is set. Clients may send the key with either `x-api-key: <key>` or `Authorization: Bearer <key>`.

Set `AEROSPACE_PROGNOSTICS_RATE_LIMIT_PER_MINUTE` to a positive integer to enable an in-memory fixed-window rate limit for protected endpoints. Requests are bucketed by authenticated API key when authentication is enabled, otherwise by client host. The default value is `0`, which disables rate limiting for local development and CI smoke checks.

For public deployment, run the API behind a managed gateway, load balancer, or ingress layer that provides TLS termination, secret rotation, centralized rate limiting, request size limits, and audit logs. The built-in controls are a local safety layer and testable serving contract, not a replacement for platform perimeter controls.

## Supply Chain Evidence

`generate-sbom` reads the locked `uv.lock` environment and writes a CycloneDX-style JSON software bill of materials. The CI workflow runs this command after `pip-audit`, so every pushed commit proves both that dependencies are vulnerability-checked and that the dependency inventory remains generatable from the lockfile.

## Promotion And Rollback

Every packaged artifact includes a `promotion` metadata block with:

- Stable `artifact_id` derived from model identity, selected policy, dataset shape, and official-test metrics.
- `stage=candidate` until a human or release process marks it as promoted outside the binary artifact.
- Official-test RMSE and NASA score, train/test row and unit counts, feature policy, HGB policy, RUL cap, and random seed.
- Rollback instructions stating that rollback is a pointer/env change to the previous promoted artifact, not a retraining event.

Promotion procedure:

1. Build a candidate artifact, metadata JSON, and model card with `cmapss-package-hgb-policy`.
2. Confirm the metadata JSON matches the intended subset, model policy, metrics, and `artifact_id`.
3. Run `cmapss-validate-artifact` against the candidate artifact, metadata JSON, and a representative telemetry CSV.
4. Run the benchmark command against representative telemetry and confirm the p95 latency budget passes.
5. Run local or CI smoke checks against the serving container with the candidate mounted through `AEROSPACE_PROGNOSTICS_MODEL_PATH`.
6. Promote by updating the deployment artifact pointer, secret, mounted path, or model registry entry to the candidate artifact.
7. Keep the previously promoted artifact available until the new model has passed telemetry drift, prediction-distribution, latency, and error-rate checks.

Rollback procedure:

1. Restore the previous promoted artifact pointer or container environment value.
2. Restart or redeploy the serving process.
3. Confirm `GET /version` reports the prior artifact metadata, `GET /health` returns `ok`, and `GET /ready` returns `ready` with the prior artifact ID.
4. Preserve request logs and monitoring summaries from the failed candidate for post-incident review.

## Production Readiness Notes

The artifact contains the trained scikit-learn model, feature policy, rolling-window configuration, operating-regime transformer when needed, train-fitted standardizer, input schema, feature schema, train-fit telemetry reference statistics, promotion metadata, and run metadata. This makes inference reproducible without refitting preprocessing at request time.

Prediction outputs are bounded to `[0, rul_cap]` at inference time. The first real FD001 package smoke test used the validation-selected HGB policy artifact:

| Subset | Model | Feature Policy | Output Bounds | Official Test RMSE | Official Test NASA Score |
|---|---|---|---|---:|---:|
| FD001 | `hist_gradient_boosting_regime_engineered_w10_r6_default` | `regime_engineered` | `[0, 125]` | 12.990750 | 252.462482 |

Current scope:

- Local model artifact packaging with `joblib`.
- Markdown model cards for deployment-candidate review.
- Batch inference from CSV.
- Artifact validation command for promotion checks and prediction smoke tests.
- Artifact benchmark command for model size and inference-latency promotion gates.
- FastAPI inference surface with request validation, liveness, readiness, and version endpoints.
- Model-specific inference schema endpoint for client contract discovery.
- Public readiness identity for loaded artifacts without exposing full version metadata.
- Structured JSON request logs, request IDs, latency headers, and scrapeable serving metrics.
- Request telemetry drift summaries and prediction-distribution monitoring.
- Promotion metadata with stable artifact IDs and rollback runbook.
- Optional API-key authentication and per-client serving rate limits.
- Lockfile-derived CycloneDX-style SBOM generation in CI.
- Tests for artifact round-trip and API prediction behavior.
- Dockerfile scaffold for containerized serving.
- CI Docker image build check.
- CI container startup smoke test against `GET /health`.
