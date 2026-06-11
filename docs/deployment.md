# Production ML Deployment Track

This track turns the research pipeline into a deployable ML system. The first production slice packages the current validation-selected HGB policy baseline, serves it behind a FastAPI contract, and keeps raw data plus generated model artifacts out of Git.

## Supported Flow

```powershell
uv run aerospace-prognostics cmapss-package-hgb-policy --data-dir data/raw/cmapss --subset FD001 --output-path artifacts/models/cmapss_fd001_hgb_policy.joblib --metadata-json artifacts/models/cmapss_fd001_hgb_policy_metadata.json --model-card-markdown artifacts/models/cmapss_fd001_hgb_policy_model_card.md
uv run aerospace-prognostics cmapss-predict-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --input-csv artifacts/examples/fd001_telemetry.csv --output-json artifacts/predictions/fd001_predictions.json
uv run aerospace-prognostics cmapss-validate-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --metadata-json artifacts/models/cmapss_fd001_hgb_policy_metadata.json --input-csv artifacts/examples/fd001_telemetry.csv --output-json artifacts/models/cmapss_fd001_hgb_policy_validation.json
uv run aerospace-prognostics cmapss-benchmark-artifact --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --input-csv artifacts/examples/fd001_telemetry.csv --max-p95-latency-ms 100 --output-json artifacts/models/cmapss_fd001_hgb_policy_benchmark.json
uv run aerospace-prognostics generate-sbom --lockfile uv.lock --output-json artifacts/sbom/cyclonedx.json
uv run aerospace-prognostics cmapss-promotion-report --validation-json artifacts/models/cmapss_fd001_hgb_policy_validation.json --benchmark-json artifacts/models/cmapss_fd001_hgb_policy_benchmark.json --model-card-markdown artifacts/models/cmapss_fd001_hgb_policy_model_card.md --sbom-json artifacts/sbom/cyclonedx.json --output-json artifacts/models/cmapss_fd001_hgb_policy_promotion.json --output-markdown artifacts/models/cmapss_fd001_hgb_policy_promotion.md
uv run aerospace-prognostics serve-api --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib --host 127.0.0.1 --port 8000
```

Run `cmapss-validate-artifact` before promotion. It verifies that the joblib artifact exists, loads with a supported schema version, carries required promotion metadata, optionally matches the exported metadata JSON, and optionally produces at least one prediction from a telemetry CSV. The command exits non-zero when any validation check fails.

Run `cmapss-benchmark-artifact` with a representative telemetry CSV before promotion. It records artifact size, input rows, prediction count, and repeated inference latency summaries, and it exits non-zero if `--max-p95-latency-ms` is supplied and the p95 latency exceeds that budget.

Run `cmapss-promotion-report` after validation, benchmarking, model-card generation, and SBOM generation. It composes a JSON and optional markdown release-gate report, checks that validation and benchmark evidence refer to the same artifact ID, and exits non-zero when any supplied gate fails.

Run `cmapss-release-bundle` after promotion evidence and, for containerized candidates, after the serving-image manifest is generated. It composes a production-grade release candidate record with SHA-256 digests for the exact model and evidence files, artifact identity checks between metadata and promotion evidence, SBOM/model-card gates, promotion-gate status, optional dashboard payload/HTML evidence, and optional serving-container manifest validation. This is the final bundle a reviewer can inspect before publishing a private GHCR image or promoting a mounted model pointer.

Run `generate-release-provenance` after the release bundle. It emits an in-toto statement with a SLSA provenance predicate, binding the release bundle subjects to the source repository, Git SHA, Git ref, workflow name, run ID, and builder identity. CI writes `artifacts/release/cmapss_fd001_provenance.json` on every push. GitHub-native artifact attestations are the preferred hosted attestation layer when available, but GitHub currently limits private-repository artifact attestations on Free/Pro/Team plans; keep this internal provenance path active while the project remains private.

The packaging command can also write a markdown model card. The card summarizes intended use, official-test metrics, feature policy, inference contract, serving monitoring, limitations, promotion gate, and rollback strategy so candidate-review evidence is readable without opening the binary artifact.

Run `dashboard-fleet-payload` after batch prediction and promotion/release evidence generation when preparing demo assets. It writes a stable `aerospace-prognostics/fleet-dashboard/v1` JSON contract with fleet assets, RUL risk levels, summary counts, prediction provenance, and optional promotion/release evidence. This keeps the future dashboard UI decoupled from model internals and lets public demos run against fixture or sample outputs before the Phase 3 models are final.

Run `dashboard-render-html` against that payload to create a standalone static dashboard artifact. The HTML is self-contained and dependency-free, so it can be opened locally, attached to release evidence, or used as the first public demo surface before the project graduates to Streamlit or a Next.js/FastAPI product interface.

## Container Serving

The serving image does not bundle a model artifact. A clean container starts and reports `missing_model` from `GET /health`, which lets CI validate the image without committing generated artifacts. The Docker image includes a `HEALTHCHECK` that probes `GET /health` through a small stdlib Python helper, so the slim image does not depend on curl or wget for liveness. The image installs only the default runtime dependency set and intentionally excludes the Phase 2 PyTorch training dependency; CI checks that `torch` is absent from the built image. CI also stamps OCI labels for source repository, Git revision, build time, version, title, description, and license, then writes `artifacts/container/serving_image_manifest.json` from `docker image inspect` so the image can be traced back to Git and release evidence. `GET /ready` returns HTTP 503 until a model artifact is loaded, so orchestrators can distinguish a live container from one that is ready for prediction traffic. When ready, the endpoint returns a minimal artifact identity with schema version, dataset, subset, model name, artifact ID, artifact SHA-256, and stage; full metadata remains behind authenticated `GET /version`.

Run the image with an explicit model mount and environment variable when serving predictions:

```powershell
docker run --rm -p 8000:8000 `
  -v ${PWD}/artifacts/models:/models `
  -e AEROSPACE_PROGNOSTICS_MODEL_PATH=/models/cmapss_fd001_hgb_policy.joblib `
  -e AEROSPACE_PROGNOSTICS_MODEL_SHA256=<expected-artifact-sha256> `
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

`GET /schema` returns the loaded artifact's concrete inference contract: required telemetry columns, row limits, grouping behavior, prediction fields, output RUL bounds, monitoring block names, artifact ID, and artifact SHA-256. This makes the deployed model contract discoverable by clients and smoke-test jobs without exposing the binary artifact.

Every API request receives `x-request-id` and `x-process-time-ms` response headers. If the caller sends `x-request-id`, the service preserves it; otherwise it generates one. Requests are logged as one-line JSON records with method, route, status code, request ID, and latency. `GET /metrics` exposes lightweight Prometheus-style counters and gauges for local container smoke checks and basic deployment monitoring, including request totals, response totals by route/status, latency summaries, prediction volume, prediction RUL distribution, telemetry drift alert requests, drift alert columns, and maximum observed standardized mean shift.

Prediction responses include a `monitoring` block. It compares request telemetry means against train-fit artifact reference statistics with standardized mean-shift scores, lists columns above the drift threshold, and summarizes the request's prediction distribution. The same compact monitoring summary is emitted as structured JSON logs for downstream alerting.

Set `AEROSPACE_PROGNOSTICS_MODEL_SHA256` or pass `--model-sha256` to `serve-api` to require an exact artifact digest at startup. The service hashes the mounted joblib before loading it and fails startup if the digest does not match, which protects promotion pointers and container mounts from silently serving the wrong binary artifact. The same digest is reported from readiness and schema responses so operators can confirm runtime artifact identity after deployment.

## Authentication And Rate Limiting

`GET /health` and `GET /ready` are intentionally unauthenticated so container orchestrators can check liveness and readiness without a secret. `GET /version`, `GET /schema`, `GET /metrics`, and `POST /predict` enforce optional API-key authentication when `AEROSPACE_PROGNOSTICS_API_KEY` is set. Clients may send the key with either `x-api-key: <key>` or `Authorization: Bearer <key>`.

Set `AEROSPACE_PROGNOSTICS_RATE_LIMIT_PER_MINUTE` to a positive integer to enable an in-memory fixed-window rate limit for protected endpoints. Requests are bucketed by authenticated API key when authentication is enabled, otherwise by client host. The default value is `0`, which disables rate limiting for local development and CI smoke checks.

For public deployment, run the API behind a managed gateway, load balancer, or ingress layer that provides TLS termination, secret rotation, centralized rate limiting, request size limits, and audit logs. The built-in controls are a local safety layer and testable serving contract, not a replacement for platform perimeter controls.

## Supply Chain Evidence

`generate-sbom` reads the locked `uv.lock` environment and writes a CycloneDX-style JSON software bill of materials. The CI workflow runs this command after `pip-audit`, so every pushed commit proves both that dependencies are vulnerability-checked and that the dependency inventory remains generatable from the lockfile.

The dependency audit intentionally ignores only `CVE-2025-3000` for `torch` while `pip-audit` reports no fixed release. Fixable findings should be resolved in `uv.lock`; for example, the current lockfile pins `pip` to `26.1.2` after the audit reported a fixed version. The serving image now uses the lighter default runtime dependency set and does not install `torch`; keep the explicit audit ignore only for the development and Phase 2 deep-learning environment until PyTorch publishes a patched package that is compatible with the project.

CI also runs `scripts/ci_release_evidence_smoke.py`. The script uses tiny fixture telemetry to build a candidate package and then exercises prediction, validation, benchmark, SBOM, promotion-report, dashboard payload, dashboard HTML, release-bundle, and provenance CLIs end to end. This is a plumbing smoke test rather than production model evidence, but it protects the release-gate workflow from silently breaking.

After the Docker image build, CI runs two serving smoke checks. The first starts the image without a model and confirms liveness plus not-ready behavior. The second mounts the tiny CI artifact into the container, enables API-key authentication, checks readiness and schema discovery, posts a prediction request, and confirms metrics are exposed through the authenticated path.

CI writes a serving-image manifest before container smoke checks. The manifest records image ID, repo tags, selected OCI labels, Docker healthcheck settings, whether `torch` was present in the runtime image, and validation booleans for required labels, healthcheck presence, revision matching, and dependency-surface expectations. CI then builds `artifacts/release/cmapss_fd001_release_bundle.json`, which links the model promotion evidence, dashboard payload, static dashboard HTML, and exact serving-image manifest, and `artifacts/release/cmapss_fd001_provenance.json`, which records source and workflow provenance for that release bundle. The workflow uploads a `ci-fd001-release-evidence` artifact containing reviewable JSON, Markdown, dashboard HTML, SBOM, and container manifest evidence, while leaving raw telemetry and the generated joblib model binary out of the upload.

## Container Publishing

The `Publish Container` workflow builds and pushes the serving image to GitHub Container Registry only for deliberate release events: `v*` Git tags or a manually dispatched workflow, optionally with an explicit image tag. Normal pushes to `main` validate the image in CI but do not publish it. Published image references use the private package path `ghcr.io/<owner>/<repo>/serving`, which inherits repository/package access controls while the repository remains private.

Each publish run tags the image with an immutable `sha-<12-char-git-sha>` reference and, when present, a human release tag such as `v0.1.0-rc1`. The workflow recomputes the serving-image manifest against the exact image reference that will be pushed, checks that training-only `torch` is absent, and logs in to GHCR with the scoped GitHub Actions token.

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
5. Generate the SBOM and compose a promotion report that passes validation, latency, model-card, and supply-chain gates.
6. Run local or CI smoke checks against the serving container with the candidate mounted through `AEROSPACE_PROGNOSTICS_MODEL_PATH`.
7. Promote by updating the deployment artifact pointer, secret, mounted path, or model registry entry to the candidate artifact.
8. Keep the previously promoted artifact available until the new model has passed telemetry drift, prediction-distribution, latency, and error-rate checks.

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
- Promotion evidence report command for validation, benchmark, model-card, and SBOM gates.
- FastAPI inference surface with request validation, liveness, readiness, and version endpoints.
- Model-specific inference schema endpoint for client contract discovery.
- Public readiness identity for loaded artifacts without exposing full version metadata.
- Runtime artifact SHA-256 identity in readiness and schema responses.
- Structured JSON request logs, request IDs, latency headers, and scrapeable serving metrics.
- Prometheus-style prediction and telemetry-drift serving metrics.
- Request telemetry drift summaries and prediction-distribution monitoring.
- Promotion metadata with stable artifact IDs and rollback runbook.
- Optional serving startup SHA-256 verification for mounted model artifacts.
- Optional API-key authentication and per-client serving rate limits.
- Lockfile-derived CycloneDX-style SBOM generation in CI.
- CI release-evidence smoke test for the prediction, validation, benchmark, SBOM, promotion-report, dashboard, release-bundle, and provenance path.
- Tests for artifact round-trip and API prediction behavior.
- Dockerfile scaffold for containerized serving.
- Docker liveness healthcheck backed by the serving `/health` endpoint.
- OCI image labels for source, Git revision, build time, version, title, description, and license.
- CI Docker image build check.
- CI serving image healthcheck metadata check.
- CI serving image dependency-surface check that excludes training-only `torch`.
- CI serving image manifest generation from Docker inspect metadata.
- CI release-candidate bundle tying model artifact evidence, SBOM, promotion gates, file digests, and serving image metadata into one reviewable record.
- CI release provenance statement using in-toto statement shape and SLSA provenance predicate metadata.
- CI upload of reviewable release evidence without raw telemetry or model binaries.
- CI container startup smoke test against `GET /health`.
- CI mounted-model container smoke test for authenticated schema, prediction, readiness, and metrics.
- Private GHCR serving-image publishing for intentional release tags.
