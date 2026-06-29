# Local Product Deployment

The local deployment stack runs the same package in two roles:

- FastAPI inference service at `http://127.0.0.1:8000`.
- Streamlit operations console at `http://127.0.0.1:8501`.

It expects the no-download quickstart artifacts to exist before containers start.

```powershell
uv run aerospace-prognostics quickstart-cmapss-demo
docker compose up --build
```

The `app-db` service initializes and seeds
`artifacts/app/aerospace_prognostics.sqlite` from the quickstart evidence bundle.
The `api` service mounts the quickstart model artifact from
`artifacts/quickstart_cmapss/models/fd001.joblib`. The `console` service mounts
the full `artifacts/` tree so prediction history, release evidence, and SQLite
state are shared with the host. Inside Compose, the console probes the API at
`http://api:8000` and surfaces health/readiness in the System tab.
The Predict tab can score telemetry through the API service when it is ready, or
fall back to direct local-artifact inference. The History tab persists run
records, prediction rows, optional observed RUL outcomes, operator decisions,
audit events, interval availability and width diagnostics, and outcome-backed
coverage/MAE diagnostics in the shared SQLite database. It can filter runs by
model, artifact, asset, risk, operator decision, date bounds, and drift-alert
presence. The Registry tab reads the same database to show stored model
artifacts, release evidence, release-gate report cards, recent prediction
usage, operational interval availability, and observed-outcome calibration
summaries for each artifact.
The Fleet tab combines the release dashboard payload with a persisted asset
registry derived from stored prediction runs, so operators can track latest
RUL, interval bounds, risk level, status, source run, and attention reasons by
asset. Registry rows include a computed priority score, priority band, and
priority reasons so turbofan engines and spacecraft anomaly channels can be
reviewed in one ordered queue.

## Endpoints

- Console: `http://127.0.0.1:8501`
- API health: `http://127.0.0.1:8000/health`
- API readiness: `http://127.0.0.1:8000/ready`
- API schema: `http://127.0.0.1:8000/schema`
- API metrics: `http://127.0.0.1:8000/metrics`

Protected API routes use an API key. The default local key is
`local-dev-secret`; override it with `AEROSPACE_PROGNOSTICS_API_KEY`.
The console's API target can be overridden with
`AEROSPACE_PROGNOSTICS_API_BASE_URL` when running outside Compose.
The console reads the same API key environment variable and sends it on
API-backed prediction requests.

## Read-Only Console Mode

Set `AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true` for hosted demos or review
environments where users should inspect evidence without mutating app state.
Read-only mode disables telemetry upload, prediction persistence, outcome
imports, operator decisions, file-writing run-evidence exports, and automatic
quickstart database seeding. Existing dashboards, registry records, evidence,
history, and in-memory downloads remain visible, including prediction rows,
outcome templates, model-review bundles, and evidence JSON for stored runs.

```powershell
$env:AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY="true"
docker compose up --build
```

## Custom Artifact Registration

Generated model artifacts and release evidence can be registered without using
the quickstart seeding path. The model artifact and inspection JSON are
required; release bundle, provenance, promotion, and dashboard payload JSON
files are optional and appear in the Registry and Evidence views when present.
The Registry tab can download a JSON review bundle for the selected artifact
without writing files, so hosted reviewers can inspect artifact identity,
release evidence, report-card diagnostics, and recent prediction usage from one
portable document.

```powershell
uv run aerospace-prognostics app-register-artifact `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --model-artifact artifacts/models/cmapss_fd001_hgb_policy.joblib `
  --inspection-json artifacts/models/cmapss_fd001_hgb_policy_inspection.json `
  --release-bundle-json artifacts/release/cmapss_fd001_release_bundle.json `
  --provenance-json artifacts/release/cmapss_fd001_provenance.json `
  --promotion-json artifacts/models/cmapss_fd001_hgb_policy_promotion.json `
  --dashboard-payload-json artifacts/dashboard/fleet_payload.json
```

## Fleet Asset Registry

Stored prediction runs automatically refresh the `fleet_assets` table. The
registry treats C-MAPSS units as turbofan RUL assets, and ranked SMAP/MSL
anomaly comparison reports or operational anomaly event CSVs can be synced as
spacecraft channel assets through the same `domain`, `asset_type`, and
structured metadata fields.
The Fleet tab can download the current registry as CSV or JSON without writing
files, which keeps hosted read-only reviews useful. The registry view and
exports can be filtered by risk level, domain, status, or attention-required
assets. Exports include priority score, priority band, and priority reasons for
review handoff. Registry JSON also includes a priority-policy summary with band
counts, reason counts, review-queue count, and top prioritized assets. In
writable local mode, the Fleet tab can also upload a spacecraft anomaly event
CSV, download the expected event schema template, and preview event counts,
channel counts, active alerts, and threshold crossings before refreshing
spacecraft channel assets without leaving the console.

Use the sync command to backfill or refresh assets from existing runs:

```powershell
uv run aerospace-prognostics app-sync-fleet-assets `
  --database artifacts/app/aerospace_prognostics.sqlite
```

To refresh only one run:

```powershell
uv run aerospace-prognostics app-sync-fleet-assets `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --run-id run-...
```

To add spacecraft anomaly channels from the ranked Track B comparison output:

```powershell
uv run aerospace-prognostics app-sync-anomaly-assets `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --comparison-csv artifacts/results/smap_msl_anomaly_model_comparison.csv `
  --source-name phase2_smap_msl
```

To refresh spacecraft channel assets from operational anomaly events, provide a
CSV with `channel_id` and `spacecraft` plus optional `event_time_utc`,
`severity`, `active`, `anomaly_score`, `threshold`, `model_name`, `source`, and
`note` columns:

```powershell
uv run aerospace-prognostics app-sync-anomaly-events `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --events-csv artifacts/ops/spacecraft_anomaly_events.csv `
  --source-name ops_stream
```

For local review handoff, export both JSON evidence and flat CSV rows:

```powershell
uv run aerospace-prognostics app-export-fleet-assets `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --output-dir artifacts/app_exports
```

For a narrower review set:

```powershell
uv run aerospace-prognostics app-export-fleet-assets `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --output-dir artifacts/app_exports `
  --risk-level critical `
  --attention-only
```

To produce a priority-policy validation artifact for review or release
evidence, export the JSON report and Markdown summary:

```powershell
uv run aerospace-prognostics app-export-priority-policy `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --output-dir artifacts/app_exports
```

For CI or release checks, make the command fail when any priority-policy
scenario check fails:

```powershell
uv run aerospace-prognostics app-export-priority-policy `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --output-dir artifacts/app_exports `
  --fail-on-policy-fail
```

## Outcome Imports

Observed RUL outcomes can be attached through the History tab or imported from
the CLI when a run has later ground truth. The History tab can download a
fillable outcome template for the selected run, and the CLI can export the same
template for handoff or offline review. Fill the `actual_rul` values and keep
the `unit_number` values aligned with units already present in the prediction
run.

```powershell
uv run aerospace-prognostics app-export-outcome-template `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --run-id run-... `
  --output-csv artifacts/outcomes/fd001_outcomes_template.csv
```

```powershell
uv run aerospace-prognostics app-record-outcomes `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --run-id run-... `
  --outcomes-csv artifacts/outcomes/fd001_outcomes.csv `
  --source-name fd001_outcomes.csv `
  --actor reliability-engineer
```

After import, the History and Registry tabs report outcome availability, MAE,
signed error, and interval coverage against observed RUL.

## Operator Decisions

Operator decisions can be appended from the History tab or CLI when a prediction
run needs review, acceptance, watch-listing, escalation, or rejection. The CLI
path is useful for incident workflows and automation because it records the
same auditable event stream as the console.

```powershell
uv run aerospace-prognostics app-record-decision `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --run-id run-... `
  --status escalated `
  --actor flight-ops `
  --note "Escalate to reliability engineering" `
  --ticket PHM-99 `
  --severity high
```

## Run Evidence Exports

Persisted prediction runs can be downloaded as JSON evidence from the History
tab without writing to the database or filesystem, which keeps hosted read-only
reviews useful. In writable local mode, they can also be exported from the
History tab or from the CLI for incident follow-up or handoff to another
engineer. The file export writes a JSON evidence document with the run metadata,
prediction rows, outcomes, and audit events, plus a separate prediction-row CSV
for spreadsheet inspection.

```powershell
uv run aerospace-prognostics app-export-run `
  --database artifacts/app/aerospace_prognostics.sqlite `
  --run-id run-... `
  --output-dir artifacts/app_exports
```

The command prints SHA-256 digests for both generated files so they can be
attached to tickets or release notes without losing traceability.

```powershell
$env:AEROSPACE_PROGNOSTICS_API_KEY="replace-this-local-key"
docker compose up --build
```

Ports can also be changed without editing `compose.yaml`:

```powershell
$env:AEROSPACE_PROGNOSTICS_API_PORT="8010"
$env:AEROSPACE_PROGNOSTICS_CONSOLE_PORT="8510"
docker compose up --build
```

## Smoke Checks

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/ready
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501/_stcore/health
```

The readiness payload should report the mounted C-MAPSS FD001 model. If the API
container exits during startup, regenerate the quickstart artifacts and confirm
that `artifacts/quickstart_cmapss/models/fd001.joblib` exists.

If Docker reports that it cannot connect to
`dockerDesktopLinuxEngine`, start Docker Desktop and wait for the engine to be
ready before rerunning `docker compose up --build`.

## Shutdown

```powershell
docker compose down
```

Local databases, predictions, and quickstart evidence remain under `artifacts/`.
They are intentionally ignored by Git.
