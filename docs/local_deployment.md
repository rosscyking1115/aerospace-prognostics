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
records, prediction rows, operator decisions, and audit events in the shared
SQLite database.

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
