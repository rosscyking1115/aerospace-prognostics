# First-Run Guide

Use this guide when opening the project for the first time or handing it to a
reviewer. It gives three paths, ordered from fastest evidence inspection to the
full local product stack.

## Prerequisites

- Python dependency management through `uv`.
- Docker Desktop only for the Compose stack or read-only demo image.
- No NASA or JPL telemetry download is required for the first run.

## Path A: Console-Only Quickstart

Best when you want the fastest local proof of the evidence workflow and
operations console.

```powershell
uv sync --dev
uv run aerospace-prognostics quickstart-cmapss-demo
uv run aerospace-prognostics app-init-db
uv run streamlit run src/aerospace_prognostics/app/streamlit_app.py
```

Open the Streamlit URL printed by the command, usually
`http://localhost:8501`.

This path creates fixture-based deployment evidence under
`artifacts/quickstart_cmapss` and a seeded SQLite app database under
`artifacts/app`. The console can show fleet triage, prediction history, release
evidence, model registry records, and read/write operator workflows.

## Path B: Local Product Stack

Best when you want the API and console running together like a small product
deployment.

```powershell
uv run aerospace-prognostics quickstart-cmapss-demo
docker compose up --build
```

Open:

- Console: `http://127.0.0.1:8501`
- API readiness: `http://127.0.0.1:8000/ready`
- API schema: `http://127.0.0.1:8000/schema`
- API metrics: `http://127.0.0.1:8000/metrics`

The Compose stack starts a FastAPI inference service, a Streamlit operations
console, and a database seeding service over the same local quickstart evidence.
The default local API key is `local-dev-secret`; override it with
`AEROSPACE_PROGNOSTICS_API_KEY` before starting Compose.

## Path C: Read-Only Demo Image

Best when you want a private review link or hosted demo surface that cannot
mutate app state.

```powershell
docker build -f Dockerfile.demo -t aerospace-prognostics-demo:local .
docker run --rm --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=128m -p 8501:8501 aerospace-prognostics-demo:local
```

Open `http://127.0.0.1:8501`.

The image bakes in the no-download quickstart evidence and starts the console in
read-only mode. Reviewers can inspect fleet triage, model evidence, prediction
history, SBOM, provenance, and downloadable JSON/CSV evidence without writing to
the seeded database.

## What To Check First

After any path starts successfully, check these surfaces:

- System tab: API and database status.
- Fleet tab: priority-ranked turbofan assets with RUL interval evidence.
- Registry tab: model artifact identity, release gates, and evidence counts.
- Evidence tab: model card, promotion report, release bundle, SBOM, and
  provenance.
- History tab: persisted prediction runs, outcomes, operator decisions, and
  downloadable run evidence.

## Generated Files

The first-run paths write generated evidence and app state under `artifacts/`.
These files are intentionally ignored by Git. Keep raw telemetry under `data/`
or another documented local path, and do not commit generated model binaries,
SQLite databases, or downloaded datasets.

## Troubleshooting

- If the console says quickstart evidence is missing, rerun
  `uv run aerospace-prognostics quickstart-cmapss-demo`.
- If Compose cannot connect to Docker, start Docker Desktop and wait for the
  Linux engine to become ready.
- If `GET /ready` returns unavailable, confirm
  `artifacts/quickstart_cmapss/models/fd001.joblib` exists and restart the
  stack.
- If port `8501` or `8000` is already in use, set
  `AEROSPACE_PROGNOSTICS_CONSOLE_PORT` or `AEROSPACE_PROGNOSTICS_API_PORT`
  before running Compose.

For deeper commands, see [Command Catalog](command_catalog.md). For deployment
details, see [Local Product Deployment](local_deployment.md) and
[Hosted Read-Only Demo](hosted_demo.md).
