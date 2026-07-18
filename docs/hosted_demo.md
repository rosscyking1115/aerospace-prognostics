# Hosted Read-Only Demo

Use `Dockerfile.demo` when you want a private hosted console that does not rely
on local bind mounts. The image bakes in the no-download C-MAPSS fixture,
release evidence, model card, SBOM, provenance, dashboard payload, and seeded
SQLite registry during the container build. At runtime it starts only the
Streamlit console in read-only mode.

```powershell
docker build -f Dockerfile.demo -t aerospace-prognostics-demo:local .
docker run --rm --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=128m -p 8501:8501 aerospace-prognostics-demo:local
```

Then open `http://127.0.0.1:8501`.

## What The Demo Allows

- Browse fleet triage, model registry, release evidence, model card, SBOM, and
  provenance records.
- Inspect seeded prediction history and deployment gate evidence.
- Download stored prediction rows, outcome templates, run evidence JSON,
  model-review bundles, and fleet registry JSON/CSV without mutating the
  seeded database.
- Verify the console health endpoint at `/_stcore/health`.

## What Read-Only Mode Blocks

- Telemetry upload and prediction persistence.
- Outcome imports.
- Operator decisions.
- File-writing run-evidence exports.
- Automatic database seeding from the Streamlit process.

This keeps a hosted review or pilot demo inspectable without turning public
visitors into database writers.

## Private Hosting Checklist

1. Keep the GitHub repository private until the demo copy, screenshots, and
   license posture are ready for public release.
2. Build from `Dockerfile.demo`; for Render, use the tracked root
   `render.yaml` blueprint.
3. Set the service port to `8501`.
4. Use `/_stcore/health` as the health check path.
5. Keep `AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true` in the service
   environment, even though the image already sets it by default.
6. Set `AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN` to a strong secret before
   sharing the hosted URL. When this variable is set, the Streamlit console
   shows a token gate before any PHM evidence screens render.
7. Run the container filesystem as read-only and provide a writable tmpfs at
   `/tmp` for framework cache files.
8. Prefer platform authentication or an allowlist for private review links when
   sharing outside the owner account. The app-level token gate is the accepted
   control for the current internal private demo; edge access control is the
   stronger boundary for larger or sensitive reviewer groups.
9. Rebuild the image whenever the quickstart evidence contract changes.

The Render Blueprint path is described by [render.yaml](../render.yaml): a
read-only demo image behind the app-level access-token gate
(`AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN`), with `/_stcore/health` as the
health check.

For the fuller local product stack with both FastAPI and Streamlit, use
`compose.yaml` instead. The demo image is intentionally single-service so it can
run on simple container hosting before the product graduates to a managed
multi-service deployment.

## Public Read-Only Demo On Streamlit Community Cloud

The private Render path above gates the console behind an access token because
it was built while the repository was private. For a public, click-through demo
there is nothing left to gate: read-only mode already blocks every write
(uploads, prediction persistence, operator decisions, seeding), so a public
visitor can inspect the evidence but cannot change it.

Streamlit Community Cloud does not build `Dockerfile.demo`, so the baked
evidence and seeded database are not present at startup. The root
[`streamlit_app.py`](../streamlit_app.py) entry point handles this: it defaults
the console to read-only and, once per process, generates the quickstart
evidence and seeds the console database if they are missing
(`aerospace_prognostics.app.bootstrap.ensure_demo_workspace`). The generation
is the no-download C-MAPSS quickstart and takes a few seconds on a cold start.

To deploy:

1. On [share.streamlit.io](https://share.streamlit.io), create an app from this
   repository on the `main` branch.
2. Set the main file path to `streamlit_app.py`.
3. Leave `AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN` unset so the demo is
   public. Read-only mode is on by default; no other configuration is required.
4. Dependencies install from the committed `pyproject.toml` and `uv.lock`; the
   demo runtime set excludes the training-only PyTorch dependency.

The result is the same read-only console the screenshot in the README shows,
with data provisioned on boot rather than baked into an image.
