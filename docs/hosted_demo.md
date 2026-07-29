# Read-Only Demo Image

> [!NOTE]
> **There is no hosted instance of this console.** A private Render service ran
> one until July 2026; it was retired rather than repaired after its build broke
> (see §8 of [release-check-2026-07-27.md](release-check-2026-07-27.md)). Nothing
> here is a live URL — this page describes an image you build and run yourself.
> The image itself is built, contract-checked and smoke-tested in CI on every
> push, so the path below is verified even though nothing is deployed.

Use `Dockerfile.demo` when you want a self-contained console that does not rely
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

## If You Do Host It Somewhere

Kept as platform-neutral guidance, not as a description of anything running.
Any container host that builds a Dockerfile will do.

1. Build from `Dockerfile.demo`.
2. Set the service port to `8501`.
3. Use `/_stcore/health` as the health check path.
4. Keep `AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true` in the service
   environment, even though the image already sets it by default.
5. Set `AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN` to a strong secret before
   sharing the URL. When this variable is set, the Streamlit console shows a
   token gate before any PHM evidence screens render.
6. Run the container filesystem as read-only and provide a writable tmpfs at
   `/tmp` for framework cache files. CI runs the image exactly this way, so the
   configuration is verified rather than assumed.
7. Prefer platform authentication or an allowlist over the app-level token gate
   when sharing outside the owner account. The token gate is a reasonable
   control for a small private audience; edge access control is the stronger
   boundary for larger or sensitive reviewer groups.
8. Rebuild the image whenever the quickstart evidence contract changes.
9. **Pin your build inputs.** `Dockerfile.demo` currently uses the floating
   `python:3.12-slim` tag and installs `uv` unpinned, so an identical commit can
   build today and fail tomorrow. That is the failure shape the retired Render
   service showed, and it is unaddressed.

For the fuller local product stack with both FastAPI and Streamlit, use
`compose.yaml` instead. The demo image is intentionally single-service so it can
run on simple container hosting before the product graduates to a managed
multi-service deployment.

## Deploying To Streamlit Community Cloud

Instructions only — no such app is running either.

The token gate described above exists because the console was originally hosted
while the repository was private. For a public, click-through demo there is
nothing left to gate: read-only mode already blocks every write
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
