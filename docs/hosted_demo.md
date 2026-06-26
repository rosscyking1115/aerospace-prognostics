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
- Download stored prediction rows, outcome templates, run evidence JSON, and
  model-review bundles without mutating the seeded database.
- Verify the console health endpoint at `/_stcore/health`.

## What Read-Only Mode Blocks

- Telemetry upload and prediction persistence.
- Outcome imports.
- Operator decisions.
- File-writing run-evidence exports.
- Automatic database seeding from the Streamlit process.

This keeps a hosted portfolio or pilot demo inspectable without turning public
visitors into database writers.

## Private Hosting Checklist

1. Keep the GitHub repository private until the demo copy, screenshots, and
   license posture are ready for public release.
2. Build from `Dockerfile.demo`.
3. Set the service port to `8501`.
4. Use `/_stcore/health` as the health check path.
5. Keep `AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true` in the service
   environment, even though the image already sets it by default.
6. Run the container filesystem as read-only and provide a writable tmpfs at
   `/tmp` for framework cache files.
7. Add platform authentication or an allowlist for private review links.
8. Rebuild the image whenever the quickstart evidence contract changes.

For the fuller local product stack with both FastAPI and Streamlit, use
`compose.yaml` instead. The demo image is intentionally single-service so it can
run on simple container hosting before the product graduates to a managed
multi-service deployment.
