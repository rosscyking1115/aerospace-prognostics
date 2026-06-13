# Product Roadmap

This project is intended to become a usable aerospace PHM tool, not only a lab
demonstration. The research pipeline remains the modelling foundation, but the
product direction is an operations console that helps teams inspect model
artifacts, run fleet predictions, review uncertainty, and keep deployment
evidence attached to every model release.

## Product Shape

The professional user-facing tool should support three surfaces:

- Local operations console for engineers and reviewers.
- API service for integration with other tools and automation.
- Hosted web deployment for demos, pilots, and portfolio presentation.

The current foundation already includes the API, Docker image, release evidence,
SBOM, artifact inspection, static dashboard output, and no-download quickstart.
The missing layer is persistent application state and a polished interactive UI.

## Milestones

1. Interactive local console
   - Streamlit app over quickstart and release artifacts.
   - Fleet table with RUL intervals, risk level, attention reasons, and evidence.
   - Telemetry CSV upload and batch prediction against a packaged artifact.
   - Screenshot/GIF for the README.

2. Local persistence
   - SQLite database for uploaded telemetry, prediction runs, model artifacts,
     dashboard payloads, release evidence, and audit events.
   - `app-init-db` command so generated quickstart artifacts can seed the app database.
   - Migration path designed so SQLite can become Postgres without changing the
     product data model.

Current status: initial SQLite persistence is active for model artifacts, release
evidence, telemetry uploads, prediction runs, and per-asset predictions. Audit
events and import/export beyond quickstart seeding remain future work.

3. Local deployment stack
   - Docker Compose with FastAPI inference service, dashboard, mounted model
     artifact storage, and database volume.
   - Seeded quickstart data for a one-command local product demo.
   - Health checks for API, dashboard, and database readiness.

4. Hosted deployment
   - Private hosted demo first, then public read-only demo when the repo is ready.
   - API-key or OAuth-protected write paths for uploads and inference.
   - Release artifacts and model binaries kept out of Git and mounted or fetched
     from controlled storage.

5. Professional workflow
   - Model registry view with artifact identity, schema version, promotion status,
     SBOM, provenance, model card, and rollback notes.
   - Prediction-run history with input hash, model ID, latency, risk counts, and
     operator notes.
   - Fleet asset registry that can combine C-MAPSS engine RUL and spacecraft
     anomaly alerts in one console.

## Near-Term Decision

Use Streamlit for the first app because it gets a usable professional console in
front of us quickly. Once the domain flow is right, either keep Streamlit for the
internal tool or move the same API/database contract behind a Next.js frontend
for a more polished public product surface.
