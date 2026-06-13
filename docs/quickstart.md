# No-Download Quickstart

Run this first when you want to see the deployment track work without downloading NASA
C-MAPSS or JPL SMAP/MSL data.

```powershell
uv sync --dev
uv run aerospace-prognostics quickstart-cmapss-demo
uv run streamlit run src/aerospace_prognostics/app/streamlit_app.py
```

The command writes a tiny C-MAPSS-compatible fixture under `artifacts/quickstart_cmapss`
and runs the same public CLI path used by CI:

- package a validation-selected HGB policy artifact;
- write metadata and a model card;
- inspect the packaged artifact contract;
- predict from fixture telemetry;
- validate and benchmark the artifact;
- generate an SBOM;
- build promotion evidence;
- render the standalone fleet dashboard HTML;
- build a release bundle;
- generate in-toto/SLSA-style provenance.

Key outputs:

- `artifacts/quickstart_cmapss/dashboard/fleet_dashboard.html`
- `artifacts/quickstart_cmapss/release/fd001_release_bundle.md`
- `artifacts/quickstart_cmapss/release/fd001_provenance.md`
- `artifacts/quickstart_cmapss/models/fd001_inspection.json`
- `artifacts/quickstart_cmapss/models/fd001_model_card.md`

The Streamlit app opens a local operations console over the same evidence bundle.
It shows the fleet triage table, artifact contract, release evidence, and a
batch-prediction panel for C-MAPSS telemetry CSVs.

The prediction and dashboard payloads include train-residual RUL interval bounds, so
the quickstart exercises the same uncertainty-aware triage contract used by the
serving API.

This quickstart is a plumbing demo, not model-quality evidence. The fixture is intentionally
tiny so reviewers can verify the lifecycle quickly before moving to the real benchmark data.

The legacy script entry point remains available:

```powershell
uv run python scripts/quickstart_cmapss_demo.py
```
