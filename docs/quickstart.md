# No-Download Quickstart

Run this first when you want to see the deployment track work without downloading NASA
C-MAPSS or JPL SMAP/MSL data.

```powershell
uv sync --dev
uv run python scripts/quickstart_cmapss_demo.py
```

The script writes a tiny C-MAPSS-compatible fixture under `artifacts/quickstart_cmapss`
and runs the same public CLI path used by CI:

- package a validation-selected HGB policy artifact;
- write metadata and a model card;
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
- `artifacts/quickstart_cmapss/models/fd001_model_card.md`

This quickstart is a plumbing demo, not model-quality evidence. The fixture is intentionally
tiny so reviewers can verify the lifecycle quickly before moving to the real benchmark data.
