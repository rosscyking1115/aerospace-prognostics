# Public Proof Assets

These tracked visual assets are a first public-launch proof set for the project.
They are small enough to live in Git and are tied to the no-download quickstart
evidence rather than raw telemetry or generated model binaries.

## Assets

![Read-only Streamlit console](assets/public-proof/streamlit_readonly_console.png)

The read-only console screenshot was captured from the local Streamlit console
with `AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true`. It shows quickstart
evidence loaded, the fleet triage table, critical RUL intervals, and the
read-only sidebar state used by the hosted-demo image.

![Fleet console snapshot](assets/public-proof/fleet_console_snapshot.svg)

The static fleet console snapshot summarizes
`artifacts/quickstart_cmapss/dashboard/fleet_payload.json` and
`artifacts/quickstart_cmapss/models/fd001_benchmark.json`. It shows the product
story we want reviewers and future users to understand quickly: a PHM operations
surface with fleet triage, release gates, artifact identity, and latency
evidence. It stays as a compact fallback asset for surfaces that prefer SVG.

![Quickstart RUL diagnostic](assets/public-proof/quickstart_rul_diagnostic.svg)

The RUL diagnostic summarizes
`artifacts/quickstart_cmapss/predictions/fd001_predictions.json`. Both sample
FD001 engines are critical quickstart cases with predicted RUL of `1.0` and
90 percent intervals from `0.0` to `2.0`, using the
`train_residual_absolute_quantile` interval method.

## Regeneration Path

Use the quickstart command to regenerate the source evidence:

```bash
uv run aerospace-prognostics quickstart-cmapss-demo
```

To refresh the Streamlit screenshot, run the console in read-only mode and
capture the Fleet tab:

```bash
AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true \
uv run streamlit run src/aerospace_prognostics/app/streamlit_app.py \
  --server.port 8503 \
  --server.headless true
```

Then refresh the visual proof assets from the regenerated quickstart payloads and
read-only console before making the repository public.

## Current Limits

- These assets are public proof assets, not certification evidence.
- The Streamlit screenshot is captured from local read-only hosted-demo mode, not
  from a public hosted URL.
- Before public launch, capture a short GIF or fresh screenshot from the actual
  hosted demo URL once that environment exists.
