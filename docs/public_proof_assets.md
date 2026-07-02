# Public Proof Assets

These tracked visual assets are a first public-launch proof set for the project.
They are small enough to live in Git and are tied to the no-download quickstart
evidence rather than raw telemetry or generated model binaries.

## Assets

![Fleet console snapshot](assets/public-proof/fleet_console_snapshot.svg)

The fleet console snapshot summarizes
`artifacts/quickstart_cmapss/dashboard/fleet_payload.json` and
`artifacts/quickstart_cmapss/models/fd001_benchmark.json`. It shows the product
story we want reviewers and future users to understand quickly: a PHM operations
surface with fleet triage, release gates, artifact identity, and latency
evidence.

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

Then refresh the visual proof assets from the regenerated quickstart payloads
before making the repository public.

## Current Limits

- These assets are public proof assets, not certification evidence.
- The screenshot-style console asset is a curated static snapshot from quickstart
  evidence; the live Streamlit console remains the professional review surface.
- Before public launch, capture a real hosted console screenshot or short GIF
  from the read-only demo image and replace the static console snapshot.
