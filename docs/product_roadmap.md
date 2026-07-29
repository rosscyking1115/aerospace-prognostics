# Engineering Roadmap

> [!NOTE]
> This file previously described a product roadmap. The productization track is
> frozen; what remains is a neutral engineering roadmap for the reference
> pipeline.

## Current direction

The project wraps familiar PHM benchmarks in a reviewable ML delivery system:

- reproducible research workflows;
- model packaging and promotion evidence;
- FastAPI serving;
- Streamlit operations console;
- Docker Compose integration;
- SBOM, provenance, drift summaries, and CI smoke checks.

It is a reference implementation, not a commercial PHM product.

## Completed engineering evidence

- C-MAPSS ingestion, baselines, sequence-model experiments, diagnostics, and
  calibration evidence.
- SMAP/MSL anomaly baselines and alert-policy reports.
- FastAPI service with health, readiness, schema discovery, API-key protection,
  metrics, and drift summaries.
- Streamlit console with prediction history, model registry, fleet registry,
  outcome import, operator decisions, and evidence downloads.
- SQLite app state and Docker Compose local stack.
- Read-only demo image, built and smoke-tested in CI. The token-gated hosted
  console it once ran on was retired in July 2026; no instance is deployed.
- Release-evidence workflow with model card, validation, benchmark, SBOM,
  promotion report, release bundle, and provenance.
- ESA-ADB source and evaluator-contract intake, now extracted to the
  [telemeval](https://github.com/rosscyking1115/telemeval) library.

## Active roadmap

1. Keep the MLOps envelope stable and easy to inspect.
2. Keep the README and screenshots current and honest.
3. Keep Phase 3 research evidence-focused: uncertainty, monotonicity,
   calibration, and benchmark-protocol correctness.
4. Track telemeval upstream (ESA-ADB metric hierarchy) as the anomaly-evaluation
   layer.
5. Do not resume product-launch planning (frozen).

## What to protect

The value on show is the engineering wrapper around common datasets — tests and
CI, release gates, serving contracts, container evidence, app-state and review
workflows, and honest limitations. Do not reposition this as a commercial
operations system.
