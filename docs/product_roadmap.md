# MLOps Portfolio Roadmap

> [!NOTE]
> This file used to describe a product roadmap. The productization track is now
> frozen; this roadmap is retained as a portfolio engineering roadmap.

## Current Direction

The project demonstrates how familiar PHM benchmarks can be wrapped in a
reviewable ML delivery system:

- reproducible research workflows;
- model packaging and promotion evidence;
- FastAPI serving;
- Streamlit review console;
- Docker Compose integration;
- SBOM, provenance, drift summaries, and CI smoke checks.

The goal is hiring signal for ML-engineer/MLOps roles, not a commercial PHM
tool.

## Completed Portfolio Evidence

- C-MAPSS ingestion, baselines, sequence-model experiments, diagnostics, and
  calibration evidence.
- SMAP/MSL anomaly baselines and alert-policy reports.
- FastAPI service with health, readiness, schema discovery, API-key protection,
  metrics, and drift summaries.
- Streamlit console with prediction history, model registry, fleet registry,
  outcome import, operator decisions, and evidence downloads.
- SQLite app state and Docker Compose local stack.
- Read-only hosted demo image and token-gated hosted review URL.
- Release-evidence workflow with model card, validation, benchmark, SBOM,
  promotion report, release bundle, and provenance.
- ESA-ADB source and evaluator-contract intake before benchmark claims.

## Active Roadmap

1. Keep the MLOps envelope stable and easy to inspect.
2. Improve README/screenshots for a short reviewer journey.
3. Keep Phase 3 research evidence-focused: uncertainty, monotonicity,
   calibration, and benchmark protocol correctness.
4. Add lightweight ESA-ADB smoke evidence only after local data extraction is
   available.
5. Avoid resuming product-launch planning.

## Hiring Signal To Protect

The strongest proof is the engineering wrapper around common datasets:

- tests and CI;
- release gates;
- serving contracts;
- container evidence;
- app state and review workflows;
- honest limitations.

Do not reposition this as a commercial operations system.
