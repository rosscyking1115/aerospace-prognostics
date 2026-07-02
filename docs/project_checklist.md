# Project Checklist

This is the living execution checklist for Aerospace Prognostics. It turns the
current roadmap into small, reviewable build slices and should be updated after
each completed slice.

## Current North Star

Build a production-grade aerospace Prognostics & Health Management tool:

- an operations console for fleet triage and release review;
- a FastAPI inference service for integration and automation;
- a release-evidence workflow that keeps model artifacts, metrics, SBOM,
  provenance, dashboards, and operational decisions tied together.

The project should be useful as a professional engineering reference and, over
time, as a real aerospace PHM tool surface. It is not a notebook-only lab demo.

## Completed Foundation

- [x] Phase 1 C-MAPSS classical baseline workflow with manifests, EDA, leakage-safe
      preprocessing, validation-selected HGB policy, and tracked result notes.
- [x] Phase 2 C-MAPSS sequence-model workflow with sequence exports, CNN, LSTM,
      TCN, Transformer baselines, comparison reports, diagnostics, and
      calibration checks.
- [x] SMAP/MSL spacecraft anomaly baseline workflow with classical, robust,
      Isolation Forest, and LSTM forecast baselines.
- [x] Production deployment evidence pipeline with model packaging, artifact
      inspection, validation, benchmarking, model cards, SBOM, promotion reports,
      release bundles, and provenance.
- [x] FastAPI serving surface with health, readiness, schema discovery, API-key
      authentication, rate limiting, request metrics, drift summaries, and Docker
      smoke checks.
- [x] Docker Compose local product stack for API, console, mounted quickstart
      artifacts, and shared SQLite app state.
- [x] Hosted read-only demo image path with baked-in quickstart evidence and a
      seeded SQLite registry.
- [x] Streamlit operations console with prediction history, fleet registry, model
      registry, outcome import, operator decisions, audit events, API readiness,
      and evidence downloads.
- [x] Cross-domain fleet registry for C-MAPSS turbofan RUL assets and SMAP/MSL
      spacecraft anomaly channels.
- [x] Fleet priority policy evidence with JSON/Markdown validation and release
      evidence integration.
- [x] Recent app-store helper extractions:
      priority policy, model registry review card, anomaly assets, turbofan
      assets, fleet registry, prediction-run helpers, and prediction-run event
      helpers.
- [x] App database schema/init helpers extracted from `store.py` into a focused
      database module with direct schema validation tests.

## Active Workstream

- [ ] Continue architecture deepening by extracting app store responsibilities
      into focused modules while preserving existing CLI, console, and database
      behavior.

Every active slice should stay small enough to finish with:

- focused tests;
- full lint/tests when code changes;
- code review;
- commit and push;
- GitHub Actions watch to green.

## Next Build Slices

1. [x] Extract app database schema/init helpers from `store.py`.
2. [ ] Extract release evidence row helpers from `store.py`.
3. [ ] Extract prediction-run query/report helpers from `store.py`.
4. [ ] Split Streamlit tabs once store interfaces are stable.
5. [ ] Reshape the public README and move the long command catalog into focused
       docs.
6. [ ] Add visual proof assets: console screenshot/GIF plus a prediction or
       anomaly diagnostic plot.

## Later Milestones

- [ ] Public-facing packaging:
      shorter README, public result summary, architecture diagram, launch-ready
      screenshots, and clear first-run path.
- [ ] Hosted deployment:
      private hosted demo first, then public read-only demo when the repo is
      ready.
- [ ] Phase 3 research differentiators:
      calibrated uncertainty evidence, monotonic degradation diagnostics, and
      constrained losses only after diagnostics exist.
- [ ] ESA-ADB intake:
      official protocol note covering data access, splits, labels, metrics, and
      smallest protocol-correct first run before model implementation.

## Working Rules

- [ ] Keep the GitHub repository private until the public-facing README,
      screenshots, and hosted demo story are ready.
- [ ] Update this checklist after each completed slice.
- [ ] Prefer production-grade behavior and evidence over portfolio-style mimicry.
- [ ] Keep raw telemetry and generated model artifacts out of Git.
