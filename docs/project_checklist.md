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

## Current Status

The launch package is internally ready for the next research phase while the
repository remains private:

- Render hosts the read-only Streamlit demo at
  <https://aerospace-prognostics-private-demo.onrender.com>.
- The hosted health endpoint has returned `200 ok`, and the hosted proof
  screenshot is tracked under `docs/assets/public-proof/`.
- `AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN` is configured on Render, so the
  PHM console is app-level token gated before evidence screens render.
- The repository stays private and `UNLICENSED` until the public launch license,
  launch copy, and final public demo posture are chosen.
- Stronger edge access, such as Cloudflare Access or an IP allowlist, is a
  later hardening step before broader external sharing, not the blocker for the
  current internal private-demo milestone.

## Completed Foundation

- [x] Phase 1 C-MAPSS classical baseline workflow with manifests, EDA, leakage-safe
      preprocessing, validation-selected HGB policy, and tracked result notes.
- [x] Phase 2 C-MAPSS sequence-model workflow with sequence exports, CNN, LSTM,
      TCN, Transformer baselines, comparison reports, diagnostics, and
      calibration checks.
- [x] SMAP/MSL spacecraft anomaly baseline workflow with classical, robust,
      Isolation Forest, and LSTM forecast baselines.
- [x] Combined Phase 2 completion audit for C-MAPSS and SMAP/MSL run manifests,
      with JSON/Markdown evidence that both track bundles verify together.
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
- [x] Release evidence row helpers extracted from `store.py` into a focused
      release-evidence module with direct row-building tests.
- [x] Prediction-run query/report shaping helpers extracted from `store.py` into
      the prediction-run helper module with direct contract tests.
- [x] Streamlit System and Roadmap tab renderers extracted from
      `streamlit_app.py` into a focused tab module with renderer contract tests.
- [x] Streamlit Evidence tab renderer extracted into the focused tab module with
      release-evidence renderer contract tests.
- [x] Streamlit Registry tab renderer extracted into the focused tab module with
      model-registry renderer contract tests.
- [x] Streamlit History tab renderer extracted into the focused tab module with
      prediction-run history renderer contract tests.
- [x] Streamlit Predict tab renderer extracted into the focused tab module with
      batch-prediction renderer contract tests.
- [x] Streamlit Fleet tab renderer extracted into the focused tab module with
      fleet registry and priority-policy renderer contract tests.
- [x] Public-facing README reshape completed, with detailed research,
      deployment, and app commands moved into `docs/command_catalog.md`.
- [x] Initial public proof assets added with a tracked console snapshot,
      quickstart RUL diagnostic plot, and evidence-source documentation.
- [x] README visual proof replaced with a real read-only Streamlit console
      screenshot captured from local hosted-demo mode.
- [x] First-run guide added with console-only, local Compose stack, and
      read-only demo image paths for launch packaging.
- [x] Architecture guide added with README diagram, system boundaries,
      evidence flow, runtime modes, and security controls.
- [x] Public results summary added with C-MAPSS, SMAP/MSL, deployment-evidence,
      and limitation notes.
- [x] Pre-Phase-3 readiness audit added to separate repo-local completion gates
      from external launch blockers before new research work starts.
- [x] Private hosting handoff added with a Render Blueprint path, access-control
      warning, and final hosted-proof readiness command.
- [x] Render-hosted read-only demo URL verified with a live health check and a
      fresh proof screenshot from the hosted environment.
- [x] Optional Streamlit app-level access token gate added for hosted demos,
      with Render `sync: false` secret configuration.
- [x] Hosted Render demo redeployed with
      `AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN` configured as the current
      private-demo access control.

## Active Workstream

- [ ] Start the next research/build slice from this reconciled plan, with
      public-launch decisions kept separate from internal engineering readiness.

Every active slice should stay small enough to finish with:

- focused tests;
- full lint/tests when code changes;
- code review;
- commit and push;
- GitHub Actions watch to green.

## Next Build Slices

1. [x] Extract app database schema/init helpers from `store.py`.
2. [x] Extract release evidence row helpers from `store.py`.
3. [x] Extract prediction-run query/report helpers from `store.py`.
4. [x] Continue splitting Streamlit tabs once store interfaces are stable.
5. [x] Reshape the public README and move the long command catalog into focused
       docs.
6. [x] Add initial visual proof assets: static console proof snapshot plus a
       prediction diagnostic plot.
7. [x] Replace the static README console proof snapshot with a real read-only
       Streamlit console screenshot after validating the desktop layout.
8. [x] Capture a short GIF or fresh screenshot from the actual hosted demo URL
       once that public environment exists.
9. [x] Add a clear first-run guide that tells reviewers when to use the
       console-only quickstart, local product stack, or read-only demo image.
10. [x] Add a compact README architecture diagram and deeper architecture guide.
11. [x] Add a concise public results summary with honest limitations.
12. [x] Add a combined Phase 2 completion audit command for both manifest-backed
        research tracks.
13. [x] Add a pre-Phase-3 readiness audit command that reports remaining launch
        and productization blockers.
14. [x] Choose the license/posture for public launch or document a private-only
        license decision for the next internal phase.
15. [x] Add a private hosting handoff and tracked Render Blueprint for the
        read-only demo.
16. [x] Create a hosted read-only demo URL and capture fresh visual proof from
        that environment.
17. [x] Add an app-level token gate for the hosted Streamlit console and wire
        the Render Blueprint to request the secret out of band.
18. [x] Configure the hosted demo's app-level access token in Render and
        redeploy the private read-only console.
19. [ ] Optional hardening before broader external sharing: put the hosted demo
        behind edge access control such as Cloudflare Access, Render inbound IP
        rules, or an equivalent allowlist. The default Render service URL
        remains internet-reachable at the network layer.

## Later Milestones

- [ ] Public-facing packaging:
      final public license, launch copy, and public-hosted proof after the
      private demo is ready.
- [ ] Hosted deployment:
      optional edge-gated private review link first, then public read-only demo
      when the repo is ready.
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
