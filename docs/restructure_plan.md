# Restructure And Replan

This document resets the project from "keep adding capability" to "shape the
work into a maintainable product-grade PHM tool." The repository is now past
the point where a single linear project plan is enough: it has research
workflows, deployment evidence, an API, a console, CI supply-chain checks, and a
hosted demo image. The next work should deepen the architecture and sharpen the
public product story before adding more modelling scope.

The living execution checklist is tracked in
[docs/project_checklist.md](project_checklist.md). Update it after each completed
slice so the roadmap stays connected to shipped work.

## Current State

The project already has these working surfaces:

- C-MAPSS Phase 1 classical baselines, manifests, EDA, and reports.
- C-MAPSS Phase 2 sequence-model workflows and comparison diagnostics.
- SMAP/MSL anomaly-baseline workflows with robust, classical, and LSTM forecast
  methods.
- Deployment track with packaged model artifacts, validation, benchmarking,
  model cards, SBOM, promotion reports, release bundles, and provenance.
- FastAPI serving with health, readiness, schema, metrics, API-key protection,
  rate limiting, drift summaries, and Docker smoke tests.
- Streamlit operations console with SQLite-backed model registry, prediction
  history, audit events, release evidence, read-only mode, and hosted demo image.

That is a strong foundation. The issue is not missing capability; the issue is
that the project shape now needs to catch up with the capability.

## Architecture Friction

Measured hotspots:

- `src/aerospace_prognostics/cli.py`: 3,306 lines and 58 subcommands.
- `src/aerospace_prognostics/experiments/cmapss_deep_baseline.py`: 2,158 lines.
- `src/aerospace_prognostics/deployment/artifacts.py`: 1,550 lines.
- `src/aerospace_prognostics/app/store.py`: 1,462 lines.
- `src/aerospace_prognostics/app/streamlit_app.py`: 951 lines.

The first restructuring goal is not to make files small for its own sake. The
goal is to make the project easier to understand, test, and extend by putting
stable public interfaces around deeper modules.

## Product Direction

The public identity should stay:

> A production-grade aerospace Prognostics & Health Management reference
> pipeline: benchmarked turbofan RUL prediction, spacecraft telemetry anomaly
> detection, calibrated uncertainty, and deployment evidence.

The product-facing tool should be framed as an operations console and release
evidence workflow, not only as a modelling lab. That means the next work should
prioritize:

- a crisp first-run path;
- a shorter public README;
- visual proof through screenshots, diagrams, and result assets;
- clear separation between research commands, deployment commands, and product
  console concerns;
- Phase 3 differentiators: calibrated uncertainty, monotonic degradation, and
  ESA-ADB protocol correctness.

## Replanned Workstreams

### 1. Architecture Deepening

Objective: reduce friction without changing behavior.

Immediate candidates:

- Split CLI registration and command handlers into domain command modules:
  `cli_app`, `cli_cmapss`, `cli_smap_msl`, `cli_deployment`, and `cli_workflows`.
- Split app persistence into focused modules:
  schema/init, model registry, prediction history, audit/events, and summaries.
- Split the Streamlit app into tab renderers once the backing store interfaces
  are stable.
- Keep current command names and tests intact during each move.

Done when:

- existing CLI commands still pass unchanged;
- `cli.py` becomes a thin parser/dispatcher;
- no product behavior changes are hidden inside the restructuring.

### 2. Public-Facing Packaging

Objective: make the project understandable in five minutes.

Immediate candidates:

- Replace the development-heavy README front page with a public-facing overview.
- Move long command catalogs into focused docs pages.
- Add `docs/public_results.md` for current benchmark tables, limitations, and
  honest interpretation.
- Add an architecture diagram covering ingestion, training, evaluation,
  packaging, serving, monitoring, and demo surfaces.
- Add at least two visual assets: console screenshot/GIF and a prediction or
  anomaly diagnostic plot.

Done when:

- README answers "what is this, why trust it, how do I run it" quickly;
- detailed commands still exist in docs;
- public-launch checklist items become directly reviewable.

### 3. Product Demo Maturity

Objective: turn the console into a credible tool surface.

Immediate candidates:

- Add a fleet asset registry that can eventually combine C-MAPSS assets and
  spacecraft anomaly channels.
- Add search/filter/sort workflows around model artifacts and prediction runs.
- Add downloadable release-review bundles from the console in read-only-safe
  form where possible.
- Add screenshots and smoke checks that prove the visual demo path works.

Done when:

- the demo feels like a tool a reviewer can inspect, not a notebook replacement;
- hosted read-only deployment remains write-safe and CI-proven.

### 4. Phase 3 Research Differentiator

Objective: add the parts that make the project intellectually distinctive.

Immediate candidates:

- Add calibrated intervals for the deployable C-MAPSS artifact or leading Phase
  2 candidate.
- Report empirical coverage, interval width, and failure modes.
- Add monotonic degradation violation diagnostics before adding new losses.
- Add one constrained training loss only after diagnostics exist.

Done when:

- uncertainty and monotonicity are reported with evidence, not just claimed;
- results are tied back to asymmetric late-failure cost.

### 5. ESA-ADB Intake Before Implementation

Objective: avoid implementing the wrong benchmark protocol.

Immediate candidates:

- Create an ESA-ADB intake note from official repo/docs/paper.
- Identify data access, size, splits, labels, official metrics, and expected
  output format.
- Decide the smallest protocol-correct first run before writing model code.

Done when:

- we can state exactly how a result will be evaluated before training anything.

## Next Execution Order

1. Start with CLI architecture deepening because it is high-friction and low
   product-risk if done test-first.
2. Then split app store reads/writes because the console is now central to the
   tool identity.
3. Then reshape README and docs around the public-review journey.
4. Then add Phase 3 uncertainty/monotonicity work.
5. Then do ESA-ADB intake and protocol implementation.

## First Implementation Slice

Refactor one small CLI domain at a time.

Recommended first slice:

1. Create a command module for app/database commands.
2. Move parser registration and handlers for:
   - `app-init-db`
   - `app-register-artifact`
   - `app-record-outcomes`
   - `app-export-run`
3. Keep the root `aerospace-prognostics` CLI behavior unchanged.
4. Run only the app CLI tests first, then the full suite.

This is a good first slice because it exercises the new command-module pattern
without touching the heavier C-MAPSS or Phase 2 command families.
