# Repository Launch Strategy

This project should stay private until the public version has a clear tool identity, reproducible evidence, and a first-run experience that makes the work easy to trust. The goal is not to look like a production ML system; it is to show a production-grade aerospace PHM workflow that happens to be built as a portfolio project.

## Current Position

The project already has a stronger engineering foundation than most C-MAPSS or spacecraft anomaly portfolio repositories:

- Reproducible C-MAPSS ingestion, manifesting, checksum verification, EDA, classical baselines, and Phase 2 sequence-model workflows.
- SMAP/MSL anomaly-detection plumbing with classical, robust, and LSTM forecasting baselines.
- Tested CLI workflows rather than notebook-only experiments.
- Deployment artifacts: packaged model, model card, validation report, benchmark report, promotion report, SBOM, release bundle, and in-toto/SLSA-style provenance.
- FastAPI serving with liveness, readiness, schema discovery, metrics, API-key protection, rate limiting, drift summaries, Docker image checks, and mounted-model smoke tests.

The main gap before public launch is not code volume. It is packaging the project so a reviewer can understand the value in five minutes and then inspect the depth for an hour.

## Success Patterns From Aerospace Repositories

High-visibility aerospace and space-adjacent repositories tend to win for one of four reasons:

- They solve a concrete operational workflow: autopilot firmware, rocket simulation, aircraft geometry, trajectory analysis, multidisciplinary optimization, or 3D geospatial visualization.
- They provide a crisp first-run path: install, run an example, see a meaningful plot, table, aircraft, map, trajectory, or simulation.
- They create trust through documentation, tests, examples, papers, institutional links, benchmark alignment, and stable APIs.
- They are useful beyond a single experiment: users can apply them to their own vehicles, telemetry, missions, or engineering questions.

Direct PHM, C-MAPSS, and spacecraft anomaly repositories are usually much smaller and often notebook-centered. That is an opening: a serious end-to-end PHM tool can stand out even without chasing huge autopilot-scale star counts.

## Public Identity

Use this positioning for the public repo:

> A production-grade aerospace Prognostics & Health Management reference pipeline: benchmarked turbofan RUL prediction, spacecraft telemetry anomaly detection, calibrated uncertainty, and deployment evidence.

Avoid claiming certification, operational readiness, or state-of-the-art leaderboard performance. The stronger claim is that the project handles the whole lifecycle honestly:

- data integrity;
- benchmark metrics;
- model diagnostics;
- asymmetric cost;
- uncertainty and calibration;
- serving contracts;
- observability;
- supply-chain evidence;
- release provenance;
- deployment limitations.

## Public README Shape

Before making the repo public, reshape the README around the reviewer journey:

1. One-sentence identity and a compact architecture diagram.
2. "Run this first" quickstart with a tiny fixture or prebuilt sample artifact.
3. Current headline results table for C-MAPSS, including Phase 1 HGB and best Phase 2 candidate.
4. Spacecraft anomaly section with SMAP/MSL baseline results and an explicit note that ESA-ADB is the serious benchmark target.
5. Deployment section showing the API contract, Docker run command, health/readiness behavior, and promotion evidence.
6. Evidence section linking model cards, release bundle shape, SBOM, provenance, and CI status.
7. Roadmap with Phase 3 uncertainty, physics-informed constraints, ESA-ADB, and dashboard work.

The README should be shorter than the current development log. Move long command catalogs into docs and keep the front page persuasive, runnable, and honest.

## Demo Assets To Create

The public repo needs visual proof:

- A dashboard screenshot or GIF showing a fleet table, RUL estimates, confidence bands, and anomaly flags.
- A compact C-MAPSS prediction diagnostic plot: actual vs predicted RUL, signed error, and late-error emphasis.
- A telemetry anomaly plot with labelled windows and model score threshold.
- A release-evidence screenshot or markdown excerpt showing validation, benchmark, SBOM, and provenance gates.
- An architecture diagram mapping data ingestion, training, evaluation, packaging, serving, and monitoring.

## Standing-Out Features

Prioritize these features because they differentiate the project from typical C-MAPSS notebooks:

- Calibrated RUL intervals, not only point predictions.
- Monotonic degradation constraints and clear reporting of violations.
- Asymmetric late-failure cost awareness in training, selection, and reporting.
- ESA-ADB benchmark integration with official evaluation rather than ad hoc F1.
- Operator-facing diagnostics: which units, cycles, sensors, or telemetry windows drove concern.
- Promotion evidence and rollback path for model artifacts.
- A small but polished inference contract that can be called from a dashboard or CI smoke test.

## Public Launch Checklist

- [ ] Keep the repository private until raw data, generated artifacts, and secrets are confirmed absent.
- [ ] Add a license decision before launch.
- [ ] Replace the development-heavy README with a public-facing README and move command catalogs into docs.
- [ ] Add architecture diagram and at least two visual result assets.
- [ ] Add a tiny fixture-based quickstart that runs without downloading NASA/JPL data.
- [x] Publish one release candidate bundle from CI artifacts and document how to inspect it.
- [ ] Add a dashboard or recorded dashboard demo.
- [ ] Finish Phase 3 uncertainty and physics-informed reporting before making strong public claims.
- [ ] Add a technical report PDF or long-form markdown report.
- [ ] Prepare a launch article that explains the engineering and domain choices without overselling the model.

## Near-Term Actions

The next few project steps should serve both the roadmap and the public-launch story:

1. Build the dashboard skeleton against fixture/sample outputs so the user experience exists before final Phase 3 results.
2. Add uncertainty intervals for the deployable C-MAPSS artifact or the leading Phase 2 model.
3. Turn the current README command list into focused docs pages.
4. Add a `docs/public_results.md` page with the current benchmark tables and honest limitations.
5. Start ESA-ADB integration research before implementing it, because benchmark protocol correctness matters more than model novelty.
