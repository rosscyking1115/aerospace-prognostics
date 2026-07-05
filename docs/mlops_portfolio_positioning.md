# MLOps Portfolio Positioning

This project is an ML-engineer/MLOps portfolio reference implementation. It is
not a product launch plan.

## Hiring-Manager Impression

The strongest signal is not that the project trains on C-MAPSS. C-MAPSS is one
of the most common predictive-maintenance benchmarks. The strongest signal is
that a familiar benchmark is wrapped in a serious ML delivery envelope:

- reproducible data and artifact workflows;
- tested CLIs instead of notebook-only experiments;
- model packaging and validation gates;
- FastAPI inference with health, readiness, auth, metrics, and drift summaries;
- Streamlit review console with prediction history, registry views, and
  downloadable evidence;
- Docker Compose and read-only demo image paths;
- SBOM, provenance, model card, promotion report, and release bundle evidence;
- CI that checks lint, tests, dependency audit, SBOM, image contracts, and smoke
  paths.

For ML-engineer/MLOps roles, this is the right edge: turn modelling work into a
reviewable, testable, deployable system.

## Strongest Proof To Surface

1. **Release evidence workflow**
   - Model card, benchmark report, validation report, SBOM, provenance, release
     bundle, and promotion report.
   - This shows lifecycle discipline beyond training metrics.

2. **Serving and observability**
   - FastAPI health/readiness/version/schema endpoints, API-key protection,
     request metrics, drift summaries, and Docker health checks.
   - This maps directly to ML platform and applied ML engineering work.

3. **Review console**
   - Streamlit console for prediction history, model registry, fleet registry,
     outcome imports, operator decisions, and evidence downloads.
   - This shows practical thinking about model review and operational handoff.

4. **Testing and CI**
   - Hundreds of tests covering data loading, feature generation, baselines,
     model workflows, API behavior, app state, evidence generation, Docker image
     contracts, and hosted-demo paths.
   - This is a sharper hiring signal than one more benchmark score.

5. **Honest benchmark framing**
   - C-MAPSS and SMAP/MSL are used as reproducible public datasets.
   - The README should say the model is not the novelty; the MLOps wrapper is.

## Weakest Proof To Avoid Over-Selling

- Do not claim operational aerospace readiness.
- Do not claim product-market readiness.
- Do not present C-MAPSS scores as novel research.
- Do not describe the hosted demo as a commercial deployment.
- Do not resume productization planning as the next phase.

## What To Show First

For a hiring reviewer, lead with this order:

1. README screenshot and architecture diagram.
2. `Run This First` no-download quickstart.
3. CI badge and test count.
4. Release evidence and SBOM/provenance docs.
5. FastAPI serving docs and Docker smoke evidence.
6. Model and anomaly benchmark results with limitations.
7. ESA-ADB protocol intake as evidence of benchmark discipline, not a rushed
   result claim.

## Next Portfolio Work

- Keep the productization track frozen.
- Polish the README and screenshots for a five-minute reviewer path.
- Add a short technical report explaining the MLOps lifecycle.
- Keep Phase 3 research focused on evidence quality: uncertainty, calibration,
  monotonicity, and ESA-ADB protocol correctness.
- Keep raw datasets and generated artifacts out of Git.
