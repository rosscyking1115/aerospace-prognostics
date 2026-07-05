# Pre-Phase-3 Readiness

> [!WARNING]
> Quarantined historical gate. Do not use this as an active productization or
> launch prerequisite.

This document and its matching CLI audit are retained as historical evidence of
the earlier launch-readiness track. The project has been reframed as an
ML-engineer/MLOps portfolio reference implementation, so Phase 3 should not be
blocked on product-launch readiness.

## Frozen Decision

- Do not execute Phase 3 as a product launch.
- Do not treat hosted demo hardening, launch copy, or public proof packaging as
  prerequisites for research.
- Keep the working MLOps envelope: FastAPI, Streamlit, Docker, CI, SBOM,
  provenance, model cards, drift summaries, and release evidence.
- Use [docs/project_checklist.md](project_checklist.md) and
  [docs/mlops_portfolio_positioning.md](mlops_portfolio_positioning.md) as the
  active planning sources.

## Historical Command

The command still exists for backwards compatibility and old evidence checks:

```powershell
uv run aerospace-prognostics pre-phase3-readiness-audit --output-json artifacts/pre_phase3_readiness.json --output-markdown artifacts/pre_phase3_readiness.md
```

It should be interpreted as a historical audit of the old launch-readiness
posture, not as the next execution gate.

## Current Posture

- Repository code is MIT licensed.
- The hosted Streamlit demo remains token gated for review convenience.
- Raw telemetry and generated model artifacts remain out of Git.
- Productization is frozen; portfolio evidence and MLOps rigor are the active
  direction.
