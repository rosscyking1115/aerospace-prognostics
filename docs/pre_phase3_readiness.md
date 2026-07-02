# Pre-Phase-3 Readiness

Use this gate before starting Phase 3 research differentiators. Phase 3 should
not begin just because Phase 2 modelling is complete; the product and launch
foundation should also be in a reviewable state.

Run the audit:

```powershell
uv run aerospace-prognostics pre-phase3-readiness-audit --output-json artifacts/pre_phase3_readiness.json --output-markdown artifacts/pre_phase3_readiness.md
```

The audit separates two kinds of work:

- repo-local gates that the codebase can prove directly, such as launch docs,
  proof assets, CI-hosted-demo checks, artifact hygiene, and Phase 2 completion
  evidence;
- external gates that need a decision or hosted environment, such as a
  private-review license posture and a private hosted demo URL.

When a private hosted demo URL and internal license posture are available, pass
them explicitly:

```powershell
uv run aerospace-prognostics pre-phase3-readiness-audit `
  --hosted-demo-url https://PRIVATE_REVIEW_URL `
  --license-decision "private-review-only until public launch license is chosen"
```

Current expected blockers before Phase 3:

- create the private hosted read-only demo URL and capture fresh visual proof
  from that environment.

The current license posture is tracked in
[docs/license_posture.md](license_posture.md): private review only, not
open-source licensed yet, and `UNLICENSED` until a public-launch license is
chosen.
