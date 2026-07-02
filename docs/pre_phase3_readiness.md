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
  private-review license posture plus a private hosted demo URL with fresh
  visual proof from that environment.

When a private hosted demo URL, hosted proof asset, and internal license posture
are available, pass them explicitly:

```powershell
uv run aerospace-prognostics pre-phase3-readiness-audit `
  --hosted-demo-url https://PRIVATE_REVIEW_URL `
  --hosted-demo-proof docs/assets/public-proof/hosted_demo_private_review.png `
  --license-decision "private-review-only until public launch license is chosen"
```

Current expected blockers before Phase 3:

- create the private hosted read-only demo URL and capture fresh visual proof
  from that environment.

The setup handoff is tracked in
[docs/private_hosting_handoff.md](private_hosting_handoff.md). A hosted URL is
not sufficient by itself; the final audit also requires the hosted proof asset
passed with `--hosted-demo-proof`.

The current license posture is tracked in
[docs/license_posture.md](license_posture.md): private review only, not
open-source licensed yet, and `UNLICENSED` until a public-launch license is
chosen.
