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

Latest hosted-demo verification:

- URL: <https://aerospace-prognostics-private-demo.onrender.com>
- Health check: `/_stcore/health` returned `200 ok`.
- Proof asset:
  [docs/assets/public-proof/hosted_demo_private_review.png](assets/public-proof/hosted_demo_private_review.png)
- Readiness audit with the hosted URL and proof asset reported
  `status=ready`, `gates=10`, `blockers=0`.

The setup handoff is tracked in
[docs/private_hosting_handoff.md](private_hosting_handoff.md). A hosted URL is
not sufficient by itself; the final audit also requires the hosted proof asset
passed with `--hosted-demo-proof`.

The default Render service URL is internet-reachable. Add Cloudflare Access or
an equivalent allowlist before treating the deployment as a private reviewer
URL. The app also supports an optional token gate through
`AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN`, which should be set on Render as
soon as the updated Blueprint deploys.

The current license posture is tracked in
[docs/license_posture.md](license_posture.md): private review only, not
open-source licensed yet, and `UNLICENSED` until a public-launch license is
chosen.
