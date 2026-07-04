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
- Access: `AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN` is configured on Render,
  so the Streamlit evidence screens are app-level token gated.
- Proof asset:
  [docs/assets/public-proof/hosted_demo_private_review.png](assets/public-proof/hosted_demo_private_review.png)
- Readiness audit with the hosted URL and proof asset reported
  `status=ready`, `gates=10`, `blockers=0`.

The setup handoff is tracked in
[docs/private_hosting_handoff.md](private_hosting_handoff.md). A hosted URL is
not sufficient by itself; the final audit also requires the hosted proof asset
passed with `--hosted-demo-proof`.

The default Render service URL remains internet-reachable at the network layer.
For the current internal private-demo milestone, the configured app-level token
gate is the accepted access control. Add Cloudflare Access, Render inbound IP
rules, or an equivalent allowlist before broader external sharing or any review
where edge-level private access is required.

The current license posture is tracked in
[docs/license_posture.md](license_posture.md): private review only, not
open-source licensed yet, and `UNLICENSED` until a public-launch license is
chosen.
