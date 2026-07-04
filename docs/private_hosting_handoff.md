# Private Hosting Handoff

This records the private read-only Streamlit demo handoff. The Render-hosted
demo now exists, passes the Streamlit health check, has a tracked hosted proof
asset, and is protected by the app-level
`AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN` gate. Stronger edge access remains
recommended before broader external sharing.

## Recommended Path

Render is the first private demo host because the repository already has a
Docker-based, single-service Streamlit image and a tracked `render.yaml`
blueprint:

- Render builds Docker services from a `Dockerfile` in the repository.
- Render web services can use an HTTP health check path.
- The blueprint uses `autoDeployTrigger: checksPass` so deploys wait for CI.
- The demo service exposes Streamlit on port `8501` and checks
  `/_stcore/health`.

Render's default service URL is internet-reachable at the network layer. The
configured app-level token gate satisfies the current internal private-demo
milestone. For reviewer groups or any broader external sharing, add one of
these stronger controls:

- Cloudflare Access on a custom hostname with an allow policy for reviewer
  emails.
- Render inbound IP rules if the chosen workspace plan supports them and the
  reviewers have stable IP addresses.
- A different hosting provider that supplies equivalent private app
  authentication.

## Current Completed Setup

1. Keep the GitHub repository private.
2. In Render, connect the GitHub account and grant access to this repository.
3. Create a Blueprint from the root `render.yaml`.
4. Confirm the service builds from `Dockerfile.demo`.
5. Confirm the service environment includes:
   - `PORT=8501`
   - `AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true`
   - `AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN=<strong-secret>`
   - `STREAMLIT_SERVER_HEADLESS=true`
   - `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`
6. Confirm the health check path is `/_stcore/health`.
7. Open the hosted URL, unlock the app-level token gate, and confirm the
   sidebar shows read-only mode.
8. Capture a fresh screenshot or short GIF from the hosted URL.
9. Save the proof asset under:

```text
docs/assets/public-proof/hosted_demo_private_review.png
```

## Optional Access Hardening

1. Add a custom domain if using Cloudflare Access.
2. Protect the hosted app with Cloudflare Access, Render inbound IP rules, or an
   equivalent allowlist before sharing it outside the owner account. The
   app-level access token keeps casual visitors out of the Streamlit console,
   but edge access control is the preferred private-review boundary for larger
   reviewer groups.
3. Capture a new proof asset from the edge-protected URL if the review context
   requires that stronger boundary.

## Final Readiness Command

After the hosted URL and proof asset exist, run:

```powershell
uv run aerospace-prognostics pre-phase3-readiness-audit `
  --hosted-demo-url https://PRIVATE_REVIEW_URL `
  --hosted-demo-proof docs/assets/public-proof/hosted_demo_private_review.png `
  --output-json artifacts/pre_phase3_readiness.json `
  --output-markdown artifacts/pre_phase3_readiness.md
```

The audit should report `status=ready` only after the hosted URL and proof asset
are both supplied.

## App-Level Token Gate

The console is public by default when `AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN`
is unset. Set that environment variable in Render to require a token before the
Streamlit app renders the sidebar, tabs, downloads, or evidence screens. The
token is stored only in the host environment and compared with a constant-time
comparison inside the app.

For Render Blueprints, `render.yaml` declares the key with `sync: false`, so
Render should prompt for the secret value instead of committing it to Git.

## Source Notes

- Render Docker deploys:
  <https://render.com/docs/docker>
- Render Blueprint reference:
  <https://render.com/docs/blueprint-spec>
- Render health checks:
  <https://render.com/docs/health-checks>
- Render GitHub connection:
  <https://render.com/docs/github>
- Cloudflare Access self-hosted applications:
  <https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/self-hosted-public-app/>
