# Private Hosting Handoff

This is the final external handoff before Phase 3 can start. The repository is
ready to build a private read-only Streamlit demo image, but the actual hosted
URL must be created in a hosting account and protected before the readiness
audit can pass.

## Recommended Path

Use Render for the first private demo because the repository already has a
Docker-based, single-service Streamlit image and a tracked `render.yaml`
blueprint:

- Render builds Docker services from a `Dockerfile` in the repository.
- Render web services can use an HTTP health check path.
- The blueprint uses `autoDeployTrigger: checksPass` so deploys wait for CI.
- The demo service exposes Streamlit on port `8501` and checks
  `/_stcore/health`.

Render's default service URL is internet-reachable. Do not count that URL as a
private review URL by itself. Protect it with one of these controls:

- Cloudflare Access on a custom hostname with an allow policy for reviewer
  emails.
- Render inbound IP rules if the chosen workspace plan supports them and the
  reviewers have stable IP addresses.
- A different hosting provider that supplies equivalent private app
  authentication.

## User Setup Steps

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
7. Add a custom domain if using Cloudflare Access.
8. Protect the hosted app with Cloudflare Access or an equivalent allowlist
   before sharing it outside the owner account. The app-level access token keeps
   casual visitors out of the Streamlit console, but edge access control is the
   preferred private-review boundary.
9. Open the protected URL, unlock the app-level token gate, and confirm the
   sidebar shows read-only mode.
10. Capture a fresh screenshot or short GIF from the protected hosted URL.
11. Save the proof asset under:

```text
docs/assets/public-proof/hosted_demo_private_review.png
```

## Final Readiness Command

After the protected URL and proof asset exist, run:

```powershell
uv run aerospace-prognostics pre-phase3-readiness-audit `
  --hosted-demo-url https://PRIVATE_REVIEW_URL `
  --hosted-demo-proof docs/assets/public-proof/hosted_demo_private_review.png `
  --output-json artifacts/pre_phase3_readiness.json `
  --output-markdown artifacts/pre_phase3_readiness.md
```

The audit should report `status=ready` only after the protected URL and proof
asset are both supplied.

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
