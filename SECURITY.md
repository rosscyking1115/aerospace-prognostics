# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to **rosscyking@gmail.com**
rather than opening a public issue. Include a description, affected component
(research CLIs, FastAPI serving, Streamlit console, or container images), the
version/commit, and a minimal reproduction if possible. You can expect an
acknowledgement within a few days.

## Scope notes

This is a reference PHM pipeline, not a production service. The most
security-relevant surfaces are the FastAPI inference service (API-key auth,
rate limiting, request metrics) and the container images. No secrets are
stored in the repository — API keys and the console access token are supplied
at runtime via environment variables (see `.env.example`); the hosted demo is
token-gated and read-only. Raw telemetry and generated artifacts are kept out
of Git.
