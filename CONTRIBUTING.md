# Contributing

This is a reference PHM pipeline. Issues and pull requests are welcome.

## Development setup

```powershell
uv sync --dev
uv run ruff check .
uv run pytest
```

All changes must keep `ruff check` and the full test suite green. CI also runs
a dependency audit, SBOM generation, and container/serving smoke checks.

## Ground rules

- **Raw telemetry and generated artifacts stay out of Git.** Keep datasets
  under `data/` (gitignored) and record source URLs/checksums when adding
  download scripts.
- **No secrets in the repo.** Supply API keys and the console token at runtime
  via environment variables — see `.env.example`.
- **Honest results.** Benchmark numbers are reproducible checkpoints, not
  operational claims; state limitations and dataset scope. Do not present
  benchmark scores as certification evidence.
- **Evaluation lives in [telemeval](https://github.com/rosscyking1115/telemeval).**
  This repo consumes it as a dependency; anomaly-metric changes belong there,
  not here.
- **Productization is frozen.** The historical launch/roadmap docs under
  `docs/` are quarantined context, not active plans; don't build on them.

## Reporting security issues

See [SECURITY.md](SECURITY.md) — report privately to rosscyking@gmail.com.
