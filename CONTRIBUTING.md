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

## Dependency advisories

`pip-audit` runs in CI and blocks on any known vulnerability. When one lands,
the fix depends on whether the affected package is one we actually import.

**If it is a direct dependency**, raise the floor in `[project] dependencies`.

**If it is transitive**, use `[tool.uv] constraint-dependencies` instead — *not*
`[project] dependencies`. The distinction matters and is easy to get wrong:
adding a transitive package to `dependencies` does pin it, but it also declares
that this project imports the package, which is false. That misstates the
dependency surface for anyone reading `pyproject.toml`, for the SBOM, and for
whoever later tries to work out why the package is listed. A constraint bounds
the resolver without claiming an import.

Worked example, committed 2026-07-28: eight advisories landed against
`gitpython 3.1.50`, which arrives transitively via Streamlit. Nothing in this
repository had changed — the same lockfile passed CI on 18 July and the advisory
database moved under it, because `pip-audit` queries that database live. The fix
was `constraint-dependencies = ["gitpython>=3.1.55"]`.

**Check the parent's own range before constraining.** Streamlit declares
`gitpython!=3.1.19,<4,>=3.0.7`, so `3.1.55` sits comfortably inside it — making
this a constraint rather than an override, with nothing forced past what its
parent permits. Had the parent capped below the fixed version, the correct
response would have been to stop and say so, not to override: forcing a package
past its parent's declared range trades a known advisory for an unknown
incompatibility.

## Reporting security issues

See [SECURITY.md](SECURITY.md) — report privately to rosscyking@gmail.com.
