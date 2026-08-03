"""Smoke test for deployment release-evidence generation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def run(
    *,
    root: str | Path = Path("artifacts") / "ci_release_evidence",
    release_name: str = "ci-fd001-candidate",
    repository: str = "rosscyking1115/aerospace-prognostics",
    git_sha: str = "0123456789abcdef0123456789abcdef01234567",
    git_ref: str = "refs/heads/main",
    workflow: str = "CI",
    run_id: str = "1",
) -> int:
    """Generate a tiny promotion-evidence bundle through the public CLI."""
    from aerospace_prognostics.deployment.quickstart import run_cmapss_quickstart

    return run_cmapss_quickstart(
        root=root,
        release_name=release_name,
        repository=repository,
        git_sha=git_sha,
        git_ref=git_ref,
        workflow=workflow,
        run_id=run_id,
    )


if __name__ == "__main__":
    raise SystemExit(run())
