"""Run a no-download C-MAPSS deployment quickstart demo."""

from __future__ import annotations

from pathlib import Path

from aerospace_prognostics.deployment.quickstart import run_cmapss_quickstart

if __name__ == "__main__":
    raise SystemExit(
        run_cmapss_quickstart(
            root=Path("artifacts") / "quickstart_cmapss",
            release_name="quickstart-fd001-demo",
            repository="local/aerospace-prognostics",
            git_sha="0" * 40,
            git_ref="refs/heads/local-quickstart",
            workflow="local-quickstart",
            run_id="local",
        )
    )
