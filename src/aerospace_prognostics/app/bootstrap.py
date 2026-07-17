"""Boot-time demo provisioning for hosts without a pre-baked image.

The Docker demo image (`Dockerfile.demo`) bakes the quickstart evidence bundle
and a seeded SQLite database at build time, so the read-only console has data
to show the moment it starts. Hosts that do not run that image -- notably
Streamlit Community Cloud, which has no Docker step -- start with an empty
workspace, so the console would render its "generate the quickstart evidence"
placeholder instead of the fleet view.

`ensure_demo_workspace` closes that gap. On boot it generates the quickstart
evidence if it is missing, then initialises and seeds the console database,
server-side. This is deliberately independent of read-only mode: read-only
governs whether a *visitor* can mutate state through the console, while this
function provisions the demo's own baseline data before anyone visits. The
function is idempotent, so on a pre-baked image (evidence already present) it
reseeds an existing database without regenerating anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aerospace_prognostics.app.dashboard_state import load_quickstart_workspace
from aerospace_prognostics.app.store import (
    initialize_app_database,
    seed_quickstart_workspace,
)
from aerospace_prognostics.deployment.quickstart import run_cmapss_quickstart


def ensure_demo_workspace(
    workspace_root: str | Path,
    database_path: str | Path,
) -> dict[str, Any]:
    """Generate quickstart evidence and seed the console database if missing.

    Returns a small status dict describing what happened, so callers (and
    tests) can assert on it. Intended to run once per process, before the
    console renders.
    """

    workspace_root = Path(workspace_root)
    database_path = Path(database_path)

    workspace = load_quickstart_workspace(workspace_root)
    generated_evidence = False
    if not workspace.is_ready:
        run_cmapss_quickstart(root=workspace_root)
        workspace = load_quickstart_workspace(workspace_root)
        generated_evidence = True

    initialize_app_database(database_path)
    seed_result = seed_quickstart_workspace(database_path, workspace)

    return {
        "workspace_ready": workspace.is_ready,
        "generated_evidence": generated_evidence,
        "seed_result": seed_result,
    }
