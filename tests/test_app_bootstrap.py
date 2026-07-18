from __future__ import annotations

from pathlib import Path

from aerospace_prognostics.app.bootstrap import ensure_demo_workspace
from aerospace_prognostics.app.store import database_summary


def test_ensure_demo_workspace_generates_and_seeds_from_empty(tmp_path: Path) -> None:
    workspace_root = tmp_path / "quickstart_cmapss"
    database_path = tmp_path / "app" / "aerospace_prognostics.sqlite"

    result = ensure_demo_workspace(workspace_root, database_path)

    assert result["generated_evidence"] is True
    assert result["workspace_ready"] is True
    assert result["seed_result"]["model_artifacts"] == 1
    assert database_path.exists()

    # Matches the baked Dockerfile.demo seeding (app-init-db): the seed registers
    # the model artifact and release evidence. The fleet-triage view is driven by
    # the workspace dashboard payload, not the persisted asset registry, so the
    # registry table stays empty until a sync -- which read-only mode blocks.
    summary = database_summary(database_path, read_only=True)
    assert summary["model_artifacts"] >= 1
    assert summary["release_evidence"] >= 1


def test_ensure_demo_workspace_is_idempotent(tmp_path: Path) -> None:
    workspace_root = tmp_path / "quickstart_cmapss"
    database_path = tmp_path / "app" / "aerospace_prognostics.sqlite"

    ensure_demo_workspace(workspace_root, database_path)
    second = ensure_demo_workspace(workspace_root, database_path)

    # The bundle already exists, so the second call must not regenerate it.
    assert second["generated_evidence"] is False
    assert second["workspace_ready"] is True
    summary = database_summary(database_path, read_only=True)
    assert summary["model_artifacts"] >= 1
