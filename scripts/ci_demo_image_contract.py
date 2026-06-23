"""Verify the hosted demo image contains the expected read-only demo state."""

from __future__ import annotations

import argparse
from pathlib import Path

from aerospace_prognostics.app.dashboard_state import load_quickstart_workspace
from aerospace_prognostics.app.store import database_summary, list_model_artifacts


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("artifacts") / "quickstart_cmapss",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    args = parser.parse_args(argv)

    workspace = load_quickstart_workspace(args.workspace)
    if not workspace.is_ready or workspace.missing_paths:
        missing = ", ".join(str(path) for path in workspace.missing_paths)
        raise RuntimeError(f"demo quickstart workspace is incomplete: {missing}")

    dashboard_payload = workspace.dashboard_payload or {}
    if dashboard_payload.get("schema_version") != "aerospace-prognostics/fleet-dashboard/v1":
        raise RuntimeError("demo dashboard payload has an unexpected schema")
    if int(dashboard_payload.get("summary", {}).get("asset_count", 0)) <= 0:
        raise RuntimeError("demo dashboard payload has no fleet assets")

    release_bundle = workspace.release_bundle or {}
    if release_bundle.get("status") != "ok":
        raise RuntimeError(f"demo release bundle is not ok: {release_bundle.get('status')}")

    provenance = workspace.provenance or {}
    if provenance.get("status") != "ok":
        raise RuntimeError(f"demo provenance is not ok: {provenance.get('status')}")

    summary = database_summary(args.database)
    minimum_counts = {
        "model_artifacts": 1,
        "release_evidence": 5,
    }
    for key, expected in minimum_counts.items():
        actual = int(summary[key])
        if actual < expected:
            raise RuntimeError(f"expected {key}>={expected}, got {actual}")

    artifacts = list_model_artifacts(args.database)
    if not artifacts:
        raise RuntimeError("expected at least one registered model artifact")
    artifact = artifacts[0]
    if int(artifact.get("evidence_count") or 0) < 5:
        raise RuntimeError(f"expected artifact evidence_count>=5, got {artifact!r}")

    print(f"workspace={workspace.root}")
    print(f"database={summary['database_path']}")
    print(f"artifact_id={artifact['artifact_id']}")
    print("demo_image_contract=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
