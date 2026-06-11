from __future__ import annotations

from pathlib import Path


def test_ci_release_bundle_includes_dashboard_artifacts() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    bundle_step = workflow.split("- name: Build release candidate bundle", maxsplit=1)[1].split(
        "- name: Generate release provenance",
        maxsplit=1,
    )[0]

    assert "--dashboard-payload-json" in bundle_step
    assert "artifacts/ci_release_evidence/dashboard/fleet_payload.json" in bundle_step
    assert "--dashboard-html" in bundle_step
    assert "artifacts/ci_release_evidence/dashboard/fleet_dashboard.html" in bundle_step
