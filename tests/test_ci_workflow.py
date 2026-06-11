from __future__ import annotations

from pathlib import Path


def test_ci_release_bundle_includes_dashboard_artifacts() -> None:
    workflow = _ci_workflow_text()

    bundle_step = workflow.split("- name: Build release candidate bundle", maxsplit=1)[1].split(
        "- name: Generate release provenance",
        maxsplit=1,
    )[0]

    assert "--dashboard-payload-json" in bundle_step
    assert "artifacts/ci_release_evidence/dashboard/fleet_payload.json" in bundle_step
    assert "--dashboard-html" in bundle_step
    assert "artifacts/ci_release_evidence/dashboard/fleet_dashboard.html" in bundle_step


def test_ci_uploads_reviewable_release_evidence_without_model_binary() -> None:
    workflow = _ci_workflow_text()

    upload_step = workflow.split("- name: Upload release evidence artifacts", maxsplit=1)[
        1
    ].split("- name: Smoke serving image", maxsplit=1)[0]

    assert "actions/upload-artifact@v4" in upload_step
    assert "if-no-files-found: error" in upload_step
    assert "ci-fd001-release-evidence" in upload_step
    assert "artifacts/release/cmapss_fd001_release_bundle.json" in upload_step
    assert "artifacts/release/cmapss_fd001_provenance.json" in upload_step
    assert "artifacts/ci_release_evidence/dashboard/fleet_dashboard.html" in upload_step
    assert "artifacts/container/serving_image_manifest.json" in upload_step
    assert ".joblib" not in upload_step
    assert "fd001_input.csv" not in upload_step


def _ci_workflow_text() -> str:
    return (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
