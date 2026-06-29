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

    assert "actions/upload-artifact@v7" in upload_step
    assert "if-no-files-found: error" in upload_step
    assert "ci-fd001-release-evidence" in upload_step
    assert "artifacts/release/cmapss_fd001_release_bundle.json" in upload_step
    assert "artifacts/release/cmapss_fd001_provenance.json" in upload_step
    assert (
        "artifacts/ci_release_evidence/release/fleet_priority_policy_validation.json"
        in upload_step
    )
    assert (
        "artifacts/ci_release_evidence/release/fleet_priority_policy_validation.md"
        in upload_step
    )
    assert "artifacts/ci_release_evidence/dashboard/fleet_dashboard.html" in upload_step
    assert "artifacts/container/serving_image_manifest.json" in upload_step
    assert ".joblib" not in upload_step
    assert "fd001_input.csv" not in upload_step


def test_ci_builds_and_smokes_hosted_demo_image() -> None:
    workflow = _ci_workflow_text()

    cleanup_step = workflow.split(
        "- name: Free serving image before hosted demo image",
        maxsplit=1,
    )[1].split("- name: Build hosted demo image", maxsplit=1)[0]
    demo_build_step = workflow.split("- name: Build hosted demo image", maxsplit=1)[
        1
    ].split("- name: Verify hosted demo image contract", maxsplit=1)[0]
    demo_contract_step = workflow.split(
        "- name: Verify hosted demo image contract",
        maxsplit=1,
    )[1].split("- name: Smoke hosted demo image", maxsplit=1)[0]
    demo_smoke_step = workflow.split("- name: Smoke hosted demo image", maxsplit=1)[1]

    assert "docker image rm aerospace-prognostics:ci" in cleanup_step
    assert "docker builder prune --force" in cleanup_step
    assert "--file Dockerfile.demo" in demo_build_step
    assert "--tag aerospace-prognostics-demo:ci" in demo_build_step
    assert "AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true" in demo_contract_step
    assert "find_spec('torch') is None" in demo_contract_step
    assert "--read-only" in demo_contract_step
    assert "--tmpfs /tmp:rw,nosuid,nodev,noexec,size=128m" in demo_contract_step
    assert '--volume "$PWD/scripts:/ci-scripts:ro"' in demo_contract_step
    assert "python /ci-scripts/ci_demo_image_contract.py" in demo_contract_step
    assert "--read-only" in demo_smoke_step
    assert "--tmpfs /tmp:rw,nosuid,nodev,noexec,size=128m" in demo_smoke_step
    assert "--publish 8502:8501" in demo_smoke_step
    assert "http://127.0.0.1:8502/_stcore/health" in demo_smoke_step
    assert "docker logs" in demo_smoke_step


def _ci_workflow_text() -> str:
    return (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
