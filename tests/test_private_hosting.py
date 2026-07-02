from __future__ import annotations

from pathlib import Path


def test_render_blueprint_targets_read_only_demo_image() -> None:
    blueprint = _repo_root() / "render.yaml"
    text = blueprint.read_text(encoding="utf-8")

    assert "runtime: docker" in text
    assert "dockerfilePath: ./Dockerfile.demo" in text
    assert "autoDeployTrigger: checksPass" in text
    assert "healthCheckPath: /_stcore/health" in text
    assert "PORT" in text
    assert 'value: "8501"' in text
    assert "AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY" in text
    assert 'value: "true"' in text


def test_private_hosting_handoff_names_proof_and_access_control() -> None:
    handoff = (_repo_root() / "docs" / "private_hosting_handoff.md").read_text(
        encoding="utf-8",
    )

    assert "Cloudflare Access" in handoff
    assert "render.yaml" in handoff
    assert "docs/assets/public-proof/hosted_demo_private_review.png" in handoff
    assert "--hosted-demo-proof" in handoff


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
