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
    assert "AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN" in text
    assert "sync: false" in text
    assert 'value: "true"' in text


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
