from __future__ import annotations

from pathlib import Path


def test_compose_stack_defines_product_services() -> None:
    compose = _compose_text()

    assert "services:" in compose
    assert "  app-db:" in compose
    assert "  api:" in compose
    assert "  console:" in compose
    assert "image: aerospace-prognostics:local" in compose


def test_compose_stack_mounts_quickstart_artifacts_and_app_state() -> None:
    compose = _compose_text()

    assert "./artifacts:/app/artifacts" in compose
    assert "./artifacts/quickstart_cmapss/models:/models:ro" in compose
    assert "AEROSPACE_PROGNOSTICS_MODEL_PATH: /models/fd001.joblib" in compose
    assert "/app/artifacts/app/aerospace_prognostics.sqlite" in compose
    assert "/app/artifacts/quickstart_cmapss" in compose


def test_compose_stack_exposes_api_and_console_with_healthchecks() -> None:
    compose = _compose_text()

    assert '"${AEROSPACE_PROGNOSTICS_API_PORT:-8000}:8000"' in compose
    assert '"${AEROSPACE_PROGNOSTICS_CONSOLE_PORT:-8501}:8501"' in compose
    assert "aerospace_prognostics.serving.healthcheck" in compose
    assert "http://127.0.0.1:8501/_stcore/health" in compose
    assert "condition: service_healthy" in compose
    assert "condition: service_completed_successfully" in compose


def test_compose_stack_sets_local_api_security_defaults() -> None:
    compose = _compose_text()

    assert "AEROSPACE_PROGNOSTICS_API_KEY:" in compose
    assert "local-dev-secret" in compose
    assert "AEROSPACE_PROGNOSTICS_RATE_LIMIT_PER_MINUTE:" in compose


def _compose_text() -> str:
    return (Path(__file__).resolve().parents[1] / "compose.yaml").read_text(encoding="utf-8")
