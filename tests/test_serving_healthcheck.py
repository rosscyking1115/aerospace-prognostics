from __future__ import annotations

from aerospace_prognostics.serving.healthcheck import (
    DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS,
    HEALTHCHECK_TIMEOUT_ENV,
    _healthcheck_timeout,
    health_payload_is_live,
)


def test_health_payload_is_live_accepts_loaded_and_missing_model_states() -> None:
    assert health_payload_is_live({"status": "ok", "model_loaded": True})
    assert health_payload_is_live({"status": "missing_model", "model_loaded": False})


def test_health_payload_is_live_rejects_inconsistent_payloads() -> None:
    assert not health_payload_is_live({"status": "ready", "model_loaded": True})
    assert not health_payload_is_live({"status": "ok", "model_loaded": False})
    assert not health_payload_is_live({"status": "missing_model", "model_loaded": True})
    assert not health_payload_is_live({"status": "ok"})
    assert not health_payload_is_live([])


def test_healthcheck_timeout_env_falls_back_for_invalid_values(monkeypatch) -> None:
    monkeypatch.delenv(HEALTHCHECK_TIMEOUT_ENV, raising=False)
    assert _healthcheck_timeout() == DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS

    monkeypatch.setenv(HEALTHCHECK_TIMEOUT_ENV, "3.5")
    assert _healthcheck_timeout() == 3.5

    monkeypatch.setenv(HEALTHCHECK_TIMEOUT_ENV, "0")
    assert _healthcheck_timeout() == DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS

    monkeypatch.setenv(HEALTHCHECK_TIMEOUT_ENV, "not-a-number")
    assert _healthcheck_timeout() == DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS
