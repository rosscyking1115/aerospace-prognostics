from __future__ import annotations

import io
import urllib.error
from unittest.mock import patch

import pytest

from aerospace_prognostics.app.api_client import (
    ApiRequestError,
    check_api_service,
    predict_telemetry,
)


def test_check_api_service_reports_ready_model() -> None:
    responses = [
        _response(200, b'{"status":"ok","model_loaded":true}'),
        _response(
            200,
            (
                b'{"status":"ready","model_loaded":true,'
                b'"model":{"artifact_id":"fd001-demo"}}'
            ),
        ),
    ]

    with patch("urllib.request.urlopen", side_effect=responses):
        status = check_api_service("http://api:8000/")

    assert status.base_url == "http://api:8000"
    assert status.is_live
    assert status.is_ready
    assert status.model_loaded
    assert status.readiness.payload["model"]["artifact_id"] == "fd001-demo"


def test_check_api_service_reports_unready_health_and_readiness() -> None:
    responses = [
        _response(200, b'{"status":"missing_model","model_loaded":false}'),
        urllib.error.HTTPError(
            "http://api:8000/ready",
            503,
            "Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"status":"missing_model","model_loaded":false}'),
        ),
    ]

    with patch("urllib.request.urlopen", side_effect=responses):
        status = check_api_service("http://api:8000")

    assert status.is_live
    assert not status.is_ready
    assert not status.model_loaded
    assert status.readiness.status_code == 503
    assert status.readiness.payload["status"] == "missing_model"
    assert status.readiness.error == "API probe failed with status 503"


def test_check_api_service_reports_unreachable_service() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        status = check_api_service("http://api:8000")

    assert not status.is_live
    assert not status.is_ready
    assert status.health.status_code is None
    assert status.readiness.error == "connection refused"


def test_predict_telemetry_posts_records_with_api_key() -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["method"] = request.get_method()
        captured["api_key"] = request.get_header("X-api-key")
        captured["body"] = request.data
        return _response(
            200,
            (
                b'{"dataset":"C-MAPSS","subset":"FD001","model_name":"hgb",'
                b'"rul_cap":125,"predictions":[{"unit_number":1,"predicted_rul":42.0}],'
                b'"monitoring":{"predictions":{"count":1}}}'
            ),
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        prediction = predict_telemetry(
            "http://api:8000/",
            telemetry=[{"unit_number": 1, "time_in_cycles": 2}],
            api_key="secret",
            timeout_seconds=7.0,
        )

    assert captured["url"] == "http://api:8000/predict"
    assert captured["timeout"] == 7.0
    assert captured["method"] == "POST"
    assert captured["api_key"] == "secret"
    assert captured["body"] == b'{"telemetry": [{"unit_number": 1, "time_in_cycles": 2}]}'
    assert prediction["predictions"][0]["predicted_rul"] == 42.0


def test_predict_telemetry_raises_request_error_for_http_error() -> None:
    error = urllib.error.HTTPError(
        "http://api:8000/predict",
        401,
        "Unauthorized",
        hdrs=None,
        fp=io.BytesIO(b'{"detail":"invalid or missing API key"}'),
    )

    with (
        patch("urllib.request.urlopen", side_effect=error),
        pytest.raises(ApiRequestError) as exc_info,
    ):
        predict_telemetry(
            "http://api:8000",
            telemetry=[{"unit_number": 1, "time_in_cycles": 2}],
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.payload["detail"] == "invalid or missing API key"
    assert str(exc_info.value) == (
        "API prediction failed with status 401: invalid or missing API key"
    )


def test_predict_telemetry_summarizes_validation_error_details() -> None:
    error = urllib.error.HTTPError(
        "http://api:8000/predict",
        422,
        "Unprocessable Entity",
        hdrs=None,
        fp=io.BytesIO(
            b'{"detail":[{"loc":["body","telemetry"],"msg":"Field required"},'
            b'{"loc":["body","other"],"msg":"Extra inputs are not permitted"}]}'
        ),
    )

    with (
        patch("urllib.request.urlopen", side_effect=error),
        pytest.raises(ApiRequestError) as exc_info,
    ):
        predict_telemetry(
            "http://api:8000",
            telemetry=[],
        )

    assert exc_info.value.status_code == 422
    assert str(exc_info.value) == (
        "API prediction failed with status 422: "
        "Field required (1 more validation errors)"
    )


def test_predict_telemetry_raises_request_error_for_unreachable_service() -> None:
    with (
        patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
        pytest.raises(ApiRequestError) as exc_info,
    ):
        predict_telemetry(
            "http://api:8000",
            telemetry=[{"unit_number": 1, "time_in_cycles": 2}],
        )

    assert exc_info.value.status_code is None
    assert "connection refused" in str(exc_info.value)


class _response:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._payload = payload

    def __enter__(self) -> _response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload
