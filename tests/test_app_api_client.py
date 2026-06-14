from __future__ import annotations

import io
import urllib.error
from unittest.mock import patch

from aerospace_prognostics.app.api_client import check_api_service


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


def test_check_api_service_reports_unreachable_service() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        status = check_api_service("http://api:8000")

    assert not status.is_live
    assert not status.is_ready
    assert status.health.status_code is None
    assert status.readiness.error == "connection refused"


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
