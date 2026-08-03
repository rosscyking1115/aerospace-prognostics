"""Small HTTP client for the local Aerospace Prognostics API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApiEndpointStatus:
    """Status payload for one API endpoint probe."""

    ok: bool
    status_code: int | None
    payload: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class ApiServiceStatus:
    """Aggregated API service status for the app console."""

    base_url: str
    health: ApiEndpointStatus
    readiness: ApiEndpointStatus

    @property
    def is_live(self) -> bool:
        return self.health.ok

    @property
    def is_ready(self) -> bool:
        return self.readiness.ok

    @property
    def model_loaded(self) -> bool:
        return bool(self.readiness.payload.get("model_loaded"))


class ApiRequestError(RuntimeError):
    """Raised when an API request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


def check_api_service(base_url: str, *, timeout_seconds: float = 1.0) -> ApiServiceStatus:
    """Probe the API health and readiness endpoints."""
    normalized_base_url = base_url.rstrip("/")
    health = _get_json(f"{normalized_base_url}/health", timeout_seconds=timeout_seconds)
    readiness = (
        _get_json(f"{normalized_base_url}/ready", timeout_seconds=timeout_seconds)
        if health.status_code is not None
        else health
    )
    return ApiServiceStatus(
        base_url=normalized_base_url,
        health=health,
        readiness=readiness,
    )


def predict_telemetry(
    base_url: str,
    *,
    telemetry: list[dict[str, Any]],
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Call the deployed API prediction endpoint."""
    normalized_base_url = base_url.rstrip("/")
    payload = {"telemetry": telemetry}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(
        f"{normalized_base_url}/predict",
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = _json_payload(response.read())
            if not 200 <= int(response.status) < 300:
                raise ApiRequestError(
                    _api_error_message(
                        "API prediction failed",
                        status_code=int(response.status),
                        payload=response_payload,
                    ),
                    status_code=int(response.status),
                    payload=response_payload,
                )
            return response_payload
    except urllib.error.HTTPError as exc:
        payload = _json_payload(exc.read())
        raise ApiRequestError(
            _api_error_message(
                "API prediction failed",
                status_code=int(exc.code),
                payload=payload,
            ),
            status_code=int(exc.code),
            payload=payload,
        ) from exc
    except (OSError, TimeoutError) as exc:
        raise ApiRequestError(f"API prediction request failed: {exc}") from exc


def _get_json(url: str, *, timeout_seconds: float) -> ApiEndpointStatus:
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = _json_payload(response.read())
            status_code = int(response.status)
            return ApiEndpointStatus(
                ok=200 <= status_code < 300,
                status_code=status_code,
                payload=payload,
            )
    except urllib.error.HTTPError as exc:
        payload = _json_payload(exc.read())
        return ApiEndpointStatus(
            ok=False,
            status_code=int(exc.code),
            payload=payload,
            error=_api_error_message(
                "API probe failed",
                status_code=int(exc.code),
                payload=payload,
            ),
        )
    except (OSError, TimeoutError) as exc:
        return ApiEndpointStatus(
            ok=False,
            status_code=None,
            payload={},
            error=str(exc),
        )


def _json_payload(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _api_error_message(
    prefix: str,
    *,
    status_code: int,
    payload: dict[str, Any],
) -> str:
    message = f"{prefix} with status {status_code}"
    detail = payload.get("detail")
    if isinstance(detail, str) and detail:
        return f"{message}: {detail}"
    if isinstance(detail, list) and detail:
        first_detail = detail[0]
        if isinstance(first_detail, dict):
            first_message = first_detail.get("msg")
            if isinstance(first_message, str) and first_message:
                remaining = len(detail) - 1
                suffix = f" ({remaining} more validation errors)" if remaining else ""
                return f"{message}: {first_message}{suffix}"
    return message
