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
            error=str(exc),
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
