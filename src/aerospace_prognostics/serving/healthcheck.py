"""Container healthcheck helper for the serving API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

HEALTHCHECK_URL_ENV = "AEROSPACE_PROGNOSTICS_HEALTHCHECK_URL"
HEALTHCHECK_TIMEOUT_ENV = "AEROSPACE_PROGNOSTICS_HEALTHCHECK_TIMEOUT_SECONDS"
DEFAULT_HEALTHCHECK_URL = "http://127.0.0.1:8000/health"
DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS = 2.0


def health_payload_is_live(payload: Any) -> bool:
    """Return whether a /health payload represents a live serving process."""
    if not isinstance(payload, dict):
        return False
    status = payload.get("status")
    model_loaded = payload.get("model_loaded")
    if not isinstance(model_loaded, bool):
        return False
    return (status == "ok" and model_loaded) or (
        status == "missing_model" and not model_loaded
    )


def main() -> int:
    """Probe the serving API liveness endpoint for Docker HEALTHCHECK."""
    url = os.getenv(HEALTHCHECK_URL_ENV, DEFAULT_HEALTHCHECK_URL)
    timeout = _healthcheck_timeout()
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, OSError, TimeoutError, URLError, json.JSONDecodeError):
        return 1
    return 0 if health_payload_is_live(payload) else 1


def _healthcheck_timeout() -> float:
    value = os.getenv(HEALTHCHECK_TIMEOUT_ENV)
    if value is None:
        return DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError:
        return DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS
    return timeout if timeout > 0 else DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS


if __name__ == "__main__":
    raise SystemExit(main())
