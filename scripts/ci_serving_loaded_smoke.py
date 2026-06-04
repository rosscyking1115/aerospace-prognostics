"""CI smoke test for a container serving a mounted model artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args(argv)

    health = _wait_for_json(args.base_url, "/health")
    if health != {"status": "ok", "model_loaded": True}:
        raise RuntimeError(f"unexpected health payload: {health!r}")

    ready = _get_json(args.base_url, "/ready")
    if ready["status"] != "ready" or not ready["model_loaded"]:
        raise RuntimeError(f"unexpected readiness payload: {ready!r}")

    unauth_schema_status = _status_for_get(args.base_url, "/schema")
    if unauth_schema_status != 401:
        raise RuntimeError(
            "expected unauthenticated /schema to return 401, "
            f"got {unauth_schema_status}"
        )

    headers = {"x-api-key": args.api_key}
    schema = _get_json(args.base_url, "/schema", headers=headers)
    if schema["artifact_id"] != ready["model"]["artifact_id"]:
        raise RuntimeError("schema artifact_id does not match readiness artifact_id")
    if schema["artifact_sha256"] != ready["model"]["artifact_sha256"]:
        raise RuntimeError("schema artifact_sha256 does not match readiness artifact_sha256")

    telemetry = json.loads(pd.read_csv(args.input_csv).to_json(orient="records"))
    prediction = _post_json(
        args.base_url,
        "/predict",
        {"telemetry": telemetry},
        headers=headers,
    )
    if len(prediction["predictions"]) != 2:
        raise RuntimeError(f"expected 2 predictions, got {prediction!r}")
    if prediction["monitoring"]["predictions"]["count"] != 2:
        raise RuntimeError(f"unexpected monitoring payload: {prediction!r}")

    metrics = _get_text(args.base_url, "/metrics", headers=headers)
    if "aerospace_prognostics_requests_total" not in metrics:
        raise RuntimeError(f"metrics payload is missing request counter: {metrics!r}")
    if "aerospace_prognostics_predictions_total 2" not in metrics:
        raise RuntimeError(f"metrics payload is missing prediction counter: {metrics!r}")
    if "aerospace_prognostics_prediction_rul_mean" not in metrics:
        raise RuntimeError(f"metrics payload is missing prediction RUL gauge: {metrics!r}")

    print(f"artifact_id={ready['model']['artifact_id']}")
    print(f"artifact_sha256={ready['model']['artifact_sha256']}")
    print(f"predictions={len(prediction['predictions'])}")
    print("loaded_container_smoke=ok")
    return 0


def _wait_for_json(base_url: str, path: str) -> dict[str, object]:
    last_error: Exception | None = None
    for _ in range(30):
        try:
            return _get_json(base_url, path)
        except (HTTPError, OSError, TimeoutError, URLError) as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"endpoint did not become available: {path}") from last_error


def _status_for_get(base_url: str, path: str) -> int:
    try:
        with urlopen(f"{base_url}{path}", timeout=10) as response:
            return response.status
    except HTTPError as exc:
        return exc.code


def _get_json(
    base_url: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    with urlopen(Request(f"{base_url}{path}", headers=headers or {}), timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {path}, got {payload!r}")
    return payload


def _post_json(
    base_url: str,
    path: str,
    payload: dict[str, object],
    *,
    headers: dict[str, str],
) -> dict[str, object]:
    request_headers = {"content-type": "application/json", **headers}
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(response_payload, dict):
        raise RuntimeError(f"expected JSON object from {path}, got {response_payload!r}")
    return response_payload


def _get_text(
    base_url: str,
    path: str,
    *,
    headers: dict[str, str],
) -> str:
    with urlopen(Request(f"{base_url}{path}", headers=headers), timeout=10) as response:
        return response.read().decode("utf-8")


if __name__ == "__main__":
    raise SystemExit(run())
