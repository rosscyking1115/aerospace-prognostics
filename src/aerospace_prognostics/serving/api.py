"""FastAPI application for deployed C-MAPSS RUL inference."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from hmac import compare_digest
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from aerospace_prognostics.data.integrity import file_sha256
from aerospace_prognostics.deployment.artifacts import (
    CmapssHgbPolicyModelArtifact,
    load_cmapss_model_artifact,
)

LOGGER = logging.getLogger("aerospace_prognostics.serving")
API_KEY_ENV = "AEROSPACE_PROGNOSTICS_API_KEY"
RATE_LIMIT_ENV = "AEROSPACE_PROGNOSTICS_RATE_LIMIT_PER_MINUTE"
MODEL_SHA256_ENV = "AEROSPACE_PROGNOSTICS_MODEL_SHA256"


class PredictRequest(BaseModel):
    """Raw telemetry rows to score."""

    telemetry: list[dict[str, float | int]] = Field(..., min_length=1, max_length=10000)


class PredictResponse(BaseModel):
    """Prediction response for one request."""

    dataset: str
    subset: str
    model_name: str
    rul_cap: int
    predictions: list[dict[str, Any]]
    monitoring: dict[str, Any]


class ServingMetrics:
    """In-memory request counters for lightweight container smoke and local serving."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._latency_seconds_sum = 0.0
        self._latency_seconds_max = 0.0
        self._status_counts: dict[tuple[str, str, int], int] = defaultdict(int)
        self._prediction_request_count = 0
        self._prediction_count = 0
        self._prediction_rul_sum = 0.0
        self._prediction_rul_min: float | None = None
        self._prediction_rul_max: float | None = None
        self._drift_alert_request_count = 0
        self._drift_alert_column_count = 0
        self._drift_max_standardized_abs_mean_shift = 0.0

    def record(self, method: str, path: str, status_code: int, latency_seconds: float) -> None:
        with self._lock:
            self._request_count += 1
            self._latency_seconds_sum += latency_seconds
            self._latency_seconds_max = max(self._latency_seconds_max, latency_seconds)
            self._status_counts[(method, path, status_code)] += 1

    def record_prediction(self, monitoring: dict[str, Any]) -> None:
        prediction_summary = _dict_value(monitoring, "predictions")
        telemetry_summary = _dict_value(monitoring, "telemetry")
        prediction_count = _int_metric_value(prediction_summary.get("count"))
        prediction_mean = _float_metric_value(prediction_summary.get("mean"))
        prediction_min = _float_metric_value(prediction_summary.get("min"))
        prediction_max = _float_metric_value(prediction_summary.get("max"))
        alert_column_count = _int_metric_value(telemetry_summary.get("alert_column_count"))
        max_shift = _float_metric_value(
            telemetry_summary.get("max_standardized_abs_mean_shift")
        )
        with self._lock:
            self._prediction_request_count += 1
            self._prediction_count += prediction_count
            if prediction_count and prediction_mean is not None:
                self._prediction_rul_sum += prediction_mean * prediction_count
            if prediction_min is not None:
                self._prediction_rul_min = (
                    prediction_min
                    if self._prediction_rul_min is None
                    else min(self._prediction_rul_min, prediction_min)
                )
            if prediction_max is not None:
                self._prediction_rul_max = (
                    prediction_max
                    if self._prediction_rul_max is None
                    else max(self._prediction_rul_max, prediction_max)
                )
            if alert_column_count:
                self._drift_alert_request_count += 1
                self._drift_alert_column_count += alert_column_count
            if max_shift is not None:
                self._drift_max_standardized_abs_mean_shift = max(
                    self._drift_max_standardized_abs_mean_shift,
                    max_shift,
                )

    def prometheus_text(self) -> str:
        with self._lock:
            prediction_mean = (
                self._prediction_rul_sum / self._prediction_count
                if self._prediction_count
                else 0.0
            )
            lines = [
                "# HELP aerospace_prognostics_requests_total Total HTTP requests served.",
                "# TYPE aerospace_prognostics_requests_total counter",
                f"aerospace_prognostics_requests_total {self._request_count}",
                "# HELP aerospace_prognostics_request_latency_seconds_sum "
                "Cumulative HTTP request latency.",
                "# TYPE aerospace_prognostics_request_latency_seconds_sum counter",
                (
                    "aerospace_prognostics_request_latency_seconds_sum "
                    f"{self._latency_seconds_sum:.9f}"
                ),
                "# HELP aerospace_prognostics_request_latency_seconds_max "
                "Maximum observed HTTP request latency.",
                "# TYPE aerospace_prognostics_request_latency_seconds_max gauge",
                (
                    "aerospace_prognostics_request_latency_seconds_max "
                    f"{self._latency_seconds_max:.9f}"
                ),
                "# HELP aerospace_prognostics_http_responses_total HTTP responses by route.",
                "# TYPE aerospace_prognostics_http_responses_total counter",
            ]
            for (method, path, status_code), count in sorted(self._status_counts.items()):
                lines.append(
                    "aerospace_prognostics_http_responses_total"
                    f'{{method="{method}",path="{path}",status_code="{status_code}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP aerospace_prognostics_prediction_requests_total "
                    "Prediction requests served.",
                    "# TYPE aerospace_prognostics_prediction_requests_total counter",
                    (
                        "aerospace_prognostics_prediction_requests_total "
                        f"{self._prediction_request_count}"
                    ),
                    "# HELP aerospace_prognostics_predictions_total "
                    "Individual RUL predictions returned.",
                    "# TYPE aerospace_prognostics_predictions_total counter",
                    f"aerospace_prognostics_predictions_total {self._prediction_count}",
                    "# HELP aerospace_prognostics_prediction_rul_mean "
                    "Mean predicted RUL across served predictions.",
                    "# TYPE aerospace_prognostics_prediction_rul_mean gauge",
                    f"aerospace_prognostics_prediction_rul_mean {prediction_mean:.9f}",
                    "# HELP aerospace_prognostics_prediction_rul_min "
                    "Minimum predicted RUL observed.",
                    "# TYPE aerospace_prognostics_prediction_rul_min gauge",
                    (
                        "aerospace_prognostics_prediction_rul_min "
                        f"{(self._prediction_rul_min or 0.0):.9f}"
                    ),
                    "# HELP aerospace_prognostics_prediction_rul_max "
                    "Maximum predicted RUL observed.",
                    "# TYPE aerospace_prognostics_prediction_rul_max gauge",
                    (
                        "aerospace_prognostics_prediction_rul_max "
                        f"{(self._prediction_rul_max or 0.0):.9f}"
                    ),
                    "# HELP aerospace_prognostics_telemetry_drift_alert_requests_total "
                    "Prediction requests with at least one telemetry drift alert column.",
                    "# TYPE aerospace_prognostics_telemetry_drift_alert_requests_total counter",
                    (
                        "aerospace_prognostics_telemetry_drift_alert_requests_total "
                        f"{self._drift_alert_request_count}"
                    ),
                    "# HELP aerospace_prognostics_telemetry_drift_alert_columns_total "
                    "Telemetry drift alert columns observed across prediction requests.",
                    "# TYPE aerospace_prognostics_telemetry_drift_alert_columns_total counter",
                    (
                        "aerospace_prognostics_telemetry_drift_alert_columns_total "
                        f"{self._drift_alert_column_count}"
                    ),
                    "# HELP aerospace_prognostics_telemetry_drift_max_standardized_abs_mean_shift "
                    "Maximum telemetry standardized absolute mean shift observed.",
                    (
                        "# TYPE "
                        "aerospace_prognostics_telemetry_drift_max_standardized_abs_mean_shift "
                        "gauge"
                    ),
                    (
                        "aerospace_prognostics_telemetry_drift_max_standardized_abs_mean_shift "
                        f"{self._drift_max_standardized_abs_mean_shift:.9f}"
                    ),
                ]
            )
            return "\n".join(lines) + "\n"


class ServingSecurity:
    """Optional API key and per-client rate limiting for deployment serving."""

    def __init__(
        self,
        *,
        api_key: str | None,
        rate_limit_per_minute: int,
        window_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._rate_limit_per_minute = rate_limit_per_minute
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._request_times: dict[str, deque[float]] = defaultdict(deque)

    def enforce(self, request: Request) -> None:
        api_key = self._require_api_key(request)
        self._enforce_rate_limit(request, api_key)

    def _require_api_key(self, request: Request) -> str | None:
        if not self._api_key:
            return None
        supplied_key = _api_key_from_request(request)
        if supplied_key is None or not compare_digest(supplied_key, self._api_key):
            raise HTTPException(
                status_code=401,
                detail="invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return supplied_key

    def _enforce_rate_limit(self, request: Request, api_key: str | None) -> None:
        if self._rate_limit_per_minute <= 0:
            return
        client_key = api_key or (request.client.host if request.client else "unknown")
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            request_times = self._request_times[client_key]
            while request_times and request_times[0] <= cutoff:
                request_times.popleft()
            if len(request_times) >= self._rate_limit_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="rate limit exceeded",
                    headers={"Retry-After": str(int(self._window_seconds))},
                )
            request_times.append(now)


def create_app(
    artifact_path: str | Path | None = None,
    *,
    api_key: str | None = None,
    rate_limit_per_minute: int | None = None,
    expected_artifact_sha256: str | None = None,
) -> FastAPI:
    """Create a FastAPI app, optionally loading a model artifact at startup.

    ``rate_limit_per_minute=0`` disables request throttling. Positive values
    enable a per-client fixed-window limit for secured deployment smoke tests.
    """
    app = FastAPI(
        title="Aerospace Prognostics API",
        version="0.1.0",
        description="Deployment API for C-MAPSS remaining useful life prediction.",
    )
    configured_path = artifact_path or os.getenv("AEROSPACE_PROGNOSTICS_MODEL_PATH")
    configured_sha256 = (
        expected_artifact_sha256
        if expected_artifact_sha256 is not None
        else os.getenv(MODEL_SHA256_ENV)
    )
    artifact_sha256 = None
    if configured_path is not None and str(configured_path):
        artifact_sha256 = (
            _verify_model_artifact_sha256(configured_path, configured_sha256)
            if configured_sha256
            else file_sha256(configured_path)
        )
    artifact = (
        load_cmapss_model_artifact(configured_path)
        if configured_path is not None and str(configured_path)
        else None
    )
    app.state.artifact = artifact
    app.state.artifact_path = str(configured_path) if configured_path is not None else None
    app.state.artifact_sha256 = artifact_sha256
    app.state.metrics = ServingMetrics()
    app.state.security = ServingSecurity(
        api_key=api_key if api_key is not None else os.getenv(API_KEY_ENV),
        rate_limit_per_minute=_configured_rate_limit(rate_limit_per_minute),
    )

    @app.middleware("http")
    async def observe_requests(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            latency_seconds = time.perf_counter() - started
            route_path = _route_path(request)
            app.state.metrics.record(request.method, route_path, status_code, latency_seconds)
            _log_request(request, request_id, status_code, latency_seconds, route_path)
            raise

        latency_seconds = time.perf_counter() - started
        route_path = _route_path(request)
        app.state.metrics.record(request.method, route_path, status_code, latency_seconds)
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{latency_seconds * 1000:.3f}"
        _log_request(request, request_id, status_code, latency_seconds, route_path)
        return response

    @app.get("/health")
    def health() -> dict[str, bool | str]:
        loaded = app.state.artifact is not None
        return {"status": "ok" if loaded else "missing_model", "model_loaded": loaded}

    @app.get("/ready", response_model=None)
    def ready() -> dict[str, bool | str] | JSONResponse:
        payload = _readiness_payload(app.state.artifact, app.state.artifact_sha256)
        if app.state.artifact is None:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/version")
    def version(request: Request) -> dict[str, Any]:
        app.state.security.enforce(request)
        model = _require_artifact(app.state.artifact)
        return model.metadata()

    @app.get("/schema")
    def schema(request: Request) -> dict[str, Any]:
        app.state.security.enforce(request)
        model = _require_artifact(app.state.artifact)
        return _inference_schema_payload(model, app.state.artifact_sha256)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        app.state.security.enforce(request)
        model = _require_artifact(app.state.artifact)
        frame = pd.DataFrame(payload.telemetry)
        try:
            predictions = model.predict_from_frame(frame)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        monitoring = model.monitoring_summary(frame, predictions)
        app.state.metrics.record_prediction(monitoring)
        LOGGER.info(
            json.dumps(
                {
                    "event": "prediction_monitoring",
                    "dataset": model.dataset,
                    "subset": model.subset,
                    "model_name": model.model_name,
                    "prediction_count": monitoring["predictions"]["count"],
                    "prediction_mean": monitoring["predictions"]["mean"],
                    "telemetry_alert_column_count": monitoring["telemetry"][
                        "alert_column_count"
                    ],
                    "telemetry_alert_columns": monitoring["telemetry"]["alert_columns"],
                    "telemetry_max_standardized_abs_mean_shift": monitoring["telemetry"][
                        "max_standardized_abs_mean_shift"
                    ],
                },
                sort_keys=True,
            )
        )
        return PredictResponse(
            dataset=model.dataset,
            subset=model.subset,
            model_name=model.model_name,
            rul_cap=model.rul_cap,
            predictions=[prediction.to_dict() for prediction in predictions],
            monitoring=monitoring,
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> str:
        app.state.security.enforce(request)
        return app.state.metrics.prometheus_text()

    return app


def _verify_model_artifact_sha256(path: str | Path, expected_sha256: str) -> str:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{MODEL_SHA256_ENV} must be a 64-character hex SHA-256 digest")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            "model artifact sha256 mismatch: "
            f"expected {expected}, got {actual} for {Path(path)}"
        )
    return actual


def _configured_rate_limit(rate_limit_per_minute: int | None) -> int:
    if rate_limit_per_minute is not None:
        if rate_limit_per_minute < 0:
            raise ValueError(
                "rate_limit_per_minute must be greater than or equal to 0"
            )
        return rate_limit_per_minute

    value = os.getenv(RATE_LIMIT_ENV, "0")
    try:
        rate_limit = int(value)
    except ValueError as exc:
        raise ValueError(f"{RATE_LIMIT_ENV} must be an integer") from exc
    if rate_limit < 0:
        raise ValueError(f"{RATE_LIMIT_ENV} must be greater than or equal to 0")
    return rate_limit


def _api_key_from_request(request: Request) -> str | None:
    header_key = request.headers.get("x-api-key")
    if header_key:
        return header_key
    authorization = request.headers.get("authorization")
    scheme, _, token = authorization.partition(" ") if authorization else ("", "", "")
    if scheme.lower() == "bearer" and token:
        return token
    return None


def _readiness_payload(
    artifact: CmapssHgbPolicyModelArtifact | None,
    artifact_sha256: str | None,
) -> dict[str, Any]:
    loaded = artifact is not None
    payload: dict[str, Any] = {
        "status": "ready" if loaded else "missing_model",
        "model_loaded": loaded,
    }
    if artifact is not None:
        payload["model"] = {
            "schema_version": artifact.schema_version,
            "dataset": artifact.dataset,
            "subset": artifact.subset,
            "model_name": artifact.model_name,
            "artifact_id": artifact.promotion_metadata.get("artifact_id"),
            "artifact_sha256": artifact_sha256,
            "stage": artifact.promotion_metadata.get("stage"),
        }
    return payload


def _inference_schema_payload(
    model: CmapssHgbPolicyModelArtifact,
    artifact_sha256: str | None,
) -> dict[str, Any]:
    return {
        "dataset": model.dataset,
        "subset": model.subset,
        "model_name": model.model_name,
        "artifact_id": model.promotion_metadata.get("artifact_id"),
        "artifact_sha256": artifact_sha256,
        "request": {
            "content_type": "application/json",
            "body_field": "telemetry",
            "min_rows": 1,
            "max_rows": 10000,
            "row_columns": [
                _inference_column_schema(column) for column in model.input_columns
            ],
            "unit_grouping": (
                "rows are grouped by unit_number; one prediction is returned per unit "
                "using the latest time_in_cycles row"
            ),
        },
        "response": {
            "prediction_fields": [
                {"name": "unit_number", "type": "integer"},
                {
                    "name": "predicted_rul",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": float(model.rul_cap),
                },
                {
                    "name": "predicted_rul_lower",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": float(model.rul_cap),
                    "nullable": True,
                },
                {
                    "name": "predicted_rul_upper",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": float(model.rul_cap),
                    "nullable": True,
                },
                {"name": "interval_method", "type": "string", "nullable": True},
                {"name": "interval_quantile_level", "type": "number", "nullable": True},
            ],
            "monitoring_fields": ["telemetry", "predictions"],
        },
    }


def _inference_column_schema(column: str) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "name": column,
        "type": "integer" if column in {"unit_number", "time_in_cycles"} else "number",
        "required": True,
        "nullable": False,
    }
    if column == "time_in_cycles":
        schema["minimum"] = 1
    return schema


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _int_metric_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_metric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path or request.url.path)


def _log_request(
    request: Request,
    request_id: str,
    status_code: int,
    latency_seconds: float,
    route_path: str,
) -> None:
    LOGGER.info(
        json.dumps(
            {
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": route_path,
                "status_code": status_code,
                "latency_ms": round(latency_seconds * 1000, 3),
            },
            sort_keys=True,
        )
    )


def _require_artifact(
    artifact: CmapssHgbPolicyModelArtifact | None,
) -> CmapssHgbPolicyModelArtifact:
    if artifact is None:
        raise HTTPException(status_code=503, detail="model artifact is not loaded")
    return artifact


app = create_app()
