"""FastAPI application for deployed C-MAPSS RUL inference."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from aerospace_prognostics.deployment.artifacts import (
    CmapssHgbPolicyModelArtifact,
    load_cmapss_model_artifact,
)

LOGGER = logging.getLogger("aerospace_prognostics.serving")


class PredictRequest(BaseModel):
    """Raw telemetry rows to score."""

    telemetry: list[dict[str, float | int]] = Field(..., min_length=1, max_length=10000)


class PredictResponse(BaseModel):
    """Prediction response for one request."""

    dataset: str
    subset: str
    model_name: str
    rul_cap: int
    predictions: list[dict[str, float | int]]
    monitoring: dict[str, Any]


class ServingMetrics:
    """In-memory request counters for lightweight container smoke and local serving."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._latency_seconds_sum = 0.0
        self._latency_seconds_max = 0.0
        self._status_counts: dict[tuple[str, str, int], int] = defaultdict(int)

    def record(self, method: str, path: str, status_code: int, latency_seconds: float) -> None:
        with self._lock:
            self._request_count += 1
            self._latency_seconds_sum += latency_seconds
            self._latency_seconds_max = max(self._latency_seconds_max, latency_seconds)
            self._status_counts[(method, path, status_code)] += 1

    def prometheus_text(self) -> str:
        with self._lock:
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
            return "\n".join(lines) + "\n"


def create_app(artifact_path: str | Path | None = None) -> FastAPI:
    """Create a FastAPI app, optionally loading a model artifact at startup."""

    app = FastAPI(
        title="Aerospace Prognostics API",
        version="0.1.0",
        description="Deployment API for C-MAPSS remaining useful life prediction.",
    )
    configured_path = artifact_path or os.getenv("AEROSPACE_PROGNOSTICS_MODEL_PATH")
    artifact = (
        load_cmapss_model_artifact(configured_path)
        if configured_path is not None and str(configured_path)
        else None
    )
    app.state.artifact = artifact
    app.state.artifact_path = str(configured_path) if configured_path is not None else None
    app.state.metrics = ServingMetrics()

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

    @app.get("/version")
    def version() -> dict[str, Any]:
        model = _require_artifact(app.state.artifact)
        return model.metadata()

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model = _require_artifact(app.state.artifact)
        frame = pd.DataFrame(request.telemetry)
        try:
            predictions = model.predict_from_frame(frame)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        monitoring = model.monitoring_summary(frame, predictions)
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
    def metrics() -> str:
        return app.state.metrics.prometheus_text()

    return app


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
