"""FastAPI application for deployed C-MAPSS RUL inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aerospace_prognostics.deployment.artifacts import (
    CmapssHgbPolicyModelArtifact,
    load_cmapss_model_artifact,
)


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
        return PredictResponse(
            dataset=model.dataset,
            subset=model.subset,
            model_name=model.model_name,
            rul_cap=model.rul_cap,
            predictions=[prediction.to_dict() for prediction in predictions],
        )

    return app


def _require_artifact(
    artifact: CmapssHgbPolicyModelArtifact | None,
) -> CmapssHgbPolicyModelArtifact:
    if artifact is None:
        raise HTTPException(status_code=503, detail="model artifact is not loaded")
    return artifact


app = create_app()
