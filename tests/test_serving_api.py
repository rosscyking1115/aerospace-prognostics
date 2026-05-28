from __future__ import annotations

import json

from fastapi.testclient import TestClient

from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.deployment.artifacts import (
    save_cmapss_model_artifact,
    train_cmapss_hgb_policy_artifact,
)
from aerospace_prognostics.serving.api import create_app
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_serving_api_health_version_and_predict(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1).artifact
    artifact_path = save_cmapss_model_artifact(artifact, tmp_path / "fd001.joblib")
    client = TestClient(create_app(artifact_path))
    bundle = load_cmapss_subset(tmp_path, "FD001")
    telemetry = json.loads(bundle.test.to_json(orient="records"))

    health = client.get("/health")
    version = client.get("/version")
    response = client.post("/predict", json={"telemetry": telemetry})

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert version.status_code == 200
    assert version.json()["subset"] == "FD001"
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == artifact.model_name
    assert [prediction["unit_number"] for prediction in payload["predictions"]] == [1, 2]


def test_serving_api_reports_validation_errors(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1).artifact
    artifact_path = save_cmapss_model_artifact(artifact, tmp_path / "fd001.joblib")
    client = TestClient(create_app(artifact_path))

    response = client.post(
        "/predict",
        json={"telemetry": [{"unit_number": 1, "time_in_cycles": 1}]},
    )

    assert response.status_code == 422
    assert "missing columns" in response.json()["detail"]
