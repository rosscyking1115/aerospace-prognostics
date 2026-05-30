from __future__ import annotations

import json
import logging

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
    response = client.post(
        "/predict",
        headers={"x-request-id": "test-request-1"},
        json={"telemetry": telemetry},
    )
    metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert version.status_code == 200
    assert version.json()["subset"] == "FD001"
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "test-request-1"
    assert float(response.headers["x-process-time-ms"]) >= 0.0
    payload = response.json()
    assert payload["model_name"] == artifact.model_name
    assert [prediction["unit_number"] for prediction in payload["predictions"]] == [1, 2]
    assert metrics.status_code == 200
    assert "aerospace_prognostics_requests_total 3" in metrics.text
    assert (
        'aerospace_prognostics_http_responses_total{method="POST",path="/predict",'
        'status_code="200"} 1'
    ) in metrics.text


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


def test_serving_api_logs_structured_request_records(tmp_path, caplog) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1).artifact
    artifact_path = save_cmapss_model_artifact(artifact, tmp_path / "fd001.joblib")
    client = TestClient(create_app(artifact_path))

    with caplog.at_level(logging.INFO, logger="aerospace_prognostics.serving"):
        response = client.get("/health", headers={"x-request-id": "log-test"})

    assert response.status_code == 200
    payloads = [json.loads(record.message) for record in caplog.records]
    assert {
        "event": "http_request",
        "request_id": "log-test",
        "method": "GET",
        "path": "/health",
        "status_code": 200,
    }.items() <= payloads[-1].items()


def test_serving_api_reports_missing_model_and_metrics() -> None:
    client = TestClient(create_app())

    health = client.get("/health")
    predict = client.post("/predict", json={"telemetry": [{"unit_number": 1}]})
    metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json() == {"status": "missing_model", "model_loaded": False}
    assert predict.status_code == 503
    assert (
        'aerospace_prognostics_http_responses_total{method="POST",path="/predict",'
        'status_code="503"} 1'
    ) in metrics.text
