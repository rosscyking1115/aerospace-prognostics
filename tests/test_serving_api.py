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
    ready = client.get("/ready")
    version = client.get("/version")
    response = client.post(
        "/predict",
        headers={"x-request-id": "test-request-1"},
        json={"telemetry": telemetry},
    )
    metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    ready_payload = ready.json()
    assert ready_payload["status"] == "ready"
    assert ready_payload["model_loaded"] is True
    assert ready_payload["model"] == {
        "schema_version": artifact.schema_version,
        "dataset": "C-MAPSS",
        "subset": "FD001",
        "model_name": artifact.model_name,
        "artifact_id": artifact.promotion_metadata["artifact_id"],
        "stage": "candidate",
    }
    assert version.status_code == 200
    assert version.json()["subset"] == "FD001"
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "test-request-1"
    assert float(response.headers["x-process-time-ms"]) >= 0.0
    payload = response.json()
    assert payload["model_name"] == artifact.model_name
    assert [prediction["unit_number"] for prediction in payload["predictions"]] == [1, 2]
    assert payload["monitoring"]["predictions"]["count"] == 2
    assert "sensor_1" in payload["monitoring"]["telemetry"]["columns"]
    assert metrics.status_code == 200
    assert "aerospace_prognostics_requests_total 4" in metrics.text
    assert (
        'aerospace_prognostics_http_responses_total{method="POST",path="/predict",'
        'status_code="200"} 1'
    ) in metrics.text


def test_serving_api_exposes_model_specific_inference_schema(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1).artifact
    artifact_path = save_cmapss_model_artifact(artifact, tmp_path / "fd001.joblib")
    client = TestClient(create_app(artifact_path))

    response = client.get("/schema")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "C-MAPSS"
    assert payload["subset"] == "FD001"
    assert payload["artifact_id"] == artifact.promotion_metadata["artifact_id"]
    assert payload["request"]["body_field"] == "telemetry"
    assert payload["request"]["min_rows"] == 1
    assert payload["request"]["max_rows"] == 10000
    assert [column["name"] for column in payload["request"]["row_columns"]] == list(
        artifact.input_columns
    )
    assert payload["request"]["row_columns"][0] == {
        "name": "unit_number",
        "type": "integer",
        "required": True,
        "nullable": False,
    }
    assert payload["request"]["row_columns"][1] == {
        "name": "time_in_cycles",
        "type": "integer",
        "required": True,
        "nullable": False,
        "minimum": 1,
    }
    assert payload["response"]["prediction_fields"] == [
        {"name": "unit_number", "type": "integer"},
        {
            "name": "predicted_rul",
            "type": "number",
            "minimum": 0.0,
            "maximum": float(artifact.rul_cap),
        },
    ]


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


def test_serving_api_enforces_optional_api_key(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1).artifact
    artifact_path = save_cmapss_model_artifact(artifact, tmp_path / "fd001.joblib")
    client = TestClient(create_app(artifact_path, api_key="test-secret"))

    health = client.get("/health")
    ready = client.get("/ready")
    missing_key = client.get("/version")
    bad_key = client.get("/metrics", headers={"x-api-key": "wrong"})
    schema_missing_key = client.get("/schema")
    valid_key = client.get("/version", headers={"authorization": "Bearer test-secret"})
    schema_valid_key = client.get("/schema", headers={"x-api-key": "test-secret"})

    assert health.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["model"]["artifact_id"] == artifact.promotion_metadata["artifact_id"]
    assert missing_key.status_code == 401
    assert missing_key.headers["www-authenticate"] == "ApiKey"
    assert bad_key.status_code == 401
    assert schema_missing_key.status_code == 401
    assert valid_key.status_code == 200
    assert valid_key.json()["subset"] == "FD001"
    assert schema_valid_key.status_code == 200
    assert schema_valid_key.json()["artifact_id"] == artifact.promotion_metadata["artifact_id"]


def test_serving_api_enforces_optional_rate_limit(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1).artifact
    artifact_path = save_cmapss_model_artifact(artifact, tmp_path / "fd001.joblib")
    client = TestClient(
        create_app(artifact_path, api_key="test-secret", rate_limit_per_minute=1)
    )
    bundle = load_cmapss_subset(tmp_path, "FD001")
    telemetry = json.loads(bundle.test.to_json(orient="records"))

    first = client.post(
        "/predict",
        headers={"x-api-key": "test-secret"},
        json={"telemetry": telemetry},
    )
    second = client.post(
        "/predict",
        headers={"x-api-key": "test-secret"},
        json={"telemetry": telemetry},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"
    assert second.json()["detail"] == "rate limit exceeded"


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


def test_serving_api_logs_prediction_monitoring_summary(tmp_path, caplog) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1).artifact
    artifact_path = save_cmapss_model_artifact(artifact, tmp_path / "fd001.joblib")
    bundle = load_cmapss_subset(tmp_path, "FD001")
    telemetry = bundle.test.copy()
    for column in [column for column in telemetry.columns if column.startswith("sensor_")]:
        telemetry[column] = telemetry[column] + 100.0
    client = TestClient(create_app(artifact_path))

    with caplog.at_level(logging.INFO, logger="aerospace_prognostics.serving"):
        response = client.post(
            "/predict",
            json={"telemetry": json.loads(telemetry.to_json(orient="records"))},
        )

    assert response.status_code == 200
    monitoring = response.json()["monitoring"]
    assert monitoring["telemetry"]["alert_column_count"] >= 21
    payloads = [json.loads(record.message) for record in caplog.records]
    prediction_logs = [
        payload for payload in payloads if payload.get("event") == "prediction_monitoring"
    ]
    assert prediction_logs[-1]["prediction_count"] == 2
    assert prediction_logs[-1]["telemetry_alert_column_count"] >= 21


def test_serving_api_reports_missing_model_and_metrics() -> None:
    client = TestClient(create_app())

    health = client.get("/health")
    ready = client.get("/ready")
    predict = client.post("/predict", json={"telemetry": [{"unit_number": 1}]})
    metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json() == {"status": "missing_model", "model_loaded": False}
    assert ready.status_code == 503
    assert ready.json() == {"status": "missing_model", "model_loaded": False}
    assert predict.status_code == 503
    assert (
        'aerospace_prognostics_http_responses_total{method="GET",path="/ready",'
        'status_code="503"} 1'
    ) in metrics.text
    assert (
        'aerospace_prognostics_http_responses_total{method="POST",path="/predict",'
        'status_code="503"} 1'
    ) in metrics.text
