from __future__ import annotations

import pandas as pd

from aerospace_prognostics.app.api_client import ApiEndpointStatus, ApiServiceStatus
from aerospace_prognostics.app.dashboard_state import (
    load_quickstart_workspace,
    predict_cmapss_telemetry,
)
from aerospace_prognostics.app.streamlit_app import (
    _artifact_prediction_runs_frame,
    _audit_events_frame,
    _decision_status_index,
    _failed_gates_frame,
    _float_display,
    _model_artifacts_frame,
    _percent_display,
    _prediction_runs_frame,
    _release_evidence_frame,
    _telemetry_records,
    _with_api_artifact_metadata,
)
from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.deployment.artifacts import (
    save_cmapss_model_artifact,
    train_cmapss_hgb_policy_artifact,
)
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_load_quickstart_workspace_reports_missing_artifacts(tmp_path) -> None:
    workspace = load_quickstart_workspace(tmp_path / "missing")

    assert workspace.is_ready is False
    assert workspace.dashboard_payload is None
    assert workspace.artifact_inspection is None
    assert workspace.model_artifact_path.name == "fd001.joblib"
    assert len(workspace.missing_paths) >= 5


def test_predict_cmapss_telemetry_returns_predictions_and_monitoring(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)
    artifact_path = save_cmapss_model_artifact(
        packaged.artifact,
        tmp_path / "models" / "fd001.joblib",
    )
    telemetry = load_cmapss_subset(tmp_path, "FD001").test

    prediction_document = predict_cmapss_telemetry(artifact_path, telemetry)

    assert prediction_document["dataset"] == "C-MAPSS"
    assert prediction_document["subset"] == "FD001"
    assert prediction_document["artifact"]["artifact_id"].startswith("fd001-")
    assert len(prediction_document["predictions"]) == 2
    assert prediction_document["predictions"][0]["predicted_rul_lower"] is not None
    assert prediction_document["monitoring"]["predictions"]["count"] == 2


def test_telemetry_records_preserves_rows_for_api_payload() -> None:
    records = _telemetry_records(
        pd.DataFrame([{"unit_number": 1, "time_in_cycles": 2, "sensor_1": 3.5}])
    )

    assert records == [{"unit_number": 1, "time_in_cycles": 2, "sensor_1": 3.5}]


def test_with_api_artifact_metadata_adds_readiness_identity() -> None:
    api_status = ApiServiceStatus(
        base_url="http://api:8000",
        health=ApiEndpointStatus(ok=True, status_code=200, payload={}),
        readiness=ApiEndpointStatus(
            ok=True,
            status_code=200,
            payload={
                "model": {
                    "artifact_id": "fd001-demo",
                    "artifact_sha256": "abc123",
                    "stage": "candidate",
                }
            },
        ),
    )

    enriched = _with_api_artifact_metadata({"predictions": []}, api_status)

    assert enriched["artifact"]["artifact_id"] == "fd001-demo"
    assert enriched["artifact"]["artifact_sha256"] == "abc123"


def test_audit_events_frame_preserves_operator_event_columns() -> None:
    frame = _audit_events_frame(
        [
            {
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "event_type": "operator_decision",
                "status": "watch",
                "actor": "flight-ops",
                "note": "Monitor next cycle",
                "payload": {"ticket": "PHM-42"},
            }
        ]
    )

    assert list(frame.columns) == ["created_at_utc", "event_type", "status", "actor", "note"]
    assert frame.iloc[0]["status"] == "watch"
    assert frame.iloc[0]["actor"] == "flight-ops"


def test_registry_frames_preserve_artifact_evidence_and_usage_columns() -> None:
    artifacts = _model_artifacts_frame(
        [
            {
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "artifact_id": "fd001-demo",
                "stage": "candidate",
                "dataset": "C-MAPSS",
                "subset": "FD001",
                "model_name": "hgb",
                "schema_version": "1.0",
                "evidence_count": 5,
                "prediction_run_count": 2,
                "latest_prediction_at": "2026-01-01T00:01:00+00:00",
            }
        ]
    )
    evidence = _release_evidence_frame(
        [
            {
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "evidence_type": "release_bundle",
                "status": "ok",
                "source_path": "release.json",
                "payload": {"status": "ok"},
            }
        ]
    )
    usage = _artifact_prediction_runs_frame(
        [
            {
                "created_at_utc": "2026-01-01T00:01:00+00:00",
                "run_id": "run-1",
                "source_name": "telemetry.csv",
                "prediction_count": 2,
                "interval_availability_rate": 1.0,
                "mean_interval_width": 18.5,
                "outcome_count": 2,
                "outcome_interval_coverage_rate": 0.5,
                "mean_absolute_error": 7.5,
                "content_sha256": "abc123",
            }
        ]
    )

    assert list(artifacts.columns) == [
        "created_at_utc",
        "artifact_id",
        "stage",
        "dataset",
        "subset",
        "model_name",
        "schema_version",
        "evidence_count",
        "prediction_run_count",
        "latest_prediction_at",
    ]
    assert evidence.iloc[0]["evidence_type"] == "release_bundle"
    assert usage.iloc[0]["run_id"] == "run-1"
    assert list(usage.columns) == [
        "created_at_utc",
        "run_id",
        "source_name",
        "prediction_count",
        "interval_availability_rate",
        "mean_interval_width",
        "outcome_count",
        "outcome_interval_coverage_rate",
        "mean_absolute_error",
        "content_sha256",
    ]
    assert usage.iloc[0]["interval_availability_rate"] == 100.0
    assert usage.iloc[0]["mean_interval_width"] == 18.5
    assert usage.iloc[0]["outcome_interval_coverage_rate"] == 50.0
    assert usage.iloc[0]["mean_absolute_error"] == 7.5


def test_prediction_runs_frame_formats_interval_availability() -> None:
    frame = _prediction_runs_frame(
        [
            {
                "created_at_utc": "2026-01-01T00:01:00+00:00",
                "run_id": "run-1",
                "source_name": "telemetry.csv",
                "model_name": "hgb",
                "prediction_count": 2,
                "min_predicted_rul": 10.0,
                "mean_predicted_rul": 15.0,
                "max_predicted_rul": 20.0,
                "interval_availability_rate": 0.5,
                "mean_interval_width": 18.5,
                "outcome_count": 1,
                "outcome_interval_coverage_rate": 1.0,
                "mean_absolute_error": 5.0,
                "drift_alert_count": 1,
                "decision_status": "watch",
                "audit_event_count": 2,
            }
        ]
    )

    assert frame.iloc[0]["interval_availability_rate"] == 50.0
    assert frame.iloc[0]["mean_interval_width"] == 18.5
    assert frame.iloc[0]["outcome_interval_coverage_rate"] == 100.0
    assert frame.iloc[0]["mean_absolute_error"] == 5.0


def test_display_helpers_format_optional_interval_metrics() -> None:
    assert _percent_display(0.875) == "88%"
    assert _percent_display(None) == "n/a"
    assert _float_display(12.345) == "12.3"
    assert _float_display(None) == "n/a"


def test_failed_gates_frame_formats_gate_names() -> None:
    frame = _failed_gates_frame(["promotion.latency_benchmark"])

    assert list(frame.columns) == ["gate"]
    assert frame.iloc[0]["gate"] == "promotion.latency_benchmark"


def test_decision_status_index_defaults_to_review_required() -> None:
    assert _decision_status_index("accepted") == 1
    assert _decision_status_index(None) == 0
    assert _decision_status_index("unknown") == 0
