from __future__ import annotations

import json
import sqlite3

from aerospace_prognostics.app.dashboard_state import QuickstartWorkspace
from aerospace_prognostics.app.store import (
    SCHEMA_VERSION,
    database_summary,
    initialize_app_database,
    list_prediction_runs,
    load_prediction_run,
    record_prediction_run,
    seed_quickstart_workspace,
)
from aerospace_prognostics.cli import main
from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.deployment.artifacts import (
    save_cmapss_model_artifact,
    train_cmapss_hgb_policy_artifact,
)
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_initialize_app_database_creates_schema(tmp_path) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")

    summary = database_summary(database_path)

    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["model_artifacts"] == 0
    assert summary["prediction_runs"] == 0


def test_app_init_db_command_creates_database_without_seed(tmp_path, capsys) -> None:
    database_path = tmp_path / "app.sqlite"

    exit_code = main(
        [
            "app-init-db",
            "--database",
            str(database_path),
            "--no-seed-quickstart",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert database_path.exists()
    assert f"database={database_path}" in output
    assert f"schema_version={SCHEMA_VERSION}" in output


def test_seed_quickstart_workspace_persists_model_and_evidence(tmp_path) -> None:
    workspace = _write_fake_workspace(tmp_path / "quickstart")
    database_path = tmp_path / "app.sqlite"

    inserted = seed_quickstart_workspace(database_path, workspace)
    second_insert = seed_quickstart_workspace(database_path, workspace)
    summary = database_summary(database_path)

    assert inserted["model_artifacts"] == 1
    assert inserted["release_evidence"] == 5
    assert second_insert["model_artifacts"] == 1
    assert second_insert["release_evidence"] == 0
    assert summary["model_artifacts"] == 1
    assert summary["release_evidence"] == 5


def test_record_prediction_run_persists_upload_run_and_prediction_rows(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)
    artifact_path = save_cmapss_model_artifact(
        packaged.artifact,
        tmp_path / "models" / "fd001.joblib",
    )
    telemetry = load_cmapss_subset(tmp_path, "FD001").test
    predictions = packaged.artifact.predict_from_frame(telemetry)
    prediction_document = {
        "dataset": packaged.artifact.dataset,
        "subset": packaged.artifact.subset,
        "model_name": packaged.artifact.model_name,
        "predictions": [prediction.to_dict() for prediction in predictions],
        "artifact": {
            "artifact_id": packaged.artifact.promotion_metadata["artifact_id"],
        },
        "monitoring": packaged.artifact.monitoring_summary(telemetry, predictions),
    }
    database_path = initialize_app_database(tmp_path / "app.sqlite")

    run_id = record_prediction_run(
        database_path,
        telemetry=telemetry,
        prediction_document=prediction_document,
        model_artifact_path=artifact_path,
        source_name="test.csv",
    )
    summary = database_summary(database_path)

    assert run_id.startswith("run-")
    assert summary["telemetry_uploads"] == 1
    assert summary["prediction_runs"] == 1
    assert summary["predictions"] == 2
    with sqlite3.connect(database_path) as connection:
        stored_run_id = connection.execute("select run_id from prediction_runs").fetchone()[0]
    assert stored_run_id == run_id


def test_prediction_run_history_loads_recent_runs_and_predictions(tmp_path) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)

    runs = list_prediction_runs(database_path)
    loaded = load_prediction_run(database_path, run_id)

    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["source_name"] == "test.csv"
    assert runs[0]["prediction_count"] == 2
    assert runs[0]["min_predicted_rul"] <= runs[0]["max_predicted_rul"]
    assert runs[0]["interval_count"] == 2
    assert runs[0]["drift_alert_count"] == 0
    assert loaded is not None
    assert loaded["run"]["run_id"] == run_id
    assert loaded["run"]["source_name"] == "test.csv"
    assert len(loaded["predictions"]) == 2
    assert loaded["predictions"][0]["predicted_rul"] <= loaded["predictions"][1]["predicted_rul"]


def test_load_prediction_run_returns_none_for_unknown_run(tmp_path) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")

    loaded = load_prediction_run(database_path, "run-missing")

    assert loaded is None


def _write_prediction_run(tmp_path):
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)
    artifact_path = save_cmapss_model_artifact(
        packaged.artifact,
        tmp_path / "models" / "fd001.joblib",
    )
    telemetry = load_cmapss_subset(tmp_path, "FD001").test
    predictions = packaged.artifact.predict_from_frame(telemetry)
    prediction_document = {
        "dataset": packaged.artifact.dataset,
        "subset": packaged.artifact.subset,
        "model_name": packaged.artifact.model_name,
        "predictions": [prediction.to_dict() for prediction in predictions],
        "artifact": {
            "artifact_id": packaged.artifact.promotion_metadata["artifact_id"],
        },
        "monitoring": packaged.artifact.monitoring_summary(telemetry, predictions),
    }
    database_path = initialize_app_database(tmp_path / "app.sqlite")
    run_id = record_prediction_run(
        database_path,
        telemetry=telemetry,
        prediction_document=prediction_document,
        model_artifact_path=artifact_path,
        source_name="test.csv",
    )
    return database_path, run_id


def _write_fake_workspace(root):
    root.mkdir(parents=True)
    model_dir = root / "models"
    release_dir = root / "release"
    dashboard_dir = root / "dashboard"
    predictions_dir = root / "predictions"
    model_dir.mkdir()
    release_dir.mkdir()
    dashboard_dir.mkdir()
    predictions_dir.mkdir()
    model_artifact_path = model_dir / "fd001.joblib"
    model_artifact_path.write_bytes(b"model")
    telemetry_csv_path = predictions_dir / "fd001_input.csv"
    telemetry_csv_path.write_text("unit_number,time_in_cycles\n1,1\n", encoding="utf-8")
    inspection = {
        "artifact_identity": {
            "artifact_id": "fd001-demo",
            "schema_version": "1.2",
        },
        "model": {
            "dataset": "C-MAPSS",
            "subset": "FD001",
            "model_name": "hist_gradient_boosting",
        },
        "promotion": {"stage": "candidate"},
    }
    release_bundle = {"status": "ok", "release_name": "fd001-demo"}
    provenance = {"status": "ok", "summary": {"workflow": "local"}}
    promotion = {"status": "ok", "artifact_identity": {"artifact_id": "fd001-demo"}}
    dashboard_payload = {"schema_version": "aerospace-prognostics/fleet-dashboard/v1"}
    artifact_inspection_path = model_dir / "fd001_inspection.json"
    release_bundle_path = release_dir / "fd001_release_bundle.json"
    provenance_path = release_dir / "fd001_provenance.json"
    promotion_report_path = model_dir / "fd001_promotion.json"
    dashboard_payload_path = dashboard_dir / "fleet_payload.json"
    for path, payload in (
        (artifact_inspection_path, inspection),
        (release_bundle_path, release_bundle),
        (provenance_path, provenance),
        (promotion_report_path, promotion),
        (dashboard_payload_path, dashboard_payload),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    return QuickstartWorkspace(
        root=root,
        model_artifact_path=model_artifact_path,
        telemetry_csv_path=telemetry_csv_path,
        dashboard_payload_path=dashboard_payload_path,
        artifact_inspection_path=artifact_inspection_path,
        release_bundle_path=release_bundle_path,
        provenance_path=provenance_path,
        promotion_report_path=promotion_report_path,
        dashboard_payload=dashboard_payload,
        artifact_inspection=inspection,
        release_bundle=release_bundle,
        provenance=provenance,
        promotion_report=promotion,
        missing_paths=(),
    )
