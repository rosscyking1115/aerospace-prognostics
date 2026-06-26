from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

import aerospace_prognostics.app.store as store
from aerospace_prognostics.app.dashboard_state import QuickstartWorkspace
from aerospace_prognostics.app.store import (
    SCHEMA_VERSION,
    database_summary,
    export_prediction_run_evidence,
    initialize_app_database,
    list_model_artifacts,
    list_prediction_run_events,
    list_prediction_runs,
    load_model_artifact,
    load_prediction_run,
    record_prediction_outcomes,
    record_prediction_run,
    record_prediction_run_event,
    register_model_artifact_evidence,
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
    assert summary["prediction_outcomes"] == 0


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
    assert "prediction_outcomes=0" in output


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


def test_register_model_artifact_evidence_persists_custom_artifact(tmp_path) -> None:
    workspace = _write_fake_workspace(tmp_path / "custom_artifact")
    database_path = tmp_path / "app.sqlite"

    result = register_model_artifact_evidence(
        database_path,
        model_artifact_path=workspace.model_artifact_path,
        inspection=workspace.artifact_inspection,
        inspection_source_path=workspace.artifact_inspection_path,
        release_evidence=(
            ("release_bundle", workspace.release_bundle_path, workspace.release_bundle),
            ("promotion_report", workspace.promotion_report_path, workspace.promotion_report),
        ),
    )
    loaded = load_model_artifact(database_path, "fd001-demo")
    summary = database_summary(database_path)

    assert result["artifact_id"] == "fd001-demo"
    assert result["model_artifacts"] == 1
    assert result["release_evidence"] == 3
    assert summary["model_artifacts"] == 1
    assert summary["release_evidence"] == 3
    assert loaded is not None
    assert loaded["artifact"]["artifact_id"] == "fd001-demo"
    assert loaded["artifact"]["artifact_sha256"] is not None
    assert {
        evidence["evidence_type"] for evidence in loaded["release_evidence"]
    } == {"artifact_inspection", "promotion_report", "release_bundle"}


def test_app_register_artifact_command_persists_custom_evidence(
    tmp_path,
    capsys,
) -> None:
    workspace = _write_fake_workspace(tmp_path / "custom_artifact")
    database_path = tmp_path / "app.sqlite"

    exit_code = main(
        [
            "app-register-artifact",
            "--database",
            str(database_path),
            "--model-artifact",
            str(workspace.model_artifact_path),
            "--inspection-json",
            str(workspace.artifact_inspection_path),
            "--release-bundle-json",
            str(workspace.release_bundle_path),
            "--provenance-json",
            str(workspace.provenance_path),
            "--promotion-json",
            str(workspace.promotion_report_path),
            "--dashboard-payload-json",
            str(workspace.dashboard_payload_path),
        ]
    )
    output = capsys.readouterr().out
    loaded = load_model_artifact(database_path, "fd001-demo")

    assert exit_code == 0
    assert f"database={database_path}" in output
    assert "artifact_id=fd001-demo" in output
    assert f"model_artifact={workspace.model_artifact_path}" in output
    assert "model_artifacts_registered=1" in output
    assert "release_evidence_registered=5" in output
    assert "model_artifacts=1" in output
    assert "release_evidence=5" in output
    assert loaded is not None
    assert len(loaded["release_evidence"]) == 5


def test_model_registry_lists_artifacts_evidence_and_prediction_usage(tmp_path) -> None:
    workspace = _write_fake_workspace(tmp_path / "quickstart")
    database_path = tmp_path / "app.sqlite"
    seed_quickstart_workspace(database_path, workspace)
    run_id = _write_prediction_run_for_artifact(
        tmp_path,
        database_path=database_path,
        artifact_id="fd001-demo",
    )

    artifacts = list_model_artifacts(database_path)
    loaded = load_model_artifact(database_path, "fd001-demo")

    assert len(artifacts) == 1
    assert artifacts[0]["artifact_id"] == "fd001-demo"
    assert artifacts[0]["evidence_count"] == 5
    assert artifacts[0]["prediction_run_count"] == 1
    assert artifacts[0]["latest_prediction_at"] is not None
    assert loaded is not None
    assert loaded["artifact"]["artifact_id"] == "fd001-demo"
    assert loaded["artifact"]["inspection"]["model"]["subset"] == "FD001"
    assert len(loaded["release_evidence"]) == 5
    assert loaded["prediction_runs"][0]["run_id"] == run_id
    assert loaded["report_card"]["artifact_id"] == "fd001-demo"
    assert loaded["report_card"]["gate_count"] == 4
    assert loaded["report_card"]["passed_gate_count"] == 3
    assert loaded["report_card"]["failed_gates"] == ["promotion.latency_benchmark"]
    assert loaded["report_card"]["p95_latency_ms"] == 42.0
    assert loaded["report_card"]["max_p95_latency_ms"] == 25.0
    assert loaded["report_card"]["interval_diagnostic_kind"] == (
        "operational_interval_availability"
    )
    assert loaded["report_card"]["prediction_count_total"] == 2
    assert loaded["report_card"]["interval_count_total"] == 2
    assert loaded["report_card"]["missing_interval_count"] == 0
    assert loaded["report_card"]["interval_availability_rate"] == 1.0
    assert loaded["report_card"]["interval_complete"] is True
    assert loaded["report_card"]["mean_interval_width"] is not None
    assert loaded["report_card"]["max_interval_width"] is not None
    assert loaded["report_card"]["provenance_workflow"] == "local"


def test_read_only_queries_use_existing_database_without_initializing(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = _write_fake_workspace(tmp_path / "quickstart")
    database_path = tmp_path / "app.sqlite"
    seed_quickstart_workspace(database_path, workspace)
    run_id = _write_prediction_run_for_artifact(
        tmp_path,
        database_path=database_path,
        artifact_id="fd001-demo",
    )

    def fail_initialize(path):
        raise AssertionError(f"unexpected write initializer call for {path}")

    monkeypatch.setattr(store, "initialize_app_database", fail_initialize)

    summary = store.database_summary(database_path, read_only=True)
    artifacts = store.list_model_artifacts(database_path, read_only=True)
    loaded_artifact = store.load_model_artifact(
        database_path,
        "fd001-demo",
        read_only=True,
    )
    runs = store.list_prediction_runs(database_path, read_only=True)
    loaded_run = store.load_prediction_run(database_path, run_id, read_only=True)
    events = store.list_prediction_run_events(database_path, run_id, read_only=True)

    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["model_artifacts"] == 1
    assert artifacts[0]["artifact_id"] == "fd001-demo"
    assert loaded_artifact is not None
    assert loaded_artifact["artifact"]["artifact_id"] == "fd001-demo"
    assert runs[0]["run_id"] == run_id
    assert loaded_run is not None
    assert loaded_run["run"]["run_id"] == run_id
    assert events[0]["event_type"] == "prediction_recorded"


def test_read_only_summary_requires_existing_database(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="app database not found"):
        database_summary(tmp_path / "missing.sqlite", read_only=True)


def test_load_model_artifact_returns_none_for_unknown_artifact(tmp_path) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")

    loaded = load_model_artifact(database_path, "artifact-missing")

    assert loaded is None


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
    assert summary["prediction_outcomes"] == 0
    assert summary["prediction_run_events"] == 1
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
    assert runs[0]["interval_availability_rate"] == 1.0
    assert runs[0]["mean_interval_width"] is not None
    assert runs[0]["max_interval_width"] is not None
    assert runs[0]["drift_alert_count"] == 0
    assert runs[0]["audit_event_count"] == 1
    assert runs[0]["decision_status"] is None
    assert loaded is not None
    assert loaded["run"]["run_id"] == run_id
    assert loaded["run"]["source_name"] == "test.csv"
    assert loaded["run"]["audit_event_count"] == 1
    assert len(loaded["audit_events"]) == 1
    assert loaded["audit_events"][0]["event_type"] == "prediction_recorded"
    assert len(loaded["predictions"]) == 2
    assert loaded["predictions"][0]["predicted_rul"] <= loaded["predictions"][1]["predicted_rul"]


def test_prediction_outcomes_attach_actuals_and_calibration_metrics(tmp_path) -> None:
    workspace = _write_fake_workspace(tmp_path / "quickstart")
    database_path = tmp_path / "app.sqlite"
    seed_quickstart_workspace(database_path, workspace)
    run_id = _write_prediction_run_for_artifact(
        tmp_path,
        database_path=database_path,
        artifact_id="fd001-demo",
    )
    loaded_before = load_prediction_run(database_path, run_id)
    assert loaded_before is not None
    outcome_frame = pd.DataFrame(
        {
            "unit_number": [
                row["unit_number"] for row in loaded_before["predictions"]
            ],
            "actual_rul": [
                row["predicted_rul"] for row in loaded_before["predictions"]
            ],
        }
    )

    result = record_prediction_outcomes(
        database_path,
        run_id=run_id,
        outcomes=outcome_frame,
        source_name="rul_outcomes.csv",
        actor="reliability-engineer",
        observed_at_utc="2026-01-01T00:00:00+00:00",
    )
    runs = list_prediction_runs(database_path)
    loaded_after = load_prediction_run(database_path, run_id)
    artifact = load_model_artifact(database_path, "fd001-demo")
    summary = database_summary(database_path)

    assert result["outcome_count"] == 2
    assert result["event_id"].startswith("event-")
    assert summary["prediction_outcomes"] == 2
    assert summary["prediction_run_events"] == 2
    assert runs[0]["outcome_count"] == 2
    assert runs[0]["outcome_availability_rate"] == 1.0
    assert runs[0]["mean_absolute_error"] == 0.0
    assert runs[0]["mean_signed_error"] == 0.0
    assert runs[0]["interval_outcome_count"] == 2
    assert runs[0]["interval_covered_count"] == 2
    assert runs[0]["outcome_interval_coverage_rate"] == 1.0
    assert loaded_after is not None
    assert loaded_after["predictions"][0]["actual_rul"] is not None
    assert loaded_after["predictions"][0]["absolute_error"] == 0.0
    assert loaded_after["predictions"][0]["interval_covered"] == 1
    assert loaded_after["predictions"][0]["outcome_source"] == "rul_outcomes.csv"
    assert loaded_after["audit_events"][0]["event_type"] == "outcomes_recorded"
    assert artifact is not None
    assert artifact["report_card"]["outcome_diagnostic_kind"] == (
        "observed_rul_outcome_coverage"
    )
    assert artifact["report_card"]["outcome_count_total"] == 2
    assert artifact["report_card"]["outcome_availability_rate"] == 1.0
    assert artifact["report_card"]["mean_absolute_error"] == 0.0
    assert artifact["report_card"]["mean_signed_error"] == 0.0
    assert artifact["report_card"]["interval_outcome_count_total"] == 2
    assert artifact["report_card"]["interval_covered_count_total"] == 2
    assert artifact["report_card"]["outcome_interval_coverage_rate"] == 1.0


def test_app_record_outcomes_command_attaches_observed_rul_csv(tmp_path, capsys) -> None:
    workspace = _write_fake_workspace(tmp_path / "quickstart")
    database_path = tmp_path / "app.sqlite"
    seed_quickstart_workspace(database_path, workspace)
    run_id = _write_prediction_run_for_artifact(
        tmp_path,
        database_path=database_path,
        artifact_id="fd001-demo",
    )
    loaded_before = load_prediction_run(database_path, run_id)
    assert loaded_before is not None
    outcomes_csv = tmp_path / "outcomes.csv"
    pd.DataFrame(
        {
            "unit_number": [
                row["unit_number"] for row in loaded_before["predictions"]
            ],
            "actual_rul": [
                row["predicted_rul"] for row in loaded_before["predictions"]
            ],
        }
    ).to_csv(outcomes_csv, index=False)

    exit_code = main(
        [
            "app-record-outcomes",
            "--database",
            str(database_path),
            "--run-id",
            run_id,
            "--outcomes-csv",
            str(outcomes_csv),
            "--source-name",
            "verified_outcomes.csv",
            "--actor",
            "reliability-engineer",
            "--observed-at-utc",
            "2026-01-01T00:00:00+00:00",
        ]
    )
    output = capsys.readouterr().out
    loaded_after = load_prediction_run(database_path, run_id)

    assert exit_code == 0
    assert f"database={database_path}" in output
    assert f"run_id={run_id}" in output
    assert f"outcomes_csv={outcomes_csv}" in output
    assert "outcome_count=2" in output
    assert "event_id=event-" in output
    assert "prediction_outcomes=2" in output
    assert loaded_after is not None
    assert loaded_after["predictions"][0]["actual_rul"] is not None
    assert loaded_after["audit_events"][0]["actor"] == "reliability-engineer"


def test_app_record_outcomes_command_rejects_malformed_csv(tmp_path) -> None:
    outcomes_csv = tmp_path / "bad_outcomes.csv"
    outcomes_csv.write_text('unit_number,actual_rul\n"1,12\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="outcomes CSV could not be read as a valid CSV",
    ):
        main(
            [
                "app-record-outcomes",
                "--database",
                str(tmp_path / "app.sqlite"),
                "--run-id",
                "run-1",
                "--outcomes-csv",
                str(outcomes_csv),
            ]
        )


def test_export_prediction_run_evidence_writes_json_and_prediction_csv(tmp_path) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)
    loaded_before = load_prediction_run(database_path, run_id)
    assert loaded_before is not None
    outcomes = pd.DataFrame(
        {
            "unit_number": [row["unit_number"] for row in loaded_before["predictions"]],
            "actual_rul": [row["predicted_rul"] for row in loaded_before["predictions"]],
        }
    )
    record_prediction_outcomes(
        database_path,
        run_id=run_id,
        outcomes=outcomes,
        source_name="verified_outcomes.csv",
        actor="reliability-engineer",
    )
    record_prediction_run_event(
        database_path,
        run_id=run_id,
        event_type="operator_decision",
        status="watch",
        actor="flight-ops",
        note="Review after next cycle",
    )

    result = export_prediction_run_evidence(
        database_path,
        run_id=run_id,
        output_dir=tmp_path / "exports",
    )
    manifest = json.loads(
        (tmp_path / "exports" / f"{run_id}_evidence.json").read_text(encoding="utf-8")
    )
    exported_predictions = pd.read_csv(result["predictions_csv"])

    assert result["run_id"] == run_id
    assert result["prediction_count"] == 2
    assert result["outcome_count"] == 2
    assert result["audit_event_count"] == 3
    assert result["evidence_sha256"]
    assert result["predictions_sha256"]
    assert manifest["schema_version"] == (
        "aerospace-prognostics/prediction-run-evidence/v1"
    )
    assert manifest["run"]["run_id"] == run_id
    assert manifest["files"]["predictions_csv"]["rows"] == 2
    assert len(manifest["predictions"]) == 2
    assert len(manifest["audit_events"]) == 3
    assert list(exported_predictions["unit_number"]) == [
        row["unit_number"] for row in manifest["predictions"]
    ]
    assert "actual_rul" in exported_predictions.columns


def test_app_export_run_command_writes_review_evidence(tmp_path, capsys) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)
    output_dir = tmp_path / "run_exports"

    exit_code = main(
        [
            "app-export-run",
            "--database",
            str(database_path),
            "--run-id",
            run_id,
            "--output-dir",
            str(output_dir),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"database={database_path}" in output
    assert f"run_id={run_id}" in output
    assert f"output_dir={output_dir}" in output
    assert "evidence_json=" in output
    assert "predictions_csv=" in output
    assert "prediction_count=2" in output
    assert "outcome_count=0" in output
    assert "audit_event_count=1" in output
    assert (output_dir / f"{run_id}_evidence.json").exists()
    assert (output_dir / f"{run_id}_predictions.csv").exists()


def test_export_prediction_run_evidence_rejects_unknown_run(tmp_path) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")

    try:
        export_prediction_run_evidence(
            database_path,
            run_id="run-missing",
            output_dir=tmp_path / "exports",
        )
    except ValueError as exc:
        assert "unknown prediction run" in str(exc)
    else:
        raise AssertionError("expected unknown run export to fail")


def test_prediction_outcomes_reject_unknown_prediction_units(tmp_path) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)

    try:
        record_prediction_outcomes(
            database_path,
            run_id=run_id,
            outcomes=pd.DataFrame({"unit_number": [999], "actual_rul": [10.0]}),
            source_name="bad_outcomes.csv",
        )
    except ValueError as exc:
        assert "outcome unit_number values are not in run" in str(exc)
    else:  # pragma: no cover - defensive assertion for clearer failures
        raise AssertionError("record_prediction_outcomes should reject unknown units")


@pytest.mark.parametrize(
    ("outcomes", "expected_message"),
    [
        (
            pd.DataFrame({"unit_number": ["unit-1"], "actual_rul": [10.0]}),
            "outcome rows require numeric, non-null unit_number and actual_rul",
        ),
        (
            pd.DataFrame({"unit_number": [1], "actual_rul": [None]}),
            "outcome rows require numeric, non-null unit_number and actual_rul",
        ),
        (
            pd.DataFrame({"unit_number": [1.5], "actual_rul": [10.0]}),
            "outcome unit_number values must be whole numbers",
        ),
        (
            pd.DataFrame({"unit_number": [1], "actual_rul": [-1.0]}),
            "actual_rul values must be nonnegative",
        ),
    ],
)
def test_prediction_outcomes_reject_invalid_outcome_values(
    tmp_path,
    outcomes: pd.DataFrame,
    expected_message: str,
) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)

    with pytest.raises(ValueError, match=expected_message):
        record_prediction_outcomes(
            database_path,
            run_id=run_id,
            outcomes=outcomes,
            source_name="bad_outcomes.csv",
        )


def test_interval_diagnostics_report_missing_prediction_bounds(tmp_path) -> None:
    workspace = _write_fake_workspace(tmp_path / "quickstart")
    database_path = tmp_path / "app.sqlite"
    seed_quickstart_workspace(database_path, workspace)
    telemetry = pd.DataFrame(
        {
            "unit_number": [1, 2],
            "time_in_cycles": [10, 11],
        }
    )
    prediction_document = {
        "dataset": "C-MAPSS",
        "subset": "FD001",
        "model_name": "manual",
        "artifact": {"artifact_id": "fd001-demo"},
        "predictions": [
            {"unit_number": 1, "predicted_rul": 12.0},
            {"unit_number": 2, "predicted_rul": 20.0},
        ],
        "monitoring": {},
    }

    run_id = record_prediction_run(
        database_path,
        telemetry=telemetry,
        prediction_document=prediction_document,
        model_artifact_path=workspace.model_artifact_path,
        source_name="manual.csv",
    )
    runs = list_prediction_runs(database_path)
    loaded = load_model_artifact(database_path, "fd001-demo")

    assert runs[0]["run_id"] == run_id
    assert runs[0]["interval_count"] == 0
    assert runs[0]["interval_availability_rate"] == 0.0
    assert runs[0]["mean_interval_width"] is None
    assert loaded is not None
    assert loaded["report_card"]["prediction_count_total"] == 2
    assert loaded["report_card"]["interval_count_total"] == 0
    assert loaded["report_card"]["missing_interval_count"] == 2
    assert loaded["report_card"]["interval_availability_rate"] == 0.0
    assert loaded["report_card"]["interval_complete"] is False
    assert loaded["report_card"]["mean_interval_width"] is None


def test_prediction_run_events_append_operator_decisions(tmp_path) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)

    event_id = record_prediction_run_event(
        database_path,
        run_id=run_id,
        event_type="operator_decision",
        status="watch",
        actor="flight-ops",
        note="Monitor on next cycle",
        payload={"ticket": "PHM-42"},
    )
    runs = list_prediction_runs(database_path)
    loaded = load_prediction_run(database_path, run_id)
    events = list_prediction_run_events(database_path, run_id)

    assert event_id.startswith("event-")
    assert runs[0]["audit_event_count"] == 2
    assert runs[0]["decision_status"] == "watch"
    assert runs[0]["decision_note"] == "Monitor on next cycle"
    assert loaded is not None
    assert loaded["run"]["decision_status"] == "watch"
    assert events[0]["event_type"] == "operator_decision"
    assert events[0]["payload"] == {"ticket": "PHM-42"}


def test_prediction_run_event_rejects_unknown_run(tmp_path) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")

    try:
        record_prediction_run_event(
            database_path,
            run_id="run-missing",
            event_type="operator_decision",
        )
    except ValueError as exc:
        assert "unknown prediction run" in str(exc)
    else:
        raise AssertionError("expected unknown run to fail")


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


def _write_prediction_run_for_artifact(tmp_path, *, database_path, artifact_id: str):
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
        "artifact": {"artifact_id": artifact_id},
        "monitoring": packaged.artifact.monitoring_summary(telemetry, predictions),
    }
    return record_prediction_run(
        database_path,
        telemetry=telemetry,
        prediction_document=prediction_document,
        model_artifact_path=artifact_path,
        source_name="test.csv",
    )


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
        "uncertainty": {
            "interval_method": "train_residual_absolute_quantile",
            "interval_confidence": 0.9,
        },
    }
    release_bundle = {
        "status": "ok",
        "release_name": "fd001-demo",
        "gates": {
            "promotion_report_ok": True,
            "promotion_gates_passed": True,
        },
    }
    provenance = {"status": "ok", "summary": {"workflow": "local"}}
    promotion = {
        "status": "failed",
        "artifact_identity": {"artifact_id": "fd001-demo"},
        "gates": {
            "artifact_validation": True,
            "latency_benchmark": False,
        },
        "evidence": {
            "benchmark": {
                "latency_ms": {"p95": 42.0},
                "max_p95_latency_ms": 25.0,
            }
        },
    }
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
