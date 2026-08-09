from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

import aerospace_prognostics.app.store as store
from aerospace_prognostics.app.dashboard_state import QuickstartWorkspace
from aerospace_prognostics.app.fleet_registry import fleet_asset_export_rows
from aerospace_prognostics.app.store import (
    SCHEMA_VERSION,
    build_fleet_asset_registry_bundle,
    build_fleet_priority_policy_validation,
    build_model_artifact_review_bundle,
    build_prediction_run_evidence,
    database_summary,
    export_fleet_asset_registry,
    export_fleet_priority_policy_validation,
    export_prediction_outcome_template,
    export_prediction_run_evidence,
    initialize_app_database,
    inspect_anomaly_events_csv,
    list_fleet_assets,
    list_model_artifacts,
    list_prediction_run_events,
    list_prediction_runs,
    load_model_artifact,
    load_prediction_run,
    record_prediction_outcomes,
    record_prediction_run,
    record_prediction_run_event,
    register_model_artifact_evidence,
    render_fleet_priority_policy_validation_markdown,
    seed_quickstart_workspace,
    sync_fleet_assets_from_anomaly_comparison,
    sync_fleet_assets_from_anomaly_events,
    sync_fleet_assets_from_prediction_run,
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
    assert summary["fleet_assets"] == 0


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
    assert "fleet_assets=0" in output


def _quickstart_evidence_diagnostics(workspace: QuickstartWorkspace) -> str:
    """Describe why an evidence item might have been skipped.

    ``register_model_artifact_evidence`` silently skips an item when its payload
    is ``None`` or its file does not exist, so a short count is indistinguishable
    from a correct one unless the inputs are reported. See the known-flake note
    on ``test_seed_quickstart_workspace_persists_model_and_evidence``.
    """
    items = (
        ("artifact_inspection", workspace.artifact_inspection_path, workspace.artifact_inspection),
        ("release_bundle", workspace.release_bundle_path, workspace.release_bundle),
        ("release_provenance", workspace.provenance_path, workspace.provenance),
        ("promotion_report", workspace.promotion_report_path, workspace.promotion_report),
        ("dashboard_payload", workspace.dashboard_payload_path, workspace.dashboard_payload),
    )
    lines = [
        f"  {name}: exists={path.exists()} payload_loaded={payload is not None} path={path}"
        for name, path, payload in items
    ]
    return "evidence inputs at assertion time:\n" + "\n".join(lines)


def test_seed_quickstart_workspace_persists_model_and_evidence(tmp_path) -> None:
    """Seeding is idempotent and registers all five evidence documents.

    **Known flake, unreproduced.** This test failed once on 2026-07-27 during a
    full-suite run whose preceding commit changed only markdown, then passed in
    isolation and on every rerun since — 20 consecutive attempts across two
    branches, including 10 runs targeting this file alone. Ruled out as causes:

    - *test-order dependence* — no ``pytest-randomly``/``xdist``/``pytest-order``
      is installed, so collection order is deterministic;
    - *shared state* — every case uses its own ``tmp_path``;
    - *time dependence in the idempotency key* — the conflict key is
      ``evidence_id = f"{type}:{artifact_id}:{sha256(file)}"``, which is
      content-addressed, and ``artifact_id`` is the fixture literal
      ``fd001-demo``. The timestamp is a stored column only, never part of
      ``on conflict(evidence_id) do nothing``;
    - *parallelism* — the suite runs single-process.

    What was never established is which assertion failed: the traceback was not
    captured at the time. The remaining untested hypothesis is a transient
    Windows filesystem-visibility stall — an anti-virus or indexer holding a
    just-written file so ``Path.exists()`` briefly returns ``False`` — which
    would silently drop one evidence item and yield 4 instead of 5.

    The assertions below therefore report the evidence inputs on failure, so the
    next occurrence diagnoses itself rather than needing another 20 runs. Do not
    replace this with a retry or an ``xfail``; that would hide the signal instead
    of sharpening it.
    """
    workspace = _write_fake_workspace(tmp_path / "quickstart")
    database_path = tmp_path / "app.sqlite"

    inserted = seed_quickstart_workspace(database_path, workspace)
    second_insert = seed_quickstart_workspace(database_path, workspace)
    summary = database_summary(database_path)

    diagnostics = _quickstart_evidence_diagnostics(workspace)

    assert inserted["model_artifacts"] == 1, diagnostics
    assert inserted["release_evidence"] == 5, (
        f"first seed registered {inserted['release_evidence']} of 5 evidence documents.\n"
        f"{diagnostics}"
    )
    assert second_insert["model_artifacts"] == 1, diagnostics
    assert second_insert["release_evidence"] == 0, (
        f"re-seeding inserted {second_insert['release_evidence']} rows; seeding must be "
        f"idempotent because evidence_id is content-addressed.\n{diagnostics}"
    )
    assert summary["model_artifacts"] == 1, diagnostics
    assert summary["release_evidence"] == 5, (
        f"database holds {summary['release_evidence']} evidence rows, expected 5.\n"
        f"{diagnostics}"
    )


def test_a_missing_evidence_file_is_silently_skipped_not_reported(tmp_path) -> None:
    """Pin the mechanism behind the known flake above.

    ``register_model_artifact_evidence`` skips any evidence item whose file is
    absent, without raising or warning. That is the behaviour that would turn a
    transient filesystem stall into a bare ``4 != 5`` with no explanation, so it
    is worth having demonstrated rather than merely hypothesised.

    This test does not reproduce the flake — it proves the failure *mode* the
    flake would take, and that the diagnostics added above would name the
    culprit.
    """
    workspace = _write_fake_workspace(tmp_path / "quickstart")
    database_path = tmp_path / "app.sqlite"
    workspace.provenance_path.unlink()

    inserted = seed_quickstart_workspace(database_path, workspace)

    # Silently four, not five, and no error of any kind.
    assert inserted["release_evidence"] == 4

    diagnostics = _quickstart_evidence_diagnostics(workspace)
    assert "release_provenance: exists=False" in diagnostics
    assert "release_bundle: exists=True" in diagnostics


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


def test_build_model_artifact_review_bundle_is_read_only_safe(tmp_path) -> None:
    workspace = _write_fake_workspace(tmp_path / "quickstart")
    database_path = tmp_path / "app.sqlite"
    seed_quickstart_workspace(database_path, workspace)
    run_id = _write_prediction_run_for_artifact(
        tmp_path,
        database_path=database_path,
        artifact_id="fd001-demo",
    )

    bundle = build_model_artifact_review_bundle(
        database_path,
        artifact_id="fd001-demo",
        read_only=True,
    )

    assert bundle["schema_version"] == "aerospace-prognostics/model-artifact-review/v1"
    assert bundle["database"]["schema_version"] == SCHEMA_VERSION
    assert bundle["artifact"]["artifact_id"] == "fd001-demo"
    assert bundle["report_card"]["artifact_id"] == "fd001-demo"
    assert bundle["counts"] == {"release_evidence": 5, "prediction_runs": 1}
    assert len(bundle["release_evidence"]) == 5
    assert bundle["prediction_runs"][0]["run_id"] == run_id


def test_build_model_artifact_review_bundle_rejects_unknown_artifact(tmp_path) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")

    with pytest.raises(ValueError, match="unknown model artifact"):
        build_model_artifact_review_bundle(
            database_path,
            artifact_id="missing-artifact",
            read_only=True,
        )


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
    assets = store.list_fleet_assets(database_path, read_only=True)
    fleet_bundle = store.build_fleet_asset_registry_bundle(
        database_path,
        read_only=True,
    )
    review_bundle = store.build_model_artifact_review_bundle(
        database_path,
        artifact_id="fd001-demo",
        read_only=True,
    )

    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["model_artifacts"] == 1
    assert artifacts[0]["artifact_id"] == "fd001-demo"
    assert loaded_artifact is not None
    assert loaded_artifact["artifact"]["artifact_id"] == "fd001-demo"
    assert runs[0]["run_id"] == run_id
    assert loaded_run is not None
    assert loaded_run["run"]["run_id"] == run_id
    assert events[0]["event_type"] == "prediction_recorded"
    assert assets[0]["latest_run_id"] == run_id
    assert fleet_bundle["summary"]["asset_count"] == 2
    assert fleet_bundle["assets"][0]["latest_run_id"] == run_id
    assert review_bundle["artifact"]["artifact_id"] == "fd001-demo"


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
    assets = list_fleet_assets(database_path)

    assert run_id.startswith("run-")
    assert summary["telemetry_uploads"] == 1
    assert summary["prediction_runs"] == 1
    assert summary["predictions"] == 2
    assert summary["prediction_outcomes"] == 0
    assert summary["prediction_run_events"] == 1
    assert summary["fleet_assets"] == 2
    assert len(assets) == 2
    assert {asset["latest_run_id"] for asset in assets} == {run_id}
    assert {asset["asset_type"] for asset in assets} == {"engine"}
    assert {asset["domain"] for asset in assets} == {"turbofan_rul"}
    assert assets[0]["metadata"]["artifact_id"] == (
        packaged.artifact.promotion_metadata["artifact_id"]
    )
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


def test_prediction_run_history_filters_operational_review_views(tmp_path) -> None:
    database_path = tmp_path / "app.sqlite"
    critical_run_id = _write_manual_prediction_run(
        database_path,
        subset="FD001",
        model_name="hgb_policy",
        artifact_id="artifact-critical",
        source_name="critical.csv",
        predicted_rul=15.0,
        monitoring={"drift": {"alert_columns": ["sensor_2"]}},
        created_at_utc="2026-01-01T00:00:00+00:00",
    )
    nominal_run_id = _write_manual_prediction_run(
        database_path,
        subset="FD002",
        model_name="lstm_forecast",
        artifact_id="artifact-nominal",
        source_name="nominal.csv",
        predicted_rul=90.0,
        monitoring={"drift": {"alert_columns": []}},
        created_at_utc="2026-02-01T00:00:00+00:00",
    )
    record_prediction_run_event(
        database_path,
        run_id=nominal_run_id,
        event_type="operator_decision",
        status="accepted",
        actor="ops",
    )

    assert [
        run["run_id"]
        for run in list_prediction_runs(database_path, model_names=("hgb_policy",))
    ] == [critical_run_id]
    assert [
        run["run_id"]
        for run in list_prediction_runs(
            database_path,
            artifact_ids=("artifact-nominal",),
        )
    ] == [nominal_run_id]
    assert [
        run["run_id"]
        for run in list_prediction_runs(database_path, asset_ids=("FD001-unit-1",))
    ] == [critical_run_id]
    assert [
        run["run_id"]
        for run in list_prediction_runs(database_path, risk_levels=("critical",))
    ] == [critical_run_id]
    assert [
        run["run_id"]
        for run in list_prediction_runs(database_path, risk_levels=("nominal",))
    ] == [nominal_run_id]
    assert [
        run["run_id"]
        for run in list_prediction_runs(database_path, decision_statuses=("accepted",))
    ] == [nominal_run_id]
    assert [
        run["run_id"]
        for run in list_prediction_runs(
            database_path,
            start_created_at_utc="2026-01-15T00:00:00+00:00",
        )
    ] == [nominal_run_id]
    assert [
        run["run_id"]
        for run in list_prediction_runs(
            database_path,
            end_created_at_utc="2026-01-15T00:00:00+00:00",
        )
    ] == [critical_run_id]
    assert [
        run["run_id"] for run in list_prediction_runs(database_path, drift_only=True)
    ] == [critical_run_id]


def test_app_sync_fleet_assets_command_refreshes_prediction_assets(
    tmp_path,
    capsys,
) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)

    result = sync_fleet_assets_from_prediction_run(database_path, run_id=run_id)
    exit_code = main(
        [
            "app-sync-fleet-assets",
            "--database",
            str(database_path),
            "--run-id",
            run_id,
        ]
    )
    output = capsys.readouterr().out
    assets = list_fleet_assets(database_path)

    assert result == {"run_id": run_id, "runs_synced": 1, "updated_assets": 2}
    assert exit_code == 0
    assert f"database={database_path}" in output
    assert f"run_id={run_id}" in output
    assert "runs_synced=1" in output
    assert "updated_assets=2" in output
    assert "fleet_assets=2" in output
    assert len(assets) == 2
    assert all(asset["latest_run_id"] == run_id for asset in assets)


def test_sync_fleet_assets_from_anomaly_comparison_adds_spacecraft_assets(
    tmp_path,
) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")
    comparison_csv = _write_anomaly_comparison_csv(tmp_path / "comparison.csv")

    result = sync_fleet_assets_from_anomaly_comparison(
        database_path,
        comparison_csv=comparison_csv,
        source_name="phase2_smap_msl",
    )
    assets = list_fleet_assets(database_path, domains=("spacecraft_anomaly",))
    attention_assets = list_fleet_assets(
        database_path,
        domains=("spacecraft_anomaly",),
        attention_only=True,
    )
    bundle = build_fleet_asset_registry_bundle(
        database_path,
        domains=("spacecraft_anomaly",),
    )
    exported_rows = fleet_asset_export_rows(assets)

    assert result["channels_synced"] == 2
    assert result["updated_assets"] == 2
    assert [asset["asset_id"] for asset in assets] == [
        "smap-channel-p-1",
        "msl-channel-m-1",
    ]
    assert assets[0]["asset_type"] == "spacecraft_channel"
    assert assets[0]["domain"] == "spacecraft_anomaly"
    assert assets[0]["source_dataset"] == "SMAP/MSL"
    assert assets[0]["latest_risk_level"] == "critical"
    assert assets[0]["latest_status"] == "anomaly_review"
    assert assets[0]["latest_run_id"] is None
    assert assets[0]["priority_score"] > assets[1]["priority_score"]
    assert assets[0]["priority_band"] == "immediate_review"
    assert any("Miss rate" in reason for reason in assets[0]["priority_reasons"])
    assert assets[0]["metadata"]["model_name"] == "robust_zscore"
    assert assets[0]["metadata"]["source_name"] == "phase2_smap_msl"
    assert assets[0]["metadata"]["f1"] == 0.2
    assert "High anomaly miss rate" in assets[0]["latest_attention_reasons"]
    assert assets[1]["latest_risk_level"] == "nominal"
    assert assets[1]["latest_attention_reasons"] == []
    assert [asset["asset_id"] for asset in attention_assets] == ["smap-channel-p-1"]
    assert bundle["summary"]["domain_counts"] == {"spacecraft_anomaly": 2}
    assert bundle["summary"]["attention_required_count"] == 1
    assert exported_rows[0]["channel_id"] == "P-1"
    assert exported_rows[0]["spacecraft"] == "SMAP"
    assert exported_rows[0]["f1"] == 0.2
    assert exported_rows[0]["predicted_positives"] == 2
    assert exported_rows[0]["priority_band"] == "immediate_review"
    assert "Miss rate" in exported_rows[0]["priority_reasons"]


def test_fleet_asset_registry_prioritizes_cross_domain_review_queue(
    tmp_path,
) -> None:
    database_path = tmp_path / "app.sqlite"
    _write_manual_prediction_run(
        database_path,
        subset="FD001",
        model_name="hgb_policy",
        artifact_id="artifact-critical-engine",
        source_name="critical_engine.csv",
        predicted_rul=15.0,
        monitoring={"drift": {"alert_columns": []}},
        created_at_utc="2026-01-01T00:00:00+00:00",
    )
    sync_fleet_assets_from_anomaly_comparison(
        database_path,
        comparison_csv=_write_anomaly_comparison_csv(tmp_path / "comparison.csv"),
        source_name="phase2_smap_msl",
    )

    assets = list_fleet_assets(database_path)
    exported = fleet_asset_export_rows(assets)

    assert [asset["asset_id"] for asset in assets][:2] == [
        "smap-channel-p-1",
        "FD001-unit-1",
    ]
    assert assets[0]["domain"] == "spacecraft_anomaly"
    assert assets[1]["domain"] == "turbofan_rul"
    assert assets[0]["priority_score"] > assets[1]["priority_score"]
    assert assets[1]["priority_band"] == "immediate_review"
    assert any("RUL risk floor" in reason for reason in assets[1]["priority_reasons"])
    assert exported[0]["priority_score"] == assets[0]["priority_score"]
    assert "priority_reasons" in exported[0]


def test_sync_fleet_assets_from_anomaly_events_updates_live_channel_assets(
    tmp_path,
) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")
    events_csv = _write_anomaly_events_csv(tmp_path / "events.csv")

    result = sync_fleet_assets_from_anomaly_events(
        database_path,
        events_csv=events_csv,
        source_name="ops_stream",
    )
    assets = list_fleet_assets(database_path, domains=("spacecraft_anomaly",))
    exported_rows = fleet_asset_export_rows(assets)

    assert result == {
        "source_path": str(events_csv),
        "source_name": "ops_stream",
        "events_processed": 3,
        "channels_synced": 2,
        "updated_assets": 2,
    }
    assert [asset["asset_id"] for asset in assets] == [
        "smap-channel-p-1",
        "msl-channel-m-1",
    ]
    assert assets[0]["latest_risk_level"] == "critical"
    assert assets[0]["latest_status"] == "anomaly_review"
    assert assets[0]["metadata"]["event_kind"] == "live_anomaly_event"
    assert assets[0]["metadata"]["event_time_utc"] == "2026-01-02T00:00:00+00:00"
    assert assets[0]["metadata"]["source_name"] == "ops_stream"
    assert assets[0]["metadata"]["active"] is True
    assert assets[0]["metadata"]["anomaly_score"] == 0.95
    assert "Severity critical" in assets[0]["latest_attention_reasons"]
    assert "Active anomaly event" in assets[0]["latest_attention_reasons"]
    assert any(
        "Live anomaly severity critical" in reason
        for reason in assets[0]["priority_reasons"]
    )
    assert any("crossed threshold" in reason for reason in assets[0]["priority_reasons"])
    assert assets[1]["latest_risk_level"] == "nominal"
    assert exported_rows[0]["event_time_utc"] == "2026-01-02T00:00:00+00:00"
    assert exported_rows[0]["severity"] == "critical"
    assert exported_rows[0]["active"] is True
    assert exported_rows[0]["anomaly_score"] == 0.95
    assert exported_rows[0]["threshold"] == 0.8


def test_inspect_anomaly_events_csv_summarizes_ingest_preview(tmp_path) -> None:
    events_csv = _write_anomaly_events_csv(tmp_path / "events.csv")

    preview = inspect_anomaly_events_csv(events_csv)

    assert preview["source_path"] == str(events_csv)
    assert preview["events_processed"] == 3
    assert preview["channels_synced"] == 2
    assert preview["risk_counts"] == {"critical": 1, "nominal": 1}
    assert preview["severity_counts"] == {"critical": 1, "info": 1}
    assert preview["active_events"] == 1
    assert preview["threshold_crossings"] == 1
    assert [event["channel_id"] for event in preview["latest_events"]] == ["M-1", "P-1"]
    assert preview["latest_events"][1]["event_time_utc"] == "2026-01-02T00:00:00+00:00"


def test_inspect_anomaly_events_csv_rejects_missing_required_columns(tmp_path) -> None:
    events_csv = tmp_path / "events.csv"
    pd.DataFrame([{"channel_id": "P-1"}]).to_csv(events_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns: spacecraft"):
        inspect_anomaly_events_csv(events_csv)


def test_app_sync_anomaly_assets_command_refreshes_spacecraft_assets(
    tmp_path,
    capsys,
) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")
    comparison_csv = _write_anomaly_comparison_csv(tmp_path / "comparison.csv")

    exit_code = main(
        [
            "app-sync-anomaly-assets",
            "--database",
            str(database_path),
            "--comparison-csv",
            str(comparison_csv),
            "--source-name",
            "phase2_smap_msl",
        ]
    )
    output = capsys.readouterr().out
    assets = list_fleet_assets(database_path, domains=("spacecraft_anomaly",))

    assert exit_code == 0
    assert f"database={database_path}" in output
    assert f"comparison_csv={comparison_csv}" in output
    assert "source_name=phase2_smap_msl" in output
    assert "channels_synced=2" in output
    assert "updated_assets=2" in output
    assert "fleet_assets=2" in output
    assert {asset["asset_type"] for asset in assets} == {"spacecraft_channel"}


def test_app_sync_anomaly_events_command_refreshes_live_spacecraft_assets(
    tmp_path,
    capsys,
) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")
    events_csv = _write_anomaly_events_csv(tmp_path / "events.csv")

    exit_code = main(
        [
            "app-sync-anomaly-events",
            "--database",
            str(database_path),
            "--events-csv",
            str(events_csv),
            "--source-name",
            "ops_stream",
        ]
    )
    output = capsys.readouterr().out
    assets = list_fleet_assets(database_path, domains=("spacecraft_anomaly",))

    assert exit_code == 0
    assert f"database={database_path}" in output
    assert f"events_csv={events_csv}" in output
    assert "source_name=ops_stream" in output
    assert "events_processed=3" in output
    assert "channels_synced=2" in output
    assert "updated_assets=2" in output
    assert "fleet_assets=2" in output
    assert {asset["asset_type"] for asset in assets} == {"spacecraft_channel"}


def test_build_fleet_asset_registry_bundle_is_read_only_safe(tmp_path) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)

    bundle = build_fleet_asset_registry_bundle(database_path, read_only=True)

    assert bundle["schema_version"] == (
        "aerospace-prognostics/fleet-asset-registry/v1"
    )
    assert bundle["database"]["schema_version"] == SCHEMA_VERSION
    assert bundle["summary"]["asset_count"] == 2
    assert bundle["summary"]["domain_counts"] == {"turbofan_rul": 2}
    assert bundle["priority_policy"]["band_counts"] == {"immediate_review": 2}
    assert bundle["priority_policy"]["review_queue_count"] == 2
    assert len(bundle["priority_policy"]["top_assets"]) == 2
    assert "Risk level is critical" in bundle["priority_policy"]["reason_counts"]
    assert bundle["files"]["assets_csv"] == {"rows": 2}
    assert {asset["latest_run_id"] for asset in bundle["assets"]} == {run_id}


def test_fleet_asset_registry_filters_assets_for_review(tmp_path) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)
    labels = _label_fleet_assets_for_filtering(database_path)

    critical_assets = list_fleet_assets(database_path, risk_levels=("critical",))
    attention_assets = list_fleet_assets(database_path, attention_only=True)
    bundle = build_fleet_asset_registry_bundle(
        database_path,
        risk_levels=("critical",),
        statuses=("maintenance_review",),
        attention_only=True,
    )

    assert [asset["asset_id"] for asset in critical_assets] == [labels["critical"]]
    assert [asset["asset_id"] for asset in attention_assets] == [labels["critical"]]
    assert bundle["filters"] == {
        "risk_levels": ["critical"],
        "domains": [],
        "statuses": ["maintenance_review"],
        "attention_only": True,
    }
    assert bundle["summary"]["asset_count"] == 1
    assert bundle["assets"][0]["asset_id"] == labels["critical"]
    assert bundle["assets"][0]["latest_run_id"] == run_id


def test_export_fleet_asset_registry_writes_json_and_csv(tmp_path) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)
    output_dir = tmp_path / "registry_exports"

    result = export_fleet_asset_registry(database_path, output_dir=output_dir)
    bundle = json.loads(
        (output_dir / "fleet_asset_registry.json").read_text(encoding="utf-8")
    )
    exported_assets = pd.read_csv(result["assets_csv"])

    assert result["output_dir"] == str(output_dir)
    assert result["asset_count"] == 2
    assert result["registry_sha256"]
    assert result["assets_sha256"]
    assert bundle["schema_version"] == (
        "aerospace-prognostics/fleet-asset-registry/v1"
    )
    assert bundle["summary"]["asset_count"] == 2
    assert bundle["priority_policy"]["top_assets"][0]["priority_reasons"]
    assert "band_counts" in bundle["priority_policy"]
    assert bundle["files"]["assets_csv"]["rows"] == 2
    assert bundle["files"]["assets_csv"]["sha256"] == result["assets_sha256"]
    assert {asset["latest_run_id"] for asset in bundle["assets"]} == {run_id}
    assert list(exported_assets["latest_run_id"]) == [run_id, run_id]
    assert "attention_reasons" in exported_assets.columns


def test_app_export_fleet_assets_command_writes_registry_evidence(
    tmp_path,
    capsys,
) -> None:
    database_path, _run_id = _write_prediction_run(tmp_path)
    labels = _label_fleet_assets_for_filtering(database_path)
    output_dir = tmp_path / "fleet_exports"

    exit_code = main(
        [
            "app-export-fleet-assets",
            "--database",
            str(database_path),
            "--output-dir",
            str(output_dir),
            "--risk-level",
            "critical",
            "--attention-only",
        ]
    )
    output = capsys.readouterr().out
    bundle = json.loads(
        (output_dir / "fleet_asset_registry.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert f"database={database_path}" in output
    assert f"output_dir={output_dir}" in output
    assert "registry_json=" in output
    assert "assets_csv=" in output
    assert "asset_count=1" in output
    assert '"risk_levels": ["critical"]' in output
    assert bundle["filters"]["attention_only"] is True
    assert bundle["assets"][0]["asset_id"] == labels["critical"]
    assert (output_dir / "fleet_asset_registry.json").exists()
    assert (output_dir / "fleet_assets.csv").exists()


def test_fleet_priority_policy_validation_checks_cross_domain_queue(
    tmp_path,
) -> None:
    database_path = _write_priority_policy_validation_fleet(tmp_path)
    output_dir = tmp_path / "policy_exports"

    report = build_fleet_priority_policy_validation(database_path, read_only=True)
    result = export_fleet_priority_policy_validation(
        database_path,
        output_dir=output_dir,
    )
    exported = json.loads(
        (output_dir / "fleet_priority_policy_validation.json").read_text(
            encoding="utf-8"
        )
    )
    markdown = render_fleet_priority_policy_validation_markdown(report)
    exported_markdown = (
        output_dir / "fleet_priority_policy_validation.md"
    ).read_text(encoding="utf-8")
    checks = {check["check_id"]: check for check in report["scenario_checks"]}

    assert report["schema_version"] == (
        "aerospace-prognostics/fleet-priority-policy-validation/v1"
    )
    assert report["overall_status"] == "pass"
    assert report["asset_count"] == 3
    assert checks["critical_assets_are_immediate_review"]["status"] == "pass"
    assert checks["watch_assets_are_review_or_better"]["status"] == "pass"
    assert checks["priority_order_is_descending"]["status"] == "pass"
    assert checks["cross_domain_review_queue"]["applicable"] is True
    assert checks["cross_domain_review_queue"]["status"] == "pass"
    assert checks["live_anomaly_events_are_explained"]["status"] == "pass"
    assert result["overall_status"] == "pass"
    assert result["validation_sha256"]
    assert result["markdown_sha256"]
    assert exported["failed_checks"] == []
    assert "# Fleet Priority Policy Validation" in markdown
    assert exported_markdown == render_fleet_priority_policy_validation_markdown(
        exported
    )


def test_app_export_priority_policy_command_writes_validation_evidence(
    tmp_path,
    capsys,
) -> None:
    database_path = _write_priority_policy_validation_fleet(tmp_path)
    output_dir = tmp_path / "policy_exports"

    exit_code = main(
        [
            "app-export-priority-policy",
            "--database",
            str(database_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    output = capsys.readouterr().out
    report = json.loads(
        (output_dir / "fleet_priority_policy_validation.json").read_text(
            encoding="utf-8"
        )
    )

    assert exit_code == 0
    assert f"database={database_path}" in output
    assert f"output_dir={output_dir}" in output
    assert "validation_json=" in output
    assert "validation_markdown=" in output
    assert "overall_status=pass" in output
    assert "failed_checks=[]" in output
    assert "asset_count=3" in output
    assert report["overall_status"] == "pass"
    assert (output_dir / "fleet_priority_policy_validation.md").exists()


def test_app_export_priority_policy_command_can_fail_release_gate(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    from aerospace_prognostics import cli_app

    def fake_export(database_path, *, output_dir):
        return {
            "output_dir": str(output_dir),
            "validation_json": str(output_dir / "fleet_priority_policy_validation.json"),
            "validation_sha256": "sha",
            "validation_markdown": str(
                output_dir / "fleet_priority_policy_validation.md"
            ),
            "markdown_sha256": "markdown-sha",
            "overall_status": "fail",
            "failed_checks": ["critical_assets_are_immediate_review"],
            "asset_count": 2,
        }

    database_path = tmp_path / "app.sqlite"
    output_dir = tmp_path / "policy_exports"
    monkeypatch.setattr(
        cli_app,
        "export_fleet_priority_policy_validation",
        fake_export,
    )

    exit_code = main(
        [
            "app-export-priority-policy",
            "--database",
            str(database_path),
            "--output-dir",
            str(output_dir),
            "--fail-on-policy-fail",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "overall_status=fail" in output
    assert 'failed_checks=["critical_assets_are_immediate_review"]' in output


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
    assert manifest["files"]["predictions_csv"]["sha256"] == result["predictions_sha256"]
    assert len(manifest["predictions"]) == 2
    assert len(manifest["audit_events"]) == 3
    assert list(exported_predictions["unit_number"]) == [
        row["unit_number"] for row in manifest["predictions"]
    ]
    assert "actual_rul" in exported_predictions.columns


def test_build_prediction_run_evidence_is_read_only_safe(tmp_path) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)

    manifest = build_prediction_run_evidence(
        database_path,
        run_id=run_id,
        read_only=True,
    )

    assert manifest["schema_version"] == (
        "aerospace-prognostics/prediction-run-evidence/v1"
    )
    assert manifest["database"]["schema_version"] == SCHEMA_VERSION
    assert manifest["run"]["run_id"] == run_id
    assert len(manifest["predictions"]) == 2
    assert len(manifest["audit_events"]) == 1
    assert manifest["files"]["predictions_csv"] == {"rows": 2}


def test_export_prediction_outcome_template_writes_fillable_csv(tmp_path) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)
    output_csv = tmp_path / "templates" / "outcomes.csv"

    result = export_prediction_outcome_template(
        database_path,
        run_id=run_id,
        output_csv=output_csv,
    )
    template = pd.read_csv(output_csv)
    loaded = load_prediction_run(database_path, run_id)
    assert loaded is not None

    assert result["run_id"] == run_id
    assert result["outcome_template_csv"] == str(output_csv)
    assert result["outcome_template_sha256"]
    assert result["prediction_count"] == len(loaded["predictions"])
    assert list(template.columns) == ["unit_number", "actual_rul"]
    assert list(template["unit_number"]) == [
        row["unit_number"] for row in loaded["predictions"]
    ]
    assert template["actual_rul"].isna().all()


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


def test_app_export_outcome_template_command_writes_template_csv(
    tmp_path,
    capsys,
) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)
    output_csv = tmp_path / "exports" / "outcomes_template.csv"

    exit_code = main(
        [
            "app-export-outcome-template",
            "--database",
            str(database_path),
            "--run-id",
            run_id,
            "--output-csv",
            str(output_csv),
        ]
    )
    output = capsys.readouterr().out
    template = pd.read_csv(output_csv)

    assert exit_code == 0
    assert f"database={database_path}" in output
    assert f"run_id={run_id}" in output
    assert f"outcome_template_csv={output_csv}" in output
    assert "outcome_template_sha256=" in output
    assert "prediction_count=2" in output
    assert list(template.columns) == ["unit_number", "actual_rul"]
    assert len(template) == 2


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


def test_app_record_decision_command_appends_operator_event(
    tmp_path,
    capsys,
) -> None:
    database_path, run_id = _write_prediction_run(tmp_path)
    payload_json = tmp_path / "decision_payload.json"
    payload_json.write_text('{"shift": "night"}', encoding="utf-8")

    exit_code = main(
        [
            "app-record-decision",
            "--database",
            str(database_path),
            "--run-id",
            run_id,
            "--status",
            "escalated",
            "--actor",
            "flight-ops",
            "--note",
            "Escalate to reliability engineering",
            "--ticket",
            "PHM-99",
            "--severity",
            "high",
            "--payload-json",
            str(payload_json),
        ]
    )
    output = capsys.readouterr().out
    runs = list_prediction_runs(database_path)
    events = list_prediction_run_events(database_path, run_id)

    assert exit_code == 0
    assert f"database={database_path}" in output
    assert f"run_id={run_id}" in output
    assert "event_id=event-" in output
    assert "decision_status=escalated" in output
    assert "actor=flight-ops" in output
    assert "prediction_run_events=2" in output
    assert runs[0]["decision_status"] == "escalated"
    assert runs[0]["decision_note"] == "Escalate to reliability engineering"
    assert events[0]["event_type"] == "operator_decision"
    assert events[0]["actor"] == "flight-ops"
    assert events[0]["payload"] == {
        "shift": "night",
        "ticket": "PHM-99",
        "severity": "high",
    }


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


def _write_priority_policy_validation_fleet(tmp_path):
    database_path = initialize_app_database(tmp_path / "app.sqlite")
    _write_manual_prediction_run(
        database_path,
        subset="FD001",
        model_name="hist_gradient_boosting",
        artifact_id="fd001-critical",
        source_name="critical_engine.csv",
        predicted_rul=5.0,
        monitoring={"risk": {"critical_threshold": 20.0, "watch_threshold": 60.0}},
        created_at_utc="2026-01-02T02:00:00+00:00",
    )
    sync_fleet_assets_from_anomaly_events(
        database_path,
        events_csv=_write_anomaly_events_csv(tmp_path / "anomaly_events.csv"),
        source_name="ops-console",
    )
    return database_path


def _label_fleet_assets_for_filtering(database_path):
    with sqlite3.connect(database_path) as connection:
        asset_ids = [
            str(row[0])
            for row in connection.execute(
                "select asset_id from fleet_assets order by asset_id asc"
            ).fetchall()
        ]
        critical_asset, nominal_asset = asset_ids
        connection.execute(
            """
            update fleet_assets
            set latest_risk_level = ?,
                latest_status = ?,
                latest_attention_json = ?
            where asset_id = ?
            """,
            (
                "critical",
                "maintenance_review",
                json.dumps(["RUL at or below critical threshold"]),
                critical_asset,
            ),
        )
        connection.execute(
            """
            update fleet_assets
            set latest_risk_level = ?,
                latest_status = ?,
                latest_attention_json = ?
            where asset_id = ?
            """,
            ("nominal", "nominal", json.dumps([]), nominal_asset),
        )
    return {"critical": critical_asset, "nominal": nominal_asset}


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


def _write_manual_prediction_run(
    database_path,
    *,
    subset: str,
    model_name: str,
    artifact_id: str,
    source_name: str,
    predicted_rul: float,
    monitoring: dict[str, object],
    created_at_utc: str,
) -> str:
    telemetry = pd.DataFrame({"unit_number": [1], "time_in_cycles": [10]})
    prediction_document = {
        "dataset": "C-MAPSS",
        "subset": subset,
        "model_name": model_name,
        "artifact": {"artifact_id": artifact_id},
        "monitoring": monitoring,
        "predictions": [
            {
                "unit_number": 1,
                "predicted_rul": predicted_rul,
                "predicted_rul_lower": predicted_rul,
                "predicted_rul_upper": predicted_rul + 10,
                "interval_method": "fixture",
                "interval_quantile_level": 0.9,
            }
        ],
    }
    run_id = record_prediction_run(
        database_path,
        telemetry=telemetry,
        prediction_document=prediction_document,
        model_artifact_path=f"models/{artifact_id}.joblib",
        source_name=source_name,
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "update prediction_runs set created_at_utc = ? where run_id = ?",
            (created_at_utc, run_id),
        )
        connection.execute(
            "update predictions set created_at_utc = ? where run_id = ?",
            (created_at_utc, run_id),
        )
        connection.execute(
            """
            update fleet_assets
            set first_seen_at_utc = ?,
                last_seen_at_utc = ?
            where latest_run_id = ?
            """,
            (created_at_utc, created_at_utc, run_id),
        )
    return run_id


def _write_anomaly_comparison_csv(path):
    rows = [
        {
            "channel_id": "P-1",
            "spacecraft": "SMAP",
            "source": "classical",
            "model_name": "robust_zscore",
            "rank_by_f1": 1,
            "precision": 0.25,
            "recall": 0.4,
            "f1": 0.2,
            "point_adjusted_f1": 0.6,
            "false_alarm_rate": 0.2,
            "miss_rate": 0.6,
            "support": 5,
            "predicted_positives": 2,
            "train_rows": 10,
            "test_rows": 12,
            "anomaly_points": 5,
        },
        {
            "channel_id": "P-1",
            "spacecraft": "SMAP",
            "source": "lstm",
            "model_name": "lstm_forecast_dynamic_threshold",
            "rank_by_f1": 2,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "point_adjusted_f1": 0.5,
            "false_alarm_rate": 0.1,
            "miss_rate": 0.5,
            "support": 5,
            "predicted_positives": 2,
            "train_rows": 10,
            "test_rows": 12,
            "anomaly_points": 5,
        },
        {
            "channel_id": "M-1",
            "spacecraft": "MSL",
            "source": "classical",
            "model_name": "pca_reconstruction",
            "rank_by_f1": 1,
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.8,
            "point_adjusted_f1": 0.9,
            "false_alarm_rate": 0.01,
            "miss_rate": 0.1,
            "support": 3,
            "predicted_positives": 0,
            "train_rows": 10,
            "test_rows": 12,
            "anomaly_points": 3,
        },
    ]
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_anomaly_events_csv(path):
    rows = [
        {
            "channel_id": "P-1",
            "spacecraft": "SMAP",
            "event_time_utc": "2026-01-01T00:00:00+00:00",
            "severity": "medium",
            "active": False,
            "anomaly_score": 0.5,
            "threshold": 0.8,
            "model_name": "robust_zscore",
            "source": "ops",
            "note": "earlier cleared event",
        },
        {
            "channel_id": "P-1",
            "spacecraft": "SMAP",
            "event_time_utc": "2026-01-02T00:00:00+00:00",
            "severity": "critical",
            "active": True,
            "anomaly_score": 0.95,
            "threshold": 0.8,
            "model_name": "robust_zscore",
            "source": "ops",
            "note": "battery telemetry excursion",
        },
        {
            "channel_id": "M-1",
            "spacecraft": "MSL",
            "event_time_utc": "2026-01-02T01:00:00+00:00",
            "severity": "info",
            "active": False,
            "anomaly_score": 0.1,
            "threshold": 0.8,
            "model_name": "pca_reconstruction",
            "source": "ops",
            "note": "nominal heartbeat",
        },
    ]
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


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
            "interval_quantile_level": 0.9,
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
