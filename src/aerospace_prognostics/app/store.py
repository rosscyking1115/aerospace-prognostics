"""SQLite persistence for the Aerospace PHM console."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aerospace_prognostics.app.anomaly_assets import (
    anomaly_asset_attention_reasons as _anomaly_asset_attention_reasons,
)
from aerospace_prognostics.app.anomaly_assets import (
    anomaly_asset_id as _anomaly_asset_id,
)
from aerospace_prognostics.app.anomaly_assets import (
    anomaly_asset_risk_level as _anomaly_asset_risk_level,
)
from aerospace_prognostics.app.anomaly_assets import (
    anomaly_asset_status as _anomaly_asset_status,
)
from aerospace_prognostics.app.anomaly_assets import (
    anomaly_event_attention_reasons as _anomaly_event_attention_reasons,
)
from aerospace_prognostics.app.anomaly_assets import (
    anomaly_event_risk_level as _anomaly_event_risk_level,
)
from aerospace_prognostics.app.anomaly_assets import (
    anomaly_event_threshold_crossed as _anomaly_event_threshold_crossed,
)
from aerospace_prognostics.app.anomaly_assets import (
    best_anomaly_asset_rows as _best_anomaly_asset_rows,
)
from aerospace_prognostics.app.anomaly_assets import (
    latest_anomaly_event_rows as _latest_anomaly_event_rows,
)
from aerospace_prognostics.app.dashboard_state import QuickstartWorkspace
from aerospace_prognostics.app.database import REQUIRED_TABLES as REQUIRED_TABLES
from aerospace_prognostics.app.database import SCHEMA_VERSION, initialize_app_database
from aerospace_prognostics.app.database import connect as _connect
from aerospace_prognostics.app.database import metadata_value as _metadata_value
from aerospace_prognostics.app.database import prepare_database as _prepare_database
from aerospace_prognostics.app.fleet_registry import (
    fleet_asset_export_rows as _fleet_asset_export_rows,
)
from aerospace_prognostics.app.fleet_registry import (
    fleet_asset_filters as _fleet_asset_filters,
)
from aerospace_prognostics.app.fleet_registry import (
    fleet_asset_registry_summary as _fleet_asset_registry_summary,
)
from aerospace_prognostics.app.model_registry import (
    model_artifact_report_card as _model_report_card,
)
from aerospace_prognostics.app.prediction_runs import (
    build_prediction_run_evidence_payload as _build_prediction_run_evidence_payload,
)
from aerospace_prognostics.app.prediction_runs import (
    event_from_row as _event_from_row,
)
from aerospace_prognostics.app.prediction_runs import (
    outcome_rows as _outcome_rows,
)
from aerospace_prognostics.app.prediction_runs import (
    outcome_template_frame as _outcome_template_frame,
)
from aerospace_prognostics.app.prediction_runs import (
    prediction_rows as _prediction_rows,
)
from aerospace_prognostics.app.prediction_runs import (
    prediction_run_detail as _prediction_run_detail,
)
from aerospace_prognostics.app.prediction_runs import (
    prediction_run_event_record as _prediction_run_event_record,
)
from aerospace_prognostics.app.prediction_runs import (
    prediction_run_export_summary as _prediction_run_export_summary,
)
from aerospace_prognostics.app.prediction_runs import (
    run_summary_from_row as _run_summary_from_row,
)
from aerospace_prognostics.app.prediction_runs import (
    with_interval_availability as _with_interval_availability,
)
from aerospace_prognostics.app.priority_policy import (
    fleet_asset_priority as _fleet_asset_priority,
)
from aerospace_prognostics.app.priority_policy import (
    fleet_priority_policy_summary as _fleet_priority_policy_summary,
)
from aerospace_prognostics.app.priority_policy import (
    fleet_priority_policy_validation_checks as _fleet_priority_policy_validation_checks,
)
from aerospace_prognostics.app.priority_policy import (
    render_fleet_priority_policy_validation_markdown,
)
from aerospace_prognostics.app.release_evidence import (
    evidence_from_row as _evidence_from_row,
)
from aerospace_prognostics.app.release_evidence import (
    release_evidence_record as _release_evidence_record,
)
from aerospace_prognostics.app.turbofan_assets import (
    turbofan_asset_attention_reasons as _turbofan_asset_attention_reasons,
)
from aerospace_prognostics.app.turbofan_assets import (
    turbofan_asset_risk_level as _turbofan_asset_risk_level,
)
from aerospace_prognostics.app.turbofan_assets import (
    turbofan_asset_status as _turbofan_asset_status,
)
from aerospace_prognostics.artifact_io import write_json_payload


def seed_quickstart_workspace(
    database_path: str | Path,
    workspace: QuickstartWorkspace,
) -> dict[str, int]:
    """Seed database records from quickstart model and release evidence files."""
    if not workspace.model_artifact_path.exists():
        return {"model_artifacts": 0, "release_evidence": 0}
    return register_model_artifact_evidence(
        database_path,
        model_artifact_path=workspace.model_artifact_path,
        inspection=workspace.artifact_inspection or {},
        inspection_source_path=workspace.artifact_inspection_path,
        release_evidence=_workspace_evidence_without_inspection(workspace),
    )


def register_model_artifact_evidence(
    database_path: str | Path,
    *,
    model_artifact_path: str | Path,
    inspection: dict[str, Any],
    inspection_source_path: str | Path | None = None,
    release_evidence: Iterable[tuple[str, str | Path, dict[str, Any] | None]] = (),
) -> dict[str, int | str]:
    """Register one model artifact and optional JSON release evidence."""
    artifact_path = Path(model_artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"model artifact not found: {artifact_path}")
    db_path = initialize_app_database(database_path)
    inserted: dict[str, int | str] = {
        "artifact_id": _artifact_id_from_inspection(inspection, artifact_path),
        "model_artifacts": 0,
        "release_evidence": 0,
    }
    timestamp = _now()
    with _connect(db_path) as connection:
        _upsert_model_artifact(
            connection,
            artifact_id=str(inserted["artifact_id"]),
            artifact_path=artifact_path,
            inspection=inspection,
            timestamp=timestamp,
        )
        inserted["model_artifacts"] = 1
        evidence_items = list(release_evidence)
        if inspection_source_path is not None:
            evidence_items.insert(
                0,
                ("artifact_inspection", Path(inspection_source_path), inspection),
            )
        for evidence_type, path, payload in evidence_items:
            evidence_path = Path(path)
            if payload is None or not evidence_path.exists():
                continue
            inserted["release_evidence"] = int(inserted["release_evidence"]) + (
                _insert_release_evidence(
                    connection,
                    artifact_id=str(inserted["artifact_id"]),
                    evidence_type=evidence_type,
                    path=evidence_path,
                    payload=payload,
                    timestamp=timestamp,
                )
            )
    return inserted


def record_prediction_run(
    database_path: str | Path,
    *,
    telemetry: pd.DataFrame,
    prediction_document: dict[str, Any],
    model_artifact_path: str | Path,
    source_name: str,
) -> str:
    """Persist one telemetry upload and its prediction rows."""
    db_path = initialize_app_database(database_path)
    timestamp = _now()
    telemetry_hash = _dataframe_sha256(telemetry)
    upload_id = f"upload-{telemetry_hash[:16]}"
    artifact = prediction_document.get("artifact")
    artifact = artifact if isinstance(artifact, dict) else {}
    artifact_id = artifact.get("artifact_id")
    run_material = {
        "upload_id": upload_id,
        "artifact_id": artifact_id,
        "model_artifact_path": str(model_artifact_path),
        "prediction_document": prediction_document,
        "timestamp": timestamp,
    }
    run_id = f"run-{_sha256_text(_json_dumps(run_material))[:16]}"
    predictions = _prediction_rows(prediction_document)
    with _connect(db_path) as connection:
        connection.execute(
            """
            insert into telemetry_uploads (
                upload_id,
                source_name,
                row_count,
                column_count,
                content_sha256,
                created_at_utc
            )
            values (?, ?, ?, ?, ?, ?)
            on conflict(upload_id) do update set
                source_name = excluded.source_name,
                row_count = excluded.row_count,
                column_count = excluded.column_count
            """,
            (
                upload_id,
                source_name,
                int(len(telemetry)),
                int(len(telemetry.columns)),
                telemetry_hash,
                timestamp,
            ),
        )
        connection.execute(
            """
            insert into prediction_runs (
                run_id,
                upload_id,
                artifact_id,
                model_artifact_path,
                dataset,
                subset,
                model_name,
                prediction_count,
                monitoring_json,
                created_at_utc
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                upload_id,
                artifact_id,
                str(model_artifact_path),
                prediction_document.get("dataset"),
                prediction_document.get("subset"),
                prediction_document.get("model_name"),
                len(predictions),
                _json_dumps(prediction_document.get("monitoring", {})),
                timestamp,
            ),
        )
        connection.executemany(
            """
            insert into predictions (
                run_id,
                asset_id,
                unit_number,
                predicted_rul,
                predicted_rul_lower,
                predicted_rul_upper,
                interval_method,
                interval_quantile_level,
                created_at_utc
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    run_id,
                    f"{prediction_document.get('subset', 'unknown')}-unit-{row['unit_number']}",
                    row["unit_number"],
                    row["predicted_rul"],
                    row.get("predicted_rul_lower"),
                    row.get("predicted_rul_upper"),
                    row.get("interval_method"),
                    row.get("interval_quantile_level"),
                    timestamp,
                )
                for row in predictions
            ),
        )
        _upsert_fleet_assets_for_run(connection, run_id=run_id, timestamp=timestamp)
        _insert_prediction_run_event(
            connection,
            run_id=run_id,
            event_type="prediction_recorded",
            status="recorded",
            actor="system",
            note="Prediction run persisted",
            payload={
                "source_name": source_name,
                "prediction_count": len(predictions),
                "artifact_id": artifact_id,
            },
            timestamp=timestamp,
        )
    return run_id


def record_prediction_run_event(
    database_path: str | Path,
    *,
    run_id: str,
    event_type: str,
    actor: str = "operator",
    status: str | None = None,
    note: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str:
    """Append an auditable event to a persisted prediction run."""
    db_path = initialize_app_database(database_path)
    timestamp = _now()
    with _connect(db_path) as connection:
        if not _prediction_run_exists(connection, run_id):
            raise ValueError(f"unknown prediction run: {run_id}")
        return _insert_prediction_run_event(
            connection,
            run_id=run_id,
            event_type=event_type,
            status=status,
            actor=actor,
            note=note,
            payload=payload,
            timestamp=timestamp,
        )


def record_prediction_outcomes(
    database_path: str | Path,
    *,
    run_id: str,
    outcomes: pd.DataFrame,
    source_name: str,
    actor: str = "operator",
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Attach observed RUL outcomes to an existing prediction run."""
    db_path = initialize_app_database(database_path)
    timestamp = _now()
    outcome_rows = _outcome_rows(outcomes)
    with _connect(db_path) as connection:
        if not _prediction_run_exists(connection, run_id):
            raise ValueError(f"unknown prediction run: {run_id}")
        expected_units = {
            int(row[0])
            for row in connection.execute(
                "select unit_number from predictions where run_id = ?",
                (run_id,),
            ).fetchall()
        }
        unknown_units = sorted(
            int(row["unit_number"])
            for row in outcome_rows
            if int(row["unit_number"]) not in expected_units
        )
        if unknown_units:
            units = ", ".join(str(unit) for unit in unknown_units)
            raise ValueError(f"outcome unit_number values are not in run {run_id}: {units}")
        connection.executemany(
            """
            insert into prediction_outcomes (
                run_id,
                unit_number,
                actual_rul,
                outcome_source,
                observed_at_utc,
                recorded_at_utc
            )
            values (?, ?, ?, ?, ?, ?)
            on conflict(run_id, unit_number) do update set
                actual_rul = excluded.actual_rul,
                outcome_source = excluded.outcome_source,
                observed_at_utc = excluded.observed_at_utc,
                recorded_at_utc = excluded.recorded_at_utc
            """,
            (
                (
                    run_id,
                    int(row["unit_number"]),
                    float(row["actual_rul"]),
                    source_name,
                    observed_at_utc,
                    timestamp,
                )
                for row in outcome_rows
            ),
        )
        event_id = _insert_prediction_run_event(
            connection,
            run_id=run_id,
            event_type="outcomes_recorded",
            status="recorded",
            actor=actor,
            note="Observed outcomes attached",
            payload={
                "source_name": source_name,
                "outcome_count": len(outcome_rows),
                "observed_at_utc": observed_at_utc,
            },
            timestamp=timestamp,
        )
    return {"outcome_count": len(outcome_rows), "event_id": event_id}


def sync_fleet_assets_from_prediction_run(
    database_path: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Refresh fleet-asset registry rows from one or all prediction runs."""
    db_path = initialize_app_database(database_path)
    timestamp = _now()
    with _connect(db_path) as connection:
        if run_id is not None and not _prediction_run_exists(connection, run_id):
            raise ValueError(f"unknown prediction run: {run_id}")
        run_ids = (
            [run_id]
            if run_id is not None
            else [
                str(row[0])
                for row in connection.execute(
                    "select run_id from prediction_runs order by created_at_utc asc"
                ).fetchall()
            ]
        )
        updated_assets = 0
        for current_run_id in run_ids:
            updated_assets += _upsert_fleet_assets_for_run(
                connection,
                run_id=str(current_run_id),
                timestamp=timestamp,
            )
    return {
        "run_id": run_id,
        "runs_synced": len(run_ids),
        "updated_assets": updated_assets,
    }


def sync_fleet_assets_from_anomaly_comparison(
    database_path: str | Path,
    *,
    comparison_csv: str | Path,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Refresh spacecraft anomaly channel assets from a ranked comparison CSV."""
    comparison_path = Path(comparison_csv)
    if not comparison_path.exists():
        raise FileNotFoundError(f"anomaly comparison CSV not found: {comparison_path}")
    rows = _best_anomaly_asset_rows(comparison_path)
    db_path = initialize_app_database(database_path)
    timestamp = _now()
    with _connect(db_path) as connection:
        updated_assets = 0
        for row in rows:
            updated_assets += _upsert_anomaly_fleet_asset(
                connection,
                row=row,
                comparison_path=comparison_path,
                source_name=source_name or str(comparison_path),
                timestamp=timestamp,
            )
    return {
        "source_path": str(comparison_path),
        "source_name": source_name or str(comparison_path),
        "channels_synced": len(rows),
        "updated_assets": updated_assets,
    }


def sync_fleet_assets_from_anomaly_events(
    database_path: str | Path,
    *,
    events_csv: str | Path,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Refresh spacecraft anomaly channel assets from operational event rows."""
    events_path = Path(events_csv)
    if not events_path.exists():
        raise FileNotFoundError(f"anomaly events CSV not found: {events_path}")
    rows, event_count = _latest_anomaly_event_rows(events_path)
    db_path = initialize_app_database(database_path)
    timestamp = _now()
    with _connect(db_path) as connection:
        updated_assets = 0
        for row in rows:
            updated_assets += _upsert_anomaly_event_fleet_asset(
                connection,
                row=row,
                events_path=events_path,
                source_name=source_name or str(events_path),
                timestamp=timestamp,
            )
    return {
        "source_path": str(events_path),
        "source_name": source_name or str(events_path),
        "events_processed": event_count,
        "channels_synced": len(rows),
        "updated_assets": updated_assets,
    }


def inspect_anomaly_events_csv(events_csv: str | Path) -> dict[str, Any]:
    """Validate and summarize operational spacecraft anomaly event rows."""
    events_path = Path(events_csv)
    if not events_path.exists():
        raise FileNotFoundError(f"anomaly events CSV not found: {events_path}")
    rows, event_count = _latest_anomaly_event_rows(events_path)
    risk_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    active_events = 0
    threshold_crossings = 0
    for row in rows:
        risk_level = _anomaly_event_risk_level(row)
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        severity = str(row.get("severity") or "unspecified")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        if bool(row.get("active")):
            active_events += 1
        if _anomaly_event_threshold_crossed(row):
            threshold_crossings += 1
    return {
        "source_path": str(events_path),
        "events_processed": event_count,
        "channels_synced": len(rows),
        "risk_counts": dict(sorted(risk_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "active_events": active_events,
        "threshold_crossings": threshold_crossings,
        "latest_events": rows,
    }


def list_fleet_assets(
    database_path: str | Path,
    *,
    limit: int = 100,
    risk_levels: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    attention_only: bool = False,
    read_only: bool = False,
) -> list[dict[str, Any]]:
    """Return the current fleet asset registry ordered by operational priority."""
    db_path = _prepare_database(database_path, read_only=read_only)
    where_clauses: list[str] = []
    parameters: list[Any] = []
    _extend_in_filter(
        where_clauses,
        parameters,
        column="latest_risk_level",
        values=risk_levels,
    )
    _extend_in_filter(where_clauses, parameters, column="domain", values=domains)
    _extend_in_filter(where_clauses, parameters, column="latest_status", values=statuses)
    if attention_only:
        where_clauses.append(
            "(latest_risk_level in ('critical', 'watch') or latest_attention_json != '[]')"
        )
    where_sql = f"where {' and '.join(where_clauses)}" if where_clauses else ""
    with _connect(db_path, read_only=read_only) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            select
                asset_id,
                asset_type,
                domain,
                source_dataset,
                source_subset,
                external_id,
                latest_run_id,
                latest_rul_prediction,
                latest_rul_lower,
                latest_rul_upper,
                latest_risk_level,
                latest_status,
                latest_attention_json,
                first_seen_at_utc,
                last_seen_at_utc,
                metadata_json
            from fleet_assets
            {where_sql}
            order by last_seen_at_utc desc
            """,
            tuple(parameters),
        ).fetchall()
    assets = [_fleet_asset_from_row(row) for row in rows]
    return sorted(
        assets,
        key=lambda asset: (
            float(asset.get("priority_score") or 0),
            str(asset.get("last_seen_at_utc") or ""),
            str(asset.get("asset_id") or ""),
        ),
        reverse=True,
    )[: int(limit)]


def build_fleet_asset_registry_bundle(
    database_path: str | Path,
    *,
    assets_csv_path: str | Path | None = None,
    risk_levels: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    attention_only: bool = False,
    read_only: bool = False,
) -> dict[str, Any]:
    """Build a portable fleet-asset registry payload without writing files."""
    normalized_filters = _fleet_asset_filters(
        risk_levels=risk_levels,
        domains=domains,
        statuses=statuses,
        attention_only=attention_only,
    )
    assets = list_fleet_assets(
        database_path,
        limit=10000,
        risk_levels=normalized_filters["risk_levels"],
        domains=normalized_filters["domains"],
        statuses=normalized_filters["statuses"],
        attention_only=bool(normalized_filters["attention_only"]),
        read_only=read_only,
    )
    summary = database_summary(database_path, read_only=read_only)
    csv_file: dict[str, Any] = {"rows": len(assets)}
    if assets_csv_path is not None:
        csv_file["path"] = str(Path(assets_csv_path))
    return {
        "schema_version": "aerospace-prognostics/fleet-asset-registry/v1",
        "exported_at_utc": _now(),
        "database": {
            "path": str(Path(database_path)),
            "schema_version": summary["schema_version"],
        },
        "filters": normalized_filters,
        "summary": _fleet_asset_registry_summary(assets),
        "priority_policy": _fleet_priority_policy_summary(assets),
        "assets": assets,
        "files": {
            "assets_csv": csv_file,
        },
    }


def export_fleet_asset_registry(
    database_path: str | Path,
    *,
    output_dir: str | Path,
    risk_levels: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    attention_only: bool = False,
) -> dict[str, Any]:
    """Export the fleet asset registry as JSON evidence and CSV rows."""
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    assets_path = export_dir / "fleet_assets.csv"
    evidence_path = export_dir / "fleet_asset_registry.json"

    bundle = build_fleet_asset_registry_bundle(
        database_path,
        assets_csv_path=assets_path,
        risk_levels=risk_levels,
        domains=domains,
        statuses=statuses,
        attention_only=attention_only,
    )
    assets = list(bundle["assets"])
    pd.DataFrame(_fleet_asset_export_rows(assets)).to_csv(assets_path, index=False)
    bundle["files"]["assets_csv"]["sha256"] = _file_sha256(assets_path)
    write_json_payload(bundle, evidence_path, default=str)
    return {
        "output_dir": str(export_dir),
        "registry_json": str(evidence_path),
        "registry_sha256": _file_sha256(evidence_path),
        "assets_csv": str(assets_path),
        "assets_sha256": _file_sha256(assets_path),
        "asset_count": len(assets),
        "filters": bundle["filters"],
        "risk_counts": bundle["summary"]["risk_counts"],
    }


def build_fleet_priority_policy_validation(
    database_path: str | Path,
    *,
    read_only: bool = False,
) -> dict[str, Any]:
    """Build validation evidence for the fleet review-priority policy."""
    assets = list_fleet_assets(database_path, limit=10000, read_only=read_only)
    summary = database_summary(database_path, read_only=read_only)
    checks = _fleet_priority_policy_validation_checks(assets)
    failed_checks = [
        check["check_id"] for check in checks if check.get("status") == "fail"
    ]
    return {
        "schema_version": (
            "aerospace-prognostics/fleet-priority-policy-validation/v1"
        ),
        "exported_at_utc": _now(),
        "database": {
            "path": str(Path(database_path)),
            "schema_version": summary["schema_version"],
        },
        "asset_count": len(assets),
        "priority_policy": _fleet_priority_policy_summary(assets),
        "scenario_checks": checks,
        "failed_checks": failed_checks,
        "overall_status": "fail" if failed_checks else "pass",
    }


def export_fleet_priority_policy_validation(
    database_path: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export priority-policy validation evidence as JSON and Markdown."""
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    validation_json = export_dir / "fleet_priority_policy_validation.json"
    validation_markdown = export_dir / "fleet_priority_policy_validation.md"

    report = build_fleet_priority_policy_validation(database_path)
    write_json_payload(report, validation_json, default=str)
    validation_markdown.write_text(
        render_fleet_priority_policy_validation_markdown(report),
        encoding="utf-8",
    )
    return {
        "output_dir": str(export_dir),
        "validation_json": str(validation_json),
        "validation_sha256": _file_sha256(validation_json),
        "validation_markdown": str(validation_markdown),
        "markdown_sha256": _file_sha256(validation_markdown),
        "overall_status": report["overall_status"],
        "failed_checks": report["failed_checks"],
        "asset_count": report["asset_count"],
    }


def list_model_artifacts(
    database_path: str | Path,
    *,
    limit: int = 50,
    read_only: bool = False,
) -> list[dict[str, Any]]:
    """Return model artifacts with evidence and prediction usage counts."""
    db_path = _prepare_database(database_path, read_only=read_only)
    with _connect(db_path, read_only=read_only) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            select
                model_artifacts.artifact_id,
                model_artifacts.artifact_path,
                model_artifacts.artifact_sha256,
                model_artifacts.dataset,
                model_artifacts.subset,
                model_artifacts.model_name,
                model_artifacts.schema_version,
                model_artifacts.stage,
                model_artifacts.created_at_utc,
                count(distinct release_evidence.evidence_id) as evidence_count,
                count(distinct prediction_runs.run_id) as prediction_run_count,
                max(prediction_runs.created_at_utc) as latest_prediction_at
            from model_artifacts
            left join release_evidence
                on release_evidence.artifact_id = model_artifacts.artifact_id
            left join prediction_runs
                on prediction_runs.artifact_id = model_artifacts.artifact_id
            group by model_artifacts.artifact_id
            order by model_artifacts.created_at_utc desc
            limit ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def load_model_artifact(
    database_path: str | Path,
    artifact_id: str,
    *,
    read_only: bool = False,
) -> dict[str, Any] | None:
    """Load one model artifact with release evidence and prediction usage."""
    db_path = _prepare_database(database_path, read_only=read_only)
    with _connect(db_path, read_only=read_only) as connection:
        connection.row_factory = sqlite3.Row
        artifact_row = connection.execute(
            """
            select
                artifact_id,
                artifact_path,
                artifact_sha256,
                dataset,
                subset,
                model_name,
                schema_version,
                stage,
                inspection_json,
                created_at_utc
            from model_artifacts
            where artifact_id = ?
            """,
            (artifact_id,),
        ).fetchone()
        if artifact_row is None:
            return None
        evidence_rows = connection.execute(
            """
            select
                evidence_id,
                evidence_type,
                source_path,
                status,
                payload_json,
                created_at_utc
            from release_evidence
            where artifact_id = ?
            order by evidence_type asc, created_at_utc desc
            """,
            (artifact_id,),
        ).fetchall()
        run_rows = connection.execute(
            """
            select
                prediction_runs.run_id,
                prediction_runs.created_at_utc,
                prediction_runs.model_name,
                prediction_runs.prediction_count,
                telemetry_uploads.source_name,
                telemetry_uploads.content_sha256,
                count(predictions.unit_number) as prediction_row_count,
                sum(
                    case
                        when predictions.predicted_rul_lower is not null
                            and predictions.predicted_rul_upper is not null
                        then 1 else 0
                    end
                ) as interval_count,
                avg(predictions.predicted_rul_upper - predictions.predicted_rul_lower)
                    as mean_interval_width,
                max(predictions.predicted_rul_upper - predictions.predicted_rul_lower)
                    as max_interval_width,
                count(prediction_outcomes.actual_rul) as outcome_count,
                avg(abs(predictions.predicted_rul - prediction_outcomes.actual_rul))
                    as mean_absolute_error,
                avg(predictions.predicted_rul - prediction_outcomes.actual_rul)
                    as mean_signed_error,
                sum(
                    case
                        when prediction_outcomes.actual_rul is not null
                            and predictions.predicted_rul_lower is not null
                            and predictions.predicted_rul_upper is not null
                        then 1 else 0
                    end
                ) as interval_outcome_count,
                sum(
                    case
                        when prediction_outcomes.actual_rul is not null
                            and predictions.predicted_rul_lower is not null
                            and predictions.predicted_rul_upper is not null
                            and prediction_outcomes.actual_rul
                                between predictions.predicted_rul_lower
                                and predictions.predicted_rul_upper
                        then 1 else 0
                    end
                ) as interval_covered_count
            from prediction_runs
            join telemetry_uploads on telemetry_uploads.upload_id = prediction_runs.upload_id
            left join predictions on predictions.run_id = prediction_runs.run_id
            left join prediction_outcomes
                on prediction_outcomes.run_id = predictions.run_id
                and prediction_outcomes.unit_number = predictions.unit_number
            where prediction_runs.artifact_id = ?
            group by prediction_runs.run_id
            order by prediction_runs.created_at_utc desc
            limit 25
            """,
            (artifact_id,),
        ).fetchall()
    artifact = dict(artifact_row)
    artifact["inspection"] = _json_loads(artifact.pop("inspection_json"))
    release_evidence = [_evidence_from_row(row) for row in evidence_rows]
    prediction_runs = [_with_interval_availability(dict(row)) for row in run_rows]
    return {
        "artifact": artifact,
        "release_evidence": release_evidence,
        "prediction_runs": prediction_runs,
        "report_card": _model_report_card(
            artifact,
            release_evidence=release_evidence,
            prediction_runs=prediction_runs,
        ),
    }


def build_model_artifact_review_bundle(
    database_path: str | Path,
    *,
    artifact_id: str,
    read_only: bool = False,
) -> dict[str, Any]:
    """Build a portable model-artifact review bundle without writing files."""
    loaded = load_model_artifact(database_path, artifact_id, read_only=read_only)
    if loaded is None:
        raise ValueError(f"unknown model artifact: {artifact_id}")

    summary = database_summary(database_path, read_only=read_only)
    return {
        "schema_version": "aerospace-prognostics/model-artifact-review/v1",
        "exported_at_utc": _now(),
        "database": {
            "path": str(Path(database_path)),
            "schema_version": summary["schema_version"],
        },
        "artifact": loaded["artifact"],
        "report_card": loaded["report_card"],
        "release_evidence": loaded["release_evidence"],
        "prediction_runs": loaded["prediction_runs"],
        "counts": {
            "release_evidence": len(loaded["release_evidence"]),
            "prediction_runs": len(loaded["prediction_runs"]),
        },
    }


def list_prediction_runs(
    database_path: str | Path,
    *,
    limit: int = 50,
    model_names: Iterable[str] | None = None,
    artifact_ids: Iterable[str] | None = None,
    asset_ids: Iterable[str] | None = None,
    risk_levels: Iterable[str] | None = None,
    decision_statuses: Iterable[str] | None = None,
    start_created_at_utc: str | None = None,
    end_created_at_utc: str | None = None,
    drift_only: bool = False,
    read_only: bool = False,
) -> list[dict[str, Any]]:
    """Return recent prediction runs with upload and aggregate prediction context."""
    db_path = _prepare_database(database_path, read_only=read_only)
    where_clauses: list[str] = []
    parameters: list[Any] = []
    _extend_in_filter(
        where_clauses,
        parameters,
        column="prediction_runs.model_name",
        values=model_names,
    )
    _extend_in_filter(
        where_clauses,
        parameters,
        column="prediction_runs.artifact_id",
        values=artifact_ids,
    )
    if start_created_at_utc:
        where_clauses.append("prediction_runs.created_at_utc >= ?")
        parameters.append(start_created_at_utc)
    if end_created_at_utc:
        where_clauses.append("prediction_runs.created_at_utc <= ?")
        parameters.append(end_created_at_utc)
    _extend_prediction_run_asset_filter(where_clauses, parameters, asset_ids=asset_ids)
    _extend_prediction_run_risk_filter(where_clauses, parameters, risk_levels=risk_levels)
    _extend_prediction_run_decision_filter(
        where_clauses,
        parameters,
        decision_statuses=decision_statuses,
    )
    if drift_only:
        where_clauses.append(
            """
            (
                (
                    prediction_runs.monitoring_json like '%"alert_columns":[%'
                    and prediction_runs.monitoring_json not like '%"alert_columns":[]%'
                )
                or (
                    prediction_runs.monitoring_json like '%"alerts":[%'
                    and prediction_runs.monitoring_json not like '%"alerts":[]%'
                )
            )
            """
        )
    where_sql = f"where {' and '.join(where_clauses)}" if where_clauses else ""
    with _connect(db_path, read_only=read_only) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            select
                prediction_runs.run_id,
                prediction_runs.created_at_utc,
                prediction_runs.artifact_id,
                prediction_runs.model_artifact_path,
                prediction_runs.dataset,
                prediction_runs.subset,
                prediction_runs.model_name,
                prediction_runs.prediction_count,
                prediction_runs.monitoring_json,
                telemetry_uploads.source_name,
                telemetry_uploads.row_count,
                telemetry_uploads.column_count,
                telemetry_uploads.content_sha256,
                min(predictions.predicted_rul) as min_predicted_rul,
                avg(predictions.predicted_rul) as mean_predicted_rul,
                max(predictions.predicted_rul) as max_predicted_rul,
                sum(
                    case
                        when predictions.predicted_rul_lower is not null
                            and predictions.predicted_rul_upper is not null
                        then 1 else 0
                    end
                ) as interval_count,
                avg(predictions.predicted_rul_upper - predictions.predicted_rul_lower)
                    as mean_interval_width,
                max(predictions.predicted_rul_upper - predictions.predicted_rul_lower)
                    as max_interval_width,
                count(prediction_outcomes.actual_rul) as outcome_count,
                avg(abs(predictions.predicted_rul - prediction_outcomes.actual_rul))
                    as mean_absolute_error,
                avg(predictions.predicted_rul - prediction_outcomes.actual_rul)
                    as mean_signed_error,
                sum(
                    case
                        when prediction_outcomes.actual_rul is not null
                            and predictions.predicted_rul_lower is not null
                            and predictions.predicted_rul_upper is not null
                        then 1 else 0
                    end
                ) as interval_outcome_count,
                sum(
                    case
                        when prediction_outcomes.actual_rul is not null
                            and predictions.predicted_rul_lower is not null
                            and predictions.predicted_rul_upper is not null
                            and prediction_outcomes.actual_rul
                                between predictions.predicted_rul_lower
                                and predictions.predicted_rul_upper
                        then 1 else 0
                    end
                ) as interval_covered_count,
                (
                    select count(*)
                    from prediction_run_events
                    where prediction_run_events.run_id = prediction_runs.run_id
                ) as audit_event_count,
                (
                    select status
                    from prediction_run_events
                    where prediction_run_events.run_id = prediction_runs.run_id
                        and prediction_run_events.event_type = 'operator_decision'
                    order by prediction_run_events.created_at_utc desc
                    limit 1
                ) as decision_status,
                (
                    select note
                    from prediction_run_events
                    where prediction_run_events.run_id = prediction_runs.run_id
                        and prediction_run_events.event_type = 'operator_decision'
                    order by prediction_run_events.created_at_utc desc
                    limit 1
                ) as decision_note
            from prediction_runs
            join telemetry_uploads on telemetry_uploads.upload_id = prediction_runs.upload_id
            left join predictions on predictions.run_id = prediction_runs.run_id
            left join prediction_outcomes
                on prediction_outcomes.run_id = predictions.run_id
                and prediction_outcomes.unit_number = predictions.unit_number
            {where_sql}
            group by prediction_runs.run_id
            order by prediction_runs.created_at_utc desc
            limit ?
            """,
            (*parameters, int(limit)),
        ).fetchall()
    return [_run_summary_from_row(row) for row in rows]


def load_prediction_run(
    database_path: str | Path,
    run_id: str,
    *,
    read_only: bool = False,
) -> dict[str, Any] | None:
    """Load one prediction run and all persisted prediction rows."""
    db_path = _prepare_database(database_path, read_only=read_only)
    with _connect(db_path, read_only=read_only) as connection:
        connection.row_factory = sqlite3.Row
        run_row = connection.execute(
            """
            select
                prediction_runs.run_id,
                prediction_runs.created_at_utc,
                prediction_runs.artifact_id,
                prediction_runs.model_artifact_path,
                prediction_runs.dataset,
                prediction_runs.subset,
                prediction_runs.model_name,
                prediction_runs.prediction_count,
                prediction_runs.monitoring_json,
                telemetry_uploads.source_name,
                telemetry_uploads.row_count,
                telemetry_uploads.column_count,
                telemetry_uploads.content_sha256,
                (
                    select count(*)
                    from prediction_run_events
                    where prediction_run_events.run_id = prediction_runs.run_id
                ) as audit_event_count,
                (
                    select status
                    from prediction_run_events
                    where prediction_run_events.run_id = prediction_runs.run_id
                        and prediction_run_events.event_type = 'operator_decision'
                    order by prediction_run_events.created_at_utc desc
                    limit 1
                ) as decision_status,
                (
                    select note
                    from prediction_run_events
                    where prediction_run_events.run_id = prediction_runs.run_id
                        and prediction_run_events.event_type = 'operator_decision'
                    order by prediction_run_events.created_at_utc desc
                    limit 1
                ) as decision_note
            from prediction_runs
            join telemetry_uploads on telemetry_uploads.upload_id = prediction_runs.upload_id
            where prediction_runs.run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if run_row is None:
            return None
        prediction_rows = connection.execute(
            """
            select
                predictions.asset_id,
                predictions.unit_number,
                predictions.predicted_rul,
                predictions.predicted_rul_lower,
                predictions.predicted_rul_upper,
                predictions.interval_method,
                predictions.interval_quantile_level,
                prediction_outcomes.actual_rul,
                predictions.predicted_rul - prediction_outcomes.actual_rul as signed_error,
                abs(predictions.predicted_rul - prediction_outcomes.actual_rul)
                    as absolute_error,
                case
                    when prediction_outcomes.actual_rul is not null
                        and predictions.predicted_rul_lower is not null
                        and predictions.predicted_rul_upper is not null
                    then prediction_outcomes.actual_rul
                        between predictions.predicted_rul_lower
                        and predictions.predicted_rul_upper
                    else null
                end as interval_covered,
                prediction_outcomes.outcome_source,
                prediction_outcomes.observed_at_utc,
                prediction_outcomes.recorded_at_utc,
                predictions.created_at_utc
            from predictions
            left join prediction_outcomes
                on prediction_outcomes.run_id = predictions.run_id
                and prediction_outcomes.unit_number = predictions.unit_number
            where predictions.run_id = ?
            order by predictions.predicted_rul asc, predictions.unit_number asc
            """,
            (run_id,),
        ).fetchall()
        event_rows = connection.execute(
            """
            select
                event_id,
                event_type,
                status,
                actor,
                note,
                payload_json,
                created_at_utc
            from prediction_run_events
            where run_id = ?
            order by created_at_utc desc
            """,
            (run_id,),
        ).fetchall()
    return _prediction_run_detail(
        run_row=run_row,
        prediction_rows=prediction_rows,
        event_rows=event_rows,
    )


def export_prediction_run_evidence(
    database_path: str | Path,
    *,
    run_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export a portable evidence bundle for one prediction run."""
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = export_dir / f"{run_id}_predictions.csv"
    evidence_path = export_dir / f"{run_id}_evidence.json"

    manifest = build_prediction_run_evidence(
        database_path,
        run_id=run_id,
        predictions_csv_path=predictions_path,
    )
    predictions = list(manifest["predictions"])
    pd.DataFrame(predictions).to_csv(predictions_path, index=False)
    manifest["files"]["predictions_csv"]["sha256"] = _file_sha256(predictions_path)
    write_json_payload(manifest, evidence_path, default=str)
    return _prediction_run_export_summary(
        run_id=run_id,
        output_dir=export_dir,
        evidence_json_path=evidence_path,
        evidence_sha256=_file_sha256(evidence_path),
        predictions_csv_path=predictions_path,
        predictions_sha256=_file_sha256(predictions_path),
        manifest=manifest,
    )


def build_prediction_run_evidence(
    database_path: str | Path,
    *,
    run_id: str,
    predictions_csv_path: str | Path | None = None,
    read_only: bool = False,
) -> dict[str, Any]:
    """Build a portable prediction-run evidence payload without writing files."""
    loaded = load_prediction_run(database_path, run_id, read_only=read_only)
    if loaded is None:
        raise ValueError(f"unknown prediction run: {run_id}")

    summary = database_summary(database_path, read_only=read_only)
    return _build_prediction_run_evidence_payload(
        database_path=database_path,
        database_schema_version=summary["schema_version"],
        loaded_run=loaded,
        exported_at_utc=_now(),
        predictions_csv_path=predictions_csv_path,
    )


def export_prediction_outcome_template(
    database_path: str | Path,
    *,
    run_id: str,
    output_csv: str | Path,
) -> dict[str, Any]:
    """Export a fillable observed-RUL outcome CSV for one prediction run."""
    loaded = load_prediction_run(database_path, run_id)
    if loaded is None:
        raise ValueError(f"unknown prediction run: {run_id}")

    predictions = list(loaded["predictions"])
    template = _outcome_template_frame(predictions)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    return {
        "run_id": run_id,
        "outcome_template_csv": str(output_path),
        "outcome_template_sha256": _file_sha256(output_path),
        "prediction_count": len(predictions),
    }


def database_summary(
    database_path: str | Path,
    *,
    read_only: bool = False,
) -> dict[str, int | str]:
    """Return table counts and schema version for the app database."""
    db_path = _prepare_database(database_path, read_only=read_only)
    with _connect(db_path, read_only=read_only) as connection:
        summary: dict[str, int | str] = {
            "database_path": str(Path(database_path)),
            "schema_version": _metadata_value(connection, "schema_version") or SCHEMA_VERSION,
        }
        for table_name in (
            "model_artifacts",
            "release_evidence",
            "telemetry_uploads",
            "prediction_runs",
            "predictions",
            "prediction_outcomes",
            "prediction_run_events",
            "fleet_assets",
        ):
            summary[table_name] = int(
                connection.execute(f"select count(*) from {table_name}").fetchone()[0]
            )
    return summary


def list_prediction_run_events(
    database_path: str | Path,
    run_id: str,
    *,
    limit: int = 50,
    read_only: bool = False,
) -> list[dict[str, Any]]:
    """Return recent audit events for a prediction run."""
    db_path = _prepare_database(database_path, read_only=read_only)
    with _connect(db_path, read_only=read_only) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            select
                event_id,
                event_type,
                status,
                actor,
                note,
                payload_json,
                created_at_utc
            from prediction_run_events
            where run_id = ?
            order by created_at_utc desc
            limit ?
            """,
            (run_id, int(limit)),
        ).fetchall()
    return [_event_from_row(row) for row in rows]


def _prediction_run_exists(connection: sqlite3.Connection, run_id: str) -> bool:
    row = connection.execute(
        "select 1 from prediction_runs where run_id = ?",
        (run_id,),
    ).fetchone()
    return row is not None


def _insert_prediction_run_event(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    event_type: str,
    actor: str,
    timestamp: str,
    status: str | None = None,
    note: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str:
    event = _prediction_run_event_record(
        run_id=run_id,
        event_type=event_type,
        status=status,
        actor=actor,
        note=note,
        payload=payload,
        timestamp=timestamp,
    )
    connection.execute(
        """
        insert into prediction_run_events (
            event_id,
            run_id,
            event_type,
            status,
            actor,
            note,
            payload_json,
            created_at_utc
        )
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event["event_id"],
            event["run_id"],
            event["event_type"],
            event["status"],
            event["actor"],
            event["note"],
            event["payload_json"],
            event["created_at_utc"],
        ),
    )
    return str(event["event_id"])


def _upsert_fleet_assets_for_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    timestamp: str,
) -> int:
    connection.row_factory = sqlite3.Row
    run = connection.execute(
        """
        select
            prediction_runs.run_id,
            prediction_runs.dataset,
            prediction_runs.subset,
            prediction_runs.model_name,
            prediction_runs.artifact_id,
            prediction_runs.created_at_utc,
            telemetry_uploads.source_name,
            telemetry_uploads.content_sha256
        from prediction_runs
        join telemetry_uploads on telemetry_uploads.upload_id = prediction_runs.upload_id
        where prediction_runs.run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if run is None:
        return 0
    rows = connection.execute(
        """
        select
            asset_id,
            unit_number,
            predicted_rul,
            predicted_rul_lower,
            predicted_rul_upper,
            interval_method,
            interval_quantile_level
        from predictions
        where run_id = ?
        """,
        (run_id,),
    ).fetchall()
    updated = 0
    for row in rows:
        risk_level = _turbofan_asset_risk_level(
            predicted_rul=float(row["predicted_rul"]),
            predicted_rul_lower=_optional_float(row["predicted_rul_lower"]),
        )
        attention = _turbofan_asset_attention_reasons(
            predicted_rul=float(row["predicted_rul"]),
            predicted_rul_lower=_optional_float(row["predicted_rul_lower"]),
            predicted_rul_upper=_optional_float(row["predicted_rul_upper"]),
            risk_level=risk_level,
        )
        metadata = {
            "unit_number": int(row["unit_number"]),
            "model_name": run["model_name"],
            "artifact_id": run["artifact_id"],
            "source_name": run["source_name"],
            "input_sha256": run["content_sha256"],
            "interval_method": row["interval_method"],
            "interval_quantile_level": row["interval_quantile_level"],
        }
        cursor = connection.execute(
            """
            insert into fleet_assets (
                asset_id,
                asset_type,
                domain,
                source_dataset,
                source_subset,
                external_id,
                latest_run_id,
                latest_rul_prediction,
                latest_rul_lower,
                latest_rul_upper,
                latest_risk_level,
                latest_status,
                latest_attention_json,
                first_seen_at_utc,
                last_seen_at_utc,
                metadata_json
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(asset_id) do update set
                asset_type = excluded.asset_type,
                domain = excluded.domain,
                source_dataset = excluded.source_dataset,
                source_subset = excluded.source_subset,
                external_id = excluded.external_id,
                latest_run_id = excluded.latest_run_id,
                latest_rul_prediction = excluded.latest_rul_prediction,
                latest_rul_lower = excluded.latest_rul_lower,
                latest_rul_upper = excluded.latest_rul_upper,
                latest_risk_level = excluded.latest_risk_level,
                latest_status = excluded.latest_status,
                latest_attention_json = excluded.latest_attention_json,
                last_seen_at_utc = excluded.last_seen_at_utc,
                metadata_json = excluded.metadata_json
            where excluded.last_seen_at_utc >= fleet_assets.last_seen_at_utc
            """,
            (
                row["asset_id"],
                "engine",
                "turbofan_rul",
                run["dataset"],
                run["subset"],
                str(row["unit_number"]),
                run_id,
                float(row["predicted_rul"]),
                _optional_float(row["predicted_rul_lower"]),
                _optional_float(row["predicted_rul_upper"]),
                risk_level,
                _turbofan_asset_status(risk_level),
                _json_dumps(attention),
                run["created_at_utc"] or timestamp,
                run["created_at_utc"] or timestamp,
                _json_dumps(metadata),
            ),
        )
        updated += max(0, cursor.rowcount)
    return updated


def _fleet_asset_from_row(row: sqlite3.Row) -> dict[str, Any]:
    asset = dict(row)
    asset["latest_attention_reasons"] = _json_loads(
        asset.pop("latest_attention_json")
    )
    asset["metadata"] = _json_loads(asset.pop("metadata_json"))
    priority = _fleet_asset_priority(asset)
    asset["priority_score"] = priority["score"]
    asset["priority_band"] = priority["band"]
    asset["priority_reasons"] = priority["reasons"]
    return asset


def _extend_in_filter(
    where_clauses: list[str],
    parameters: list[Any],
    *,
    column: str,
    values: Iterable[str] | None,
) -> None:
    normalized = _normalized_filter_values(values)
    if not normalized:
        return
    placeholders = ", ".join("?" for _ in normalized)
    where_clauses.append(f"{column} in ({placeholders})")
    parameters.extend(normalized)


def _extend_prediction_run_asset_filter(
    where_clauses: list[str],
    parameters: list[Any],
    *,
    asset_ids: Iterable[str] | None,
) -> None:
    normalized = _normalized_filter_values(asset_ids)
    if not normalized:
        return
    placeholders = ", ".join("?" for _ in normalized)
    where_clauses.append(
        f"""
        exists (
            select 1
            from predictions asset_filter_predictions
            where asset_filter_predictions.run_id = prediction_runs.run_id
                and asset_filter_predictions.asset_id in ({placeholders})
        )
        """
    )
    parameters.extend(normalized)


def _extend_prediction_run_risk_filter(
    where_clauses: list[str],
    parameters: list[Any],
    *,
    risk_levels: Iterable[str] | None,
) -> None:
    normalized = _normalized_filter_values(risk_levels)
    if not normalized:
        return
    risk_conditions: list[str] = []
    risk_floor_sql = """
        case
            when risk_filter_predictions.predicted_rul_lower is not null
                and risk_filter_predictions.predicted_rul_lower
                    < risk_filter_predictions.predicted_rul
            then risk_filter_predictions.predicted_rul_lower
            else risk_filter_predictions.predicted_rul
        end
    """
    if "critical" in normalized:
        risk_conditions.append(f"({risk_floor_sql}) <= 20")
    if "watch" in normalized:
        risk_conditions.append(
            f"({risk_floor_sql}) > 20 and ({risk_floor_sql}) <= 50"
        )
    if "nominal" in normalized:
        risk_conditions.append(f"({risk_floor_sql}) > 50")
    if "unknown" in normalized:
        risk_conditions.append("risk_filter_predictions.predicted_rul is null")
    if not risk_conditions:
        return
    where_clauses.append(
        f"""
        exists (
            select 1
            from predictions risk_filter_predictions
            where risk_filter_predictions.run_id = prediction_runs.run_id
                and ({" or ".join(risk_conditions)})
        )
        """
    )


def _extend_prediction_run_decision_filter(
    where_clauses: list[str],
    parameters: list[Any],
    *,
    decision_statuses: Iterable[str] | None,
) -> None:
    normalized = _normalized_filter_values(decision_statuses)
    if not normalized:
        return
    placeholders = ", ".join("?" for _ in normalized)
    where_clauses.append(
        f"""
        (
            select status
            from prediction_run_events decision_filter_events
            where decision_filter_events.run_id = prediction_runs.run_id
                and decision_filter_events.event_type = 'operator_decision'
            order by decision_filter_events.created_at_utc desc
            limit 1
        ) in ({placeholders})
        """
    )
    parameters.extend(normalized)


def _normalized_filter_values(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _upsert_anomaly_fleet_asset(
    connection: sqlite3.Connection,
    *,
    row: dict[str, Any],
    comparison_path: Path,
    source_name: str,
    timestamp: str,
) -> int:
    risk_level = _anomaly_asset_risk_level(row)
    attention = _anomaly_asset_attention_reasons(row, risk_level=risk_level)
    metadata = {
        **row,
        "source_name": source_name,
        "comparison_csv": str(comparison_path),
        "comparison_sha256": _file_sha256(comparison_path),
    }
    asset_id = _anomaly_asset_id(row["spacecraft"], row["channel_id"])
    cursor = connection.execute(
        """
        insert into fleet_assets (
            asset_id,
            asset_type,
            domain,
            source_dataset,
            source_subset,
            external_id,
            latest_run_id,
            latest_rul_prediction,
            latest_rul_lower,
            latest_rul_upper,
            latest_risk_level,
            latest_status,
            latest_attention_json,
            first_seen_at_utc,
            last_seen_at_utc,
            metadata_json
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(asset_id) do update set
            asset_type = excluded.asset_type,
            domain = excluded.domain,
            source_dataset = excluded.source_dataset,
            source_subset = excluded.source_subset,
            external_id = excluded.external_id,
            latest_run_id = excluded.latest_run_id,
            latest_rul_prediction = excluded.latest_rul_prediction,
            latest_rul_lower = excluded.latest_rul_lower,
            latest_rul_upper = excluded.latest_rul_upper,
            latest_risk_level = excluded.latest_risk_level,
            latest_status = excluded.latest_status,
            latest_attention_json = excluded.latest_attention_json,
            last_seen_at_utc = excluded.last_seen_at_utc,
            metadata_json = excluded.metadata_json
        where excluded.last_seen_at_utc >= fleet_assets.last_seen_at_utc
        """,
        (
            asset_id,
            "spacecraft_channel",
            "spacecraft_anomaly",
            "SMAP/MSL",
            row["spacecraft"],
            row["channel_id"],
            None,
            None,
            None,
            None,
            risk_level,
            _anomaly_asset_status(risk_level),
            _json_dumps(attention),
            timestamp,
            timestamp,
            _json_dumps(metadata),
        ),
    )
    return max(0, cursor.rowcount)


def _upsert_anomaly_event_fleet_asset(
    connection: sqlite3.Connection,
    *,
    row: dict[str, Any],
    events_path: Path,
    source_name: str,
    timestamp: str,
) -> int:
    risk_level = _anomaly_event_risk_level(row)
    attention = _anomaly_event_attention_reasons(row, risk_level=risk_level)
    metadata = {
        **row,
        "source_name": source_name,
        "events_csv": str(events_path),
        "events_sha256": _file_sha256(events_path),
    }
    last_seen_at = str(row.get("event_time_utc") or timestamp)
    asset_id = _anomaly_asset_id(row["spacecraft"], row["channel_id"])
    cursor = connection.execute(
        """
        insert into fleet_assets (
            asset_id,
            asset_type,
            domain,
            source_dataset,
            source_subset,
            external_id,
            latest_run_id,
            latest_rul_prediction,
            latest_rul_lower,
            latest_rul_upper,
            latest_risk_level,
            latest_status,
            latest_attention_json,
            first_seen_at_utc,
            last_seen_at_utc,
            metadata_json
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(asset_id) do update set
            asset_type = excluded.asset_type,
            domain = excluded.domain,
            source_dataset = excluded.source_dataset,
            source_subset = excluded.source_subset,
            external_id = excluded.external_id,
            latest_run_id = excluded.latest_run_id,
            latest_rul_prediction = excluded.latest_rul_prediction,
            latest_rul_lower = excluded.latest_rul_lower,
            latest_rul_upper = excluded.latest_rul_upper,
            latest_risk_level = excluded.latest_risk_level,
            latest_status = excluded.latest_status,
            latest_attention_json = excluded.latest_attention_json,
            last_seen_at_utc = excluded.last_seen_at_utc,
            metadata_json = excluded.metadata_json
        where excluded.last_seen_at_utc >= fleet_assets.last_seen_at_utc
        """,
        (
            asset_id,
            "spacecraft_channel",
            "spacecraft_anomaly",
            "SMAP/MSL",
            row["spacecraft"],
            row["channel_id"],
            None,
            None,
            None,
            None,
            risk_level,
            _anomaly_asset_status(risk_level),
            _json_dumps(attention),
            last_seen_at,
            last_seen_at,
            _json_dumps(metadata),
        ),
    )
    return max(0, cursor.rowcount)


def _upsert_model_artifact(
    connection: sqlite3.Connection,
    *,
    artifact_id: str,
    artifact_path: Path,
    inspection: dict[str, Any],
    timestamp: str,
) -> None:
    identity = inspection.get("artifact_identity")
    identity = identity if isinstance(identity, dict) else {}
    model = inspection.get("model")
    model = model if isinstance(model, dict) else {}
    promotion = inspection.get("promotion")
    promotion = promotion if isinstance(promotion, dict) else {}
    connection.execute(
        """
        insert into model_artifacts (
            artifact_id,
            artifact_path,
            artifact_sha256,
            dataset,
            subset,
            model_name,
            schema_version,
            stage,
            inspection_json,
            created_at_utc
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(artifact_id) do update set
            artifact_path = excluded.artifact_path,
            artifact_sha256 = excluded.artifact_sha256,
            dataset = excluded.dataset,
            subset = excluded.subset,
            model_name = excluded.model_name,
            schema_version = excluded.schema_version,
            stage = excluded.stage,
            inspection_json = excluded.inspection_json
        """,
        (
            artifact_id,
            str(artifact_path),
            _file_sha256(artifact_path),
            model.get("dataset"),
            model.get("subset"),
            model.get("model_name"),
            identity.get("schema_version"),
            promotion.get("stage"),
            _json_dumps(inspection),
            timestamp,
        ),
    )


def _insert_release_evidence(
    connection: sqlite3.Connection,
    *,
    artifact_id: str,
    evidence_type: str,
    path: Path,
    payload: dict[str, Any],
    timestamp: str,
) -> int:
    evidence = _release_evidence_record(
        artifact_id=artifact_id,
        evidence_type=evidence_type,
        path=path,
        payload=payload,
        timestamp=timestamp,
    )
    cursor = connection.execute(
        """
        insert into release_evidence (
            evidence_id,
            artifact_id,
            evidence_type,
            source_path,
            status,
            payload_json,
            created_at_utc
        )
        values (?, ?, ?, ?, ?, ?, ?)
        on conflict(evidence_id) do nothing
        """,
        (
            evidence["evidence_id"],
            evidence["artifact_id"],
            evidence["evidence_type"],
            evidence["source_path"],
            evidence["status"],
            evidence["payload_json"],
            evidence["created_at_utc"],
        ),
    )
    return max(0, cursor.rowcount)


def _workspace_evidence_without_inspection(
    workspace: QuickstartWorkspace,
) -> Iterable[tuple[str, Path, dict[str, Any] | None]]:
    yield "release_bundle", workspace.release_bundle_path, workspace.release_bundle
    yield "release_provenance", workspace.provenance_path, workspace.provenance
    yield "promotion_report", workspace.promotion_report_path, workspace.promotion_report
    yield "dashboard_payload", workspace.dashboard_payload_path, workspace.dashboard_payload


def _artifact_id_from_inspection(inspection: dict[str, Any], artifact_path: Path) -> str:
    identity = inspection.get("artifact_identity")
    if isinstance(identity, dict) and identity.get("artifact_id"):
        return str(identity["artifact_id"])
    return f"artifact-{_sha256_text(str(artifact_path))[:16]}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "active", "anomaly"}


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _json_loads(payload: str | None) -> Any:
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}


def _now() -> str:
    return datetime.now(UTC).isoformat()
