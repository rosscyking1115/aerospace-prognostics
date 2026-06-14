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

from aerospace_prognostics.app.dashboard_state import QuickstartWorkspace

SCHEMA_VERSION = "aerospace-prognostics/app-db/v1"


def initialize_app_database(database_path: str | Path) -> Path:
    """Create or migrate the local app database."""

    db_path = Path(database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        connection.executescript(
            """
            create table if not exists app_metadata (
                key text primary key,
                value text not null,
                updated_at_utc text not null
            );

            create table if not exists model_artifacts (
                artifact_id text primary key,
                artifact_path text not null,
                artifact_sha256 text,
                dataset text,
                subset text,
                model_name text,
                schema_version text,
                stage text,
                inspection_json text,
                created_at_utc text not null
            );

            create table if not exists release_evidence (
                evidence_id text primary key,
                artifact_id text,
                evidence_type text not null,
                source_path text not null,
                status text,
                payload_json text not null,
                created_at_utc text not null
            );

            create table if not exists telemetry_uploads (
                upload_id text primary key,
                source_name text not null,
                row_count integer not null,
                column_count integer not null,
                content_sha256 text not null,
                created_at_utc text not null
            );

            create table if not exists prediction_runs (
                run_id text primary key,
                upload_id text not null references telemetry_uploads(upload_id),
                artifact_id text,
                model_artifact_path text not null,
                dataset text,
                subset text,
                model_name text,
                prediction_count integer not null,
                monitoring_json text,
                created_at_utc text not null
            );

            create table if not exists predictions (
                run_id text not null references prediction_runs(run_id),
                asset_id text not null,
                unit_number integer not null,
                predicted_rul real not null,
                predicted_rul_lower real,
                predicted_rul_upper real,
                interval_method text,
                interval_confidence real,
                created_at_utc text not null,
                primary key (run_id, unit_number)
            );

            create table if not exists prediction_run_events (
                event_id text primary key,
                run_id text not null references prediction_runs(run_id),
                event_type text not null,
                status text,
                actor text not null,
                note text,
                payload_json text,
                created_at_utc text not null
            );
            """
        )
        timestamp = _now()
        connection.execute(
            """
            insert into app_metadata (key, value, updated_at_utc)
            values (?, ?, ?)
            on conflict(key) do update set
                value = excluded.value,
                updated_at_utc = excluded.updated_at_utc
            """,
            ("schema_version", SCHEMA_VERSION, timestamp),
        )
    return db_path


def seed_quickstart_workspace(
    database_path: str | Path,
    workspace: QuickstartWorkspace,
) -> dict[str, int]:
    """Seed database records from quickstart model and release evidence files."""

    db_path = initialize_app_database(database_path)
    inserted = {"model_artifacts": 0, "release_evidence": 0}
    artifact_id = _artifact_id_from_workspace(workspace)
    timestamp = _now()
    with _connect(db_path) as connection:
        if workspace.model_artifact_path.exists():
            inspection = workspace.artifact_inspection or {}
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
                    str(workspace.model_artifact_path),
                    _file_sha256(workspace.model_artifact_path),
                    model.get("dataset"),
                    model.get("subset"),
                    model.get("model_name"),
                    identity.get("schema_version"),
                    promotion.get("stage"),
                    _json_dumps(inspection),
                    timestamp,
                ),
            )
            inserted["model_artifacts"] = 1

        for evidence_type, path, payload in _workspace_evidence(workspace):
            if payload is None or not path.exists():
                continue
            evidence_id = f"{evidence_type}:{artifact_id}:{_file_sha256(path)}"
            status = payload.get("status") if isinstance(payload.get("status"), str) else None
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
                    evidence_id,
                    artifact_id,
                    evidence_type,
                    str(path),
                    status,
                    _json_dumps(payload),
                    timestamp,
                ),
            )
            inserted["release_evidence"] += max(0, cursor.rowcount)
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
                interval_confidence,
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
                    row.get("interval_confidence"),
                    timestamp,
                )
                for row in predictions
            ),
        )
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


def list_prediction_runs(
    database_path: str | Path,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return recent prediction runs with upload and aggregate prediction context."""

    db_path = initialize_app_database(database_path)
    with _connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
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
                min(predictions.predicted_rul) as min_predicted_rul,
                avg(predictions.predicted_rul) as mean_predicted_rul,
                max(predictions.predicted_rul) as max_predicted_rul,
                count(predictions.predicted_rul_lower) as interval_count,
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
            group by prediction_runs.run_id
            order by prediction_runs.created_at_utc desc
            limit ?
            """,
            (int(limit),),
        ).fetchall()
    return [_run_summary_from_row(row) for row in rows]


def load_prediction_run(database_path: str | Path, run_id: str) -> dict[str, Any] | None:
    """Load one prediction run and all persisted prediction rows."""

    db_path = initialize_app_database(database_path)
    with _connect(db_path) as connection:
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
                asset_id,
                unit_number,
                predicted_rul,
                predicted_rul_lower,
                predicted_rul_upper,
                interval_method,
                interval_confidence,
                created_at_utc
            from predictions
            where run_id = ?
            order by predicted_rul asc, unit_number asc
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
    run = dict(run_row)
    run["monitoring"] = _json_loads(run.pop("monitoring_json"))
    return {
        "run": run,
        "predictions": [dict(row) for row in prediction_rows],
        "audit_events": [_event_from_row(row) for row in event_rows],
    }


def database_summary(database_path: str | Path) -> dict[str, int | str]:
    """Return table counts and schema version for the app database."""

    db_path = initialize_app_database(database_path)
    with _connect(db_path) as connection:
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
            "prediction_run_events",
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
) -> list[dict[str, Any]]:
    """Return recent audit events for a prediction run."""

    db_path = initialize_app_database(database_path)
    with _connect(db_path) as connection:
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


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.execute("pragma foreign_keys = on")
    return connection


def _metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "select value from app_metadata where key = ?",
        (key,),
    ).fetchone()
    return str(row[0]) if row is not None else None


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
    event_material = {
        "run_id": run_id,
        "event_type": event_type,
        "status": status,
        "actor": actor,
        "note": note,
        "payload": payload or {},
        "timestamp": timestamp,
    }
    event_id = f"event-{_sha256_text(_json_dumps(event_material))[:16]}"
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
            event_id,
            run_id,
            event_type,
            status,
            actor,
            note,
            _json_dumps(payload or {}),
            timestamp,
        ),
    )
    return event_id


def _workspace_evidence(
    workspace: QuickstartWorkspace,
) -> Iterable[tuple[str, Path, dict[str, Any] | None]]:
    yield "artifact_inspection", workspace.artifact_inspection_path, workspace.artifact_inspection
    yield "release_bundle", workspace.release_bundle_path, workspace.release_bundle
    yield "release_provenance", workspace.provenance_path, workspace.provenance
    yield "promotion_report", workspace.promotion_report_path, workspace.promotion_report
    yield "dashboard_payload", workspace.dashboard_payload_path, workspace.dashboard_payload


def _artifact_id_from_workspace(workspace: QuickstartWorkspace) -> str:
    inspection = workspace.artifact_inspection or {}
    identity = inspection.get("artifact_identity")
    if isinstance(identity, dict) and identity.get("artifact_id"):
        return str(identity["artifact_id"])
    return f"artifact-{_sha256_text(str(workspace.model_artifact_path))[:16]}"


def _prediction_rows(prediction_document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = prediction_document.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("prediction_document['predictions'] must be a list")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("prediction rows must be JSON objects")
        if "unit_number" not in row or "predicted_rul" not in row:
            raise ValueError("prediction rows require unit_number and predicted_rul")
        parsed.append(row)
    return parsed


def _run_summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    monitoring = _json_loads(result.pop("monitoring_json"))
    result["drift_alert_count"] = _drift_alert_count(monitoring)
    return result


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    event = dict(row)
    event["payload"] = _json_loads(event.pop("payload_json"))
    return event


def _drift_alert_count(monitoring: Any) -> int:
    if not isinstance(monitoring, dict):
        return 0
    drift = monitoring.get("drift")
    if not isinstance(drift, dict):
        return 0
    alert_columns = drift.get("alert_columns")
    if isinstance(alert_columns, list):
        return len(alert_columns)
    alerts = drift.get("alerts")
    if isinstance(alerts, list):
        return len(alerts)
    return 0


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
