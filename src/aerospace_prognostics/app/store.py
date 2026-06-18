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

            create table if not exists prediction_outcomes (
                run_id text not null,
                unit_number integer not null,
                actual_rul real not null,
                outcome_source text not null,
                observed_at_utc text,
                recorded_at_utc text not null,
                primary key (run_id, unit_number),
                foreign key (run_id, unit_number)
                    references predictions(run_id, unit_number)
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


def list_model_artifacts(
    database_path: str | Path,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return model artifacts with evidence and prediction usage counts."""

    db_path = initialize_app_database(database_path)
    with _connect(db_path) as connection:
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
) -> dict[str, Any] | None:
    """Load one model artifact with release evidence and prediction usage."""

    db_path = initialize_app_database(database_path)
    with _connect(db_path) as connection:
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
                predictions.asset_id,
                predictions.unit_number,
                predictions.predicted_rul,
                predictions.predicted_rul_lower,
                predictions.predicted_rul_upper,
                predictions.interval_method,
                predictions.interval_confidence,
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
            "prediction_outcomes",
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


def _outcome_rows(outcomes: pd.DataFrame) -> list[dict[str, Any]]:
    required_columns = {"unit_number", "actual_rul"}
    missing_columns = required_columns.difference(outcomes.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"outcomes require columns: {missing}")
    parsed: list[dict[str, Any]] = []
    for row in outcomes[["unit_number", "actual_rul"]].to_dict(orient="records"):
        if pd.isna(row["unit_number"]) or pd.isna(row["actual_rul"]):
            raise ValueError("outcome rows require unit_number and actual_rul")
        parsed.append(
            {
                "unit_number": int(row["unit_number"]),
                "actual_rul": float(row["actual_rul"]),
            }
        )
    if not parsed:
        raise ValueError("outcomes must contain at least one row")
    return parsed


def _run_summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    monitoring = _json_loads(result.pop("monitoring_json"))
    result["drift_alert_count"] = _drift_alert_count(monitoring)
    return _with_interval_availability(result)


def _with_interval_availability(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    prediction_count = _optional_int(result.get("prediction_count")) or 0
    interval_count = _optional_int(result.get("interval_count")) or 0
    result["interval_count"] = interval_count
    result["interval_availability_rate"] = (
        interval_count / prediction_count if prediction_count > 0 else None
    )
    outcome_count = _optional_int(result.get("outcome_count")) or 0
    interval_outcome_count = _optional_int(result.get("interval_outcome_count")) or 0
    interval_covered_count = _optional_int(result.get("interval_covered_count")) or 0
    result["outcome_count"] = outcome_count
    result["outcome_availability_rate"] = (
        outcome_count / prediction_count if prediction_count > 0 else None
    )
    result["interval_outcome_count"] = interval_outcome_count
    result["interval_covered_count"] = interval_covered_count
    result["outcome_interval_coverage_rate"] = (
        interval_covered_count / interval_outcome_count
        if interval_outcome_count > 0
        else None
    )
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    event = dict(row)
    event["payload"] = _json_loads(event.pop("payload_json"))
    return event


def _evidence_from_row(row: sqlite3.Row) -> dict[str, Any]:
    evidence = dict(row)
    evidence["payload"] = _json_loads(evidence.pop("payload_json"))
    return evidence


def _model_report_card(
    artifact: dict[str, Any],
    *,
    release_evidence: list[dict[str, Any]],
    prediction_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_by_type = {
        str(evidence.get("evidence_type")): evidence
        for evidence in release_evidence
        if evidence.get("evidence_type") is not None
    }
    promotion = _payload_for_evidence(evidence_by_type, "promotion_report")
    release_bundle = _payload_for_evidence(evidence_by_type, "release_bundle")
    provenance = _payload_for_evidence(evidence_by_type, "release_provenance")
    inspection = artifact.get("inspection")
    inspection = inspection if isinstance(inspection, dict) else {}
    uncertainty = inspection.get("uncertainty")
    uncertainty = uncertainty if isinstance(uncertainty, dict) else {}
    promotion_gates = _bool_dict(promotion.get("gates"))
    release_gates = _bool_dict(release_bundle.get("gates"))
    all_gates = {
        **{f"promotion.{key}": value for key, value in promotion_gates.items()},
        **{f"release.{key}": value for key, value in release_gates.items()},
    }
    latest_run_at = max(
        (
            str(run["created_at_utc"])
            for run in prediction_runs
            if run.get("created_at_utc") is not None
        ),
        default=None,
    )
    promotion_evidence = promotion.get("evidence")
    promotion_evidence = promotion_evidence if isinstance(promotion_evidence, dict) else {}
    benchmark = promotion_evidence.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    latency_ms = benchmark.get("latency_ms")
    latency_ms = latency_ms if isinstance(latency_ms, dict) else {}
    provenance_summary = provenance.get("summary")
    provenance_summary = provenance_summary if isinstance(provenance_summary, dict) else {}
    prediction_count_total = 0
    interval_count_total = 0
    weighted_interval_width = 0.0
    interval_width_count = 0
    max_interval_width = None
    outcome_count_total = 0
    weighted_absolute_error = 0.0
    weighted_signed_error = 0.0
    interval_outcome_count_total = 0
    interval_covered_count_total = 0
    for run in prediction_runs:
        prediction_count = _optional_int(run.get("prediction_count"))
        if prediction_count is None:
            prediction_count = _optional_int(run.get("prediction_row_count")) or 0
        interval_count = _optional_int(run.get("interval_count")) or 0
        mean_width = _optional_float(run.get("mean_interval_width"))
        run_max_width = _optional_float(run.get("max_interval_width"))
        outcome_count = _optional_int(run.get("outcome_count")) or 0
        mean_absolute_error = _optional_float(run.get("mean_absolute_error"))
        mean_signed_error = _optional_float(run.get("mean_signed_error"))
        interval_outcome_count = _optional_int(run.get("interval_outcome_count")) or 0
        interval_covered_count = _optional_int(run.get("interval_covered_count")) or 0
        prediction_count_total += prediction_count
        interval_count_total += interval_count
        outcome_count_total += outcome_count
        interval_outcome_count_total += interval_outcome_count
        interval_covered_count_total += interval_covered_count
        if mean_width is not None and interval_count > 0:
            weighted_interval_width += mean_width * interval_count
            interval_width_count += interval_count
        if mean_absolute_error is not None and outcome_count > 0:
            weighted_absolute_error += mean_absolute_error * outcome_count
        if mean_signed_error is not None and outcome_count > 0:
            weighted_signed_error += mean_signed_error * outcome_count
        if run_max_width is not None:
            max_interval_width = (
                run_max_width
                if max_interval_width is None
                else max(max_interval_width, run_max_width)
            )
    missing_interval_count = max(prediction_count_total - interval_count_total, 0)
    interval_availability_rate = (
        interval_count_total / prediction_count_total
        if prediction_count_total > 0
        else None
    )
    mean_interval_width = (
        weighted_interval_width / interval_width_count
        if interval_width_count > 0
        else None
    )
    mean_absolute_error = (
        weighted_absolute_error / outcome_count_total
        if outcome_count_total > 0
        else None
    )
    mean_signed_error = (
        weighted_signed_error / outcome_count_total
        if outcome_count_total > 0
        else None
    )
    outcome_availability_rate = (
        outcome_count_total / prediction_count_total
        if prediction_count_total > 0
        else None
    )
    outcome_interval_coverage_rate = (
        interval_covered_count_total / interval_outcome_count_total
        if interval_outcome_count_total > 0
        else None
    )
    return {
        "artifact_id": artifact.get("artifact_id"),
        "stage": artifact.get("stage"),
        "dataset": artifact.get("dataset"),
        "subset": artifact.get("subset"),
        "model_name": artifact.get("model_name"),
        "release_status": release_bundle.get("status"),
        "promotion_status": promotion.get("status"),
        "gate_count": len(all_gates),
        "passed_gate_count": sum(1 for value in all_gates.values() if value is True),
        "failed_gates": [key for key, value in sorted(all_gates.items()) if value is not True],
        "evidence_count": len(release_evidence),
        "prediction_run_count": len(prediction_runs),
        "latest_prediction_at": latest_run_at,
        "p95_latency_ms": latency_ms.get("p95"),
        "max_p95_latency_ms": benchmark.get("max_p95_latency_ms"),
        "interval_method": uncertainty.get("interval_method"),
        "interval_confidence": uncertainty.get("interval_confidence"),
        "interval_diagnostic_kind": "operational_interval_availability",
        "prediction_count_total": prediction_count_total,
        "interval_count_total": interval_count_total,
        "missing_interval_count": missing_interval_count,
        "interval_availability_rate": interval_availability_rate,
        "interval_complete": (
            prediction_count_total > 0 and missing_interval_count == 0
        ),
        "mean_interval_width": mean_interval_width,
        "max_interval_width": max_interval_width,
        "outcome_diagnostic_kind": "observed_rul_outcome_coverage",
        "outcome_count_total": outcome_count_total,
        "outcome_availability_rate": outcome_availability_rate,
        "mean_absolute_error": mean_absolute_error,
        "mean_signed_error": mean_signed_error,
        "interval_outcome_count_total": interval_outcome_count_total,
        "interval_covered_count_total": interval_covered_count_total,
        "outcome_interval_coverage_rate": outcome_interval_coverage_rate,
        "provenance_workflow": provenance_summary.get("workflow"),
    }


def _payload_for_evidence(
    evidence_by_type: dict[str, dict[str, Any]],
    evidence_type: str,
) -> dict[str, Any]:
    evidence = evidence_by_type.get(evidence_type)
    if not isinstance(evidence, dict):
        return {}
    payload = evidence.get("payload")
    return payload if isinstance(payload, dict) else {}


def _bool_dict(payload: Any) -> dict[str, bool]:
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, bool)}


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
