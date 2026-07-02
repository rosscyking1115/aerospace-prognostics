"""SQLite schema and connection helpers for the PHM console app."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "aerospace-prognostics/app-db/v1"
REQUIRED_TABLES = (
    "app_metadata",
    "model_artifacts",
    "release_evidence",
    "telemetry_uploads",
    "prediction_runs",
    "predictions",
    "prediction_outcomes",
    "prediction_run_events",
    "fleet_assets",
)


def initialize_app_database(database_path: str | Path) -> Path:
    """Create or migrate the local app database."""

    db_path = Path(database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as connection:
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

            create table if not exists fleet_assets (
                asset_id text primary key,
                asset_type text not null,
                domain text not null,
                source_dataset text,
                source_subset text,
                external_id text,
                latest_run_id text references prediction_runs(run_id),
                latest_rul_prediction real,
                latest_rul_lower real,
                latest_rul_upper real,
                latest_risk_level text,
                latest_status text,
                latest_attention_json text not null,
                first_seen_at_utc text not null,
                last_seen_at_utc text not null,
                metadata_json text not null
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


def prepare_database(database_path: str | Path, *, read_only: bool) -> Path:
    """Return a validated existing DB for read-only use, or initialize it."""

    if read_only:
        return validate_existing_database(database_path)
    return initialize_app_database(database_path)


def validate_existing_database(database_path: str | Path) -> Path:
    """Validate that a read-only app database exists and matches this schema."""

    db_path = Path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(f"app database not found: {db_path}")

    with connect(db_path, read_only=True) as connection:
        schema_version = metadata_value(connection, "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise RuntimeError(
                f"app database schema mismatch: expected {SCHEMA_VERSION}, "
                f"got {schema_version or 'missing'}"
            )
        table_rows = connection.execute(
            "select name from sqlite_master where type = 'table'"
        ).fetchall()
        table_names = {str(row[0]) for row in table_rows}
        missing_tables = sorted(set(REQUIRED_TABLES).difference(table_names))
        if missing_tables:
            raise RuntimeError(
                "app database is missing required tables: "
                + ", ".join(missing_tables)
            )
    return db_path


def connect(database_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open an app database connection with foreign-key enforcement enabled."""

    if read_only:
        uri = database_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    else:
        connection = sqlite3.connect(database_path)
    connection.execute("pragma foreign_keys = on")
    return connection


def metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
    """Return one app_metadata value from an open connection."""

    row = connection.execute(
        "select value from app_metadata where key = ?",
        (key,),
    ).fetchone()
    return str(row[0]) if row is not None else None


def _now() -> str:
    return datetime.now(UTC).isoformat()
