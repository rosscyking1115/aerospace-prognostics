from __future__ import annotations

import sqlite3

import pytest

from aerospace_prognostics.app.database import (
    REQUIRED_TABLES,
    SCHEMA_VERSION,
    connect,
    initialize_app_database,
    metadata_value,
    prepare_database,
    validate_existing_database,
)


def test_initialize_app_database_creates_required_tables_and_metadata(tmp_path) -> None:
    database_path = initialize_app_database(tmp_path / "nested" / "app.sqlite")

    with connect(database_path, read_only=True) as connection:
        table_rows = connection.execute(
            "select name from sqlite_master where type = 'table'"
        ).fetchall()
        table_names = {str(row[0]) for row in table_rows}

        assert set(REQUIRED_TABLES).issubset(table_names)
        assert metadata_value(connection, "schema_version") == SCHEMA_VERSION


def test_prepare_database_read_only_requires_existing_database(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="app database not found"):
        prepare_database(tmp_path / "missing.sqlite", read_only=True)


def test_validate_existing_database_rejects_schema_version_mismatch(tmp_path) -> None:
    database_path = tmp_path / "app.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            create table app_metadata (
                key text primary key,
                value text not null,
                updated_at_utc text not null
            )
            """
        )
        connection.execute(
            """
            insert into app_metadata (key, value, updated_at_utc)
            values ('schema_version', 'old-schema', '2026-01-01T00:00:00+00:00')
            """
        )

    with pytest.raises(RuntimeError, match="app database schema mismatch"):
        validate_existing_database(database_path)


def test_validate_existing_database_rejects_missing_required_tables(tmp_path) -> None:
    database_path = initialize_app_database(tmp_path / "app.sqlite")
    with sqlite3.connect(database_path) as connection:
        connection.execute("drop table fleet_assets")

    with pytest.raises(RuntimeError, match="missing required tables: fleet_assets"):
        validate_existing_database(database_path)
