"""App database command handlers for the project CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from aerospace_prognostics.app.dashboard_state import load_quickstart_workspace
from aerospace_prognostics.app.store import (
    database_summary,
    export_prediction_run_evidence,
    initialize_app_database,
    record_prediction_outcomes,
    register_model_artifact_evidence,
    seed_quickstart_workspace,
)

APP_COMMANDS = {
    "app-init-db",
    "app-register-artifact",
    "app-record-outcomes",
    "app-export-run",
}


def register_app_commands(subparsers: Any) -> None:
    app_init_db = subparsers.add_parser(
        "app-init-db",
        help="Initialize the local app database and optionally seed quickstart evidence",
    )
    app_init_db.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_init_db.add_argument(
        "--quickstart-dir",
        type=Path,
        default=Path("artifacts") / "quickstart_cmapss",
    )
    app_init_db.add_argument("--no-seed-quickstart", action="store_true")

    app_register_artifact = subparsers.add_parser(
        "app-register-artifact",
        help="Register a model artifact and release evidence in the local app database",
    )
    app_register_artifact.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_register_artifact.add_argument("--model-artifact", type=Path, required=True)
    app_register_artifact.add_argument("--inspection-json", type=Path, required=True)
    app_register_artifact.add_argument("--release-bundle-json", type=Path)
    app_register_artifact.add_argument("--provenance-json", type=Path)
    app_register_artifact.add_argument("--promotion-json", type=Path)
    app_register_artifact.add_argument("--dashboard-payload-json", type=Path)

    app_record_outcomes = subparsers.add_parser(
        "app-record-outcomes",
        help="Attach observed RUL outcomes to a persisted prediction run",
    )
    app_record_outcomes.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_record_outcomes.add_argument("--run-id", required=True)
    app_record_outcomes.add_argument("--outcomes-csv", type=Path, required=True)
    app_record_outcomes.add_argument("--source-name")
    app_record_outcomes.add_argument("--actor", default="operator")
    app_record_outcomes.add_argument("--observed-at-utc")

    app_export_run = subparsers.add_parser(
        "app-export-run",
        help="Export a persisted prediction run as portable review evidence",
    )
    app_export_run.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_export_run.add_argument("--run-id", required=True)
    app_export_run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "app_exports",
    )


def handle_app_command(args: argparse.Namespace) -> int | None:
    if args.command == "app-init-db":
        database_path = initialize_app_database(args.database)
        print(f"database={database_path}")
        if not args.no_seed_quickstart:
            workspace = load_quickstart_workspace(args.quickstart_dir)
            inserted = seed_quickstart_workspace(database_path, workspace)
            print(f"quickstart_dir={workspace.root}")
            print(f"model_artifacts_seeded={inserted['model_artifacts']}")
            print(f"release_evidence_seeded={inserted['release_evidence']}")
            if workspace.missing_paths:
                print(f"missing_quickstart_paths={len(workspace.missing_paths)}")
        summary = database_summary(database_path)
        print(f"schema_version={summary['schema_version']}")
        print(f"model_artifacts={summary['model_artifacts']}")
        print(f"release_evidence={summary['release_evidence']}")
        print(f"telemetry_uploads={summary['telemetry_uploads']}")
        print(f"prediction_runs={summary['prediction_runs']}")
        print(f"predictions={summary['predictions']}")
        print(f"prediction_outcomes={summary['prediction_outcomes']}")
        return 0

    if args.command == "app-register-artifact":
        inspection = _read_json_file(args.inspection_json)
        evidence_files = (
            ("release_bundle", args.release_bundle_json),
            ("release_provenance", args.provenance_json),
            ("promotion_report", args.promotion_json),
            ("dashboard_payload", args.dashboard_payload_json),
        )
        result = register_model_artifact_evidence(
            args.database,
            model_artifact_path=args.model_artifact,
            inspection=inspection,
            inspection_source_path=args.inspection_json,
            release_evidence=tuple(
                (evidence_type, path, _read_json_file(path))
                for evidence_type, path in evidence_files
                if path is not None
            ),
        )
        summary = database_summary(args.database)
        print(f"database={args.database}")
        print(f"artifact_id={result['artifact_id']}")
        print(f"model_artifact={args.model_artifact}")
        print(f"model_artifacts_registered={result['model_artifacts']}")
        print(f"release_evidence_registered={result['release_evidence']}")
        print(f"model_artifacts={summary['model_artifacts']}")
        print(f"release_evidence={summary['release_evidence']}")
        return 0

    if args.command == "app-record-outcomes":
        outcomes = _read_outcomes_csv(args.outcomes_csv)
        result = record_prediction_outcomes(
            args.database,
            run_id=args.run_id,
            outcomes=outcomes,
            source_name=args.source_name or str(args.outcomes_csv),
            actor=args.actor,
            observed_at_utc=args.observed_at_utc,
        )
        summary = database_summary(args.database)
        print(f"database={args.database}")
        print(f"run_id={args.run_id}")
        print(f"outcomes_csv={args.outcomes_csv}")
        print(f"outcome_count={result['outcome_count']}")
        print(f"event_id={result['event_id']}")
        print(f"prediction_outcomes={summary['prediction_outcomes']}")
        return 0

    if args.command == "app-export-run":
        result = export_prediction_run_evidence(
            args.database,
            run_id=args.run_id,
            output_dir=args.output_dir,
        )
        print(f"database={args.database}")
        print(f"run_id={args.run_id}")
        print(f"output_dir={result['output_dir']}")
        print(f"evidence_json={result['evidence_json']}")
        print(f"evidence_sha256={result['evidence_sha256']}")
        print(f"predictions_csv={result['predictions_csv']}")
        print(f"predictions_sha256={result['predictions_sha256']}")
        print(f"prediction_count={result['prediction_count']}")
        print(f"outcome_count={result['outcome_count']}")
        print(f"audit_event_count={result['audit_event_count']}")
        return 0

    return None


def _read_json_file(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _read_outcomes_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (EmptyDataError, ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"outcomes CSV could not be read as a valid CSV: {path}") from exc
