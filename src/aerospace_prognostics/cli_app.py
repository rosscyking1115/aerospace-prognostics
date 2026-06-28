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
    export_fleet_asset_registry,
    export_prediction_outcome_template,
    export_prediction_run_evidence,
    initialize_app_database,
    record_prediction_outcomes,
    record_prediction_run_event,
    register_model_artifact_evidence,
    seed_quickstart_workspace,
    sync_fleet_assets_from_anomaly_comparison,
    sync_fleet_assets_from_anomaly_events,
    sync_fleet_assets_from_prediction_run,
)

APP_COMMANDS = {
    "app-init-db",
    "app-register-artifact",
    "app-sync-fleet-assets",
    "app-sync-anomaly-assets",
    "app-sync-anomaly-events",
    "app-export-fleet-assets",
    "app-record-outcomes",
    "app-record-decision",
    "app-export-outcome-template",
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

    app_sync_fleet_assets = subparsers.add_parser(
        "app-sync-fleet-assets",
        help="Refresh fleet asset registry rows from persisted prediction runs",
    )
    app_sync_fleet_assets.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_sync_fleet_assets.add_argument("--run-id")

    app_sync_anomaly_assets = subparsers.add_parser(
        "app-sync-anomaly-assets",
        help="Refresh spacecraft anomaly channel assets from a ranked comparison CSV",
    )
    app_sync_anomaly_assets.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_sync_anomaly_assets.add_argument("--comparison-csv", type=Path, required=True)
    app_sync_anomaly_assets.add_argument("--source-name")

    app_sync_anomaly_events = subparsers.add_parser(
        "app-sync-anomaly-events",
        help="Refresh spacecraft anomaly channel assets from operational event CSV rows",
    )
    app_sync_anomaly_events.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_sync_anomaly_events.add_argument("--events-csv", type=Path, required=True)
    app_sync_anomaly_events.add_argument("--source-name")

    app_export_fleet_assets = subparsers.add_parser(
        "app-export-fleet-assets",
        help="Export the persisted fleet asset registry as review evidence",
    )
    app_export_fleet_assets.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_export_fleet_assets.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "app_exports",
    )
    app_export_fleet_assets.add_argument(
        "--risk-level",
        action="append",
        choices=("critical", "watch", "nominal", "unknown"),
        help="Filter exported assets by latest risk level; repeat for multiple values",
    )
    app_export_fleet_assets.add_argument(
        "--domain",
        action="append",
        help="Filter exported assets by domain; repeat for multiple values",
    )
    app_export_fleet_assets.add_argument(
        "--status",
        action="append",
        help="Filter exported assets by latest status; repeat for multiple values",
    )
    app_export_fleet_assets.add_argument(
        "--attention-only",
        action="store_true",
        help="Export only assets with critical/watch risk or attention reasons",
    )

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

    app_record_decision = subparsers.add_parser(
        "app-record-decision",
        help="Append an auditable operator decision to a persisted prediction run",
    )
    app_record_decision.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_record_decision.add_argument("--run-id", required=True)
    app_record_decision.add_argument(
        "--status",
        required=True,
        choices=("review_required", "accepted", "watch", "escalated", "rejected"),
    )
    app_record_decision.add_argument("--actor", default="operator")
    app_record_decision.add_argument("--note")
    app_record_decision.add_argument("--ticket")
    app_record_decision.add_argument("--severity")
    app_record_decision.add_argument("--payload-json", type=Path)

    app_export_outcome_template = subparsers.add_parser(
        "app-export-outcome-template",
        help="Export a fillable outcome CSV template for a persisted prediction run",
    )
    app_export_outcome_template.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts") / "app" / "aerospace_prognostics.sqlite",
    )
    app_export_outcome_template.add_argument("--run-id", required=True)
    app_export_outcome_template.add_argument("--output-csv", type=Path)

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
        print(f"fleet_assets={summary['fleet_assets']}")
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

    if args.command == "app-sync-fleet-assets":
        result = sync_fleet_assets_from_prediction_run(
            args.database,
            run_id=args.run_id,
        )
        summary = database_summary(args.database)
        print(f"database={args.database}")
        if args.run_id is not None:
            print(f"run_id={args.run_id}")
        print(f"runs_synced={result['runs_synced']}")
        print(f"updated_assets={result['updated_assets']}")
        print(f"fleet_assets={summary['fleet_assets']}")
        return 0

    if args.command == "app-sync-anomaly-assets":
        result = sync_fleet_assets_from_anomaly_comparison(
            args.database,
            comparison_csv=args.comparison_csv,
            source_name=args.source_name,
        )
        summary = database_summary(args.database)
        print(f"database={args.database}")
        print(f"comparison_csv={result['source_path']}")
        print(f"source_name={result['source_name']}")
        print(f"channels_synced={result['channels_synced']}")
        print(f"updated_assets={result['updated_assets']}")
        print(f"fleet_assets={summary['fleet_assets']}")
        return 0

    if args.command == "app-sync-anomaly-events":
        result = sync_fleet_assets_from_anomaly_events(
            args.database,
            events_csv=args.events_csv,
            source_name=args.source_name,
        )
        summary = database_summary(args.database)
        print(f"database={args.database}")
        print(f"events_csv={result['source_path']}")
        print(f"source_name={result['source_name']}")
        print(f"events_processed={result['events_processed']}")
        print(f"channels_synced={result['channels_synced']}")
        print(f"updated_assets={result['updated_assets']}")
        print(f"fleet_assets={summary['fleet_assets']}")
        return 0

    if args.command == "app-export-fleet-assets":
        result = export_fleet_asset_registry(
            args.database,
            output_dir=args.output_dir,
            risk_levels=args.risk_level,
            domains=args.domain,
            statuses=args.status,
            attention_only=args.attention_only,
        )
        print(f"database={args.database}")
        print(f"output_dir={result['output_dir']}")
        print(f"registry_json={result['registry_json']}")
        print(f"registry_sha256={result['registry_sha256']}")
        print(f"assets_csv={result['assets_csv']}")
        print(f"assets_sha256={result['assets_sha256']}")
        print(f"asset_count={result['asset_count']}")
        print(f"filters={json.dumps(result['filters'], sort_keys=True)}")
        return 0

    if args.command == "app-record-decision":
        payload = _decision_payload(
            payload_json=args.payload_json,
            ticket=args.ticket,
            severity=args.severity,
        )
        event_id = record_prediction_run_event(
            args.database,
            run_id=args.run_id,
            event_type="operator_decision",
            status=args.status,
            actor=args.actor,
            note=args.note,
            payload=payload,
        )
        summary = database_summary(args.database)
        print(f"database={args.database}")
        print(f"run_id={args.run_id}")
        print(f"event_id={event_id}")
        print(f"decision_status={args.status}")
        print(f"actor={args.actor}")
        print(f"prediction_run_events={summary['prediction_run_events']}")
        return 0

    if args.command == "app-export-outcome-template":
        output_csv = args.output_csv or (
            Path("artifacts") / "app_exports" / f"{args.run_id}_outcomes_template.csv"
        )
        result = export_prediction_outcome_template(
            args.database,
            run_id=args.run_id,
            output_csv=output_csv,
        )
        print(f"database={args.database}")
        print(f"run_id={args.run_id}")
        print(f"outcome_template_csv={result['outcome_template_csv']}")
        print(f"outcome_template_sha256={result['outcome_template_sha256']}")
        print(f"prediction_count={result['prediction_count']}")
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


def _decision_payload(
    *,
    payload_json: Path | None,
    ticket: str | None,
    severity: str | None,
) -> dict[str, object]:
    payload = _read_json_file(payload_json) if payload_json is not None else {}
    if ticket:
        payload["ticket"] = ticket
    if severity:
        payload["severity"] = severity
    return payload


def _read_outcomes_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (EmptyDataError, ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"outcomes CSV could not be read as a valid CSV: {path}") from exc
