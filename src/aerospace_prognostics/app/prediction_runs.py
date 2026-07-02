"""Prediction-run evidence helpers for the local PHM console."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

PREDICTION_RUN_EVIDENCE_SCHEMA_VERSION = (
    "aerospace-prognostics/prediction-run-evidence/v1"
)


def prediction_rows(prediction_document: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and return prediction rows from an API/artifact response document."""

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


def prediction_run_event_record(
    *,
    run_id: str,
    event_type: str,
    actor: str,
    timestamp: str,
    status: str | None = None,
    note: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable DB record for one prediction-run audit event."""

    normalized_payload = payload or {}
    event_material = {
        "run_id": run_id,
        "event_type": event_type,
        "status": status,
        "actor": actor,
        "note": note,
        "payload": normalized_payload,
        "timestamp": timestamp,
    }
    event_id = f"event-{_sha256_text(_json_dumps(event_material))[:16]}"
    return {
        "event_id": event_id,
        "run_id": run_id,
        "event_type": event_type,
        "status": status,
        "actor": actor,
        "note": note,
        "payload_json": _json_dumps(normalized_payload),
        "created_at_utc": timestamp,
    }


def event_from_row(row: Any) -> dict[str, Any]:
    """Convert a prediction_run_events DB row into the console event contract."""

    event = dict(row)
    event["payload"] = _json_loads(event.pop("payload_json"))
    return event


def run_summary_from_row(row: Any) -> dict[str, Any]:
    """Convert a prediction-run aggregate DB row into the console list contract."""

    result = dict(row)
    monitoring = _json_loads(result.pop("monitoring_json"))
    result["drift_alert_count"] = _drift_alert_count(monitoring)
    return with_interval_availability(result)


def prediction_run_detail(
    *,
    run_row: Any,
    prediction_rows: list[Any],
    event_rows: list[Any],
) -> dict[str, Any]:
    """Build the loaded prediction-run contract from persisted DB rows."""

    run = dict(run_row)
    run["monitoring"] = _json_loads(run.pop("monitoring_json"))
    return {
        "run": run,
        "predictions": [dict(row) for row in prediction_rows],
        "audit_events": [event_from_row(row) for row in event_rows],
    }


def prediction_run_export_summary(
    *,
    run_id: str,
    output_dir: str | Path,
    evidence_json_path: str | Path,
    evidence_sha256: str,
    predictions_csv_path: str | Path,
    predictions_sha256: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build the export result summary returned by the app-store API."""

    predictions = list(manifest["predictions"])
    outcome_count = sum(1 for row in predictions if row.get("actual_rul") is not None)
    return {
        "run_id": run_id,
        "output_dir": str(Path(output_dir)),
        "evidence_json": str(Path(evidence_json_path)),
        "evidence_sha256": evidence_sha256,
        "predictions_csv": str(Path(predictions_csv_path)),
        "predictions_sha256": predictions_sha256,
        "prediction_count": len(predictions),
        "outcome_count": outcome_count,
        "audit_event_count": len(manifest["audit_events"]),
    }


def outcome_rows(outcomes: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate and normalize observed RUL outcome rows."""

    required_columns = {"unit_number", "actual_rul"}
    missing_columns = required_columns.difference(outcomes.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"outcomes require columns: {missing}")
    numeric_outcomes = outcomes[["unit_number", "actual_rul"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric_outcomes.empty:
        raise ValueError("outcomes must contain at least one row")
    if numeric_outcomes.isnull().any().any():
        raise ValueError(
            "outcome rows require numeric, non-null unit_number and actual_rul"
        )
    if (numeric_outcomes["unit_number"] % 1 != 0).any():
        raise ValueError("outcome unit_number values must be whole numbers")
    if (numeric_outcomes["actual_rul"] < 0).any():
        raise ValueError("actual_rul values must be nonnegative")

    parsed: list[dict[str, Any]] = []
    for row in numeric_outcomes.to_dict(orient="records"):
        parsed.append(
            {
                "unit_number": int(row["unit_number"]),
                "actual_rul": float(row["actual_rul"]),
            }
        )
    return parsed


def with_interval_availability(row: dict[str, Any]) -> dict[str, Any]:
    """Add interval and observed-outcome availability metrics to a run row."""

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


def build_prediction_run_evidence_payload(
    *,
    database_path: str | Path,
    database_schema_version: int | str,
    loaded_run: dict[str, Any],
    exported_at_utc: str,
    predictions_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a portable prediction-run evidence payload from loaded DB records."""

    predictions = list(loaded_run["predictions"])
    csv_file: dict[str, Any] = {"rows": len(predictions)}
    if predictions_csv_path is not None:
        csv_file["path"] = str(Path(predictions_csv_path))
    return {
        "schema_version": PREDICTION_RUN_EVIDENCE_SCHEMA_VERSION,
        "exported_at_utc": exported_at_utc,
        "database": {
            "path": str(Path(database_path)),
            "schema_version": database_schema_version,
        },
        "run": loaded_run["run"],
        "predictions": predictions,
        "audit_events": loaded_run["audit_events"],
        "files": {
            "predictions_csv": csv_file,
        },
    }


def outcome_template_frame(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a fillable observed-RUL outcome template for prediction rows."""

    template = pd.DataFrame(predictions).reindex(columns=["unit_number"])
    template["actual_rul"] = ""
    return template


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
