"""Prediction-run evidence helpers for the local PHM console."""

from __future__ import annotations

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
