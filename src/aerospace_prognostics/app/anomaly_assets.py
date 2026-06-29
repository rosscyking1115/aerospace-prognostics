"""Spacecraft anomaly fleet asset helpers for the local PHM console."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def best_anomaly_asset_rows(path: Path) -> list[dict[str, Any]]:
    """Return the best comparison row for each anomaly channel."""

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"anomaly comparison CSV has no rows: {path}")
    required_columns = {
        "channel_id",
        "spacecraft",
        "source",
        "model_name",
        "precision",
        "recall",
        "f1",
        "point_adjusted_f1",
        "false_alarm_rate",
        "miss_rate",
        "support",
        "predicted_positives",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"anomaly comparison CSV is missing required columns: {missing}")
    if "rank_by_f1" not in frame.columns:
        frame = frame.sort_values(
            by=["channel_id", "f1", "point_adjusted_f1", "false_alarm_rate"],
            ascending=[True, False, False, True],
        )
        return [
            _anomaly_asset_row(row)
            for row in frame.groupby("channel_id", sort=True).head(1).to_dict("records")
        ]
    ranked = frame.copy()
    ranked["rank_by_f1"] = pd.to_numeric(ranked["rank_by_f1"], errors="coerce")
    if ranked["rank_by_f1"].isna().all():
        raise ValueError("anomaly comparison CSV rank_by_f1 values must be numeric")
    best = ranked.sort_values(
        by=["channel_id", "rank_by_f1", "f1", "point_adjusted_f1"],
        ascending=[True, True, False, False],
    )
    return [
        _anomaly_asset_row(row)
        for row in best.groupby("channel_id", sort=True).head(1).to_dict("records")
    ]


def latest_anomaly_event_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Return the latest live anomaly event row for each spacecraft channel."""

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"anomaly events CSV has no rows: {path}")
    required_columns = {"channel_id", "spacecraft"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"anomaly events CSV is missing required columns: {missing}")
    rows = [_anomaly_event_row(row) for row in frame.to_dict("records")]
    indexed_rows = [
        {
            **row,
            "_source_order": index,
            "_sort_time": _anomaly_event_sort_time(row),
        }
        for index, row in enumerate(rows)
    ]
    latest_rows = (
        pd.DataFrame(indexed_rows)
        .sort_values(
            by=["spacecraft", "channel_id", "_sort_time", "_source_order"],
            ascending=[True, True, True, True],
        )
        .groupby(["spacecraft", "channel_id"], sort=True)
        .tail(1)
        .sort_values(by=["spacecraft", "channel_id"])
        .to_dict("records")
    )
    cleaned = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in latest_rows
    ]
    return cleaned, len(rows)


def anomaly_asset_id(spacecraft: str, channel_id: str) -> str:
    """Build the stable fleet asset ID for a spacecraft channel."""

    return f"{spacecraft.lower()}-channel-{channel_id.lower()}"


def anomaly_asset_risk_level(row: dict[str, Any]) -> str:
    """Classify a benchmarked anomaly channel into a fleet risk level."""

    if float(row["miss_rate"]) >= 0.5 or float(row["f1"]) < 0.25:
        return "critical"
    if (
        int(row["predicted_positives"]) > 0
        or float(row["false_alarm_rate"]) >= 0.1
        or float(row["f1"]) < 0.5
    ):
        return "watch"
    return "nominal"


def anomaly_asset_attention_reasons(
    row: dict[str, Any],
    *,
    risk_level: str,
) -> list[str]:
    """Explain why a benchmarked anomaly channel needs attention."""

    reasons: list[str] = []
    if int(row["predicted_positives"]) > 0:
        reasons.append("Anomaly detections present in evaluation window")
    if float(row["miss_rate"]) >= 0.5:
        reasons.append("High anomaly miss rate")
    if float(row["false_alarm_rate"]) >= 0.1:
        reasons.append("False alarm rate at or above review threshold")
    if float(row["f1"]) < 0.5:
        reasons.append("Low pointwise F1 for anomaly channel")
    return reasons


def anomaly_asset_status(risk_level: str) -> str:
    """Map anomaly risk to an operator-facing status."""

    return {
        "critical": "anomaly_review",
        "watch": "monitor",
        "nominal": "nominal",
    }.get(risk_level, "unknown")


def anomaly_event_risk_level(row: dict[str, Any]) -> str:
    """Classify a live anomaly event into a fleet risk level."""

    severity = str(row.get("severity") or "").strip().lower()
    threshold_crossed = anomaly_event_threshold_crossed(row)
    active = bool(row.get("active"))
    if severity in {"critical", "high"}:
        return "critical"
    if severity in {"medium", "warning"} or active or threshold_crossed:
        return "watch"
    return "nominal"


def anomaly_event_attention_reasons(
    row: dict[str, Any],
    *,
    risk_level: str,
) -> list[str]:
    """Explain why a live anomaly event needs attention."""

    reasons: list[str] = []
    severity = str(row.get("severity") or "").strip().lower()
    if severity and severity not in {"info", "nominal"}:
        reasons.append(f"Severity {severity}")
    if bool(row.get("active")):
        reasons.append("Active anomaly event")
    if anomaly_event_threshold_crossed(row):
        reasons.append("Anomaly score crossed alert threshold")
    if risk_level == "critical" and not reasons:
        reasons.append("Critical anomaly event")
    return reasons


def anomaly_event_threshold_crossed(row: dict[str, Any]) -> bool:
    """Return whether an event score crosses its alert threshold."""

    score = _optional_float(row.get("anomaly_score"))
    threshold = _optional_float(row.get("threshold"))
    return score is not None and threshold is not None and score >= threshold


def _anomaly_asset_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "channel_id": str(row["channel_id"]),
        "spacecraft": str(row["spacecraft"]),
        "source": str(row["source"]),
        "model_name": str(row["model_name"]),
        "rank_by_f1": _optional_int(row.get("rank_by_f1")),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "point_adjusted_f1": float(row["point_adjusted_f1"]),
        "false_alarm_rate": float(row["false_alarm_rate"]),
        "miss_rate": float(row["miss_rate"]),
        "support": int(row["support"]),
        "predicted_positives": int(row["predicted_positives"]),
        "train_rows": _optional_int(row.get("train_rows")),
        "test_rows": _optional_int(row.get("test_rows")),
        "anomaly_points": _optional_int(row.get("anomaly_points")),
    }


def _anomaly_event_row(row: dict[str, Any]) -> dict[str, Any]:
    event_time = _optional_text(
        row.get("event_time_utc")
        or row.get("observed_at_utc")
        or row.get("timestamp_utc")
        or row.get("timestamp")
    )
    active = _optional_bool(
        row.get("active", row.get("prediction", row.get("is_anomaly")))
    )
    anomaly_score = _optional_float(
        row.get("anomaly_score", row.get("score", row.get("severity_score")))
    )
    threshold = _optional_float(row.get("threshold", row.get("alert_threshold")))
    severity = _normalized_anomaly_event_severity(row.get("severity"))
    return {
        "event_kind": "live_anomaly_event",
        "channel_id": str(row["channel_id"]),
        "spacecraft": str(row["spacecraft"]),
        "event_time_utc": event_time,
        "severity": severity,
        "active": active,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "model_name": _optional_text(row.get("model_name")),
        "source": _optional_text(row.get("source")),
        "note": _optional_text(row.get("note") or row.get("description")),
    }


def _anomaly_event_sort_time(row: dict[str, Any]) -> str:
    return str(row.get("event_time_utc") or "")


def _normalized_anomaly_event_severity(value: Any) -> str | None:
    severity = _optional_text(value)
    if severity is None:
        return None
    normalized = severity.strip().lower()
    return {
        "warn": "warning",
        "warning": "warning",
        "med": "medium",
        "moderate": "medium",
        "crit": "critical",
        "severe": "critical",
    }.get(normalized, normalized)


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


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
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
