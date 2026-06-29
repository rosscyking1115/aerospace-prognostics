"""Turbofan RUL fleet asset helpers for the local PHM console."""

from __future__ import annotations


def turbofan_asset_risk_level(
    *,
    predicted_rul: float,
    predicted_rul_lower: float | None,
) -> str:
    """Classify an engine RUL prediction into an operator-facing risk level."""

    risk_floor = (
        min(predicted_rul, predicted_rul_lower)
        if predicted_rul_lower is not None
        else predicted_rul
    )
    if risk_floor <= 20:
        return "critical"
    if risk_floor <= 50:
        return "watch"
    return "nominal"


def turbofan_asset_attention_reasons(
    *,
    predicted_rul: float,
    predicted_rul_lower: float | None,
    predicted_rul_upper: float | None,
    risk_level: str,
) -> list[str]:
    """Explain why an engine RUL asset needs operator attention."""

    reasons: list[str] = []
    if risk_level == "critical":
        reasons.append("RUL at or below critical threshold")
    elif risk_level == "watch":
        reasons.append("RUL inside watch threshold")
    if predicted_rul_lower is not None and predicted_rul_lower <= 20 < predicted_rul:
        reasons.append("Interval lower bound crosses critical threshold")
    if (
        predicted_rul_lower is not None
        and predicted_rul_upper is not None
        and predicted_rul_upper - predicted_rul_lower >= 30
    ):
        reasons.append("Wide RUL interval")
    return reasons


def turbofan_asset_status(risk_level: str) -> str:
    """Map engine RUL risk to a fleet registry status."""

    return {
        "critical": "maintenance_review",
        "watch": "monitor",
        "nominal": "nominal",
    }.get(risk_level, "unknown")
