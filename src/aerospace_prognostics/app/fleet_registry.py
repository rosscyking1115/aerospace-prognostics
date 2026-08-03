"""Fleet registry evidence helpers for the local PHM console."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def normalized_filter_values(values: Iterable[str] | None) -> list[str]:
    """Normalize optional user filter values into stable non-empty strings."""
    if values is None:
        return []
    return sorted({str(value).strip() for value in values if str(value).strip()})


def fleet_asset_filters(
    *,
    risk_levels: Iterable[str] | None,
    domains: Iterable[str] | None,
    statuses: Iterable[str] | None,
    attention_only: bool,
) -> dict[str, Any]:
    """Build normalized fleet-registry filters for evidence payloads."""
    return {
        "risk_levels": normalized_filter_values(risk_levels),
        "domains": normalized_filter_values(domains),
        "statuses": normalized_filter_values(statuses),
        "attention_only": bool(attention_only),
    }


def fleet_asset_registry_summary(assets: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize fleet-registry assets for portable evidence bundles."""
    risk_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for asset in assets:
        risk = str(asset.get("latest_risk_level") or "unknown")
        domain = str(asset.get("domain") or "unknown")
        status = str(asset.get("latest_status") or "unknown")
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
    attention_required_count = sum(
        1
        for asset in assets
        if str(asset.get("latest_risk_level")) in {"critical", "watch"}
        or bool(asset.get("latest_attention_reasons"))
    )
    return {
        "asset_count": len(assets),
        "attention_required_count": attention_required_count,
        "risk_counts": dict(sorted(risk_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
    }


def fleet_asset_export_rows(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten fleet-registry assets into operator handoff CSV rows."""
    rows: list[dict[str, Any]] = []
    for asset in assets:
        metadata = asset.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        reasons = asset.get("latest_attention_reasons")
        reasons = reasons if isinstance(reasons, list) else []
        rows.append(
            {
                "asset_id": asset.get("asset_id"),
                "asset_type": asset.get("asset_type"),
                "domain": asset.get("domain"),
                "source_dataset": asset.get("source_dataset"),
                "source_subset": asset.get("source_subset"),
                "external_id": asset.get("external_id"),
                "latest_run_id": asset.get("latest_run_id"),
                "latest_rul_prediction": asset.get("latest_rul_prediction"),
                "latest_rul_lower": asset.get("latest_rul_lower"),
                "latest_rul_upper": asset.get("latest_rul_upper"),
                "latest_risk_level": asset.get("latest_risk_level"),
                "latest_status": asset.get("latest_status"),
                "priority_score": asset.get("priority_score"),
                "priority_band": asset.get("priority_band"),
                "priority_reasons": "; ".join(
                    str(reason) for reason in asset.get("priority_reasons") or []
                ),
                "attention_reasons": "; ".join(str(reason) for reason in reasons),
                "first_seen_at_utc": asset.get("first_seen_at_utc"),
                "last_seen_at_utc": asset.get("last_seen_at_utc"),
                "model_name": metadata.get("model_name"),
                "artifact_id": metadata.get("artifact_id"),
                "source_name": metadata.get("source_name"),
                "input_sha256": metadata.get("input_sha256"),
                "channel_id": metadata.get("channel_id"),
                "spacecraft": metadata.get("spacecraft"),
                "event_time_utc": metadata.get("event_time_utc"),
                "severity": metadata.get("severity"),
                "active": metadata.get("active"),
                "anomaly_score": metadata.get("anomaly_score"),
                "threshold": metadata.get("threshold"),
                "anomaly_source": metadata.get("source"),
                "f1": metadata.get("f1"),
                "point_adjusted_f1": metadata.get("point_adjusted_f1"),
                "false_alarm_rate": metadata.get("false_alarm_rate"),
                "miss_rate": metadata.get("miss_rate"),
                "support": metadata.get("support"),
                "predicted_positives": metadata.get("predicted_positives"),
            }
        )
    return rows
