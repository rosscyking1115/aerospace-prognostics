"""Dashboard-ready fleet payloads for public demos and product surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DASHBOARD_SCHEMA_VERSION = "aerospace-prognostics/fleet-dashboard/v1"


@dataclass(frozen=True)
class FleetDashboardPayload:
    """A stable JSON contract for fleet-level dashboard views."""

    title: str
    generated_at_utc: str
    assets: list[dict[str, Any]]
    summary: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)
    schema_version: str = DASHBOARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""

        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "generated_at_utc": self.generated_at_utc,
            "summary": self.summary,
            "assets": self.assets,
            "evidence": self.evidence,
        }


def build_fleet_dashboard_payload(
    prediction_json: str | Path,
    *,
    title: str = "Aerospace PHM Fleet View",
    promotion_json: str | Path | None = None,
    release_bundle_json: str | Path | None = None,
    generated_at_utc: str | None = None,
) -> FleetDashboardPayload:
    """Build a dashboard payload from deployment prediction and evidence JSON."""

    prediction_path = Path(prediction_json)
    predictions = _read_json_object(prediction_path, "prediction JSON")
    prediction_rows = _list_of_objects(predictions.get("predictions"), "predictions")
    assets = [_asset_from_prediction(predictions, row) for row in prediction_rows]
    summary = _fleet_summary(assets)

    evidence: dict[str, Any] = {
        "prediction_json_path": str(prediction_path),
        "prediction_source": {
            "dataset": predictions.get("dataset"),
            "subset": predictions.get("subset"),
            "model_name": predictions.get("model_name"),
            "rul_cap": predictions.get("rul_cap"),
        },
    }
    if promotion_json is not None:
        promotion_path = Path(promotion_json)
        promotion = _read_json_object(promotion_path, "promotion JSON")
        evidence["promotion"] = _promotion_evidence(promotion_path, promotion)
    if release_bundle_json is not None:
        release_path = Path(release_bundle_json)
        release_bundle = _read_json_object(release_path, "release bundle JSON")
        evidence["release_bundle"] = _release_bundle_evidence(release_path, release_bundle)

    return FleetDashboardPayload(
        title=title,
        generated_at_utc=generated_at_utc or datetime.now(UTC).isoformat(),
        assets=assets,
        summary=summary,
        evidence=evidence,
    )


def write_fleet_dashboard_payload_json(
    payload: FleetDashboardPayload,
    output_json: str | Path,
) -> Path:
    """Write a dashboard payload JSON document."""

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload.to_dict(), indent=2, sort_keys=True) + "\n")
    return output_path


def _asset_from_prediction(
    prediction_document: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    unit_number = _required_int(prediction, "unit_number")
    predicted_rul = _required_float(prediction, "predicted_rul")
    rul_cap = _optional_float(prediction_document.get("rul_cap"))
    risk_level = _rul_risk_level(predicted_rul, rul_cap)
    return {
        "asset_id": f"{prediction_document.get('subset', 'unknown')}-unit-{unit_number}",
        "asset_type": "turbofan_engine",
        "dataset": prediction_document.get("dataset"),
        "subset": prediction_document.get("subset"),
        "unit_number": unit_number,
        "model_name": prediction_document.get("model_name"),
        "predicted_rul": predicted_rul,
        "rul_cap": rul_cap,
        "risk_level": risk_level,
        "status": _status_for_risk(risk_level),
    }


def _fleet_summary(assets: list[dict[str, Any]]) -> dict[str, Any]:
    risk_counts = {"critical": 0, "watch": 0, "nominal": 0, "unknown": 0}
    for asset in assets:
        risk_level = str(asset.get("risk_level", "unknown"))
        risk_counts[risk_level if risk_level in risk_counts else "unknown"] += 1
    predicted_ruls = [
        float(asset["predicted_rul"])
        for asset in assets
        if isinstance(asset.get("predicted_rul"), int | float)
    ]
    return {
        "asset_count": len(assets),
        "risk_counts": risk_counts,
        "min_predicted_rul": min(predicted_ruls) if predicted_ruls else None,
        "max_predicted_rul": max(predicted_ruls) if predicted_ruls else None,
    }


def _promotion_evidence(path: Path, promotion: dict[str, Any]) -> dict[str, Any]:
    gates = promotion.get("gates")
    gate_values = gates if isinstance(gates, dict) else {}
    return {
        "path": str(path),
        "status": promotion.get("status"),
        "artifact_identity": promotion.get("artifact_identity") or {},
        "gate_count": len(gate_values),
        "gates_passed": sum(1 for value in gate_values.values() if value is True),
    }


def _release_bundle_evidence(path: Path, release_bundle: dict[str, Any]) -> dict[str, Any]:
    evidence = release_bundle.get("evidence")
    evidence_values = evidence if isinstance(evidence, dict) else {}
    return {
        "path": str(path),
        "status": release_bundle.get("status"),
        "release_name": release_bundle.get("release_name"),
        "artifact_identity": release_bundle.get("artifact_identity") or {},
        "evidence_count": len(evidence_values),
    }


def _rul_risk_level(predicted_rul: float, rul_cap: float | None) -> str:
    if predicted_rul <= 20:
        return "critical"
    if predicted_rul <= 50:
        return "watch"
    if rul_cap is not None and predicted_rul >= rul_cap * 0.95:
        return "nominal"
    return "nominal"


def _status_for_risk(risk_level: str) -> str:
    return {
        "critical": "maintenance_review",
        "watch": "monitor",
        "nominal": "nominal",
    }.get(risk_level, "unknown")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _list_of_objects(value: Any, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{field_name}[{index}] must be an object")
        rows.append(row)
    return rows


def _required_int(payload: dict[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"prediction is missing numeric {field_name}")
    return int(value)


def _required_float(payload: dict[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"prediction is missing numeric {field_name}")
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
