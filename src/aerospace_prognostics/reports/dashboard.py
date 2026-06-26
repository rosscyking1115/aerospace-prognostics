"""Dashboard-ready fleet payloads for public demos and product surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload

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
    assets = _rank_assets(
        [_asset_from_prediction(predictions, row) for row in prediction_rows]
    )
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

    return write_json_payload(payload.to_dict(), output_json)


def render_fleet_dashboard_html(payload: FleetDashboardPayload | dict[str, Any]) -> str:
    """Render a standalone fleet dashboard HTML document."""

    payload_dict = payload.to_dict() if isinstance(payload, FleetDashboardPayload) else payload
    title = str(payload_dict.get("title") or "Aerospace PHM Fleet View")
    summary = payload_dict.get("summary") if isinstance(payload_dict.get("summary"), dict) else {}
    assets = _list_of_objects(payload_dict.get("assets"), "assets")
    evidence = (
        payload_dict.get("evidence")
        if isinstance(payload_dict.get("evidence"), dict)
        else {}
    )
    risk_counts = (
        summary.get("risk_counts")
        if isinstance(summary.get("risk_counts"), dict)
        else {"critical": 0, "watch": 0, "nominal": 0, "unknown": 0}
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_html(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #64727d;
      --line: #d9e0e5;
      --critical: #b42318;
      --watch: #a15c07;
      --nominal: #16724d;
      --accent: #1f5d7a;
      --track: #edf1f4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    .shell {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 36px;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric, .evidence, .table-wrap {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{
      padding: 14px;
      min-height: 84px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .metric strong {{
      display: block;
      margin-top: 6px;
      font-size: 26px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 14px;
      align-items: start;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 780px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
      font-size: 14px;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
      background: #fbfcfd;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .risk {{
      display: inline-flex;
      align-items: center;
      min-width: 92px;
      justify-content: center;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .risk-critical {{ color: #fff; background: var(--critical); }}
    .risk-watch {{ color: #fff; background: var(--watch); }}
    .risk-nominal {{ color: #fff; background: var(--nominal); }}
    .risk-unknown {{ color: var(--ink); background: var(--track); }}
    .rul {{
      display: grid;
      grid-template-columns: 64px minmax(120px, 1fr);
      align-items: center;
      gap: 10px;
    }}
    .bar {{
      height: 10px;
      border-radius: 999px;
      background: var(--track);
      overflow: hidden;
    }}
    .fill {{
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
    }}
    .evidence {{
      padding: 14px;
    }}
    .evidence h2 {{
      margin: 0 0 10px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 118px minmax(0, 1fr);
      gap: 8px 10px;
      font-size: 13px;
      margin-bottom: 14px;
    }}
    .kv dt {{
      color: var(--muted);
      margin: 0;
    }}
    .kv dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 820px) {{
      header {{
        display: block;
      }}
      .meta {{
        margin-top: 8px;
        text-align: left;
      }}
      .metrics {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .layout {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <h1>{_html(title)}</h1>
      <div class="meta">{_html(str(payload_dict.get("generated_at_utc", "")))}</div>
    </header>
    <section class="metrics" aria-label="Fleet summary">
      {_metric("Assets", summary.get("asset_count"))}
      {_metric("Critical", risk_counts.get("critical"))}
      {_metric("Watch", risk_counts.get("watch"))}
      {_metric("Nominal", risk_counts.get("nominal"))}
    </section>
    <section class="layout">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Priority</th>
              <th>Asset</th>
              <th>Subset</th>
              <th>Model</th>
              <th>RUL</th>
              <th>Interval</th>
              <th>Risk</th>
              <th>Status</th>
              <th>Attention</th>
            </tr>
          </thead>
          <tbody>
            {''.join(_asset_row(asset) for asset in assets)}
          </tbody>
        </table>
      </div>
      <aside class="evidence">
        <h2>Evidence</h2>
        {_evidence_block(evidence)}
      </aside>
    </section>
  </main>
</body>
</html>
"""
    return html


def write_fleet_dashboard_html(
    payload: FleetDashboardPayload | dict[str, Any],
    output_html: str | Path,
) -> Path:
    """Write a standalone fleet dashboard HTML document."""

    output_path = prepare_output_path(output_html)
    output_path.write_text(render_fleet_dashboard_html(payload), encoding="utf-8")
    return output_path


def _asset_from_prediction(
    prediction_document: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    unit_number = _required_int(prediction, "unit_number")
    predicted_rul = _required_float(prediction, "predicted_rul")
    rul_cap = _optional_float(prediction_document.get("rul_cap"))
    rul_interval = _prediction_interval(prediction)
    risk_level = _rul_risk_level(predicted_rul, rul_cap, rul_interval=rul_interval)
    attention_reasons = _attention_reasons(
        predicted_rul=predicted_rul,
        risk_level=risk_level,
        rul_cap=rul_cap,
        rul_interval=rul_interval,
    )
    return {
        "asset_id": f"{prediction_document.get('subset', 'unknown')}-unit-{unit_number}",
        "asset_type": "turbofan_engine",
        "dataset": prediction_document.get("dataset"),
        "subset": prediction_document.get("subset"),
        "unit_number": unit_number,
        "model_name": prediction_document.get("model_name"),
        "predicted_rul": predicted_rul,
        "rul_cap": rul_cap,
        "rul_interval": rul_interval,
        "risk_level": risk_level,
        "risk_score": _risk_score(risk_level, predicted_rul, rul_interval),
        "status": _status_for_risk(risk_level),
        "attention_reasons": attention_reasons,
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
    attention_required = [
        asset
        for asset in assets
        if str(asset.get("risk_level")) in {"critical", "watch"}
        or bool(asset.get("attention_reasons"))
    ]
    return {
        "asset_count": len(assets),
        "risk_counts": risk_counts,
        "min_predicted_rul": min(predicted_ruls) if predicted_ruls else None,
        "max_predicted_rul": max(predicted_ruls) if predicted_ruls else None,
        "attention_required_count": len(attention_required),
        "top_attention_assets": [
            {
                "asset_id": asset.get("asset_id"),
                "priority_rank": asset.get("priority_rank"),
                "risk_level": asset.get("risk_level"),
                "predicted_rul": asset.get("predicted_rul"),
                "attention_reasons": asset.get("attention_reasons", []),
            }
            for asset in attention_required[:5]
        ],
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


def _rank_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        assets,
        key=lambda asset: (
            -float(asset.get("risk_score") or 0.0),
            float(asset.get("predicted_rul") or 0.0),
            str(asset.get("asset_id") or ""),
        ),
    )
    for index, asset in enumerate(ranked, start=1):
        asset["priority_rank"] = index
    return ranked


def _rul_risk_level(
    predicted_rul: float,
    rul_cap: float | None,
    *,
    rul_interval: dict[str, float] | None = None,
) -> str:
    lower_bound = rul_interval.get("lower") if rul_interval else None
    risk_floor = min(predicted_rul, lower_bound) if lower_bound is not None else predicted_rul
    if risk_floor <= 20:
        return "critical"
    if risk_floor <= 50:
        return "watch"
    if rul_cap is not None and predicted_rul >= rul_cap * 0.95:
        return "nominal"
    return "nominal"


def _risk_score(
    risk_level: str,
    predicted_rul: float,
    rul_interval: dict[str, float] | None,
) -> float:
    base = {"critical": 300.0, "watch": 200.0, "nominal": 100.0}.get(risk_level, 0.0)
    lower_bound = rul_interval.get("lower") if rul_interval else predicted_rul
    interval_width = (
        (rul_interval["upper"] - rul_interval["lower"])
        if rul_interval is not None
        else 0.0
    )
    return base + max(0.0, 125.0 - lower_bound) + min(50.0, max(0.0, interval_width))


def _attention_reasons(
    *,
    predicted_rul: float,
    risk_level: str,
    rul_cap: float | None,
    rul_interval: dict[str, float] | None,
) -> list[str]:
    reasons: list[str] = []
    if risk_level == "critical":
        reasons.append("RUL at or below critical threshold")
    elif risk_level == "watch":
        reasons.append("RUL inside watch threshold")
    if rul_interval is not None:
        lower = rul_interval["lower"]
        upper = rul_interval["upper"]
        if lower <= 20 < predicted_rul:
            reasons.append("Interval lower bound crosses critical threshold")
        if upper - lower >= 30:
            reasons.append("Wide RUL interval")
    if rul_cap is not None and predicted_rul >= rul_cap * 0.95:
        reasons.append("Prediction near configured RUL cap")
    return reasons


def _status_for_risk(risk_level: str) -> str:
    return {
        "critical": "maintenance_review",
        "watch": "monitor",
        "nominal": "nominal",
    }.get(risk_level, "unknown")


def _metric(label: str, value: Any) -> str:
    return (
        '<div class="metric">'
        f"<span>{_html(label)}</span>"
        f"<strong>{_html(_display_value(value))}</strong>"
        "</div>"
    )


def _asset_row(asset: dict[str, Any]) -> str:
    risk_level = str(asset.get("risk_level") or "unknown")
    predicted_rul = _optional_float(asset.get("predicted_rul"))
    rul_cap = _optional_float(asset.get("rul_cap"))
    width = _rul_width(predicted_rul, rul_cap)
    reasons = asset.get("attention_reasons")
    return (
        "<tr>"
        f"<td>{_html(asset.get('priority_rank'))}</td>"
        f"<td>{_html(asset.get('asset_id'))}</td>"
        f"<td>{_html(asset.get('subset'))}</td>"
        f"<td>{_html(asset.get('model_name'))}</td>"
        "<td>"
        '<div class="rul">'
        f"<span>{_html(_format_number(predicted_rul))}</span>"
        '<div class="bar">'
        f'<div class="fill" style="width: {width:.2f}%"></div>'
        "</div>"
        "</div>"
        "</td>"
        f"<td>{_html(_format_interval(asset.get('rul_interval')))}</td>"
        f'<td><span class="risk risk-{_html(_risk_class(risk_level))}">'
        f"{_html(risk_level)}</span></td>"
        f"<td>{_html(asset.get('status'))}</td>"
        f"<td>{_html(_reason_summary(reasons))}</td>"
        "</tr>"
    )


def _evidence_block(evidence: dict[str, Any]) -> str:
    prediction_source = evidence.get("prediction_source")
    promotion = evidence.get("promotion")
    release_bundle = evidence.get("release_bundle")
    rows: list[tuple[str, Any]] = []
    if isinstance(prediction_source, dict):
        rows.extend(
            [
                ("Dataset", prediction_source.get("dataset")),
                ("Subset", prediction_source.get("subset")),
                ("Model", prediction_source.get("model_name")),
            ]
        )
    if isinstance(promotion, dict):
        rows.extend(
            [
                ("Promotion", promotion.get("status")),
                ("Gates", f"{promotion.get('gates_passed')}/{promotion.get('gate_count')}"),
            ]
        )
    if isinstance(release_bundle, dict):
        rows.extend(
            [
                ("Release", release_bundle.get("status")),
                ("Evidence", release_bundle.get("evidence_count")),
            ]
        )
    if not rows:
        rows.append(("Source", evidence.get("prediction_json_path")))
    return "<dl class=\"kv\">" + "".join(
        f"<dt>{_html(label)}</dt><dd>{_html(_display_value(value))}</dd>" for label, value in rows
    ) + "</dl>"


def _rul_width(predicted_rul: float | None, rul_cap: float | None) -> float:
    if predicted_rul is None:
        return 0.0
    if rul_cap is None or rul_cap <= 0:
        return 100.0
    return max(0.0, min(100.0, (predicted_rul / rul_cap) * 100.0))


def _prediction_interval(prediction: dict[str, Any]) -> dict[str, float] | None:
    interval = prediction.get("rul_interval")
    if isinstance(interval, dict):
        lower = _optional_float(interval.get("lower"))
        upper = _optional_float(interval.get("upper"))
    else:
        lower = _first_optional_float(
            prediction,
            ("predicted_rul_lower", "rul_lower", "rul_lower_bound"),
        )
        upper = _first_optional_float(
            prediction,
            ("predicted_rul_upper", "rul_upper", "rul_upper_bound"),
        )
    if lower is None or upper is None:
        return None
    return {"lower": min(lower, upper), "upper": max(lower, upper)}


def _first_optional_float(payload: dict[str, Any], field_names: tuple[str, ...]) -> float | None:
    for field_name in field_names:
        value = _optional_float(payload.get(field_name))
        if value is not None:
            return value
    return None


def _format_interval(interval: Any) -> str:
    if not isinstance(interval, dict):
        return "n/a"
    lower = _optional_float(interval.get("lower"))
    upper = _optional_float(interval.get("upper"))
    if lower is None or upper is None:
        return "n/a"
    return f"{_format_number(lower)}-{_format_number(upper)}"


def _reason_summary(reasons: Any) -> str:
    if not isinstance(reasons, list) or not reasons:
        return "none"
    return "; ".join(str(reason) for reason in reasons[:2])


def _risk_class(risk_level: str) -> str:
    return risk_level if risk_level in {"critical", "watch", "nominal", "unknown"} else "unknown"


def _display_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _html(value: Any) -> str:
    return escape(_display_value(value), quote=True)


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
