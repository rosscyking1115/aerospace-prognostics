"""Fleet priority scoring and validation policy helpers."""

from __future__ import annotations

from typing import Any


def fleet_priority_policy_summary(assets: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the current fleet priority queue."""
    band_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    review_queue_count = 0
    for asset in assets:
        band = str(asset.get("priority_band") or "unknown")
        band_counts[band] = band_counts.get(band, 0) + 1
        if band in {"immediate_review", "review"}:
            review_queue_count += 1
        reasons = asset.get("priority_reasons")
        if not isinstance(reasons, list):
            reasons = []
        for reason in reasons:
            reason_text = str(reason)
            reason_counts[reason_text] = reason_counts.get(reason_text, 0) + 1
    top_assets = [
        {
            "asset_id": asset.get("asset_id"),
            "domain": asset.get("domain"),
            "latest_risk_level": asset.get("latest_risk_level"),
            "priority_score": asset.get("priority_score"),
            "priority_band": asset.get("priority_band"),
            "priority_reasons": asset.get("priority_reasons") or [],
        }
        for asset in assets[:10]
    ]
    return {
        "review_queue_count": review_queue_count,
        "band_counts": dict(sorted(band_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "top_assets": top_assets,
    }


def fleet_priority_policy_validation_checks(
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate cross-domain fleet priority policy invariants."""
    critical_assets = [
        asset
        for asset in assets
        if str(asset.get("latest_risk_level") or "") == "critical"
    ]
    critical_misses = [
        asset
        for asset in critical_assets
        if str(asset.get("priority_band") or "") != "immediate_review"
    ]
    watch_assets = [
        asset for asset in assets if str(asset.get("latest_risk_level") or "") == "watch"
    ]
    watch_misses = [
        asset
        for asset in watch_assets
        if str(asset.get("priority_band") or "") not in {"immediate_review", "review"}
    ]
    scores = [float(asset.get("priority_score") or 0.0) for asset in assets]
    attention_assets = [
        asset
        for asset in assets
        if str(asset.get("latest_risk_level") or "") in {"critical", "watch"}
        or bool(asset.get("latest_attention_reasons"))
    ]
    unexplained_attention = [
        asset for asset in attention_assets if not asset.get("priority_reasons")
    ]
    domains = {str(asset.get("domain") or "") for asset in assets}
    operational_domains_present = {"turbofan_rul", "spacecraft_anomaly"}.issubset(
        domains
    )
    top_domains = {str(asset.get("domain") or "") for asset in assets[:10]}
    live_event_assets = [
        asset
        for asset in assets
        if _asset_metadata(asset).get("event_kind") == "live_anomaly_event"
    ]
    unexplained_live_events = [
        asset
        for asset in live_event_assets
        if not any(
            str(reason).startswith(("Live anomaly", "Anomaly score"))
            for reason in asset.get("priority_reasons") or []
        )
    ]
    return [
        _fleet_priority_policy_check(
            "critical_assets_are_immediate_review",
            title="Critical assets enter immediate review",
            applicable=bool(critical_assets),
            passed=not critical_misses,
            observed_assets=critical_assets,
            failed_assets=critical_misses,
        ),
        _fleet_priority_policy_check(
            "watch_assets_are_review_or_better",
            title="Watch assets enter the review queue",
            applicable=bool(watch_assets),
            passed=not watch_misses,
            observed_assets=watch_assets,
            failed_assets=watch_misses,
        ),
        {
            "check_id": "priority_order_is_descending",
            "title": "Assets are sorted by descending priority score",
            "status": "pass"
            if scores == sorted(scores, reverse=True)
            else "fail",
            "applicable": bool(assets),
            "observed_count": len(assets),
            "failed_asset_ids": [],
            "evidence": {
                "first_scores": scores[:10],
            },
        },
        _fleet_priority_policy_check(
            "attention_assets_have_explanations",
            title="Attention assets have operator-readable explanations",
            applicable=bool(attention_assets),
            passed=not unexplained_attention,
            observed_assets=attention_assets,
            failed_assets=unexplained_attention,
        ),
        {
            "check_id": "cross_domain_review_queue",
            "title": "Cross-domain fleets keep both domains visible in the top queue",
            "status": (
                "pass"
                if not operational_domains_present
                or {"turbofan_rul", "spacecraft_anomaly"}.issubset(top_domains)
                else "fail"
            ),
            "applicable": operational_domains_present,
            "observed_count": len(assets),
            "failed_asset_ids": []
            if not operational_domains_present
            or {"turbofan_rul", "spacecraft_anomaly"}.issubset(top_domains)
            else _asset_ids(assets[:10]),
            "evidence": {
                "domains": sorted(domains),
                "top_queue_domains": sorted(top_domains),
            },
        },
        _fleet_priority_policy_check(
            "live_anomaly_events_are_explained",
            title="Live anomaly event assets preserve event-specific reasons",
            applicable=bool(live_event_assets),
            passed=not unexplained_live_events,
            observed_assets=live_event_assets,
            failed_assets=unexplained_live_events,
        ),
    ]


def render_fleet_priority_policy_validation_markdown(
    report: dict[str, Any],
) -> str:
    """Render priority-policy validation evidence as Markdown."""
    policy = report.get("priority_policy")
    policy = policy if isinstance(policy, dict) else {}
    lines = [
        "# Fleet Priority Policy Validation",
        "",
        f"- Overall status: {_markdown_inline(report.get('overall_status'))}",
        f"- Exported at UTC: {_markdown_inline(report.get('exported_at_utc'))}",
        f"- Asset count: {_markdown_inline(report.get('asset_count'))}",
        f"- Review queue count: {_markdown_inline(policy.get('review_queue_count'))}",
        "",
        "## Scenario Checks",
        "",
        "| Check | Status | Applicable | Observed | Failed assets |",
        "| --- | --- | --- | ---: | --- |",
    ]
    checks = report.get("scenario_checks")
    checks = checks if isinstance(checks, list) else []
    for check in checks:
        check = check if isinstance(check, dict) else {}
        failed_asset_ids = check.get("failed_asset_ids")
        failed_asset_ids = failed_asset_ids if isinstance(failed_asset_ids, list) else []
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(check.get("title")),
                    _markdown_cell(check.get("status")),
                    _markdown_cell(check.get("applicable")),
                    _markdown_cell(check.get("observed_count")),
                    _markdown_cell(", ".join(str(asset) for asset in failed_asset_ids)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Priority Bands",
            "",
            "| Band | Count |",
            "| --- | ---: |",
        ]
    )
    band_counts = policy.get("band_counts")
    band_counts = band_counts if isinstance(band_counts, dict) else {}
    for band, count in sorted(band_counts.items()):
        lines.append(f"| {_markdown_cell(band)} | {_markdown_cell(count)} |")
    lines.extend(
        [
            "",
            "## Top Assets",
            "",
            "| Asset | Domain | Risk | Score | Band | Reasons |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    top_assets = policy.get("top_assets")
    top_assets = top_assets if isinstance(top_assets, list) else []
    for asset in top_assets:
        asset = asset if isinstance(asset, dict) else {}
        reasons = asset.get("priority_reasons")
        reasons = reasons if isinstance(reasons, list) else []
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(asset.get("asset_id")),
                    _markdown_cell(asset.get("domain")),
                    _markdown_cell(asset.get("latest_risk_level")),
                    _markdown_cell(asset.get("priority_score")),
                    _markdown_cell(asset.get("priority_band")),
                    _markdown_cell("; ".join(str(reason) for reason in reasons)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def fleet_asset_priority(asset: dict[str, Any]) -> dict[str, Any]:
    """Score one fleet asset for cross-domain review priority."""
    risk_level = str(asset.get("latest_risk_level") or "unknown")
    score = {
        "critical": 300.0,
        "watch": 200.0,
        "nominal": 100.0,
    }.get(risk_level, 0.0)
    reasons = [f"Risk level is {risk_level}"]
    domain = str(asset.get("domain") or "")
    if domain == "turbofan_rul":
        score += _turbofan_priority_modifier(asset, reasons)
    elif domain == "spacecraft_anomaly":
        score += _spacecraft_anomaly_priority_modifier(asset, reasons)

    attention = asset.get("latest_attention_reasons")
    attention_count = len(attention) if isinstance(attention, list) else 0
    if attention_count:
        score += min(25.0, attention_count * 5.0)
        reasons.append(f"{attention_count} attention reason(s)")

    rounded_score = round(score, 3)
    return {
        "score": rounded_score,
        "band": _fleet_asset_priority_band(rounded_score),
        "reasons": reasons,
    }


def _fleet_priority_policy_check(
    check_id: str,
    *,
    title: str,
    applicable: bool,
    passed: bool,
    observed_assets: list[dict[str, Any]],
    failed_assets: list[dict[str, Any]],
) -> dict[str, Any]:
    status = "pass" if not applicable or passed else "fail"
    return {
        "check_id": check_id,
        "title": title,
        "status": status,
        "applicable": applicable,
        "observed_count": len(observed_assets),
        "failed_asset_ids": _asset_ids(failed_assets),
        "evidence": {
            "sample_asset_ids": _asset_ids(observed_assets[:10]),
            "failed_count": len(failed_assets),
        },
    }


def _asset_ids(assets: list[dict[str, Any]]) -> list[str]:
    return [str(asset.get("asset_id")) for asset in assets if asset.get("asset_id")]


def _asset_metadata(asset: dict[str, Any]) -> dict[str, Any]:
    metadata = asset.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _turbofan_priority_modifier(asset: dict[str, Any], reasons: list[str]) -> float:
    modifier = 0.0
    predicted_rul = _optional_float(asset.get("latest_rul_prediction"))
    lower = _optional_float(asset.get("latest_rul_lower"))
    upper = _optional_float(asset.get("latest_rul_upper"))
    risk_floor = _minimum_present_float(predicted_rul, lower)
    if risk_floor is not None:
        urgency = max(0.0, 100.0 - risk_floor)
        modifier += min(100.0, urgency)
        reasons.append(f"RUL risk floor {risk_floor:.1f}")
    if lower is not None and upper is not None:
        width = upper - lower
        if width >= 30:
            modifier += min(25.0, width / 4.0)
            reasons.append(f"Wide RUL interval {width:.1f}")
    return modifier


def _spacecraft_anomaly_priority_modifier(
    asset: dict[str, Any],
    reasons: list[str],
) -> float:
    metadata = asset.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    modifier = 0.0
    if _metadata_has_live_anomaly_event(metadata):
        modifier += _spacecraft_anomaly_event_priority_modifier(metadata, reasons)
    predicted_positives = _optional_int(metadata.get("predicted_positives")) or 0
    if predicted_positives > 0:
        modifier += min(40.0, predicted_positives * 5.0)
        reasons.append(f"{predicted_positives} predicted anomaly point(s)")
    miss_rate = _optional_float(metadata.get("miss_rate"))
    if miss_rate is not None:
        modifier += min(80.0, miss_rate * 80.0)
        reasons.append(f"Miss rate {miss_rate:.3f}")
    false_alarm_rate = _optional_float(metadata.get("false_alarm_rate"))
    if false_alarm_rate is not None:
        modifier += min(40.0, false_alarm_rate * 100.0)
        reasons.append(f"False alarm rate {false_alarm_rate:.3f}")
    f1 = _optional_float(metadata.get("f1"))
    if f1 is not None and f1 < 0.6:
        modifier += min(50.0, (0.6 - f1) * 100.0)
        reasons.append(f"Low anomaly F1 {f1:.3f}")
    return modifier


def _metadata_has_live_anomaly_event(metadata: dict[str, Any]) -> bool:
    return (
        metadata.get("event_kind") == "live_anomaly_event"
        or "severity" in metadata
        or "active" in metadata
        or "anomaly_score" in metadata
        or "threshold" in metadata
    )


def _spacecraft_anomaly_event_priority_modifier(
    metadata: dict[str, Any],
    reasons: list[str],
) -> float:
    modifier = 0.0
    severity = str(metadata.get("severity") or "").strip().lower()
    severity_weights = {
        "critical": 90.0,
        "high": 70.0,
        "medium": 35.0,
        "warning": 35.0,
        "low": 10.0,
        "info": 0.0,
    }
    if severity:
        modifier += severity_weights.get(severity, 15.0)
        reasons.append(f"Live anomaly severity {severity}")
    if bool(metadata.get("active")):
        modifier += 60.0
        reasons.append("Active live anomaly event")
    score = _optional_float(metadata.get("anomaly_score"))
    threshold = _optional_float(metadata.get("threshold"))
    if score is not None and threshold is not None:
        margin = score - threshold
        if margin >= 0:
            modifier += min(50.0, 20.0 + margin * 30.0)
            reasons.append(f"Anomaly score {score:.3f} crossed threshold {threshold:.3f}")
        else:
            modifier += min(15.0, max(0.0, score) * 5.0)
            reasons.append(f"Anomaly score {score:.3f} below threshold {threshold:.3f}")
    elif score is not None:
        modifier += min(40.0, max(0.0, score) * 10.0)
        reasons.append(f"Anomaly score {score:.3f}")
    return modifier


def _minimum_present_float(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _fleet_asset_priority_band(score: float) -> str:
    if score >= 300:
        return "immediate_review"
    if score >= 200:
        return "review"
    if score >= 100:
        return "monitor"
    return "unknown"


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


def _markdown_inline(value: object, *, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value)
    if not text:
        return default
    return text.replace("\n", " ").replace("`", "'")


def _markdown_cell(value: object) -> str:
    return _markdown_inline(value, default="").replace("|", "\\|")
