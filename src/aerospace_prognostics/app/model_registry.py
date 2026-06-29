"""Model registry review helpers for the local PHM console."""

from __future__ import annotations

from typing import Any


def model_artifact_report_card(
    artifact: dict[str, Any],
    *,
    release_evidence: list[dict[str, Any]],
    prediction_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact review card for one registered model artifact."""

    evidence_by_type = {
        str(evidence.get("evidence_type")): evidence
        for evidence in release_evidence
        if evidence.get("evidence_type") is not None
    }
    promotion = _payload_for_evidence(evidence_by_type, "promotion_report")
    release_bundle = _payload_for_evidence(evidence_by_type, "release_bundle")
    provenance = _payload_for_evidence(evidence_by_type, "release_provenance")
    inspection = artifact.get("inspection")
    inspection = inspection if isinstance(inspection, dict) else {}
    uncertainty = inspection.get("uncertainty")
    uncertainty = uncertainty if isinstance(uncertainty, dict) else {}
    promotion_gates = _bool_dict(promotion.get("gates"))
    release_gates = _bool_dict(release_bundle.get("gates"))
    all_gates = {
        **{f"promotion.{key}": value for key, value in promotion_gates.items()},
        **{f"release.{key}": value for key, value in release_gates.items()},
    }
    latest_run_at = max(
        (
            str(run["created_at_utc"])
            for run in prediction_runs
            if run.get("created_at_utc") is not None
        ),
        default=None,
    )
    promotion_evidence = promotion.get("evidence")
    promotion_evidence = promotion_evidence if isinstance(promotion_evidence, dict) else {}
    benchmark = promotion_evidence.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    latency_ms = benchmark.get("latency_ms")
    latency_ms = latency_ms if isinstance(latency_ms, dict) else {}
    provenance_summary = provenance.get("summary")
    provenance_summary = provenance_summary if isinstance(provenance_summary, dict) else {}
    prediction_count_total = 0
    interval_count_total = 0
    weighted_interval_width = 0.0
    interval_width_count = 0
    max_interval_width = None
    outcome_count_total = 0
    weighted_absolute_error = 0.0
    weighted_signed_error = 0.0
    interval_outcome_count_total = 0
    interval_covered_count_total = 0
    for run in prediction_runs:
        prediction_count = _optional_int(run.get("prediction_count"))
        if prediction_count is None:
            prediction_count = _optional_int(run.get("prediction_row_count")) or 0
        interval_count = _optional_int(run.get("interval_count")) or 0
        mean_width = _optional_float(run.get("mean_interval_width"))
        run_max_width = _optional_float(run.get("max_interval_width"))
        outcome_count = _optional_int(run.get("outcome_count")) or 0
        mean_absolute_error = _optional_float(run.get("mean_absolute_error"))
        mean_signed_error = _optional_float(run.get("mean_signed_error"))
        interval_outcome_count = _optional_int(run.get("interval_outcome_count")) or 0
        interval_covered_count = _optional_int(run.get("interval_covered_count")) or 0
        prediction_count_total += prediction_count
        interval_count_total += interval_count
        outcome_count_total += outcome_count
        interval_outcome_count_total += interval_outcome_count
        interval_covered_count_total += interval_covered_count
        if mean_width is not None and interval_count > 0:
            weighted_interval_width += mean_width * interval_count
            interval_width_count += interval_count
        if mean_absolute_error is not None and outcome_count > 0:
            weighted_absolute_error += mean_absolute_error * outcome_count
        if mean_signed_error is not None and outcome_count > 0:
            weighted_signed_error += mean_signed_error * outcome_count
        if run_max_width is not None:
            max_interval_width = (
                run_max_width
                if max_interval_width is None
                else max(max_interval_width, run_max_width)
            )
    missing_interval_count = max(prediction_count_total - interval_count_total, 0)
    interval_availability_rate = (
        interval_count_total / prediction_count_total
        if prediction_count_total > 0
        else None
    )
    mean_interval_width = (
        weighted_interval_width / interval_width_count
        if interval_width_count > 0
        else None
    )
    mean_absolute_error = (
        weighted_absolute_error / outcome_count_total
        if outcome_count_total > 0
        else None
    )
    mean_signed_error = (
        weighted_signed_error / outcome_count_total
        if outcome_count_total > 0
        else None
    )
    outcome_availability_rate = (
        outcome_count_total / prediction_count_total
        if prediction_count_total > 0
        else None
    )
    outcome_interval_coverage_rate = (
        interval_covered_count_total / interval_outcome_count_total
        if interval_outcome_count_total > 0
        else None
    )
    return {
        "artifact_id": artifact.get("artifact_id"),
        "stage": artifact.get("stage"),
        "dataset": artifact.get("dataset"),
        "subset": artifact.get("subset"),
        "model_name": artifact.get("model_name"),
        "release_status": release_bundle.get("status"),
        "promotion_status": promotion.get("status"),
        "gate_count": len(all_gates),
        "passed_gate_count": sum(1 for value in all_gates.values() if value is True),
        "failed_gates": [key for key, value in sorted(all_gates.items()) if value is not True],
        "evidence_count": len(release_evidence),
        "prediction_run_count": len(prediction_runs),
        "latest_prediction_at": latest_run_at,
        "p95_latency_ms": latency_ms.get("p95"),
        "max_p95_latency_ms": benchmark.get("max_p95_latency_ms"),
        "interval_method": uncertainty.get("interval_method"),
        "interval_confidence": uncertainty.get("interval_confidence"),
        "interval_diagnostic_kind": "operational_interval_availability",
        "prediction_count_total": prediction_count_total,
        "interval_count_total": interval_count_total,
        "missing_interval_count": missing_interval_count,
        "interval_availability_rate": interval_availability_rate,
        "interval_complete": (
            prediction_count_total > 0 and missing_interval_count == 0
        ),
        "mean_interval_width": mean_interval_width,
        "max_interval_width": max_interval_width,
        "outcome_diagnostic_kind": "observed_rul_outcome_coverage",
        "outcome_count_total": outcome_count_total,
        "outcome_availability_rate": outcome_availability_rate,
        "mean_absolute_error": mean_absolute_error,
        "mean_signed_error": mean_signed_error,
        "interval_outcome_count_total": interval_outcome_count_total,
        "interval_covered_count_total": interval_covered_count_total,
        "outcome_interval_coverage_rate": outcome_interval_coverage_rate,
        "provenance_workflow": provenance_summary.get("workflow"),
    }


def _payload_for_evidence(
    evidence_by_type: dict[str, dict[str, Any]],
    evidence_type: str,
) -> dict[str, Any]:
    evidence = evidence_by_type.get(evidence_type)
    if not isinstance(evidence, dict):
        return {}
    payload = evidence.get("payload")
    return payload if isinstance(payload, dict) else {}


def _bool_dict(payload: Any) -> dict[str, bool]:
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, bool)}


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
