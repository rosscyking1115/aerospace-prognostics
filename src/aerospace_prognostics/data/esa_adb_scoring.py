"""ESA-ADB lightweight-subset baseline scoring and event-wise detection evidence.

This module is the next slice above the evaluator-contract helpers in
:mod:`aerospace_prognostics.data.esa_adb`. It codifies the official lightweight
channel subsets, provides a simple robust z-score baseline that emits the
official binary-detection contract, and scores event-wise detection quality.

Scope caveat (kept deliberately honest, per docs/phase3_esa_adb_intake.md):
this computes the *event-wise detection* top of the ESA-ADB metric hierarchy
(false-alarm-sensitive precision/recall/F0.5 over labelled events). It does not
yet reproduce the full official hierarchy (ADTQC detection timing, affiliation-
based proximity, subsystem-aware diagnosis). Results are therefore
protocol-shaped detection evidence, not a full ESA-ADB leaderboard claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

import pandas as pd
from telemeval.contract import build_metric_inputs
from telemeval.metrics.event_wise import score_event_wise

from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload

# Event-wise scoring is provided by the telemeval library (extracted from this
# project); the alias keeps this project's call sites and evidence tooling
# stable.
score_esa_adb_event_wise = score_event_wise

ESA_ADB_EVENT_WISE_EVIDENCE_SCHEMA = "aerospace-prognostics/esa-adb-event-wise-detection/v1"

# Official lightweight subsets from the ESA-ADB paper/README (see intake doc).
# Stored as channel numbers so we do not assume an on-disk channel naming scheme
# we have not yet confirmed against the real Mission archives.
ESA_ADB_LIGHTWEIGHT_CHANNELS: dict[str, tuple[int, ...]] = {
    "Mission1": tuple(range(41, 47)),  # channels 41-46
    "Mission2": tuple(range(18, 29)),  # channels 18-28
}

# Robust z-score constant: MAD * 1.4826 approximates the standard deviation for
# normally distributed data, matching the Track B SMAP/MSL robust baseline.
_MAD_TO_STD = 1.4826


def lightweight_channel_numbers(mission: str) -> tuple[int, ...]:
    """Return the official lightweight-subset channel numbers for a mission."""
    try:
        return ESA_ADB_LIGHTWEIGHT_CHANNELS[mission]
    except KeyError as exc:
        known = ", ".join(sorted(ESA_ADB_LIGHTWEIGHT_CHANNELS))
        raise ValueError(
            f"unknown ESA-ADB mission {mission!r}; known missions: {known}"
        ) from exc


def robust_zscore_detections(
    values_by_channel: Mapping[str, Sequence[float]],
    timestamps: Sequence[Any],
    *,
    threshold: float = 5.0,
) -> dict[str, pd.DataFrame]:
    """Emit official-contract binary detections from per-channel telemetry.

    A simple robust z-score baseline: for each channel, a point is flagged when
    its distance from the channel median, scaled by the MAD, exceeds
    ``threshold``. This produces the ``Timestamp, Score`` binary frames consumed
    by :func:`aerospace_prognostics.data.esa_adb.build_esa_adb_metric_inputs`.

    The algorithm is intentionally a baseline; "official-compatible" refers to
    the emitted output contract, not to an official ESA-ADB detector.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if not values_by_channel:
        raise ValueError("robust z-score scoring requires at least one channel")

    index = list(timestamps)
    n_rows = len(index)
    if n_rows == 0:
        raise ValueError("robust z-score scoring requires at least one timestamp")

    detections: dict[str, pd.DataFrame] = {}
    for channel, values in values_by_channel.items():
        series = [float(value) for value in values]
        if len(series) != n_rows:
            raise ValueError(
                f"channel {channel!r} has {len(series)} values but "
                f"{n_rows} timestamps were provided"
            )
        scores = _robust_zscore_flags(series, threshold)
        detections[str(channel)] = pd.DataFrame(
            {"Timestamp": index, "Score": scores}
        )
    return detections


def build_esa_adb_event_wise_evidence(
    scores: Mapping[str, Any],
    *,
    mission: str,
    target_channels: Sequence[str],
    lightweight: bool,
) -> dict[str, Any]:
    """Wrap event-wise detection scores in a scope-bounded evidence payload."""
    return {
        "schema_version": ESA_ADB_EVENT_WISE_EVIDENCE_SCHEMA,
        "mission": mission,
        "lightweight_subset": lightweight,
        "lightweight_channel_numbers": (
            list(lightweight_channel_numbers(mission))
            if mission in ESA_ADB_LIGHTWEIGHT_CHANNELS
            else None
        ),
        "target_channels": [str(channel) for channel in target_channels],
        "metric_scope": (
            "event-wise detection only (precision/recall/F-beta over labelled "
            "events); does not include ADTQC timing, affiliation-based "
            "proximity, or subsystem-aware diagnosis"
        ),
        "claim_status": (
            "protocol-shaped detection evidence, pending official-evaluator "
            "cross-check; not a full ESA-ADB leaderboard claim"
        ),
        "beta": scores["beta"],
        "excluded_categories": scores["excluded_categories"],
        "total_events": scores["total_events"],
        "detected_events": scores["detected_events"],
        "missed_events": scores["missed_events"],
        "predicted_alarms": scores["predicted_alarms"],
        "true_alarms": scores["true_alarms"],
        "false_alarms": scores["false_alarms"],
        "event_wise_precision": scores["event_wise_precision"],
        "event_wise_recall": scores["event_wise_recall"],
        "event_wise_fbeta": scores["event_wise_fbeta"],
        "per_event": [
            {
                "id": row["id"],
                "start_time": _isoformat(row["start_time"]),
                "end_time": _isoformat(row["end_time"]),
                "category": row["category"],
                "detected": row["detected"],
            }
            for row in scores["per_event"]
        ],
    }


def render_esa_adb_event_wise_markdown(evidence: Mapping[str, Any]) -> str:
    """Render event-wise detection evidence as human-readable Markdown."""
    lines = [
        f"# ESA-ADB Event-Wise Detection — {evidence['mission']}",
        "",
        f"> {evidence['claim_status']}",
        "",
        f"- Metric scope: {evidence['metric_scope']}.",
        f"- Lightweight subset: {evidence['lightweight_subset']}.",
        f"- Target channels: {len(evidence['target_channels'])}.",
        f"- F-beta weighting: beta = {evidence['beta']}.",
        "",
        "## Detection Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Labelled events | {evidence['total_events']} |",
        f"| Detected events | {evidence['detected_events']} |",
        f"| Missed events | {evidence['missed_events']} |",
        f"| Predicted alarms | {evidence['predicted_alarms']} |",
        f"| True alarms | {evidence['true_alarms']} |",
        f"| False alarms | {evidence['false_alarms']} |",
        f"| Event-wise precision | {evidence['event_wise_precision']:.6f} |",
        f"| Event-wise recall | {evidence['event_wise_recall']:.6f} |",
        f"| Event-wise F{evidence['beta']} | {evidence['event_wise_fbeta']:.6f} |",
        "",
        "## Per-Event Detection",
        "",
        "| Event ID | Category | Start | End | Detected |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in evidence["per_event"]:
        lines.append(
            f"| {row['id']} | {row['category']} | {row['start_time']} | "
            f"{row['end_time']} | {'yes' if row['detected'] else 'no'} |"
        )
    lines.append("")

    provenance = evidence.get("run_provenance")
    if provenance:
        lines.extend(
            [
                "## Run Provenance",
                "",
                f"- Data source: `{provenance['data_source']}`.",
                f"- Channels: {', '.join(provenance['channels'])}.",
                f"- Samples: {provenance['total_samples']} "
                f"(train {provenance['train_samples']}, test {provenance['test_samples']}).",
                f"- Test window: {provenance['test_window_start']} "
                f"to {provenance['test_window_end']}.",
                f"- Baseline: {provenance['baseline']} "
                f"at threshold {provenance['robust_threshold']}.",
                f"- Standardization fit: {provenance['standardization_fit']}.",
                f"- Resampling: {provenance['resampling']}.",
                "",
            ]
        )
    return "\n".join(lines)


def score_esa_adb_mission_from_predictions(
    labels: pd.DataFrame,
    predictions_by_channel: Mapping[str, pd.DataFrame],
    *,
    mission: str,
    lightweight: bool,
    beta: float = 0.5,
    exclude_categories: Sequence[str] = (),
) -> dict[str, Any]:
    """End-to-end event-wise evidence from labels and per-channel detections."""
    metric_inputs = build_metric_inputs(labels, predictions_by_channel)
    scores = score_event_wise(metric_inputs, beta=beta, exclude_categories=exclude_categories)
    return build_esa_adb_event_wise_evidence(
        scores,
        mission=mission,
        target_channels=metric_inputs["target_channels"],
        lightweight=lightweight,
    )


def write_esa_adb_event_wise_evidence(
    evidence: Mapping[str, Any],
    *,
    json_path: str | None = None,
    markdown_path: str | None = None,
) -> None:
    """Write event-wise detection evidence as JSON and/or Markdown artifacts."""
    if json_path is not None:
        write_json_payload(dict(evidence), json_path)
    if markdown_path is not None:
        output_path = prepare_output_path(markdown_path)
        output_path.write_text(
            render_esa_adb_event_wise_markdown(evidence) + "\n", encoding="utf-8"
        )


def _robust_zscore_flags(series: list[float], threshold: float) -> list[int]:
    center = median(series)
    deviations = [abs(value - center) for value in series]
    mad = median(deviations)
    scale = mad * _MAD_TO_STD
    if scale <= 0.0:
        # A degenerate (constant) channel yields no anomalies.
        return [0] * len(series)
    return [1 if deviation / scale > threshold else 0 for deviation in deviations]


def _isoformat(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.isoformat()
