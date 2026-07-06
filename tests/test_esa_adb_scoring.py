from __future__ import annotations

import json

import pandas as pd

from aerospace_prognostics.data.esa_adb import build_esa_adb_metric_inputs
from aerospace_prognostics.data.esa_adb_scoring import (
    ESA_ADB_EVENT_WISE_EVIDENCE_SCHEMA,
    ESA_ADB_LIGHTWEIGHT_CHANNELS,
    build_esa_adb_event_wise_evidence,
    lightweight_channel_numbers,
    render_esa_adb_event_wise_markdown,
    robust_zscore_detections,
    score_esa_adb_event_wise,
    score_esa_adb_mission_from_predictions,
    write_esa_adb_event_wise_evidence,
)


def _labels(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "ID",
            "Channel",
            "StartTime",
            "EndTime",
            "Category",
            "Dimensionality",
            "Locality",
            "Length",
        ],
    )


def test_lightweight_channel_numbers_match_official_subsets() -> None:
    assert ESA_ADB_LIGHTWEIGHT_CHANNELS["Mission1"] == tuple(range(41, 47))
    assert ESA_ADB_LIGHTWEIGHT_CHANNELS["Mission2"] == tuple(range(18, 29))
    assert lightweight_channel_numbers("Mission1") == (41, 42, 43, 44, 45, 46)


def test_lightweight_channel_numbers_rejects_unknown_mission() -> None:
    try:
        lightweight_channel_numbers("Mission9")
    except ValueError as exc:
        assert "unknown ESA-ADB mission" in str(exc)
    else:
        raise AssertionError("expected unknown mission to be rejected")


def test_robust_zscore_detections_flags_outliers_and_keeps_contract() -> None:
    timestamps = pd.to_datetime(
        [f"2024-01-01T00:0{minute}:00" for minute in range(7)]
    )
    # Non-degenerate baseline (MAD > 0) with a single clear spike at index 6.
    detections = robust_zscore_detections(
        {"channel_41": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 50.0]},
        timestamps,
        threshold=5.0,
    )

    frame = detections["channel_41"]
    assert frame.columns.tolist() == ["Timestamp", "Score"]
    assert frame["Score"].tolist() == [0, 0, 0, 0, 0, 0, 1]


def test_robust_zscore_detections_returns_zeros_for_constant_channel() -> None:
    timestamps = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:01:00"])
    detections = robust_zscore_detections(
        {"flat": [7.0, 7.0]}, timestamps, threshold=5.0
    )
    assert detections["flat"]["Score"].tolist() == [0, 0]


def test_robust_zscore_detections_rejects_length_mismatch() -> None:
    timestamps = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:01:00"])
    try:
        robust_zscore_detections({"c": [1.0]}, timestamps)
    except ValueError as exc:
        assert "timestamps were provided" in str(exc)
    else:
        raise AssertionError("expected length mismatch to be rejected")


def test_score_event_wise_detects_hit_and_penalises_false_alarm() -> None:
    timestamps = pd.to_datetime(
        [f"2024-01-01T00:0{minute}:00" for minute in range(6)]
    )
    # One anomaly event over minutes 1-2; predictions hit it (minute 1) and also
    # raise a spurious alarm at minute 4 (a false alarm).
    labels = _labels(
        [
            [
                "A-1",
                "ch1",
                pd.Timestamp("2024-01-01T00:01:00"),
                pd.Timestamp("2024-01-01T00:02:00"),
                "Anomaly",
                "Univariate",
                "Local",
                "Subsequence",
            ]
        ]
    )
    predictions = {
        "ch1": pd.DataFrame(
            {"Timestamp": timestamps, "Score": [0, 1, 0, 0, 1, 0]}
        )
    }
    metric_inputs = build_esa_adb_metric_inputs(labels, predictions)

    scores = score_esa_adb_event_wise(metric_inputs, beta=0.5)

    assert scores["total_events"] == 1
    assert scores["detected_events"] == 1
    assert scores["predicted_alarms"] == 2
    assert scores["true_alarms"] == 1
    assert scores["false_alarms"] == 1
    assert scores["event_wise_recall"] == 1.0
    assert scores["event_wise_precision"] == 0.5
    # F0.5 weights precision above recall: (1.25 * 0.5 * 1.0) / (0.25*0.5 + 1.0)
    assert round(scores["event_wise_fbeta"], 6) == round(0.625 / 1.125, 6)


def test_score_event_wise_reports_missed_event() -> None:
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"]
    )
    labels = _labels(
        [
            [
                "A-1",
                "ch1",
                pd.Timestamp("2024-01-01T00:02:00"),
                pd.Timestamp("2024-01-01T00:02:00"),
                "Anomaly",
                "Univariate",
                "Local",
                "Point",
            ]
        ]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 0, 0]})}
    metric_inputs = build_esa_adb_metric_inputs(labels, predictions)

    scores = score_esa_adb_event_wise(metric_inputs)

    assert scores["detected_events"] == 0
    assert scores["missed_events"] == 1
    assert scores["event_wise_recall"] == 0.0
    assert scores["event_wise_fbeta"] == 0.0


def test_score_event_wise_excludes_requested_categories() -> None:
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"]
    )
    labels = _labels(
        [
            [
                "A-1",
                "ch1",
                pd.Timestamp("2024-01-01T00:00:00"),
                pd.Timestamp("2024-01-01T00:00:00"),
                "Anomaly",
                "Univariate",
                "Local",
                "Point",
            ],
            [
                "G-1",
                "ch1",
                pd.Timestamp("2024-01-01T00:02:00"),
                pd.Timestamp("2024-01-01T00:02:00"),
                "Communication Gap",
                "Univariate",
                "Local",
                "Point",
            ],
        ]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [1, 0, 0]})}
    metric_inputs = build_esa_adb_metric_inputs(labels, predictions)

    scores = score_esa_adb_event_wise(
        metric_inputs, exclude_categories=("Communication Gap",)
    )

    assert scores["total_events"] == 1
    assert scores["excluded_categories"] == ["Communication Gap"]
    assert scores["detected_events"] == 1


def test_evidence_payload_carries_scope_caveats() -> None:
    scores = {
        "beta": 0.5,
        "excluded_categories": [],
        "total_events": 2,
        "detected_events": 1,
        "missed_events": 1,
        "predicted_alarms": 3,
        "true_alarms": 1,
        "false_alarms": 2,
        "event_wise_precision": 1 / 3,
        "event_wise_recall": 0.5,
        "event_wise_fbeta": 0.357143,
        "per_event": [
            {
                "id": "A-1",
                "start_time": pd.Timestamp("2024-01-01T00:01:00"),
                "end_time": pd.Timestamp("2024-01-01T00:02:00"),
                "category": "Anomaly",
                "detected": True,
            }
        ],
    }

    evidence = build_esa_adb_event_wise_evidence(
        scores, mission="Mission1", target_channels=["ch1"], lightweight=True
    )

    assert evidence["schema_version"] == ESA_ADB_EVENT_WISE_EVIDENCE_SCHEMA
    assert evidence["lightweight_channel_numbers"] == list(range(41, 47))
    assert "not a full ESA-ADB leaderboard claim" in evidence["claim_status"]
    assert "event-wise detection only" in evidence["metric_scope"]
    assert evidence["per_event"][0]["start_time"] == "2024-01-01T00:01:00"

    markdown = render_esa_adb_event_wise_markdown(evidence)
    assert "# ESA-ADB Event-Wise Detection — Mission1" in markdown
    assert "| Detected events | 1 |" in markdown


def test_score_mission_end_to_end_and_write_artifacts(tmp_path) -> None:
    timestamps = pd.to_datetime(
        [f"2024-01-01T00:0{minute}:00" for minute in range(5)]
    )
    labels = _labels(
        [
            [
                "A-1",
                "ch1",
                pd.Timestamp("2024-01-01T00:01:00"),
                pd.Timestamp("2024-01-01T00:01:00"),
                "Anomaly",
                "Univariate",
                "Local",
                "Point",
            ]
        ]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 1, 0, 0, 0]})}

    evidence = score_esa_adb_mission_from_predictions(
        labels, predictions, mission="Mission1", lightweight=True
    )
    assert evidence["detected_events"] == 1
    assert evidence["false_alarms"] == 0

    json_path = tmp_path / "out" / "evidence.json"
    markdown_path = tmp_path / "out" / "evidence.md"
    write_esa_adb_event_wise_evidence(
        evidence, json_path=str(json_path), markdown_path=str(markdown_path)
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["mission"] == "Mission1"
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# ESA-ADB Event-Wise Detection — Mission1"
    )
