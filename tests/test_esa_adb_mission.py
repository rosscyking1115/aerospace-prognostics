from __future__ import annotations

import numpy as np
import pandas as pd

from aerospace_prognostics.data.esa_adb_mission import (
    chronological_split,
    filter_events_to_range,
    filter_events_to_window,
    label_interval_mask,
    robust_train_fit_scores,
    robust_train_fit_zscores,
    select_threshold_by_validation,
)
from aerospace_prognostics.data.esa_adb_scoring import lightweight_channel_numbers


def test_lightweight_channel_numbers_cover_both_missions() -> None:
    assert lightweight_channel_numbers("Mission1") == tuple(range(41, 47))
    assert lightweight_channel_numbers("Mission2") == tuple(range(18, 29))


def test_chronological_split_returns_first_half_index() -> None:
    assert chronological_split(10) == 5
    assert chronological_split(11) == 5


def test_chronological_split_rejects_tiny_series() -> None:
    try:
        chronological_split(1)
    except ValueError as exc:
        assert "at least two rows" in str(exc)
    else:
        raise AssertionError("expected tiny series to be rejected")


def test_label_interval_mask_marks_inclusive_bounds() -> None:
    grid = np.array(
        [f"2024-01-01T00:0{minute}:00" for minute in range(6)],
        dtype="datetime64[ns]",
    )
    mask = label_interval_mask(
        grid,
        ["2024-01-01T00:01:00"],
        ["2024-01-01T00:03:00"],
    )
    assert mask.tolist() == [False, True, True, True, False, False]


def test_label_interval_mask_handles_no_intervals() -> None:
    grid = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[ns]"
    )
    mask = label_interval_mask(grid, [], [])
    assert mask.tolist() == [False, False]


def test_robust_train_fit_scores_fits_on_nominal_train_only() -> None:
    # A steady baseline with one late test-side spike. The training-side spike
    # is masked out of the fit so it does not inflate the robust scale.
    values = np.array([1.0, 2.0, 1.0, 2.0, 99.0, 1.0, 2.0, 50.0])
    is_train = np.array([True, True, True, True, True, False, False, False])
    anomaly = np.array([False, False, False, False, True, False, False, False])
    nominal_train = is_train & ~anomaly

    scores = robust_train_fit_scores(values, nominal_train, threshold=5.0)

    assert scores.dtype == np.dtype("uint8")
    assert scores[7] == 1  # test-side spike detected
    assert scores[4] == 1  # masked training spike still scored when applied
    assert scores[:4].tolist() == [0, 0, 0, 0]


def test_robust_train_fit_scores_zero_scale_yields_no_detections() -> None:
    values = np.array([5.0, 5.0, 5.0, 42.0])
    nominal_train = np.array([True, True, True, False])

    scores = robust_train_fit_scores(values, nominal_train, threshold=5.0)

    assert scores.tolist() == [0, 0, 0, 0]


def test_robust_train_fit_scores_requires_nominal_points() -> None:
    values = np.array([1.0, 2.0])
    try:
        robust_train_fit_scores(values, np.array([False, False]))
    except ValueError as exc:
        assert "no nominal training points" in str(exc)
    else:
        raise AssertionError("expected empty reference to be rejected")


def test_filter_events_to_window_drops_pure_training_events() -> None:
    labels = pd.DataFrame(
        {
            "ID": ["train-only", "spans-boundary", "test-only"],
            "Channel": ["channel_41", "channel_41", "channel_41"],
            "StartTime": pd.to_datetime(
                ["2024-01-01T00:00:00", "2024-01-01T00:04:00", "2024-01-01T00:08:00"]
            ),
            "EndTime": pd.to_datetime(
                ["2024-01-01T00:02:00", "2024-01-01T00:06:00", "2024-01-01T00:09:00"]
            ),
        }
    )
    kept = filter_events_to_window(labels, pd.Timestamp("2024-01-01T00:05:00"))
    assert kept["ID"].tolist() == ["spans-boundary", "test-only"]


def test_filter_events_to_range_keeps_only_window_overlap() -> None:
    labels = pd.DataFrame(
        {
            "ID": ["before", "inside", "after"],
            "StartTime": pd.to_datetime(
                ["2024-01-01T00:00:00", "2024-01-01T00:05:00", "2024-01-01T00:20:00"]
            ),
            "EndTime": pd.to_datetime(
                ["2024-01-01T00:01:00", "2024-01-01T00:06:00", "2024-01-01T00:21:00"]
            ),
        }
    )
    kept = filter_events_to_range(
        labels, pd.Timestamp("2024-01-01T00:03:00"), pd.Timestamp("2024-01-01T00:10:00")
    )
    assert kept["ID"].tolist() == ["inside"]


def test_robust_train_fit_zscores_and_scores_agree_at_threshold() -> None:
    values = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 50.0])
    nominal_train = np.array([True, True, True, True, True, True, False])

    zscores = robust_train_fit_zscores(values, nominal_train)
    assert zscores.dtype == np.dtype("float32")
    assert zscores[6] > 5.0  # spike stands out
    assert float(zscores[0]) < 5.0

    scores = robust_train_fit_scores(values, nominal_train, threshold=5.0)
    assert scores.tolist() == (zscores > 5.0).astype("uint8").tolist()


def _val_labels(ids_starts_ends: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [row[0] for row in ids_starts_ends],
            "StartTime": pd.to_datetime([row[1] for row in ids_starts_ends]),
            "EndTime": pd.to_datetime([row[2] for row in ids_starts_ends]),
            "Category": ["Anomaly"] * len(ids_starts_ends),
            "Dimensionality": ["Univariate"] * len(ids_starts_ends),
            "Locality": ["Local"] * len(ids_starts_ends),
            "Length": ["Point"] * len(ids_starts_ends),
        }
    )


def test_select_threshold_by_validation_prefers_fewer_false_alarms() -> None:
    val_index = pd.DatetimeIndex(
        [f"2024-01-01T00:0{minute}:00" for minute in range(6)]
    )
    # One event at minute 2. A low threshold fires three separated runs (two of
    # them false alarms); a high threshold fires only the true spike at minute 2.
    zscores = {"ch1": np.array([6.0, 0.0, 20.0, 0.0, 6.0, 0.0], dtype="float32")}
    val_labels = _val_labels([("e1", "2024-01-01T00:02:00", "2024-01-01T00:02:00")])

    result = select_threshold_by_validation(
        zscores, val_index, val_labels, thresholds=(5.0, 10.0), beta=0.5, min_events=1
    )

    assert result["best_threshold"] == 10.0
    low = next(r for r in result["per_threshold"] if r["threshold"] == 5.0)
    high = next(r for r in result["per_threshold"] if r["threshold"] == 10.0)
    assert low["false_alarms"] > high["false_alarms"]
    assert high["false_alarms"] == 0


def test_select_threshold_falls_back_when_validation_is_sparse() -> None:
    val_index = pd.DatetimeIndex(
        [f"2024-01-01T00:0{minute}:00" for minute in range(6)]
    )
    zscores = {"ch1": np.array([6.0, 0.0, 20.0, 0.0, 6.0, 0.0], dtype="float32")}
    val_labels = _val_labels([("e1", "2024-01-01T00:02:00", "2024-01-01T00:02:00")])

    # Only one validation event but min_events=10 -> fall back to the fixed default.
    result = select_threshold_by_validation(
        zscores,
        val_index,
        val_labels,
        thresholds=(5.0, 10.0),
        beta=0.5,
        min_events=10,
        fallback_threshold=5.0,
    )

    assert result["best_threshold"] == 5.0
    assert result["validation_events"] == 1
    assert "fallback" in result["selection_reason"]


def test_select_threshold_prefers_conservative_within_tolerance() -> None:
    # Three events all detected at every threshold with no false alarms, so all
    # thresholds tie on F0.5; the conservative rule takes the highest threshold.
    val_index = pd.DatetimeIndex(
        [f"2024-01-01T00:0{minute}:00" for minute in range(3)]
    )
    zscores = {"ch1": np.array([30.0, 30.0, 30.0], dtype="float32")}
    val_labels = _val_labels(
        [
            ("e1", "2024-01-01T00:00:00", "2024-01-01T00:00:00"),
            ("e2", "2024-01-01T00:01:00", "2024-01-01T00:01:00"),
            ("e3", "2024-01-01T00:02:00", "2024-01-01T00:02:00"),
        ]
    )

    result = select_threshold_by_validation(
        zscores, val_index, val_labels, thresholds=(5.0, 10.0, 20.0), beta=0.5, min_events=1
    )

    assert result["best_threshold"] == 20.0
