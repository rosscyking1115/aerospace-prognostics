from __future__ import annotations

import numpy as np
import pandas as pd

from aerospace_prognostics.data.esa_adb_mission1 import (
    chronological_split,
    filter_events_to_window,
    label_interval_mask,
    robust_train_fit_scores,
)


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
