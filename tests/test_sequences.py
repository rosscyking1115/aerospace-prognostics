from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerospace_prognostics.sequences import make_last_sequence_windows, make_sequence_windows


def test_make_sequence_windows_rolls_within_each_unit() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2, 2],
            "time_in_cycles": [1, 2, 3, 1, 2, 3],
            "sensor_1": [10, 11, 12, 20, 21, 22],
            "rul_capped": [2, 1, 0, 2, 1, 0],
        }
    )

    dataset = make_sequence_windows(frame, window_size=2, feature_columns=["sensor_1"])

    assert dataset.windows.shape == (4, 2, 1)
    assert dataset.targets.tolist() == [1, 0, 1, 0]
    assert dataset.unit_numbers.tolist() == [1, 1, 2, 2]
    assert dataset.end_cycles.tolist() == [2, 3, 2, 3]
    assert dataset.feature_columns == ("sensor_1",)
    np.testing.assert_array_equal(dataset.windows[0, :, 0], np.array([10, 11], dtype=np.float32))
    np.testing.assert_array_equal(dataset.windows[2, :, 0], np.array([20, 21], dtype=np.float32))


def test_make_sequence_windows_honours_stride() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1],
            "time_in_cycles": [1, 2, 3, 4],
            "sensor_1": [10, 11, 12, 13],
            "rul_capped": [3, 2, 1, 0],
        }
    )

    dataset = make_sequence_windows(
        frame,
        window_size=2,
        stride=2,
        feature_columns=["sensor_1"],
    )

    assert dataset.windows.shape == (2, 2, 1)
    assert dataset.targets.tolist() == [2, 0]
    assert dataset.end_cycles.tolist() == [2, 4]


def test_make_last_sequence_windows_pads_short_units() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [1, 1, 2],
            "time_in_cycles": [1, 2, 1],
            "sensor_1": [10, 11, 20],
        }
    )

    dataset = make_last_sequence_windows(
        frame,
        window_size=3,
        feature_columns=["sensor_1"],
        pad_value=-1,
    )

    assert dataset.windows.shape == (2, 3, 1)
    assert dataset.targets is None
    np.testing.assert_array_equal(
        dataset.windows[0, :, 0],
        np.array([-1, 10, 11], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dataset.windows[1, :, 0],
        np.array([-1, -1, 20], dtype=np.float32),
    )


def test_sequence_window_validation() -> None:
    frame = pd.DataFrame({"unit_number": [1], "time_in_cycles": [1], "sensor_1": [10]})

    with pytest.raises(ValueError, match="window_size must be positive"):
        make_sequence_windows(frame, window_size=0, feature_columns=["sensor_1"])

    with pytest.raises(ValueError, match="stride must be positive"):
        make_sequence_windows(frame, window_size=1, stride=0, feature_columns=["sensor_1"])

    with pytest.raises(ValueError, match="missing columns"):
        make_sequence_windows(frame, window_size=1, feature_columns=["sensor_2"])


def test_make_sequence_windows_returns_empty_dataset_for_short_units() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [1],
            "time_in_cycles": [1],
            "sensor_1": [10],
            "rul_capped": [0],
        }
    )

    dataset = make_sequence_windows(frame, window_size=2, feature_columns=["sensor_1"])

    assert dataset.windows.shape == (0, 2, 1)
    assert dataset.targets is not None
    assert dataset.targets.shape == (0,)
