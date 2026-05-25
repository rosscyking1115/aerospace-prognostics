"""Sliding-window sequence generation for time-series models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aerospace_prognostics.features import cmapss_feature_columns

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class SequenceWindowDataset:
    """Windowed time-series data with unit/cycle metadata."""

    windows: np.ndarray
    targets: np.ndarray | None
    unit_numbers: np.ndarray
    end_cycles: np.ndarray
    feature_columns: tuple[str, ...]


def make_sequence_windows(
    frame: pd.DataFrame,
    *,
    window_size: int,
    stride: int = 1,
    feature_columns: list[str] | None = None,
    target_column: str | None = "rul_capped",
) -> SequenceWindowDataset:
    """Create rolling windows within each unit.

    Each output target, when requested, is the target value at the final row of
    the corresponding window.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    columns = feature_columns or cmapss_feature_columns()
    required = ["unit_number", "time_in_cycles", *columns]
    if target_column is not None:
        required.append(target_column)
    _validate_columns(frame, required)

    windows: list[np.ndarray] = []
    targets: list[float] = []
    unit_numbers: list[int] = []
    end_cycles: list[int] = []

    for unit_number, unit_frame in _iter_unit_frames(frame):
        features = unit_frame.loc[:, columns].to_numpy(dtype=np.float32)
        if len(features) < window_size:
            continue

        target_values = (
            unit_frame.loc[:, target_column].to_numpy(dtype=np.float32)
            if target_column is not None
            else None
        )
        cycles = unit_frame["time_in_cycles"].to_numpy()
        for start in range(0, len(features) - window_size + 1, stride):
            end = start + window_size
            windows.append(features[start:end])
            if target_values is not None:
                targets.append(float(target_values[end - 1]))
            unit_numbers.append(int(unit_number))
            end_cycles.append(int(cycles[end - 1]))

    return SequenceWindowDataset(
        windows=_stack_windows(windows, window_size=window_size, feature_count=len(columns)),
        targets=np.asarray(targets, dtype=np.float32) if target_column is not None else None,
        unit_numbers=np.asarray(unit_numbers, dtype=np.int64),
        end_cycles=np.asarray(end_cycles, dtype=np.int64),
        feature_columns=tuple(columns),
    )


def make_last_sequence_windows(
    frame: pd.DataFrame,
    *,
    window_size: int,
    feature_columns: list[str] | None = None,
    pad_value: float = 0.0,
) -> SequenceWindowDataset:
    """Create one final window per unit, left-padding short units when needed."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")

    columns = feature_columns or cmapss_feature_columns()
    required = ["unit_number", "time_in_cycles", *columns]
    _validate_columns(frame, required)

    windows: list[np.ndarray] = []
    unit_numbers: list[int] = []
    end_cycles: list[int] = []

    for unit_number, unit_frame in _iter_unit_frames(frame):
        features = unit_frame.loc[:, columns].to_numpy(dtype=np.float32)
        if len(features) >= window_size:
            window = features[-window_size:]
        else:
            pad_rows = window_size - len(features)
            padding = np.full((pad_rows, len(columns)), pad_value, dtype=np.float32)
            window = np.vstack([padding, features])
        windows.append(window)
        unit_numbers.append(int(unit_number))
        end_cycles.append(int(unit_frame["time_in_cycles"].iloc[-1]))

    return SequenceWindowDataset(
        windows=_stack_windows(windows, window_size=window_size, feature_count=len(columns)),
        targets=None,
        unit_numbers=np.asarray(unit_numbers, dtype=np.int64),
        end_cycles=np.asarray(end_cycles, dtype=np.int64),
        feature_columns=tuple(columns),
    )


def _iter_unit_frames(frame: pd.DataFrame):
    sorted_frame = frame.sort_values(["unit_number", "time_in_cycles"])
    yield from sorted_frame.groupby("unit_number", sort=True)


def _validate_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")


def _stack_windows(
    windows: list[np.ndarray],
    *,
    window_size: int,
    feature_count: int,
) -> np.ndarray:
    if not windows:
        return np.empty((0, window_size, feature_count), dtype=np.float32)
    return np.stack(windows).astype(np.float32, copy=False)
