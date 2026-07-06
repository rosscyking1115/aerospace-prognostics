"""ESA-ADB lightweight real-data event-wise detection run.

This is the slice that touches real ESA Anomaly Dataset telemetry rather than
fixtures. It runs a protocol-shaped lightweight path for a mission (Mission1
channels 41-46, Mission2 channels 18-28):

- load the lightweight target channels for the mission;
- split chronologically into a first-half train and second-half test;
- fit a robust z-score baseline on nominal (non-anomaly) training points only,
  so no thresholds or standardization are fit on test rows;
- score event-wise detection on the test window with
  :func:`aerospace_prognostics.data.esa_adb_scoring.score_esa_adb_event_wise`.

Honesty guardrails (see docs/phase3_esa_adb_intake.md):

- Only the event-wise detection top of the ESA-ADB metric hierarchy is scored.
- The official zero-order-hold resampling to the mission target frequency is
  *not* applied: the lightweight channels handled here happen to share a single
  native grid and are already aligned, so this baseline scores on the native
  grid. The loader rejects a mission whose lightweight channels do not share a
  grid, since that case genuinely requires resampling. This is recorded in the
  run provenance and means the result is not a full official reproduction.
- Every artifact is labelled protocol-shaped detection evidence, not an
  ESA-ADB leaderboard claim.

The heavy archive IO lives here and is exercised through the CLI, never through
the test suite. The pure split/mask/baseline/window helpers are fixture-tested.
"""

from __future__ import annotations

import pickle
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aerospace_prognostics.data.esa_adb import ESA_ADB_ANOMALY_TYPE_COLUMNS
from aerospace_prognostics.data.esa_adb_scoring import (
    build_esa_adb_event_wise_evidence,
    lightweight_channel_numbers,
    score_esa_adb_event_wise,
)

DEFAULT_ROBUST_THRESHOLD = 5.0
DEFAULT_EXCLUDE_CATEGORIES = ("Communication Gap",)
DEFAULT_VALIDATION_MONTHS = 3
DEFAULT_THRESHOLD_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0)
_MAD_TO_STD = 1.4826


def chronological_split(n_rows: int) -> int:
    """Return the first-half/second-half split index for ``n_rows`` samples."""

    if n_rows < 2:
        raise ValueError("chronological split requires at least two rows")
    return n_rows // 2


def label_interval_mask(
    grid: np.ndarray,
    starts: Sequence[Any],
    ends: Sequence[Any],
) -> np.ndarray:
    """Mark grid points that fall inside any ``[start, end]`` label interval.

    ``grid`` must be a sorted ``datetime64`` array. Membership is inclusive of
    both interval bounds, matching the event-detection semantics.
    """

    mask = np.zeros(len(grid), dtype=bool)
    start_values = pd.to_datetime(pd.Series(starts)).to_numpy("datetime64[ns]")
    end_values = pd.to_datetime(pd.Series(ends)).to_numpy("datetime64[ns]")
    for start, end in zip(start_values, end_values, strict=True):
        lo = int(np.searchsorted(grid, start, side="left"))
        hi = int(np.searchsorted(grid, end, side="right"))
        if hi > lo:
            mask[lo:hi] = True
    return mask


def robust_train_fit_zscores(
    values: np.ndarray,
    nominal_train_mask: np.ndarray,
) -> np.ndarray:
    """Continuous robust z-scores fit on nominal training points only.

    The median and MAD scale are estimated from ``values`` where
    ``nominal_train_mask`` is true (training rows that are not inside any
    labelled anomaly), then applied to every sample. A degenerate channel
    (zero MAD on the reference points) yields all-zero scores. Kept separate
    from thresholding so a threshold sweep can reuse one fit per channel.
    """

    if len(values) != len(nominal_train_mask):
        raise ValueError("values and nominal_train_mask must be the same length")

    values = np.asarray(values, dtype="float32")
    reference = values[nominal_train_mask].astype("float64")
    if reference.size == 0:
        raise ValueError("no nominal training points available to fit the baseline")

    center = float(np.median(reference))
    scale = float(np.median(np.abs(reference - center))) * _MAD_TO_STD
    if scale <= 0.0:
        return np.zeros(len(values), dtype="float32")

    # Compute in float32 and in place to keep peak memory near one channel's
    # array rather than a float64 copy of it.
    deviation = np.abs(values - center)
    deviation /= scale
    return deviation


def robust_train_fit_scores(
    values: np.ndarray,
    nominal_train_mask: np.ndarray,
    *,
    threshold: float = DEFAULT_ROBUST_THRESHOLD,
) -> np.ndarray:
    """Binary robust z-score detections at a fixed threshold."""

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    zscores = robust_train_fit_zscores(values, nominal_train_mask)
    return (zscores > threshold).astype("uint8")


def select_threshold_by_validation(
    val_zscores_by_channel: Mapping[str, np.ndarray],
    val_index: pd.DatetimeIndex,
    val_labels: pd.DataFrame,
    thresholds: Sequence[float],
    *,
    beta: float = 0.5,
    exclude_categories: Sequence[str] = DEFAULT_EXCLUDE_CATEGORIES,
) -> dict[str, Any]:
    """Pick the threshold with the best validation-window event-wise F-beta.

    Pure and fixture-testable: given per-channel validation z-scores and the
    validation-window labels, it sweeps ``thresholds`` and returns the winner.
    Ties break toward fewer false alarms, then toward the higher (more
    conservative) threshold.
    """

    if not val_zscores_by_channel:
        raise ValueError("threshold selection requires at least one channel")
    if len(thresholds) == 0:
        raise ValueError("threshold selection requires at least one candidate")

    val_length = len(val_index)
    per_threshold: list[dict[str, Any]] = []
    for threshold in thresholds:
        global_score = np.zeros(val_length, dtype="uint8")
        for zscores in val_zscores_by_channel.values():
            np.maximum(global_score, (zscores > threshold).astype("uint8"), out=global_score)
        metric_inputs = {
            "global_labels": val_labels.reset_index(drop=True),
            "global_predictions": pd.DataFrame(
                {"Timestamp": val_index, "Score": global_score}
            ),
        }
        scores = score_esa_adb_event_wise(
            metric_inputs, beta=beta, exclude_categories=exclude_categories
        )
        per_threshold.append(
            {
                "threshold": float(threshold),
                "event_wise_fbeta": scores["event_wise_fbeta"],
                "event_wise_precision": scores["event_wise_precision"],
                "event_wise_recall": scores["event_wise_recall"],
                "false_alarms": scores["false_alarms"],
            }
        )

    best = max(
        per_threshold,
        key=lambda row: (row["event_wise_fbeta"], -row["false_alarms"], row["threshold"]),
    )
    return {"best_threshold": best["threshold"], "per_threshold": per_threshold}


def filter_events_to_range(
    labels: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Keep label rows for events overlapping the half-open ``[start, end)``."""

    start_times = pd.to_datetime(labels["StartTime"])
    end_times = pd.to_datetime(labels["EndTime"])
    keep = (end_times >= pd.Timestamp(window_start)) & (
        start_times < pd.Timestamp(window_end)
    )
    return labels.loc[keep].copy()


def filter_events_to_window(
    labels: pd.DataFrame,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    """Keep only label rows for events that overlap the test window.

    ESA-ADB reports test-split performance, so events entirely inside the
    training half must not count against test recall.
    """

    end_times = pd.to_datetime(labels["EndTime"])
    return labels.loc[end_times >= pd.Timestamp(window_start)].copy()


def run_mission_lightweight(
    archive: str | Path,
    *,
    mission: str = "Mission1",
    threshold: float = DEFAULT_ROBUST_THRESHOLD,
    threshold_selection: str = "fixed",
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    validation_months: int = DEFAULT_VALIDATION_MONTHS,
    beta: float = 0.5,
    exclude_categories: Sequence[str] = DEFAULT_EXCLUDE_CATEGORIES,
) -> dict[str, Any]:
    """Run the lightweight event-wise detection path for a mission end to end.

    Channels are loaded and scored one at a time and reduced to a single global
    max-score detection series, so peak memory stays near one channel rather
    than the whole lightweight subset. The event-wise scorer needs only the
    global series; channel-aware metrics are out of scope here.

    ``threshold_selection`` is ``"fixed"`` (use ``threshold``) or ``"validation"``
    (sweep ``threshold_grid`` on the last ``validation_months`` of the training
    half and pick the best validation-window F-beta, never touching test rows).
    """

    if threshold_selection not in ("fixed", "validation"):
        raise ValueError("threshold_selection must be 'fixed' or 'validation'")

    channel_numbers = lightweight_channel_numbers(mission)
    channels = [f"channel_{number}" for number in channel_numbers]

    reader = _MissionArchiveReader(archive, mission)
    labels, anomaly_types = reader.load_labels()

    subset_labels = labels[labels["Channel"].isin(channels)].copy()
    subset_labels["StartTime"] = _to_naive_utc(subset_labels["StartTime"])
    subset_labels["EndTime"] = _to_naive_utc(subset_labels["EndTime"])

    def merge_types(frame: pd.DataFrame) -> pd.DataFrame:
        merged = frame.merge(
            anomaly_types[["ID", *ESA_ADB_ANOMALY_TYPE_COLUMNS]],
            on="ID",
            how="left",
            validate="many_to_one",
        )
        return merged.drop(columns=["Channel"]).reset_index(drop=True)

    chosen_threshold = float(threshold)
    selection_detail: dict[str, Any] | None = None

    if threshold_selection == "validation":
        selection = _select_validation_threshold(
            reader,
            channels,
            subset_labels,
            merge_types,
            thresholds=threshold_grid,
            validation_months=validation_months,
            beta=beta,
            exclude_categories=exclude_categories,
        )
        chosen_threshold = selection["best_threshold"]
        selection_detail = selection

    grid: np.ndarray | None = None
    split = 0
    global_test_score: np.ndarray | None = None
    for _channel, channel_grid, channel_split, zscores in _iter_channel_zscores(
        reader, channels, subset_labels
    ):
        if grid is None:
            grid = channel_grid
            split = channel_split
            global_test_score = np.zeros(len(grid) - split, dtype="uint8")
        binary_test = (zscores[split:] > chosen_threshold).astype("uint8")
        np.maximum(global_test_score, binary_test, out=global_test_score)
        del zscores, binary_test

    if grid is None or global_test_score is None:
        raise ValueError("no lightweight channels were loaded")

    test_start = pd.Timestamp(grid[split])
    test_index = pd.DatetimeIndex(grid[split:])

    window_labels = filter_events_to_window(subset_labels, test_start)
    metric_inputs = {
        "global_labels": merge_types(window_labels),
        "global_predictions": pd.DataFrame(
            {"Timestamp": test_index, "Score": global_test_score}
        ),
        "target_channels": list(channels),
    }
    scores = score_esa_adb_event_wise(
        metric_inputs, beta=beta, exclude_categories=exclude_categories
    )
    evidence = build_esa_adb_event_wise_evidence(
        scores,
        mission=mission,
        target_channels=metric_inputs["target_channels"],
        lightweight=True,
    )
    evidence["run_provenance"] = {
        "data_source": str(archive),
        "channels": channels,
        "total_samples": int(len(grid)),
        "train_samples": int(split),
        "test_samples": int(len(grid) - split),
        "test_window_start": test_start.isoformat(),
        "test_window_end": pd.Timestamp(grid[-1]).isoformat(),
        "baseline": "robust z-score (median/MAD)",
        "threshold_selection": threshold_selection,
        "robust_threshold": chosen_threshold,
        "standardization_fit": "nominal training points only (no test leakage)",
        "resampling": (
            "none applied; the lightweight channels for this mission share a "
            "single native grid and are aligned; official zero-order-hold "
            "resampling to the target frequency is not yet applied"
        ),
    }
    if selection_detail is not None:
        evidence["run_provenance"]["threshold_grid"] = list(threshold_grid)
        evidence["run_provenance"]["validation_months"] = validation_months
        evidence["threshold_selection"] = selection_detail
    return evidence


def _iter_channel_zscores(
    reader: _MissionArchiveReader,
    channels: Sequence[str],
    subset_labels: pd.DataFrame,
) -> Iterator[tuple[str, np.ndarray, int, np.ndarray]]:
    """Yield ``(channel, grid, split, zscores)`` one channel at a time.

    Establishes the shared grid on the first channel and rejects any later
    channel that does not share it (that case needs real resampling).
    """

    grid: np.ndarray | None = None
    split = 0
    is_train: np.ndarray | None = None
    for channel in channels:
        channel_grid, values = reader.load_channel(channel)
        if grid is None:
            grid = channel_grid
            split = chronological_split(len(grid))
            is_train = np.zeros(len(grid), dtype=bool)
            is_train[:split] = True
        elif not np.array_equal(grid, channel_grid):
            raise ValueError(
                f"channel {channel} does not share the lightweight grid; "
                "official resampling is required for non-aligned channels"
            )

        channel_labels = subset_labels[subset_labels["Channel"] == channel]
        anomaly_mask = label_interval_mask(
            grid, channel_labels["StartTime"], channel_labels["EndTime"]
        )
        nominal_train_mask = is_train & ~anomaly_mask
        zscores = robust_train_fit_zscores(values, nominal_train_mask)
        del values, anomaly_mask, nominal_train_mask
        yield channel, grid, split, zscores


def _select_validation_threshold(
    reader: _MissionArchiveReader,
    channels: Sequence[str],
    subset_labels: pd.DataFrame,
    merge_types: Any,
    *,
    thresholds: Sequence[float],
    validation_months: int,
    beta: float,
    exclude_categories: Sequence[str],
) -> dict[str, Any]:
    """Collect validation-window z-scores per channel and pick a threshold."""

    val_zscores_by_channel: dict[str, np.ndarray] = {}
    grid: np.ndarray | None = None
    split = 0
    val_lo = 0
    val_start: pd.Timestamp | None = None
    for channel, channel_grid, channel_split, zscores in _iter_channel_zscores(
        reader, channels, subset_labels
    ):
        if grid is None:
            grid = channel_grid
            split = channel_split
            test_start = pd.Timestamp(grid[split])
            val_start = test_start - pd.DateOffset(months=validation_months)
            val_lo = int(np.searchsorted(grid, np.datetime64(val_start), side="left"))
        val_zscores_by_channel[channel] = zscores[val_lo:split].astype("float32", copy=True)
        del zscores

    if grid is None or val_start is None:
        raise ValueError("no lightweight channels were loaded for threshold selection")

    val_index = pd.DatetimeIndex(grid[val_lo:split])
    if len(val_index) == 0:
        raise ValueError("validation window is empty; cannot select a threshold")

    test_start = pd.Timestamp(grid[split])
    val_labels = merge_types(filter_events_to_range(subset_labels, val_start, test_start))
    if val_labels.empty:
        raise ValueError("no validation-window events; cannot select a threshold")

    return select_threshold_by_validation(
        val_zscores_by_channel,
        val_index,
        val_labels,
        thresholds,
        beta=beta,
        exclude_categories=exclude_categories,
    )


def _to_naive_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True)
    return parsed.dt.tz_convert(None)


class _MissionArchiveReader:
    """Read ESA mission files from a zip archive or an extracted directory."""

    def __init__(self, archive: str | Path, mission: str) -> None:
        self._path = Path(archive)
        self._mission = mission
        self._zip: zipfile.ZipFile | None = None
        self._root = self._locate_root()

    def _locate_root(self) -> str:
        if self._path.is_dir():
            for candidate in self._path.rglob("labels.csv"):
                if self._mission in str(candidate):
                    return str(candidate.parent)
            raise ValueError(f"could not find {self._mission} labels.csv under {self._path}")

        if self._path.suffix.lower() != ".zip":
            raise ValueError(f"archive must be a .zip file or a directory: {self._path}")
        self._zip = zipfile.ZipFile(self._path)
        for name in self._zip.namelist():
            if name.endswith("labels.csv") and self._mission in name:
                return name[: -len("labels.csv")]
        raise ValueError(f"could not find {self._mission} labels.csv inside {self._path}")

    def _open(self, relative: str) -> Any:
        if self._zip is not None:
            return self._zip.open(self._root + relative)
        return open(Path(self._root) / relative, "rb")

    def load_labels(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with self._open("labels.csv") as handle:
            labels = pd.read_csv(handle)
        with self._open("anomaly_types.csv") as handle:
            anomaly_types = pd.read_csv(handle)
        return labels, anomaly_types

    def load_channel(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        """Stream one pickled channel frame into a grid and float32 value array.

        ``pickle.load`` reads incrementally from the archive stream, avoiding a
        large intermediate ``bytes`` buffer for the ~180 MB channel members.
        """

        with self._open(f"channels/{channel}/{channel}") as handle:
            frame = pickle.load(handle)  # noqa: S301 - trusted local dataset
        grid = frame.index.to_numpy("datetime64[ns]")
        values = frame[channel].to_numpy("float32")
        del frame
        return grid, values
