"""Metrics for binary telemetry anomaly detection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AnomalyDetectionMetrics:
    """Confusion-matrix metrics for point-wise binary anomaly labels."""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    miss_rate: float
    support: int
    predicted_positives: int
    point_adjusted: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def score_binary_anomalies(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    point_adjusted: bool = False,
) -> AnomalyDetectionMetrics:
    """Score binary anomaly predictions with optional segment-level point adjustment."""

    true = _as_binary_vector(y_true, name="y_true")
    pred = _as_binary_vector(y_pred, name="y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if point_adjusted:
        pred = point_adjusted_predictions(true, pred)

    true_positive = int(np.sum((true == 1) & (pred == 1)))
    false_positive = int(np.sum((true == 0) & (pred == 1)))
    false_negative = int(np.sum((true == 1) & (pred == 0)))
    true_negative = int(np.sum((true == 0) & (pred == 0)))

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return AnomalyDetectionMetrics(
        true_positives=true_positive,
        false_positives=false_positive,
        false_negatives=false_negative,
        true_negatives=true_negative,
        precision=precision,
        recall=recall,
        f1=_safe_divide(2 * precision * recall, precision + recall),
        false_alarm_rate=_safe_divide(false_positive, false_positive + true_negative),
        miss_rate=_safe_divide(false_negative, true_positive + false_negative),
        support=int(np.sum(true == 1)),
        predicted_positives=int(np.sum(pred == 1)),
        point_adjusted=point_adjusted,
    )


def point_adjusted_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Mark an entire true anomaly segment as detected if any point in it is detected."""

    true = _as_binary_vector(y_true, name="y_true")
    pred = _as_binary_vector(y_pred, name="y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")

    adjusted = pred.copy()
    segment_start: int | None = None
    for index, label in enumerate(np.append(true, 0)):
        if label == 1 and segment_start is None:
            segment_start = index
        elif label == 0 and segment_start is not None:
            segment_end = index
            if np.any(pred[segment_start:segment_end] == 1):
                adjusted[segment_start:segment_end] = 1
            segment_start = None
    return adjusted


def _as_binary_vector(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional binary vector")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one label")
    if not np.all(np.isin(array, [0, 1, False, True])):
        raise ValueError(f"{name} must contain only 0/1 labels")
    return array.astype(np.int8)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
