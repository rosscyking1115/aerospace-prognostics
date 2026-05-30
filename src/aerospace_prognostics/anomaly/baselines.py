"""Simple anomaly-detection baselines for multivariate telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from aerospace_prognostics.anomaly.metrics import (
    AnomalyDetectionMetrics,
    score_binary_anomalies,
)


@dataclass(frozen=True)
class RobustZScoreModel:
    """Median/MAD model used to score robust per-channel deviations."""

    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]
    threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RobustZScoreBaselineResult:
    """Outputs from a robust z-score anomaly baseline run."""

    model: RobustZScoreModel
    metrics: AnomalyDetectionMetrics
    point_adjusted_metrics: AnomalyDetectionMetrics
    scores: tuple[float, ...]
    predictions: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model.to_dict(),
            "metrics": self.metrics.to_dict(),
            "point_adjusted_metrics": self.point_adjusted_metrics.to_dict(),
            "scores": list(self.scores),
            "predictions": list(self.predictions),
        }


def fit_robust_zscore_model(
    train_values: Sequence[Sequence[float]] | np.ndarray,
    *,
    feature_names: Sequence[str],
    threshold: float = 3.5,
) -> RobustZScoreModel:
    """Fit per-feature robust center and scale statistics on nominal training telemetry."""

    values = _as_2d_float_array(train_values, name="train_values")
    feature_tuple = tuple(feature_names)
    if len(feature_tuple) != values.shape[1]:
        raise ValueError("feature_names length must match train_values column count")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    medians = np.median(values, axis=0)
    mad = np.median(np.abs(values - medians), axis=0)
    robust_scales = 1.4826 * mad
    fallback_scales = np.std(values, axis=0, ddof=0)
    scales = np.where(
        robust_scales > 1e-12,
        robust_scales,
        np.where(fallback_scales > 1e-12, fallback_scales, 1.0),
    )
    return RobustZScoreModel(
        feature_names=feature_tuple,
        medians=tuple(float(value) for value in medians),
        scales=tuple(float(value) for value in scales),
        threshold=float(threshold),
    )


def robust_zscore_scores(
    model: RobustZScoreModel,
    values: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Score each telemetry row by the largest absolute robust z-score across features."""

    array = _as_2d_float_array(values, name="values")
    medians = np.asarray(model.medians, dtype=np.float64)
    scales = np.asarray(model.scales, dtype=np.float64)
    if array.shape[1] != len(model.feature_names):
        raise ValueError("values column count must match fitted model feature count")
    return np.max(np.abs((array - medians) / scales), axis=1)


def predict_robust_zscore_anomalies(
    model: RobustZScoreModel,
    values: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return anomaly scores and binary predictions for telemetry rows."""

    scores = robust_zscore_scores(model, values)
    return scores, (scores > model.threshold).astype(np.int8)


def run_robust_zscore_baseline(
    train_values: Sequence[Sequence[float]] | np.ndarray,
    test_values: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    feature_names: Sequence[str],
    threshold: float = 3.5,
) -> RobustZScoreBaselineResult:
    """Fit a robust z-score model on train telemetry and score labelled test telemetry."""

    model = fit_robust_zscore_model(
        train_values,
        feature_names=feature_names,
        threshold=threshold,
    )
    scores, predictions = predict_robust_zscore_anomalies(model, test_values)
    return RobustZScoreBaselineResult(
        model=model,
        metrics=score_binary_anomalies(labels, predictions),
        point_adjusted_metrics=score_binary_anomalies(labels, predictions, point_adjusted=True),
        scores=tuple(float(value) for value in scores),
        predictions=tuple(int(value) for value in predictions),
    )


def _as_2d_float_array(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional numeric array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array
