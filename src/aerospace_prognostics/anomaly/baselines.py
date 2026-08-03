"""Simple anomaly-detection baselines for multivariate telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aerospace_prognostics.anomaly.metrics import (
    AnomalyDetectionMetrics,
    score_binary_anomalies,
)

CLASSICAL_ANOMALY_BASELINE_METHODS = (
    "robust_zscore",
    "pca_reconstruction",
    "isolation_forest",
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


@dataclass(frozen=True)
class ClassicalAnomalyBaselineResult:
    """Common result container for comparable classical anomaly baselines."""

    model_name: str
    model_config: dict[str, object]
    metrics: AnomalyDetectionMetrics
    point_adjusted_metrics: AnomalyDetectionMetrics
    scores: tuple[float, ...]
    predictions: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_config": self.model_config,
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


def run_classical_anomaly_baselines(
    train_values: Sequence[Sequence[float]] | np.ndarray,
    test_values: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    feature_names: Sequence[str],
    methods: Sequence[str] = CLASSICAL_ANOMALY_BASELINE_METHODS,
    robust_threshold: float = 3.5,
    pca_components: int | None = None,
    pca_threshold_quantile: float = 0.99,
    isolation_contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[ClassicalAnomalyBaselineResult, ...]:
    """Run a compact set of classical telemetry anomaly baselines."""
    train_array = _as_2d_float_array(train_values, name="train_values")
    test_array = _as_2d_float_array(test_values, name="test_values")
    feature_tuple = tuple(feature_names)
    if train_array.shape[1] != test_array.shape[1]:
        raise ValueError("train_values and test_values must have the same column count")
    if len(feature_tuple) != train_array.shape[1]:
        raise ValueError("feature_names length must match telemetry column count")
    unknown_methods = sorted(set(methods) - set(CLASSICAL_ANOMALY_BASELINE_METHODS))
    if unknown_methods:
        raise ValueError(f"unknown anomaly baseline methods: {', '.join(unknown_methods)}")

    results: list[ClassicalAnomalyBaselineResult] = []
    for method in methods:
        if method == "robust_zscore":
            robust = run_robust_zscore_baseline(
                train_array,
                test_array,
                labels,
                feature_names=feature_tuple,
                threshold=robust_threshold,
            )
            results.append(
                ClassicalAnomalyBaselineResult(
                    model_name=method,
                    model_config=robust.model.to_dict(),
                    metrics=robust.metrics,
                    point_adjusted_metrics=robust.point_adjusted_metrics,
                    scores=robust.scores,
                    predictions=robust.predictions,
                )
            )
        elif method == "pca_reconstruction":
            results.append(
                _run_pca_reconstruction_baseline(
                    train_array,
                    test_array,
                    labels,
                    feature_names=feature_tuple,
                    n_components=pca_components,
                    threshold_quantile=pca_threshold_quantile,
                    random_state=random_state,
                )
            )
        elif method == "isolation_forest":
            results.append(
                _run_isolation_forest_baseline(
                    train_array,
                    test_array,
                    labels,
                    feature_names=feature_tuple,
                    contamination=isolation_contamination,
                    random_state=random_state,
                )
            )
    return tuple(results)


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


def _run_pca_reconstruction_baseline(
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    feature_names: Sequence[str],
    n_components: int | None,
    threshold_quantile: float,
    random_state: int,
) -> ClassicalAnomalyBaselineResult:
    if not 0 < threshold_quantile < 1:
        raise ValueError("pca_threshold_quantile must be between 0 and 1")
    component_count = n_components
    if component_count is None:
        component_count = min(train_values.shape[1], max(1, train_values.shape[1] - 1))
    if not 1 <= component_count <= train_values.shape[1]:
        raise ValueError("pca_components must be between 1 and the feature count")

    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=component_count, random_state=random_state),
    )
    pipeline.fit(train_values)
    train_errors = _pca_reconstruction_errors(pipeline, train_values)
    test_errors = _pca_reconstruction_errors(pipeline, test_values)
    threshold = float(np.quantile(train_errors, threshold_quantile))
    predictions = (test_errors > threshold).astype(np.int8)
    metrics = score_binary_anomalies(labels, predictions)
    pca = pipeline.named_steps["pca"]
    return ClassicalAnomalyBaselineResult(
        model_name="pca_reconstruction",
        model_config={
            "feature_names": tuple(feature_names),
            "n_components": int(component_count),
            "threshold": threshold,
            "threshold_quantile": float(threshold_quantile),
            "explained_variance_ratio": tuple(
                float(value) for value in pca.explained_variance_ratio_
            ),
        },
        metrics=metrics,
        point_adjusted_metrics=score_binary_anomalies(labels, predictions, point_adjusted=True),
        scores=tuple(float(value) for value in test_errors),
        predictions=tuple(int(value) for value in predictions),
    )


def _run_isolation_forest_baseline(
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    feature_names: Sequence[str],
    contamination: float,
    random_state: int,
) -> ClassicalAnomalyBaselineResult:
    if not 0 < contamination < 0.5:
        raise ValueError("isolation_contamination must be between 0 and 0.5")

    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(train_values)
    scores = -model.score_samples(test_values)
    predictions = (model.predict(test_values) == -1).astype(np.int8)
    metrics = score_binary_anomalies(labels, predictions)
    return ClassicalAnomalyBaselineResult(
        model_name="isolation_forest",
        model_config={
            "feature_names": tuple(feature_names),
            "contamination": float(contamination),
            "random_state": int(random_state),
        },
        metrics=metrics,
        point_adjusted_metrics=score_binary_anomalies(labels, predictions, point_adjusted=True),
        scores=tuple(float(value) for value in scores),
        predictions=tuple(int(value) for value in predictions),
    )


def _pca_reconstruction_errors(pipeline: object, values: np.ndarray) -> np.ndarray:
    scaled = pipeline.named_steps["standardscaler"].transform(values)
    transformed = pipeline.named_steps["pca"].transform(scaled)
    reconstructed = pipeline.named_steps["pca"].inverse_transform(transformed)
    errors = np.mean((scaled - reconstructed) ** 2, axis=1)
    return np.asarray(errors, dtype=np.float64)
