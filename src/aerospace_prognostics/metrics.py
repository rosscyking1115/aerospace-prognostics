"""Evaluation metrics and RUL target helpers for prognostics benchmarks."""

from __future__ import annotations

from collections.abc import Iterable
from math import exp, isfinite, sqrt


def _as_float_list(values: Iterable[float], *, name: str) -> list[float]:
    converted = [float(value) for value in values]
    if not converted:
        raise ValueError(f"{name} must contain at least one value")
    if not all(isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _validated_pairs(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> tuple[list[float], list[float]]:
    true_values = _as_float_list(y_true, name="y_true")
    pred_values = _as_float_list(y_pred, name="y_pred")
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")
    return true_values, pred_values


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Return root mean squared error."""
    true_values, pred_values = _validated_pairs(y_true, y_pred)
    squared_error = sum(
        (predicted - actual) ** 2
        for actual, predicted in zip(true_values, pred_values, strict=True)
    )
    return sqrt(squared_error / len(true_values))


def nasa_rul_score(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Return the NASA C-MAPSS asymmetric RUL score.

    Positive error means the model overestimated RUL, which is late and therefore
    penalised more heavily than an early warning.
    """
    true_values, pred_values = _validated_pairs(y_true, y_pred)
    score = 0.0
    for actual, predicted in zip(true_values, pred_values, strict=True):
        error = predicted - actual
        if error < 0:
            score += exp(-error / 13.0) - 1.0
        else:
            score += exp(error / 10.0) - 1.0
    return score


def piecewise_rul(rul: Iterable[float], *, cap: float = 125.0) -> list[float]:
    """Apply the standard piecewise-linear early-life RUL cap used for C-MAPSS."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    values = _as_float_list(rul, name="rul")
    return [min(value, cap) for value in values]
