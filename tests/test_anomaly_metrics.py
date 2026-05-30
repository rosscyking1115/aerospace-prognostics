from __future__ import annotations

import pytest

from aerospace_prognostics.anomaly.metrics import (
    point_adjusted_predictions,
    score_binary_anomalies,
)


def test_score_binary_anomalies_reports_confusion_counts() -> None:
    metrics = score_binary_anomalies(
        y_true=[0, 1, 1, 0, 0, 1],
        y_pred=[0, 1, 0, 1, 0, 1],
    )

    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 2
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.false_alarm_rate == pytest.approx(1 / 3)
    assert metrics.miss_rate == pytest.approx(1 / 3)
    assert metrics.support == 3
    assert metrics.predicted_positives == 3
    assert metrics.point_adjusted is False


def test_point_adjusted_predictions_marks_detected_segments() -> None:
    adjusted = point_adjusted_predictions(
        y_true=[0, 1, 1, 1, 0, 1, 1, 0],
        y_pred=[0, 0, 1, 0, 0, 0, 0, 1],
    )

    assert adjusted.tolist() == [0, 1, 1, 1, 0, 0, 0, 1]


def test_point_adjusted_metrics_can_raise_segment_recall() -> None:
    raw = score_binary_anomalies(
        y_true=[0, 1, 1, 1, 0],
        y_pred=[0, 0, 1, 0, 0],
    )
    adjusted = score_binary_anomalies(
        y_true=[0, 1, 1, 1, 0],
        y_pred=[0, 0, 1, 0, 0],
        point_adjusted=True,
    )

    assert raw.recall == pytest.approx(1 / 3)
    assert adjusted.recall == 1.0
    assert adjusted.f1 > raw.f1


def test_score_binary_anomalies_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="only 0/1 labels"):
        score_binary_anomalies(y_true=[0, 2], y_pred=[0, 1])


def test_score_binary_anomalies_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        score_binary_anomalies(y_true=[0, 1], y_pred=[0])

