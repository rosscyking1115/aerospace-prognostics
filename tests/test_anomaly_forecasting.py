from __future__ import annotations

import numpy as np
import pytest
import torch

from aerospace_prognostics.anomaly.forecasting import (
    TelemetryLstmForecaster,
    run_lstm_forecast_anomaly_baseline,
)


def test_lstm_forecaster_returns_next_step_shape() -> None:
    model = TelemetryLstmForecaster(feature_count=2, hidden_size=4)

    predictions = model(torch.zeros((2, 3, 2), dtype=torch.float32))

    assert tuple(predictions.shape) == (2, 2)


def test_run_lstm_forecast_anomaly_baseline_tracks_history_and_scores() -> None:
    train = np.array(
        [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]
    )
    test = np.array(
        [[0.0, 0.0], [0.1, 0.1], [4.0, -4.0], [4.5, -4.5], [0.2, 0.2], [0.3, 0.3]]
    )
    labels = np.array([0, 0, 1, 1, 0, 0])

    result = run_lstm_forecast_anomaly_baseline(
        train,
        test,
        labels,
        feature_names=("feature_0", "feature_1"),
        window_size=2,
        hidden_size=4,
        epochs=2,
        batch_size=2,
        random_state=7,
    )

    assert result.model_name == "lstm_forecast_robust_threshold"
    assert len(result.history) == 2
    assert len(result.scores) == len(test)
    assert result.scores[0] == 0.0
    assert result.scores[1] == 0.0
    assert result.metrics.support == 2
    assert result.model_config["threshold"] > 0


def test_run_lstm_forecast_anomaly_baseline_rejects_short_series() -> None:
    values = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="longer than window_size"):
        run_lstm_forecast_anomaly_baseline(
            values,
            values,
            np.array([0, 1]),
            feature_names=("feature_0",),
            window_size=2,
        )
