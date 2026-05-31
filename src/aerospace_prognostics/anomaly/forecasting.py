"""Forecasting-based anomaly baselines for telemetry sequences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aerospace_prognostics.anomaly.metrics import (
    AnomalyDetectionMetrics,
    score_binary_anomalies,
)


@dataclass(frozen=True)
class LstmForecastTrainingEpoch:
    """Training loss recorded for one LSTM forecast epoch."""

    epoch: int
    train_loss: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LstmForecastAnomalyResult:
    """Outputs from an LSTM next-step forecasting anomaly baseline."""

    model_name: str
    model_config: dict[str, object]
    metrics: AnomalyDetectionMetrics
    point_adjusted_metrics: AnomalyDetectionMetrics
    history: tuple[LstmForecastTrainingEpoch, ...]
    scores: tuple[float, ...]
    predictions: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_config": self.model_config,
            "metrics": self.metrics.to_dict(),
            "point_adjusted_metrics": self.point_adjusted_metrics.to_dict(),
            "history": [epoch.to_dict() for epoch in self.history],
            "scores": list(self.scores),
            "predictions": list(self.predictions),
        }


class TelemetryLstmForecaster(nn.Module):
    """Small LSTM model for one-step telemetry forecasting."""

    def __init__(
        self,
        *,
        feature_count: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            input_size=feature_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = nn.Linear(hidden_size, feature_count)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.recurrent(windows)
        return self.head(encoded[:, -1, :])


def run_lstm_forecast_anomaly_baseline(
    train_values: Sequence[Sequence[float]] | np.ndarray,
    test_values: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    feature_names: Sequence[str],
    window_size: int = 30,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    threshold_sigma: float = 3.0,
    random_state: int = 42,
    device: str = "cpu",
) -> LstmForecastAnomalyResult:
    """Train an LSTM forecaster and flag large one-step prediction errors."""

    train_array = _as_2d_float_array(train_values, name="train_values")
    test_array = _as_2d_float_array(test_values, name="test_values")
    label_array = np.asarray(labels, dtype=np.int8)
    feature_tuple = tuple(feature_names)
    _validate_lstm_forecast_inputs(
        train_array,
        test_array,
        label_array,
        feature_tuple,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        threshold_sigma=threshold_sigma,
    )

    torch.manual_seed(random_state)
    generator = torch.Generator().manual_seed(random_state)
    torch_device = torch.device(device)
    train_scaled, test_scaled, means, scales = _standardize_train_test(train_array, test_array)
    train_windows, train_targets = _forecast_windows(train_scaled, window_size=window_size)

    model = TelemetryLstmForecaster(
        feature_count=train_array.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_windows, dtype=torch.float32),
            torch.as_tensor(train_targets, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    history: list[LstmForecastTrainingEpoch] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_windows, batch_targets in loader:
            batch_windows = batch_windows.to(torch_device)
            batch_targets = batch_targets.to(torch_device)
            optimizer.zero_grad()
            predictions = model(batch_windows)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_windows)
            total_rows += len(batch_windows)
        history.append(
            LstmForecastTrainingEpoch(
                epoch=epoch,
                train_loss=total_loss / max(total_rows, 1),
            )
        )

    train_errors = _forecast_errors(model, train_scaled, window_size, torch_device, batch_size)
    test_errors = _forecast_errors(model, test_scaled, window_size, torch_device, batch_size)
    threshold = _robust_error_threshold(train_errors, sigma=threshold_sigma)
    scores = np.zeros(len(test_array), dtype=np.float64)
    scores[window_size:] = test_errors
    predictions = (scores > threshold).astype(np.int8)

    config = {
        "feature_names": feature_tuple,
        "window_size": int(window_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "threshold_sigma": float(threshold_sigma),
        "threshold": float(threshold),
        "random_state": int(random_state),
        "device": device,
        "train_means": tuple(float(value) for value in means),
        "train_scales": tuple(float(value) for value in scales),
    }
    return LstmForecastAnomalyResult(
        model_name="lstm_forecast_robust_threshold",
        model_config=config,
        metrics=score_binary_anomalies(label_array, predictions),
        point_adjusted_metrics=score_binary_anomalies(
            label_array,
            predictions,
            point_adjusted=True,
        ),
        history=tuple(history),
        scores=tuple(float(value) for value in scores),
        predictions=tuple(int(value) for value in predictions),
    )


def _validate_lstm_forecast_inputs(
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    window_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    threshold_sigma: float,
) -> None:
    if train_values.shape[1] != test_values.shape[1]:
        raise ValueError("train_values and test_values must have the same column count")
    if len(feature_names) != train_values.shape[1]:
        raise ValueError("feature_names length must match telemetry column count")
    if len(labels) != len(test_values):
        raise ValueError("labels length must match test_values row count")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(train_values) <= window_size or len(test_values) <= window_size:
        raise ValueError("train_values and test_values must each be longer than window_size")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive")


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


def _standardize_train_test(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(train_values, axis=0)
    stds = np.std(train_values, axis=0, ddof=0)
    scales = np.where(stds > 1e-12, stds, 1.0)
    return (train_values - means) / scales, (test_values - means) / scales, means, scales


def _forecast_windows(values: np.ndarray, *, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    windows = np.stack(
        [values[start : start + window_size] for start in range(len(values) - window_size)]
    )
    targets = values[window_size:]
    return windows.astype(np.float32), targets.astype(np.float32)


def _forecast_errors(
    model: nn.Module,
    values: np.ndarray,
    window_size: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    windows, targets = _forecast_windows(values, window_size=window_size)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[start : start + batch_size], dtype=torch.float32).to(
                device
            )
            predictions.append(model(batch).cpu().numpy())
    prediction_array = np.concatenate(predictions, axis=0)
    return np.mean((prediction_array - targets) ** 2, axis=1)


def _robust_error_threshold(errors: np.ndarray, *, sigma: float) -> float:
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    robust_scale = 1.4826 * mad
    fallback_scale = float(np.std(errors, ddof=0))
    scale = robust_scale if robust_scale > 1e-12 else fallback_scale
    if scale <= 1e-12:
        scale = 1.0
    return median + sigma * scale
