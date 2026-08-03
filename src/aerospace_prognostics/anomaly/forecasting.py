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

LSTM_FORECAST_THRESHOLD_METHODS = ("robust", "dynamic")


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


@dataclass(frozen=True)
class DynamicThresholdConfig:
    """Configuration for Telemanom-style nonparametric dynamic thresholding."""

    batch_size: int = 70
    window_batches: int = 30
    smoothing_fraction: float = 0.05
    z_start: float = 2.5
    z_stop: float = 12.0
    z_step: float = 0.5
    error_buffer: int = 100
    p: float = 0.13
    max_sequences: int = 5
    max_anomaly_fraction: float = 0.5

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
    threshold_method: str = "robust",
    dynamic_threshold_config: DynamicThresholdConfig | None = None,
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
        threshold_method=threshold_method,
        dynamic_threshold_config=dynamic_threshold_config,
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
    scores = np.zeros(len(test_array), dtype=np.float64)
    scores[window_size:] = test_errors
    threshold = _robust_error_threshold(train_errors, sigma=threshold_sigma)
    if threshold_method == "dynamic":
        threshold_config = dynamic_threshold_config or DynamicThresholdConfig()
        predictions = _dynamic_threshold_predictions(
            scores,
            min_index=window_size,
            config=threshold_config,
        )
    else:
        threshold_config = None
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
        "threshold_method": threshold_method,
        "threshold": float(threshold),
        "random_state": int(random_state),
        "device": device,
        "train_means": tuple(float(value) for value in means),
        "train_scales": tuple(float(value) for value in scales),
    }
    if threshold_config is not None:
        config["dynamic_threshold_config"] = threshold_config.to_dict()
    return LstmForecastAnomalyResult(
        model_name=f"lstm_forecast_{threshold_method}_threshold",
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
    threshold_method: str,
    dynamic_threshold_config: DynamicThresholdConfig | None,
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
    if threshold_method not in LSTM_FORECAST_THRESHOLD_METHODS:
        raise ValueError(
            "threshold_method must be one of "
            f"{', '.join(LSTM_FORECAST_THRESHOLD_METHODS)}"
        )
    if dynamic_threshold_config is not None:
        _validate_dynamic_threshold_config(dynamic_threshold_config)


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


def _dynamic_threshold_predictions(
    scores: np.ndarray,
    *,
    min_index: int,
    config: DynamicThresholdConfig,
) -> np.ndarray:
    _validate_dynamic_threshold_config(config)
    smoothed = _ewma(scores, span=max(1, int(len(scores) * config.smoothing_fraction)))
    predictions = np.zeros(len(scores), dtype=np.int8)
    trailing_rows = config.batch_size * config.window_batches
    n_windows = max(0, int((len(scores) - trailing_rows) / config.batch_size))

    for window_num in range(n_windows + 1):
        start_idx = window_num * config.batch_size
        end_idx = trailing_rows + start_idx
        if window_num == n_windows:
            end_idx = len(scores)
        if end_idx <= start_idx:
            continue
        window_scores = smoothed[start_idx:end_idx]
        threshold = _find_dynamic_threshold(window_scores, config=config)
        candidate_indices = np.flatnonzero(window_scores >= threshold)
        candidate_indices = _buffer_indices(
            candidate_indices,
            length=len(window_scores),
            error_buffer=config.error_buffer,
        )
        if window_num == 0:
            candidate_indices = candidate_indices[candidate_indices >= min_index]
        else:
            candidate_indices = candidate_indices[
                candidate_indices >= max(0, len(window_scores) - config.batch_size)
            ]
        candidate_indices = _prune_dynamic_sequences(
            candidate_indices,
            window_scores,
            threshold=threshold,
            p=config.p,
        )
        predictions[candidate_indices + start_idx] = 1
    predictions[:min_index] = 0
    return predictions


def _find_dynamic_threshold(errors: np.ndarray, *, config: DynamicThresholdConfig) -> float:
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors, ddof=0))
    if std_error <= 1e-12:
        return mean_error + config.z_stop

    best_score = -np.inf
    best_threshold = mean_error + config.z_stop * std_error
    for z_value in np.arange(config.z_start, config.z_stop, config.z_step):
        threshold = mean_error + float(z_value) * std_error
        anomalous = np.flatnonzero(errors >= threshold)
        anomalous = _buffer_indices(
            anomalous,
            length=len(errors),
            error_buffer=config.error_buffer,
        )
        if len(anomalous) == 0 or len(anomalous) >= len(errors) * config.max_anomaly_fraction:
            continue
        sequences = _consecutive_sequences(anomalous)
        if len(sequences) > config.max_sequences:
            continue
        pruned_errors = errors[errors < threshold]
        if len(pruned_errors) == 0:
            continue
        mean_decrease = (mean_error - float(np.mean(pruned_errors))) / max(mean_error, 1e-12)
        std_decrease = (std_error - float(np.std(pruned_errors, ddof=0))) / std_error
        score = (mean_decrease + std_decrease) / (len(sequences) ** 2 + len(anomalous))
        if score >= best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def _buffer_indices(indices: np.ndarray, *, length: int, error_buffer: int) -> np.ndarray:
    if len(indices) == 0:
        return indices.astype(int)
    offsets = np.arange(-error_buffer, error_buffer + 1)
    buffered = (indices.reshape(-1, 1) + offsets.reshape(1, -1)).ravel()
    return np.unique(buffered[(buffered >= 0) & (buffered < length)]).astype(int)


def _prune_dynamic_sequences(
    indices: np.ndarray,
    errors: np.ndarray,
    *,
    threshold: float,
    p: float,
) -> np.ndarray:
    sequences = _consecutive_sequences(indices)
    if not sequences:
        return indices.astype(int)
    sequence_maxima = np.asarray(
        [float(np.max(errors[start : end + 1])) for start, end in sequences]
    )
    non_anom = errors[errors < threshold]
    non_anom_max = float(np.max(non_anom)) if len(non_anom) else threshold
    sorted_maxima = np.append(np.sort(sequence_maxima)[::-1], non_anom_max)
    values_to_remove: set[float] = set()
    for index in range(len(sorted_maxima) - 1):
        separation = (sorted_maxima[index] - sorted_maxima[index + 1]) / max(
            abs(sorted_maxima[index]),
            1e-12,
        )
        if separation < p:
            values_to_remove.add(float(sorted_maxima[index]))
        else:
            values_to_remove.clear()
    if not values_to_remove:
        return indices.astype(int)

    kept_sequences = [
        sequence
        for sequence, maximum in zip(sequences, sequence_maxima, strict=True)
        if float(maximum) not in values_to_remove
    ]
    if not kept_sequences:
        return np.array([], dtype=int)
    return np.concatenate([np.arange(start, end + 1) for start, end in kept_sequences]).astype(int)


def _consecutive_sequences(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    sorted_indices = np.sort(np.unique(indices.astype(int)))
    sequences: list[tuple[int, int]] = []
    start = int(sorted_indices[0])
    previous = start
    for value in sorted_indices[1:]:
        current = int(value)
        if current == previous + 1:
            previous = current
            continue
        sequences.append((start, previous))
        start = current
        previous = current
    sequences.append((start, previous))
    return sequences


def _ewma(values: np.ndarray, *, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    smoothed = np.zeros_like(values, dtype=np.float64)
    if len(values) == 0:
        return smoothed
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def _validate_dynamic_threshold_config(config: DynamicThresholdConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("dynamic threshold batch_size must be positive")
    if config.window_batches <= 0:
        raise ValueError("dynamic threshold window_batches must be positive")
    if not 0 < config.smoothing_fraction <= 1:
        raise ValueError("dynamic threshold smoothing_fraction must be in (0, 1]")
    if config.z_start <= 0 or config.z_stop <= config.z_start or config.z_step <= 0:
        raise ValueError("dynamic threshold z range is invalid")
    if config.error_buffer < 0:
        raise ValueError("dynamic threshold error_buffer must be non-negative")
    if not 0 <= config.p < 1:
        raise ValueError("dynamic threshold p must be in [0, 1)")
    if config.max_sequences <= 0:
        raise ValueError("dynamic threshold max_sequences must be positive")
    if not 0 < config.max_anomaly_fraction < 1:
        raise ValueError("dynamic threshold max_anomaly_fraction must be in (0, 1)")
