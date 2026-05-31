"""Deep sequence baselines for C-MAPSS RUL prediction."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aerospace_prognostics.evaluation import RegressionRunResult
from aerospace_prognostics.metrics import nasa_rul_score, rmse

CMAPSS_DEEP_COMPARISON_MODELS = ("cnn", "lstm", "bilstm", "tcn", "transformer")


@dataclass(frozen=True)
class CmapssCnnTrainingEpoch:
    """Per-epoch training and validation metrics for a CNN baseline run."""

    epoch: int
    train_loss: float
    validation_rmse: float
    validation_nasa_score: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serialisable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class CmapssDeepPrediction:
    """Official-test prediction diagnostics for one C-MAPSS engine unit."""

    unit_number: int
    actual_rul: float
    predicted_rul: float
    error: float
    absolute_error: float
    late_error: float
    early_error: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serialisable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class CmapssCnnBaselineRun:
    """Full CNN baseline run with the selected official-test result."""

    result: RegressionRunResult
    selected_epoch: int
    history: tuple[CmapssCnnTrainingEpoch, ...]
    predictions: tuple[CmapssDeepPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""

        return {
            "result": self.result.to_dict(),
            "selected_epoch": self.selected_epoch,
            "history": [epoch.to_dict() for epoch in self.history],
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(frozen=True)
class CmapssLstmBaselineRun:
    """Full LSTM/BiLSTM baseline run with the selected official-test result."""

    result: RegressionRunResult
    selected_epoch: int
    history: tuple[CmapssCnnTrainingEpoch, ...]
    predictions: tuple[CmapssDeepPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""

        return {
            "result": self.result.to_dict(),
            "selected_epoch": self.selected_epoch,
            "history": [epoch.to_dict() for epoch in self.history],
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(frozen=True)
class CmapssTcnBaselineRun:
    """Full TCN baseline run with the selected official-test result."""

    result: RegressionRunResult
    selected_epoch: int
    history: tuple[CmapssCnnTrainingEpoch, ...]
    predictions: tuple[CmapssDeepPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""

        return {
            "result": self.result.to_dict(),
            "selected_epoch": self.selected_epoch,
            "history": [epoch.to_dict() for epoch in self.history],
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(frozen=True)
class CmapssTransformerBaselineRun:
    """Full Transformer baseline run with the selected official-test result."""

    result: RegressionRunResult
    selected_epoch: int
    history: tuple[CmapssCnnTrainingEpoch, ...]
    predictions: tuple[CmapssDeepPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""

        return {
            "result": self.result.to_dict(),
            "selected_epoch": self.selected_epoch,
            "history": [epoch.to_dict() for epoch in self.history],
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


CmapssDeepBaselineRun = (
    CmapssCnnBaselineRun
    | CmapssLstmBaselineRun
    | CmapssTcnBaselineRun
    | CmapssTransformerBaselineRun
)


class CmapssOneDimensionalCnn(nn.Module):
    """Small 1D-CNN baseline over C-MAPSS sensor windows."""

    def __init__(
        self,
        *,
        feature_count: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(feature_count, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Predict RUL from `(batch, timesteps, features)` windows."""

        return self.network(windows.transpose(1, 2)).squeeze(-1)


class CmapssLstmRegressor(nn.Module):
    """LSTM/BiLSTM baseline over C-MAPSS sensor windows."""

    def __init__(
        self,
        *,
        feature_count: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            input_size=feature_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * directions, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Predict RUL from `(batch, timesteps, features)` windows."""

        output, _ = self.recurrent(windows)
        return self.head(output[:, -1, :]).squeeze(-1)


class _CausalChomp1d(nn.Module):
    """Trim right-side padding from causal Conv1d outputs."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return values
        return values[:, :, : -self.chomp_size]


class _TcnResidualBlock(nn.Module):
    """Two-layer dilated causal convolution block with a residual path."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _CausalChomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _CausalChomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(values) + self.residual(values))


class CmapssTemporalConvolutionalRegressor(nn.Module):
    """Compact TCN baseline over C-MAPSS sensor windows."""

    def __init__(
        self,
        *,
        feature_count: int,
        hidden_channels: int = 32,
        num_levels: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be at least 1")
        if num_levels < 1:
            raise ValueError("num_levels must be at least 1")
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        blocks: list[nn.Module] = []
        input_channels = feature_count
        for level in range(num_levels):
            blocks.append(
                _TcnResidualBlock(
                    input_channels=input_channels,
                    output_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            input_channels = hidden_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Predict RUL from `(batch, timesteps, features)` windows."""

        features = self.network(windows.transpose(1, 2))
        return self.head(features[:, :, -1]).squeeze(-1)


class CmapssTransformerRegressor(nn.Module):
    """Compact Transformer encoder baseline over C-MAPSS sensor windows."""

    def __init__(
        self,
        *,
        feature_count: int,
        sequence_length: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be at least 1")
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if dim_feedforward < 1:
            raise ValueError("dim_feedforward must be at least 1")
        self.input_projection = nn.Linear(feature_count, d_model)
        self.register_buffer(
            "position_encoding",
            _sinusoidal_position_encoding(sequence_length, d_model),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Predict RUL from `(batch, timesteps, features)` windows."""

        sequence_length = windows.shape[1]
        encoded = self.input_projection(windows)
        encoded = encoded + self.position_encoding[:sequence_length].unsqueeze(0)
        features = self.encoder(encoded)
        return self.head(features[:, -1, :]).squeeze(-1)


def _sinusoidal_position_encoding(sequence_length: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(dimensions * (-np.log(10000.0) / d_model))
    encoding = torch.zeros(sequence_length, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


def run_cmapss_cnn_baseline(
    sequence_dir: str | Path,
    subset: str,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> RegressionRunResult:
    """Train and evaluate a compact CNN baseline from exported C-MAPSS sequences."""

    return run_cmapss_cnn_baseline_run(
        sequence_dir,
        subset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
    ).result


def run_cmapss_cnn_baseline_run(
    sequence_dir: str | Path,
    subset: str,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> CmapssCnnBaselineRun:
    """Train a CNN baseline and select the best epoch by validation NASA score."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if checkpoint_policy not in {"validation_nasa", "final"}:
        raise ValueError("checkpoint_policy must be 'validation_nasa' or 'final'")

    def model_factory(feature_count: int, sequence_length: int) -> nn.Module:
        return CmapssOneDimensionalCnn(
            feature_count=feature_count,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    result, selected_epoch, history, predictions = _run_sequence_baseline_run(
        sequence_dir,
        subset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
        model_factory=model_factory,
        model_name_base_factory=lambda metadata: (
            f"cnn_1d_w{metadata['window_size']}_e{epochs}_c{hidden_channels}"
        ),
    )
    return CmapssCnnBaselineRun(
        result=result,
        selected_epoch=selected_epoch,
        history=history,
        predictions=predictions,
    )


def run_cmapss_lstm_baseline(
    sequence_dir: str | Path,
    subset: str,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    bidirectional: bool = False,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> RegressionRunResult:
    """Train and evaluate an LSTM/BiLSTM baseline from exported C-MAPSS sequences."""

    return run_cmapss_lstm_baseline_run(
        sequence_dir,
        subset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
    ).result


def run_cmapss_lstm_baseline_run(
    sequence_dir: str | Path,
    subset: str,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    bidirectional: bool = False,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> CmapssLstmBaselineRun:
    """Train an LSTM/BiLSTM baseline and select the best epoch by validation NASA score."""

    if hidden_size < 1:
        raise ValueError("hidden_size must be at least 1")
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")

    def model_factory(feature_count: int, sequence_length: int) -> nn.Module:
        return CmapssLstmRegressor(
            feature_count=feature_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    model_kind = "bilstm" if bidirectional else "lstm"
    result, selected_epoch, history, predictions = _run_sequence_baseline_run(
        sequence_dir,
        subset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
        model_factory=model_factory,
        model_name_base_factory=lambda metadata: (
            f"{model_kind}_w{metadata['window_size']}_e{epochs}_h{hidden_size}_l{num_layers}"
        ),
    )
    return CmapssLstmBaselineRun(
        result=result,
        selected_epoch=selected_epoch,
        history=history,
        predictions=predictions,
    )


def run_cmapss_tcn_baseline(
    sequence_dir: str | Path,
    subset: str,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    num_levels: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> RegressionRunResult:
    """Train and evaluate a compact TCN baseline from exported C-MAPSS sequences."""

    return run_cmapss_tcn_baseline_run(
        sequence_dir,
        subset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_channels=hidden_channels,
        num_levels=num_levels,
        kernel_size=kernel_size,
        dropout=dropout,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
    ).result


def run_cmapss_tcn_baseline_run(
    sequence_dir: str | Path,
    subset: str,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    num_levels: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> CmapssTcnBaselineRun:
    """Train a compact TCN baseline and select the best epoch by validation NASA score."""

    if hidden_channels < 1:
        raise ValueError("hidden_channels must be at least 1")
    if num_levels < 1:
        raise ValueError("num_levels must be at least 1")
    if kernel_size < 1:
        raise ValueError("kernel_size must be at least 1")

    def model_factory(feature_count: int, sequence_length: int) -> nn.Module:
        return CmapssTemporalConvolutionalRegressor(
            feature_count=feature_count,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    result, selected_epoch, history, predictions = _run_sequence_baseline_run(
        sequence_dir,
        subset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
        model_factory=model_factory,
        model_name_base_factory=lambda metadata: (
            f"tcn_w{metadata['window_size']}_e{epochs}_c{hidden_channels}"
            f"_l{num_levels}_k{kernel_size}"
        ),
    )
    return CmapssTcnBaselineRun(
        result=result,
        selected_epoch=selected_epoch,
        history=history,
        predictions=predictions,
    )


def run_cmapss_transformer_baseline(
    sequence_dir: str | Path,
    subset: str,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    d_model: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 64,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> RegressionRunResult:
    """Train and evaluate a compact Transformer baseline from exported C-MAPSS sequences."""

    return run_cmapss_transformer_baseline_run(
        sequence_dir,
        subset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
    ).result


def run_cmapss_transformer_baseline_run(
    sequence_dir: str | Path,
    subset: str,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    d_model: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 64,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> CmapssTransformerBaselineRun:
    """Train a Transformer baseline and select the best epoch by validation NASA score."""

    if d_model < 1:
        raise ValueError("d_model must be at least 1")
    if num_heads < 1:
        raise ValueError("num_heads must be at least 1")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if dim_feedforward < 1:
        raise ValueError("dim_feedforward must be at least 1")

    def model_factory(feature_count: int, sequence_length: int) -> nn.Module:
        return CmapssTransformerRegressor(
            feature_count=feature_count,
            sequence_length=sequence_length,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    result, selected_epoch, history, predictions = _run_sequence_baseline_run(
        sequence_dir,
        subset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
        model_factory=model_factory,
        model_name_base_factory=lambda metadata: (
            f"transformer_w{metadata['window_size']}_e{epochs}_d{d_model}"
            f"_h{num_heads}_l{num_layers}_ff{dim_feedforward}"
        ),
    )
    return CmapssTransformerBaselineRun(
        result=result,
        selected_epoch=selected_epoch,
        history=history,
        predictions=predictions,
    )


def _run_sequence_baseline_run(
    sequence_dir: str | Path,
    subset: str,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_policy: str,
    random_state: int,
    device: str,
    model_factory: Callable[[int, int], nn.Module],
    model_name_base_factory: Callable[[dict[str, Any]], str],
) -> tuple[
    RegressionRunResult,
    int,
    tuple[CmapssCnnTrainingEpoch, ...],
    tuple[CmapssDeepPrediction, ...],
]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if checkpoint_policy not in {"validation_nasa", "final"}:
        raise ValueError("checkpoint_policy must be 'validation_nasa' or 'final'")

    _seed_everything(random_state)
    paths = _sequence_paths(sequence_dir, subset)
    train_payload = _load_sequence_npz(paths["train"])
    validation_payload = _load_sequence_npz(_validation_selection_path(paths))
    test_payload = _load_sequence_npz(paths["test"])
    metadata = _load_metadata(paths["metadata"])

    torch_device = torch.device(device)
    model = model_factory(
        int(train_payload["windows"].shape[-1]),
        int(train_payload["windows"].shape[1]),
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_payload["windows"], dtype=torch.float32),
            torch.as_tensor(train_payload["targets"], dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(random_state),
    )

    history: list[CmapssCnnTrainingEpoch] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: CmapssCnnTrainingEpoch | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        sample_count = 0
        for batch_windows, batch_targets in loader:
            batch_windows = batch_windows.to(torch_device)
            batch_targets = batch_targets.to(torch_device)
            optimizer.zero_grad()
            loss = loss_function(model(batch_windows), batch_targets)
            loss.backward()
            optimizer.step()
            batch_size_actual = len(batch_targets)
            loss_total += float(loss.detach().cpu()) * batch_size_actual
            sample_count += batch_size_actual

        validation_predictions = _predict(model, validation_payload["windows"], torch_device)
        epoch_metrics = CmapssCnnTrainingEpoch(
            epoch=epoch,
            train_loss=loss_total / sample_count,
            validation_rmse=rmse(validation_payload["targets"], validation_predictions),
            validation_nasa_score=nasa_rul_score(
                validation_payload["targets"], validation_predictions
            ),
        )
        history.append(epoch_metrics)
        if best_epoch is None or _is_better_validation_epoch(epoch_metrics, best_epoch):
            best_epoch = epoch_metrics
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }

    if not history or best_state is None or best_epoch is None:
        raise RuntimeError("training did not produce a validation checkpoint")

    selected_epoch = best_epoch
    selected_state = best_state
    selected_label = f"best_e{best_epoch.epoch}"
    if checkpoint_policy == "final":
        selected_epoch = history[-1]
        selected_state = {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        }
        selected_label = f"final_e{selected_epoch.epoch}"

    model.load_state_dict(selected_state)
    model.to(torch_device)
    test_predictions = _predict(model, test_payload["windows"], torch_device)
    model_name = f"{model_name_base_factory(metadata)}_{selected_label}"

    result = RegressionRunResult(
        dataset="C-MAPSS-sequence",
        subset=str(metadata["subset"]),
        model_name=(
            f"{model_name}_val_rmse_{selected_epoch.validation_rmse:.6f}"
            f"_val_nasa_{selected_epoch.validation_nasa_score:.6f}"
        ),
        rmse=rmse(test_payload["targets"], test_predictions),
        nasa_score=nasa_rul_score(test_payload["targets"], test_predictions),
        train_rows=len(train_payload["windows"]),
        train_units=len(np.unique(train_payload["unit_numbers"])),
        test_rows=len(test_payload["windows"]),
        test_units=len(np.unique(test_payload["unit_numbers"])),
        test_rul_values=len(test_payload["targets"]),
        rul_cap=int(metadata["rul_cap"]),
        random_state=random_state,
        standardize=bool(metadata["standardize"]),
    )
    predictions = _build_sequence_predictions(
        test_payload["unit_numbers"],
        test_payload["targets"],
        test_predictions,
    )
    return result, selected_epoch.epoch, tuple(history), predictions


def _build_sequence_predictions(
    unit_numbers: np.ndarray,
    actual_rul: np.ndarray,
    predicted_rul: np.ndarray,
) -> tuple[CmapssDeepPrediction, ...]:
    rows: list[CmapssDeepPrediction] = []
    for unit_number, actual, predicted in zip(
        unit_numbers,
        actual_rul,
        predicted_rul,
        strict=True,
    ):
        actual_value = float(actual)
        predicted_value = float(predicted)
        error = predicted_value - actual_value
        rows.append(
            CmapssDeepPrediction(
                unit_number=int(unit_number),
                actual_rul=actual_value,
                predicted_rul=predicted_value,
                error=error,
                absolute_error=abs(error),
                late_error=max(error, 0.0),
                early_error=max(-error, 0.0),
            )
        )
    return tuple(rows)


def run_all_cmapss_cnn_baselines(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[RegressionRunResult]:
    """Train CNN baselines for the requested C-MAPSS subsets."""

    return [
        run.result
        for run in run_all_cmapss_cnn_baseline_runs(
            sequence_dir,
            subsets=subsets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
    ]


def run_all_cmapss_lstm_baselines(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    bidirectional: bool = False,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[RegressionRunResult]:
    """Train LSTM/BiLSTM baselines for the requested C-MAPSS subsets."""

    return [
        run.result
        for run in run_all_cmapss_lstm_baseline_runs(
            sequence_dir,
            subsets=subsets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
    ]


def run_all_cmapss_lstm_baseline_runs(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    bidirectional: bool = False,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[CmapssLstmBaselineRun]:
    """Train full LSTM/BiLSTM baseline runs for the requested C-MAPSS subsets."""

    return [
        run_cmapss_lstm_baseline_run(
            sequence_dir,
            subset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
        for subset in subsets
    ]


def run_all_cmapss_tcn_baselines(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    num_levels: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[RegressionRunResult]:
    """Train TCN baselines for the requested C-MAPSS subsets."""

    return [
        run.result
        for run in run_all_cmapss_tcn_baseline_runs(
            sequence_dir,
            subsets=subsets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dropout=dropout,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
    ]


def run_all_cmapss_transformer_baselines(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    d_model: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 64,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[RegressionRunResult]:
    """Train Transformer baselines for the requested C-MAPSS subsets."""

    return [
        run.result
        for run in run_all_cmapss_transformer_baseline_runs(
            sequence_dir,
            subsets=subsets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
    ]


def run_all_cmapss_transformer_baseline_runs(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    d_model: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 64,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[CmapssTransformerBaselineRun]:
    """Train full Transformer baseline runs for the requested C-MAPSS subsets."""

    return [
        run_cmapss_transformer_baseline_run(
            sequence_dir,
            subset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
        for subset in subsets
    ]


def run_all_cmapss_tcn_baseline_runs(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    num_levels: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[CmapssTcnBaselineRun]:
    """Train full TCN baseline runs for the requested C-MAPSS subsets."""

    return [
        run_cmapss_tcn_baseline_run(
            sequence_dir,
            subset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dropout=dropout,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
        for subset in subsets
    ]


def run_cmapss_deep_baseline_comparison(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    models: tuple[str, ...] = ("cnn", "bilstm", "tcn"),
    epochs: int = 5,
    batch_size: int = 256,
    learning_rates: tuple[float, ...] = (1e-3,),
    hidden_sizes: tuple[int, ...] = (32,),
    num_layers: int = 1,
    tcn_levels: int = 3,
    transformer_heads: int = 4,
    transformer_dim_feedforward: int | None = None,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[RegressionRunResult]:
    """Compare selected Phase 2 deep baselines across compact hyperparameter grids."""

    return [
        run.result
        for run in run_cmapss_deep_baseline_comparison_runs(
            sequence_dir,
            subsets=subsets,
            models=models,
            epochs=epochs,
            batch_size=batch_size,
            learning_rates=learning_rates,
            hidden_sizes=hidden_sizes,
            num_layers=num_layers,
            tcn_levels=tcn_levels,
            transformer_heads=transformer_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            kernel_size=kernel_size,
            dropout=dropout,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
    ]


def run_cmapss_deep_baseline_comparison_runs(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    models: tuple[str, ...] = ("cnn", "bilstm", "tcn"),
    epochs: int = 5,
    batch_size: int = 256,
    learning_rates: tuple[float, ...] = (1e-3,),
    hidden_sizes: tuple[int, ...] = (32,),
    num_layers: int = 1,
    tcn_levels: int = 3,
    transformer_heads: int = 4,
    transformer_dim_feedforward: int | None = None,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[CmapssDeepBaselineRun]:
    """Compare deep baselines and preserve histories plus per-unit predictions."""

    _validate_deep_comparison_inputs(
        models=models,
        learning_rates=learning_rates,
        hidden_sizes=hidden_sizes,
    )
    runs: list[CmapssDeepBaselineRun] = []
    for model_name in models:
        for learning_rate in learning_rates:
            for hidden_size in hidden_sizes:
                candidate_runs = _run_deep_comparison_candidate_runs(
                    sequence_dir,
                    subsets=subsets,
                    model_name=model_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    tcn_levels=tcn_levels,
                    transformer_heads=transformer_heads,
                    transformer_dim_feedforward=transformer_dim_feedforward,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    checkpoint_policy=checkpoint_policy,
                    random_state=random_state,
                    device=device,
                )
                runs.extend(
                    _with_comparison_run_label(
                        run,
                        model_name=model_name,
                        learning_rate=learning_rate,
                        hidden_size=hidden_size,
                    )
                    for run in candidate_runs
                )
    return runs


def write_cmapss_deep_predictions_csv(
    runs: list[CmapssDeepBaselineRun] | tuple[CmapssDeepBaselineRun, ...],
    output_path: str | Path,
) -> Path:
    """Write official-test per-unit prediction diagnostics for deep C-MAPSS runs."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "subset",
        "model_name",
        "selected_epoch",
        "unit_number",
        "actual_rul",
        "predicted_rul",
        "error",
        "absolute_error",
        "late_error",
        "early_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for prediction in run.predictions:
                writer.writerow(
                    {
                        "dataset": run.result.dataset,
                        "subset": run.result.subset,
                        "model_name": run.result.model_name,
                        "selected_epoch": run.selected_epoch,
                        **prediction.to_dict(),
                    }
                )
    return path


def _validate_deep_comparison_inputs(
    *,
    models: tuple[str, ...],
    learning_rates: tuple[float, ...],
    hidden_sizes: tuple[int, ...],
) -> None:
    if not models:
        raise ValueError("models must contain at least one model")
    unknown_models = sorted(set(models) - set(CMAPSS_DEEP_COMPARISON_MODELS))
    if unknown_models:
        raise ValueError(f"unknown deep comparison models: {', '.join(unknown_models)}")
    if not learning_rates:
        raise ValueError("learning_rates must contain at least one value")
    if not hidden_sizes:
        raise ValueError("hidden_sizes must contain at least one value")
    if any(learning_rate <= 0 for learning_rate in learning_rates):
        raise ValueError("learning_rates must all be positive")
    if any(hidden_size < 1 for hidden_size in hidden_sizes):
        raise ValueError("hidden_sizes must all be at least 1")


def _run_deep_comparison_candidate_runs(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    num_layers: int,
    tcn_levels: int,
    transformer_heads: int,
    transformer_dim_feedforward: int | None,
    kernel_size: int,
    dropout: float,
    checkpoint_policy: str,
    random_state: int,
    device: str,
) -> list[CmapssDeepBaselineRun]:
    if model_name == "cnn":
        return run_all_cmapss_cnn_baseline_runs(
            sequence_dir,
            subsets=subsets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_channels=hidden_size,
            kernel_size=kernel_size,
            dropout=dropout,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
    if model_name in {"lstm", "bilstm"}:
        return run_all_cmapss_lstm_baseline_runs(
            sequence_dir,
            subsets=subsets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=model_name == "bilstm",
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
    if model_name == "tcn":
        return run_all_cmapss_tcn_baseline_runs(
            sequence_dir,
            subsets=subsets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_channels=hidden_size,
            num_levels=tcn_levels,
            kernel_size=kernel_size,
            dropout=dropout,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
    return run_all_cmapss_transformer_baseline_runs(
        sequence_dir,
        subsets=subsets,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        d_model=hidden_size,
        num_heads=transformer_heads,
        num_layers=num_layers,
        dim_feedforward=transformer_dim_feedforward or hidden_size * 2,
        dropout=dropout,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
    )


def _with_comparison_label(
    result: RegressionRunResult,
    *,
    model_name: str,
    learning_rate: float,
    hidden_size: int,
) -> RegressionRunResult:
    label = (
        f"compare_{model_name}_h{hidden_size}_lr{_format_comparison_float(learning_rate)}"
    )
    return replace(result, model_name=f"{label}_{result.model_name}")


def _with_comparison_run_label(
    run: CmapssDeepBaselineRun,
    *,
    model_name: str,
    learning_rate: float,
    hidden_size: int,
) -> CmapssDeepBaselineRun:
    return replace(
        run,
        result=_with_comparison_label(
            run.result,
            model_name=model_name,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
        ),
    )


def _format_comparison_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_all_cmapss_cnn_baseline_runs(
    sequence_dir: str | Path,
    *,
    subsets: tuple[str, ...],
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    random_state: int = 42,
    device: str = "cpu",
) -> list[CmapssCnnBaselineRun]:
    """Train full CNN baseline runs for the requested C-MAPSS subsets."""

    return [
        run_cmapss_cnn_baseline_run(
            sequence_dir,
            subset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            checkpoint_policy=checkpoint_policy,
            random_state=random_state,
            device=device,
        )
        for subset in subsets
    ]


def _seed_everything(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)


def _is_better_validation_epoch(
    candidate: CmapssCnnTrainingEpoch,
    incumbent: CmapssCnnTrainingEpoch,
) -> bool:
    if candidate.validation_nasa_score < incumbent.validation_nasa_score:
        return True
    if candidate.validation_nasa_score > incumbent.validation_nasa_score:
        return False
    return candidate.validation_rmse < incumbent.validation_rmse


def _sequence_paths(sequence_dir: str | Path, subset: str) -> dict[str, Path]:
    subset_dir = Path(sequence_dir) / subset.lower()
    return {
        "train": subset_dir / "train_sequences.npz",
        "validation": subset_dir / "validation_sequences.npz",
        "validation_selection": subset_dir / "validation_selection_sequences.npz",
        "test": subset_dir / "test_sequences.npz",
        "metadata": subset_dir / "metadata.json",
    }


def _validation_selection_path(paths: dict[str, Path]) -> Path:
    path = paths["validation_selection"]
    if path.exists():
        return path
    return paths["validation"]


def _load_sequence_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing sequence artifact: {path}")
    payload = np.load(path)
    return {
        "windows": payload["windows"].astype(np.float32),
        "targets": payload["targets"].astype(np.float32),
        "unit_numbers": payload["unit_numbers"],
    }


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing sequence metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _predict(
    model: nn.Module,
    windows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[start : start + batch_size], dtype=torch.float32).to(
                device
            )
            predictions.append(model(batch).cpu().numpy())
    if not predictions:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(predictions).astype(np.float32)
