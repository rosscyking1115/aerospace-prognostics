"""SMAP/MSL anomaly-detection experiment helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from aerospace_prognostics.anomaly.baselines import (
    CLASSICAL_ANOMALY_BASELINE_METHODS,
    run_classical_anomaly_baselines,
)
from aerospace_prognostics.anomaly.forecasting import (
    DynamicThresholdConfig,
    LstmForecastTrainingEpoch,
    run_lstm_forecast_anomaly_baseline,
)
from aerospace_prognostics.anomaly.metrics import AnomalyDetectionMetrics
from aerospace_prognostics.data.smap_msl import load_smap_msl_channel, read_smap_msl_labels


@dataclass(frozen=True)
class SmapMslClassicalBaselineRun:
    """One classical anomaly-baseline result on one SMAP/MSL channel."""

    channel_id: str
    spacecraft: str
    train_rows: int
    test_rows: int
    feature_count: int
    anomaly_sequences: int
    anomaly_points: int
    model_name: str
    model_config: dict[str, object]
    metrics: AnomalyDetectionMetrics
    point_adjusted_metrics: AnomalyDetectionMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "spacecraft": self.spacecraft,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "feature_count": self.feature_count,
            "anomaly_sequences": self.anomaly_sequences,
            "anomaly_points": self.anomaly_points,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "metrics": self.metrics.to_dict(),
            "point_adjusted_metrics": self.point_adjusted_metrics.to_dict(),
        }


@dataclass(frozen=True)
class SmapMslLstmForecastBaselineRun:
    """One LSTM forecasting anomaly-baseline result on one SMAP/MSL channel."""

    channel_id: str
    spacecraft: str
    train_rows: int
    test_rows: int
    feature_count: int
    anomaly_sequences: int
    anomaly_points: int
    model_name: str
    model_config: dict[str, object]
    metrics: AnomalyDetectionMetrics
    point_adjusted_metrics: AnomalyDetectionMetrics
    history: tuple[LstmForecastTrainingEpoch, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "spacecraft": self.spacecraft,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "feature_count": self.feature_count,
            "anomaly_sequences": self.anomaly_sequences,
            "anomaly_points": self.anomaly_points,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "metrics": self.metrics.to_dict(),
            "point_adjusted_metrics": self.point_adjusted_metrics.to_dict(),
            "history": [epoch.to_dict() for epoch in self.history],
        }


def run_smap_msl_classical_baselines(
    data_dir: Path,
    *,
    channels: tuple[str, ...] | None = None,
    max_channels: int | None = None,
    methods: tuple[str, ...] = CLASSICAL_ANOMALY_BASELINE_METHODS,
    robust_threshold: float = 3.5,
    pca_components: int | None = None,
    pca_threshold_quantile: float = 0.99,
    isolation_contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[SmapMslClassicalBaselineRun, ...]:
    """Run classical anomaly baselines directly on raw SMAP/MSL channel arrays."""

    channel_ids = _selected_channel_ids(data_dir, channels=channels, max_channels=max_channels)
    runs: list[SmapMslClassicalBaselineRun] = []
    for channel_id in channel_ids:
        channel = load_smap_msl_channel(data_dir, channel_id)
        results = run_classical_anomaly_baselines(
            channel.train_values,
            channel.test_values,
            channel.test_labels,
            feature_names=channel.feature_names,
            methods=methods,
            robust_threshold=robust_threshold,
            pca_components=pca_components,
            pca_threshold_quantile=pca_threshold_quantile,
            isolation_contamination=isolation_contamination,
            random_state=random_state,
        )
        for result in results:
            runs.append(
                SmapMslClassicalBaselineRun(
                    channel_id=channel.metadata.channel_id,
                    spacecraft=channel.metadata.spacecraft,
                    train_rows=len(channel.train_values),
                    test_rows=len(channel.test_values),
                    feature_count=len(channel.feature_names),
                    anomaly_sequences=len(channel.metadata.anomaly_sequences),
                    anomaly_points=int(channel.test_labels.sum()),
                    model_name=result.model_name,
                    model_config=result.model_config,
                    metrics=result.metrics,
                    point_adjusted_metrics=result.point_adjusted_metrics,
                )
            )
    return tuple(runs)


def run_smap_msl_lstm_forecast_baseline(
    data_dir: Path,
    *,
    channels: tuple[str, ...] | None = None,
    max_channels: int | None = None,
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
) -> tuple[SmapMslLstmForecastBaselineRun, ...]:
    """Run an LSTM next-step forecasting anomaly baseline on SMAP/MSL channels."""

    channel_ids = _selected_channel_ids(data_dir, channels=channels, max_channels=max_channels)
    runs: list[SmapMslLstmForecastBaselineRun] = []
    for channel_id in channel_ids:
        channel = load_smap_msl_channel(data_dir, channel_id)
        result = run_lstm_forecast_anomaly_baseline(
            channel.train_values,
            channel.test_values,
            channel.test_labels,
            feature_names=channel.feature_names,
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
            random_state=random_state,
            device=device,
        )
        runs.append(
            SmapMslLstmForecastBaselineRun(
                channel_id=channel.metadata.channel_id,
                spacecraft=channel.metadata.spacecraft,
                train_rows=len(channel.train_values),
                test_rows=len(channel.test_values),
                feature_count=len(channel.feature_names),
                anomaly_sequences=len(channel.metadata.anomaly_sequences),
                anomaly_points=int(channel.test_labels.sum()),
                model_name=result.model_name,
                model_config=result.model_config,
                metrics=result.metrics,
                point_adjusted_metrics=result.point_adjusted_metrics,
                history=result.history,
            )
        )
    return tuple(runs)


def write_smap_msl_classical_baselines_json(
    runs: tuple[SmapMslClassicalBaselineRun, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL classical baseline runs as JSON."""

    output_path = _prepare_output_path(path)
    payload = [run.to_dict() for run in runs]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_smap_msl_lstm_forecast_baseline_json(
    runs: tuple[SmapMslLstmForecastBaselineRun, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL LSTM forecast baseline runs as JSON."""

    output_path = _prepare_output_path(path)
    payload = [run.to_dict() for run in runs]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_smap_msl_classical_baselines_csv(
    runs: tuple[SmapMslClassicalBaselineRun, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL classical baseline runs as a compact metrics table."""

    output_path = _prepare_output_path(path)
    fieldnames = [
        "channel_id",
        "spacecraft",
        "model_name",
        "train_rows",
        "test_rows",
        "feature_count",
        "anomaly_sequences",
        "anomaly_points",
        "precision",
        "recall",
        "f1",
        "point_adjusted_f1",
        "false_alarm_rate",
        "miss_rate",
        "support",
        "predicted_positives",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "channel_id": run.channel_id,
                    "spacecraft": run.spacecraft,
                    "model_name": run.model_name,
                    "train_rows": run.train_rows,
                    "test_rows": run.test_rows,
                    "feature_count": run.feature_count,
                    "anomaly_sequences": run.anomaly_sequences,
                    "anomaly_points": run.anomaly_points,
                    "precision": f"{run.metrics.precision:.12g}",
                    "recall": f"{run.metrics.recall:.12g}",
                    "f1": f"{run.metrics.f1:.12g}",
                    "point_adjusted_f1": f"{run.point_adjusted_metrics.f1:.12g}",
                    "false_alarm_rate": f"{run.metrics.false_alarm_rate:.12g}",
                    "miss_rate": f"{run.metrics.miss_rate:.12g}",
                    "support": run.metrics.support,
                    "predicted_positives": run.metrics.predicted_positives,
                }
            )


def write_smap_msl_lstm_forecast_baseline_csv(
    runs: tuple[SmapMslLstmForecastBaselineRun, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL LSTM forecast baseline runs as a compact metrics table."""

    output_path = _prepare_output_path(path)
    fieldnames = [
        "channel_id",
        "spacecraft",
        "model_name",
        "train_rows",
        "test_rows",
        "feature_count",
        "anomaly_sequences",
        "anomaly_points",
        "epochs",
        "final_train_loss",
        "threshold",
        "precision",
        "recall",
        "f1",
        "point_adjusted_f1",
        "false_alarm_rate",
        "miss_rate",
        "support",
        "predicted_positives",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            final_train_loss = run.history[-1].train_loss if run.history else 0.0
            writer.writerow(
                {
                    "channel_id": run.channel_id,
                    "spacecraft": run.spacecraft,
                    "model_name": run.model_name,
                    "train_rows": run.train_rows,
                    "test_rows": run.test_rows,
                    "feature_count": run.feature_count,
                    "anomaly_sequences": run.anomaly_sequences,
                    "anomaly_points": run.anomaly_points,
                    "epochs": len(run.history),
                    "final_train_loss": f"{final_train_loss:.12g}",
                    "threshold": f"{float(run.model_config['threshold']):.12g}",
                    "precision": f"{run.metrics.precision:.12g}",
                    "recall": f"{run.metrics.recall:.12g}",
                    "f1": f"{run.metrics.f1:.12g}",
                    "point_adjusted_f1": f"{run.point_adjusted_metrics.f1:.12g}",
                    "false_alarm_rate": f"{run.metrics.false_alarm_rate:.12g}",
                    "miss_rate": f"{run.metrics.miss_rate:.12g}",
                    "support": run.metrics.support,
                    "predicted_positives": run.metrics.predicted_positives,
                }
            )


def _selected_channel_ids(
    data_dir: Path,
    *,
    channels: tuple[str, ...] | None,
    max_channels: int | None,
) -> tuple[str, ...]:
    if max_channels is not None and max_channels <= 0:
        raise ValueError("max_channels must be positive")
    available = tuple(metadata.channel_id for metadata in read_smap_msl_labels(data_dir))
    if channels is not None:
        missing = sorted(set(channels) - set(available))
        if missing:
            raise ValueError(f"channels not found in labels: {', '.join(missing)}")
        selected = channels
    else:
        selected = available
    if max_channels is not None:
        selected = selected[:max_channels]
    if not selected:
        raise ValueError("no SMAP/MSL channels selected")
    return selected


def _prepare_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
