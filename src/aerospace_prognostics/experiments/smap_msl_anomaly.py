"""SMAP/MSL anomaly-detection experiment helpers."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from aerospace_prognostics.anomaly.baselines import (
    CLASSICAL_ANOMALY_BASELINE_METHODS,
    run_classical_anomaly_baselines,
    run_robust_zscore_baseline,
)
from aerospace_prognostics.anomaly.metrics import AnomalyDetectionMetrics
from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload
from aerospace_prognostics.data.smap_msl import load_smap_msl_channel, read_smap_msl_labels

LSTM_FORECAST_THRESHOLD_METHODS = ("robust", "dynamic")


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
    history: tuple[object, ...]

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


@dataclass(frozen=True)
class SmapMslRobustThresholdSweepRun:
    """One robust z-score threshold result on one SMAP/MSL channel."""

    channel_id: str
    spacecraft: str
    threshold: float
    train_rows: int
    test_rows: int
    feature_count: int
    anomaly_sequences: int
    anomaly_points: int
    metrics: AnomalyDetectionMetrics
    point_adjusted_metrics: AnomalyDetectionMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "spacecraft": self.spacecraft,
            "threshold": self.threshold,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "feature_count": self.feature_count,
            "anomaly_sequences": self.anomaly_sequences,
            "anomaly_points": self.anomaly_points,
            "metrics": self.metrics.to_dict(),
            "point_adjusted_metrics": self.point_adjusted_metrics.to_dict(),
        }


@dataclass(frozen=True)
class SmapMslRobustThresholdSweepAggregate:
    """Aggregate robust z-score threshold metrics across selected SMAP/MSL channels."""

    threshold: float
    channels: int
    wins_by_f1: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_point_adjusted_f1: float
    mean_false_alarm_rate: float
    mean_miss_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "channels": self.channels,
            "wins_by_f1": self.wins_by_f1,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
            "mean_point_adjusted_f1": self.mean_point_adjusted_f1,
            "mean_false_alarm_rate": self.mean_false_alarm_rate,
            "mean_miss_rate": self.mean_miss_rate,
        }


@dataclass(frozen=True)
class SmapMslRobustThresholdOperatingPoint:
    """Selected robust threshold for a spacecraft family or global channel group."""

    scope: str
    group: str
    false_alarm_budget: float
    selected_threshold: float
    feasible: bool
    channels: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_point_adjusted_f1: float
    mean_false_alarm_rate: float
    mean_miss_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "group": self.group,
            "false_alarm_budget": self.false_alarm_budget,
            "selected_threshold": self.selected_threshold,
            "feasible": self.feasible,
            "channels": self.channels,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
            "mean_point_adjusted_f1": self.mean_point_adjusted_f1,
            "mean_false_alarm_rate": self.mean_false_alarm_rate,
            "mean_miss_rate": self.mean_miss_rate,
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


def run_smap_msl_robust_threshold_sweep(
    data_dir: Path,
    *,
    thresholds: tuple[float, ...],
    channels: tuple[str, ...] | None = None,
    max_channels: int | None = None,
) -> tuple[SmapMslRobustThresholdSweepRun, ...]:
    """Sweep robust z-score thresholds across raw SMAP/MSL channel arrays."""
    if not thresholds:
        raise ValueError("thresholds must contain at least one value")
    if any(threshold <= 0 for threshold in thresholds):
        raise ValueError("thresholds must be positive")
    channel_ids = _selected_channel_ids(data_dir, channels=channels, max_channels=max_channels)
    runs: list[SmapMslRobustThresholdSweepRun] = []
    for channel_id in channel_ids:
        channel = load_smap_msl_channel(data_dir, channel_id)
        for threshold in thresholds:
            result = run_robust_zscore_baseline(
                channel.train_values,
                channel.test_values,
                channel.test_labels,
                feature_names=channel.feature_names,
                threshold=threshold,
            )
            runs.append(
                SmapMslRobustThresholdSweepRun(
                    channel_id=channel.metadata.channel_id,
                    spacecraft=channel.metadata.spacecraft,
                    threshold=float(threshold),
                    train_rows=len(channel.train_values),
                    test_rows=len(channel.test_values),
                    feature_count=len(channel.feature_names),
                    anomaly_sequences=len(channel.metadata.anomaly_sequences),
                    anomaly_points=int(channel.test_labels.sum()),
                    metrics=result.metrics,
                    point_adjusted_metrics=result.point_adjusted_metrics,
                )
            )
    return tuple(runs)


def aggregate_smap_msl_robust_threshold_sweep(
    runs: tuple[SmapMslRobustThresholdSweepRun, ...],
) -> tuple[SmapMslRobustThresholdSweepAggregate, ...]:
    """Aggregate robust threshold sweep results by threshold."""
    if not runs:
        raise ValueError("runs must contain at least one item")
    winners = _robust_threshold_wins_by_channel(runs)
    aggregates: list[SmapMslRobustThresholdSweepAggregate] = []
    for threshold in sorted({run.threshold for run in runs}):
        threshold_runs = tuple(run for run in runs if run.threshold == threshold)
        aggregates.append(
            SmapMslRobustThresholdSweepAggregate(
                threshold=threshold,
                channels=len(threshold_runs),
                wins_by_f1=winners.get(threshold, 0),
                mean_precision=_mean(run.metrics.precision for run in threshold_runs),
                mean_recall=_mean(run.metrics.recall for run in threshold_runs),
                mean_f1=_mean(run.metrics.f1 for run in threshold_runs),
                mean_point_adjusted_f1=_mean(
                    run.point_adjusted_metrics.f1 for run in threshold_runs
                ),
                mean_false_alarm_rate=_mean(run.metrics.false_alarm_rate for run in threshold_runs),
                mean_miss_rate=_mean(run.metrics.miss_rate for run in threshold_runs),
            )
        )
    return tuple(aggregates)


def select_smap_msl_robust_threshold_operating_points(
    runs: tuple[SmapMslRobustThresholdSweepRun, ...],
    *,
    false_alarm_budget: float,
    group_by: str = "spacecraft",
) -> tuple[SmapMslRobustThresholdOperatingPoint, ...]:
    """Select robust threshold operating points under a false-alarm budget."""
    if not runs:
        raise ValueError("runs must contain at least one item")
    if false_alarm_budget < 0 or false_alarm_budget > 1:
        raise ValueError("false_alarm_budget must be between 0 and 1")
    if group_by not in {"global", "spacecraft"}:
        raise ValueError("group_by must be 'global' or 'spacecraft'")

    grouped_runs: dict[str, tuple[SmapMslRobustThresholdSweepRun, ...]]
    if group_by == "global":
        grouped_runs = {"all": runs}
    else:
        grouped_runs = {
            spacecraft: tuple(run for run in runs if run.spacecraft == spacecraft)
            for spacecraft in sorted({run.spacecraft for run in runs})
        }

    operating_points: list[SmapMslRobustThresholdOperatingPoint] = []
    for group, group_runs in grouped_runs.items():
        aggregates = aggregate_smap_msl_robust_threshold_sweep(group_runs)
        feasible = tuple(
            aggregate
            for aggregate in aggregates
            if aggregate.mean_false_alarm_rate <= false_alarm_budget
        )
        selected = min(
            feasible or aggregates,
            key=(
                _operating_point_feasible_sort_key
                if feasible
                else _operating_point_infeasible_sort_key
            ),
        )
        operating_points.append(
            SmapMslRobustThresholdOperatingPoint(
                scope=group_by,
                group=group,
                false_alarm_budget=false_alarm_budget,
                selected_threshold=selected.threshold,
                feasible=bool(feasible),
                channels=selected.channels,
                mean_precision=selected.mean_precision,
                mean_recall=selected.mean_recall,
                mean_f1=selected.mean_f1,
                mean_point_adjusted_f1=selected.mean_point_adjusted_f1,
                mean_false_alarm_rate=selected.mean_false_alarm_rate,
                mean_miss_rate=selected.mean_miss_rate,
            )
        )
    return tuple(operating_points)


def select_smap_msl_robust_threshold_policy_runs(
    runs: tuple[SmapMslRobustThresholdSweepRun, ...],
    operating_points: tuple[SmapMslRobustThresholdOperatingPoint, ...],
) -> tuple[SmapMslRobustThresholdSweepRun, ...]:
    """Select the per-channel runs implied by robust threshold operating points."""
    if not runs:
        raise ValueError("runs must contain at least one item")
    if not operating_points:
        raise ValueError("operating_points must contain at least one item")
    scopes = {operating_point.scope for operating_point in operating_points}
    if len(scopes) != 1:
        raise ValueError("operating_points must use one scope")
    scope = next(iter(scopes))
    if scope not in {"global", "spacecraft"}:
        raise ValueError("operating point scope must be 'global' or 'spacecraft'")
    if len({operating_point.group for operating_point in operating_points}) != len(
        operating_points
    ):
        raise ValueError("operating point groups must be unique")

    operating_point_by_group = {
        operating_point.group: operating_point for operating_point in operating_points
    }
    selected_runs_by_channel: dict[str, SmapMslRobustThresholdSweepRun] = {}
    for run in runs:
        group = "all" if scope == "global" else run.spacecraft
        operating_point = operating_point_by_group.get(group)
        if operating_point is None:
            raise ValueError(f"missing operating point for group: {group}")
        if run.threshold == operating_point.selected_threshold:
            selected_runs_by_channel.setdefault(run.channel_id, run)
    expected_channels = {run.channel_id for run in runs}
    selected_channels = set(selected_runs_by_channel)
    if selected_channels != expected_channels:
        missing = ", ".join(sorted(expected_channels - selected_channels))
        raise ValueError(f"policy selection is missing channels: {missing}")
    return tuple(
        selected_runs_by_channel[channel_id]
        for channel_id in dict.fromkeys(run.channel_id for run in runs)
    )


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
    dynamic_threshold_config: object | None = None,
    random_state: int = 42,
    device: str = "cpu",
) -> tuple[SmapMslLstmForecastBaselineRun, ...]:
    """Run an LSTM next-step forecasting anomaly baseline on SMAP/MSL channels."""
    channel_ids = _selected_channel_ids(data_dir, channels=channels, max_channels=max_channels)
    from aerospace_prognostics.anomaly.forecasting import run_lstm_forecast_anomaly_baseline

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
    payload = [run.to_dict() for run in runs]
    write_json_payload(payload, path)


def write_smap_msl_lstm_forecast_baseline_json(
    runs: tuple[SmapMslLstmForecastBaselineRun, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL LSTM forecast baseline runs as JSON."""
    payload = [run.to_dict() for run in runs]
    write_json_payload(payload, path)


def write_smap_msl_classical_baselines_csv(
    runs: tuple[SmapMslClassicalBaselineRun, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL classical baseline runs as a compact metrics table."""
    output_path = prepare_output_path(path)
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


def write_smap_msl_robust_threshold_sweep_json(
    runs: tuple[SmapMslRobustThresholdSweepRun, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL robust threshold sweep runs as JSON."""
    payload = [run.to_dict() for run in runs]
    write_json_payload(payload, path)


def write_smap_msl_robust_threshold_sweep_csv(
    runs: tuple[SmapMslRobustThresholdSweepRun, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL robust threshold sweep runs as a compact metrics table."""
    output_path = prepare_output_path(path)
    fieldnames = [
        "channel_id",
        "spacecraft",
        "threshold",
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
                    "threshold": f"{run.threshold:.12g}",
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


def write_smap_msl_robust_threshold_sweep_aggregate_json(
    aggregates: tuple[SmapMslRobustThresholdSweepAggregate, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL robust threshold sweep aggregate metrics as JSON."""
    payload = [aggregate.to_dict() for aggregate in aggregates]
    write_json_payload(payload, path)


def write_smap_msl_robust_threshold_sweep_aggregate_csv(
    aggregates: tuple[SmapMslRobustThresholdSweepAggregate, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL robust threshold sweep aggregate metrics as CSV."""
    if not aggregates:
        raise ValueError("aggregates must contain at least one item")
    output_path = prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregates[0].to_dict()))
        writer.writeheader()
        writer.writerows(aggregate.to_dict() for aggregate in aggregates)


def write_smap_msl_robust_threshold_operating_points_json(
    operating_points: tuple[SmapMslRobustThresholdOperatingPoint, ...],
    path: Path,
) -> None:
    """Write selected SMAP/MSL robust threshold operating points as JSON."""
    payload = [operating_point.to_dict() for operating_point in operating_points]
    write_json_payload(payload, path)


def write_smap_msl_robust_threshold_operating_points_csv(
    operating_points: tuple[SmapMslRobustThresholdOperatingPoint, ...],
    path: Path,
) -> None:
    """Write selected SMAP/MSL robust threshold operating points as CSV."""
    if not operating_points:
        raise ValueError("operating_points must contain at least one item")
    output_path = prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(operating_points[0].to_dict()))
        writer.writeheader()
        writer.writerows(operating_point.to_dict() for operating_point in operating_points)


def write_smap_msl_robust_threshold_policy_json(
    runs: tuple[SmapMslRobustThresholdSweepRun, ...],
    operating_points: tuple[SmapMslRobustThresholdOperatingPoint, ...],
    path: Path,
) -> None:
    """Write selected robust threshold policy runs as comparison-ready JSON rows."""
    payload = [
        _robust_threshold_policy_row(run, _operating_point_for_run(run, operating_points))
        for run in runs
    ]
    write_json_payload(payload, path)


def write_smap_msl_robust_threshold_policy_csv(
    runs: tuple[SmapMslRobustThresholdSweepRun, ...],
    operating_points: tuple[SmapMslRobustThresholdOperatingPoint, ...],
    path: Path,
) -> None:
    """Write selected robust threshold policy runs as a comparison-ready CSV table."""
    if not runs:
        raise ValueError("runs must contain at least one item")
    output_path = prepare_output_path(path)
    rows = [
        _robust_threshold_policy_row(run, _operating_point_for_run(run, operating_points))
        for run in runs
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_smap_msl_lstm_forecast_baseline_csv(
    runs: tuple[SmapMslLstmForecastBaselineRun, ...],
    path: Path,
) -> None:
    """Write SMAP/MSL LSTM forecast baseline runs as a compact metrics table."""
    output_path = prepare_output_path(path)
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
    available = tuple(
        dict.fromkeys(metadata.channel_id for metadata in read_smap_msl_labels(data_dir))
    )
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


def _robust_threshold_wins_by_channel(
    runs: tuple[SmapMslRobustThresholdSweepRun, ...],
) -> dict[float, int]:
    winners: dict[float, int] = {}
    for channel_id in sorted({run.channel_id for run in runs}):
        channel_runs = [run for run in runs if run.channel_id == channel_id]
        best = min(
            channel_runs,
            key=lambda run: (
                -run.metrics.f1,
                -run.point_adjusted_metrics.f1,
                run.metrics.false_alarm_rate,
                run.metrics.miss_rate,
                run.threshold,
            ),
        )
        winners[best.threshold] = winners.get(best.threshold, 0) + 1
    return winners


def _operating_point_feasible_sort_key(
    aggregate: SmapMslRobustThresholdSweepAggregate,
) -> tuple[float, float, float, float, float]:
    return (
        -aggregate.mean_f1,
        aggregate.mean_false_alarm_rate,
        -aggregate.mean_point_adjusted_f1,
        aggregate.mean_miss_rate,
        aggregate.threshold,
    )


def _operating_point_infeasible_sort_key(
    aggregate: SmapMslRobustThresholdSweepAggregate,
) -> tuple[float, float, float, float, float]:
    return (
        aggregate.mean_false_alarm_rate,
        -aggregate.mean_f1,
        -aggregate.mean_point_adjusted_f1,
        aggregate.mean_miss_rate,
        aggregate.threshold,
    )


def _operating_point_for_run(
    run: SmapMslRobustThresholdSweepRun,
    operating_points: tuple[SmapMslRobustThresholdOperatingPoint, ...],
) -> SmapMslRobustThresholdOperatingPoint:
    for operating_point in operating_points:
        if operating_point.scope == "global" and operating_point.group == "all":
            return operating_point
        if operating_point.scope == "spacecraft" and operating_point.group == run.spacecraft:
            return operating_point
    raise ValueError(f"missing operating point for channel: {run.channel_id}")


def _robust_threshold_policy_row(
    run: SmapMslRobustThresholdSweepRun,
    operating_point: SmapMslRobustThresholdOperatingPoint,
) -> dict[str, object]:
    return {
        "channel_id": run.channel_id,
        "spacecraft": run.spacecraft,
        "model_name": f"robust_zscore_{operating_point.scope}_threshold_policy",
        "threshold_policy_scope": operating_point.scope,
        "threshold_policy_group": operating_point.group,
        "false_alarm_budget": f"{operating_point.false_alarm_budget:.12g}",
        "feasible": operating_point.feasible,
        "selected_threshold": f"{operating_point.selected_threshold:.12g}",
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


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("cannot average an empty sequence")
    return sum(sequence) / len(sequence)
