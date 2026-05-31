"""Phase 2 SMAP/MSL anomaly-baseline workflow orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from aerospace_prognostics.anomaly.baselines import CLASSICAL_ANOMALY_BASELINE_METHODS
from aerospace_prognostics.anomaly.forecasting import DynamicThresholdConfig
from aerospace_prognostics.experiments.smap_msl_anomaly import (
    SmapMslClassicalBaselineRun,
    SmapMslLstmForecastBaselineRun,
    run_smap_msl_classical_baselines,
    run_smap_msl_lstm_forecast_baseline,
    write_smap_msl_classical_baselines_csv,
    write_smap_msl_classical_baselines_json,
    write_smap_msl_lstm_forecast_baseline_csv,
    write_smap_msl_lstm_forecast_baseline_json,
)
from aerospace_prognostics.reports.anomaly_model_comparison import (
    AnomalyModelComparisonRow,
    build_anomaly_model_comparison,
    write_anomaly_model_comparison_csv,
    write_anomaly_model_comparison_markdown,
)


@dataclass(frozen=True)
class Phase2SmapMslWorkflowResult:
    """Artifact paths and run results from the SMAP/MSL Phase 2 workflow."""

    artifact_dir: Path
    classical_json_path: Path
    classical_csv_path: Path
    lstm_robust_json_path: Path
    lstm_robust_csv_path: Path
    lstm_dynamic_json_path: Path
    lstm_dynamic_csv_path: Path
    comparison_csv_path: Path
    comparison_markdown_path: Path
    summary_markdown_path: Path
    classical_runs: tuple[SmapMslClassicalBaselineRun, ...]
    lstm_robust_runs: tuple[SmapMslLstmForecastBaselineRun, ...]
    lstm_dynamic_runs: tuple[SmapMslLstmForecastBaselineRun, ...]
    comparison_rows: tuple[AnomalyModelComparisonRow, ...]


def run_phase2_smap_msl_workflow(
    data_dir: str | Path,
    artifact_dir: str | Path,
    *,
    channels: tuple[str, ...] | None = None,
    max_channels: int | None = None,
    classical_methods: tuple[str, ...] = CLASSICAL_ANOMALY_BASELINE_METHODS,
    robust_threshold: float = 3.5,
    pca_components: int | None = None,
    pca_threshold_quantile: float = 0.99,
    isolation_contamination: float = 0.05,
    window_size: int = 30,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    threshold_sigma: float = 3.0,
    dynamic_threshold_config: DynamicThresholdConfig | None = None,
    random_state: int = 42,
    device: str = "cpu",
) -> Phase2SmapMslWorkflowResult:
    """Run classical, LSTM forecast, dynamic-threshold, and comparison SMAP/MSL outputs."""

    root = Path(data_dir)
    artifacts = Path(artifact_dir)
    results_dir = artifacts / "results"
    artifacts.mkdir(parents=True, exist_ok=True)

    classical_runs = run_smap_msl_classical_baselines(
        root,
        channels=channels,
        max_channels=max_channels,
        methods=classical_methods,
        robust_threshold=robust_threshold,
        pca_components=pca_components,
        pca_threshold_quantile=pca_threshold_quantile,
        isolation_contamination=isolation_contamination,
        random_state=random_state,
    )
    classical_json_path = results_dir / "smap_msl_classical_baselines.json"
    classical_csv_path = results_dir / "smap_msl_classical_baselines.csv"
    write_smap_msl_classical_baselines_json(classical_runs, classical_json_path)
    write_smap_msl_classical_baselines_csv(classical_runs, classical_csv_path)

    lstm_robust_runs = run_smap_msl_lstm_forecast_baseline(
        root,
        channels=channels,
        max_channels=max_channels,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        threshold_sigma=threshold_sigma,
        threshold_method="robust",
        random_state=random_state,
        device=device,
    )
    lstm_robust_json_path = results_dir / "smap_msl_lstm_forecast_robust.json"
    lstm_robust_csv_path = results_dir / "smap_msl_lstm_forecast_robust.csv"
    write_smap_msl_lstm_forecast_baseline_json(lstm_robust_runs, lstm_robust_json_path)
    write_smap_msl_lstm_forecast_baseline_csv(lstm_robust_runs, lstm_robust_csv_path)

    lstm_dynamic_runs = run_smap_msl_lstm_forecast_baseline(
        root,
        channels=channels,
        max_channels=max_channels,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        threshold_sigma=threshold_sigma,
        threshold_method="dynamic",
        dynamic_threshold_config=dynamic_threshold_config,
        random_state=random_state,
        device=device,
    )
    lstm_dynamic_json_path = results_dir / "smap_msl_lstm_forecast_dynamic.json"
    lstm_dynamic_csv_path = results_dir / "smap_msl_lstm_forecast_dynamic.csv"
    write_smap_msl_lstm_forecast_baseline_json(lstm_dynamic_runs, lstm_dynamic_json_path)
    write_smap_msl_lstm_forecast_baseline_csv(lstm_dynamic_runs, lstm_dynamic_csv_path)

    comparison_rows = build_anomaly_model_comparison(
        (classical_csv_path, lstm_robust_csv_path, lstm_dynamic_csv_path),
        source_labels=("classical", "lstm_robust", "lstm_dynamic"),
    )
    comparison_csv_path = results_dir / "smap_msl_anomaly_model_comparison.csv"
    comparison_markdown_path = results_dir / "smap_msl_anomaly_model_comparison.md"
    write_anomaly_model_comparison_csv(comparison_rows, comparison_csv_path)
    write_anomaly_model_comparison_markdown(comparison_rows, comparison_markdown_path)

    summary_markdown_path = artifacts / "phase2_smap_msl_summary.md"
    _write_phase2_smap_msl_summary(
        summary_markdown_path,
        classical_csv_path=classical_csv_path,
        lstm_robust_csv_path=lstm_robust_csv_path,
        lstm_dynamic_csv_path=lstm_dynamic_csv_path,
        comparison_markdown_path=comparison_markdown_path,
        comparison_rows=tuple(comparison_rows),
    )

    return Phase2SmapMslWorkflowResult(
        artifact_dir=artifacts,
        classical_json_path=classical_json_path,
        classical_csv_path=classical_csv_path,
        lstm_robust_json_path=lstm_robust_json_path,
        lstm_robust_csv_path=lstm_robust_csv_path,
        lstm_dynamic_json_path=lstm_dynamic_json_path,
        lstm_dynamic_csv_path=lstm_dynamic_csv_path,
        comparison_csv_path=comparison_csv_path,
        comparison_markdown_path=comparison_markdown_path,
        summary_markdown_path=summary_markdown_path,
        classical_runs=tuple(classical_runs),
        lstm_robust_runs=tuple(lstm_robust_runs),
        lstm_dynamic_runs=tuple(lstm_dynamic_runs),
        comparison_rows=tuple(comparison_rows),
    )


def _write_phase2_smap_msl_summary(
    path: Path,
    *,
    classical_csv_path: Path,
    lstm_robust_csv_path: Path,
    lstm_dynamic_csv_path: Path,
    comparison_markdown_path: Path,
    comparison_rows: tuple[AnomalyModelComparisonRow, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 SMAP/MSL Summary",
        "",
        f"- Classical baseline table: `{classical_csv_path.as_posix()}`",
        f"- LSTM robust-threshold table: `{lstm_robust_csv_path.as_posix()}`",
        f"- LSTM dynamic-threshold table: `{lstm_dynamic_csv_path.as_posix()}`",
        f"- Ranked anomaly comparison: `{comparison_markdown_path.as_posix()}`",
        "",
        "## Best Model By Point-Wise F1",
        "",
        "| Channel | Spacecraft | Source | Model | F1 | Point-Adjusted F1 | False Alarm Rate |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    best_rows = _best_smap_msl_rows_by_channel(comparison_rows)
    for best in best_rows:
        lines.append(
            f"| {best.channel_id} | {best.spacecraft} | {best.source} | "
            f"{best.model_name} | {best.f1:.6f} | {best.point_adjusted_f1:.6f} | "
            f"{best.false_alarm_rate:.6f} |"
        )
    _append_smap_msl_aggregate_table(
        lines,
        rows=best_rows,
        heading="Winner Counts",
        count_header="Channels Won",
    )
    _append_smap_msl_aggregate_table(
        lines,
        rows=comparison_rows,
        heading="Average Metrics By Source And Model",
        count_header="Rows",
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_smap_msl_rows_by_channel(
    comparison_rows: tuple[AnomalyModelComparisonRow, ...],
) -> tuple[AnomalyModelComparisonRow, ...]:
    best_rows: list[AnomalyModelComparisonRow] = []
    for channel_id in sorted({row.channel_id for row in comparison_rows}):
        best_rows.append(
            min(
                (row for row in comparison_rows if row.channel_id == channel_id),
                key=lambda row: row.rank_by_f1,
            )
        )
    return tuple(best_rows)


def _append_smap_msl_aggregate_table(
    lines: list[str],
    *,
    rows: tuple[AnomalyModelComparisonRow, ...],
    heading: str,
    count_header: str,
) -> None:
    lines.extend(
        [
            "",
            f"## {heading}",
            "",
            (
                f"| Source | Model | {count_header} | Mean F1 | "
                "Mean Point-Adjusted F1 | Mean False Alarm Rate |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for source, model_name, group in _group_smap_msl_rows_by_model(rows):
        mean_f1 = _mean_smap_msl_metric(row.f1 for row in group)
        mean_point_adjusted_f1 = _mean_smap_msl_metric(row.point_adjusted_f1 for row in group)
        mean_false_alarm_rate = _mean_smap_msl_metric(row.false_alarm_rate for row in group)
        lines.append(
            f"| {source} | {model_name} | {len(group)} | {mean_f1:.6f} | "
            f"{mean_point_adjusted_f1:.6f} | {mean_false_alarm_rate:.6f} |"
        )


def _group_smap_msl_rows_by_model(
    rows: tuple[AnomalyModelComparisonRow, ...],
) -> tuple[tuple[str, str, tuple[AnomalyModelComparisonRow, ...]], ...]:
    grouped: dict[tuple[str, str], list[AnomalyModelComparisonRow]] = {}
    for row in rows:
        grouped.setdefault((row.source, row.model_name), []).append(row)
    return tuple(
        (source, model_name, tuple(grouped[(source, model_name)]))
        for source, model_name in sorted(grouped)
    )


def _mean_smap_msl_metric(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("cannot average an empty metric sequence")
    return sum(sequence) / len(sequence)
