"""Command-line utilities for local dataset checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from aerospace_prognostics.anomaly.baselines import CLASSICAL_ANOMALY_BASELINE_METHODS
from aerospace_prognostics.cli_anomaly import handle_anomaly_command, register_anomaly_commands
from aerospace_prognostics.cli_app import handle_app_command, register_app_commands
from aerospace_prognostics.cli_cmapss_baselines import (
    handle_cmapss_baseline_command,
    register_cmapss_baseline_commands,
)
from aerospace_prognostics.cli_cmapss_data import (
    handle_cmapss_data_command,
    register_cmapss_data_commands,
)
from aerospace_prognostics.cli_cmapss_deep import (
    CMAPSS_DEEP_COMPARISON_MODELS,
    CMAPSS_DEEP_TRAINING_LOSSES,
    handle_cmapss_deep_command,
    register_cmapss_deep_commands,
)
from aerospace_prognostics.cli_cmapss_validation import (
    handle_cmapss_validation_command,
    register_cmapss_validation_commands,
)
from aerospace_prognostics.cli_deployment import (
    handle_deployment_command,
    register_deployment_artifact_commands,
    register_deployment_release_commands,
)
from aerospace_prognostics.cli_reports import (
    handle_report_command,
    register_cmapss_report_commands,
    register_dashboard_commands,
    register_smap_msl_report_commands,
)
from aerospace_prognostics.cli_smap_msl import (
    handle_smap_msl_data_command,
    register_smap_msl_data_commands,
    register_smap_msl_download_command,
)
from aerospace_prognostics.cli_workflows import handle_workflow_command, register_workflow_commands
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
from aerospace_prognostics.experiments.smap_msl_anomaly import (
    LSTM_FORECAST_THRESHOLD_METHODS,
    SmapMslClassicalBaselineRun,
    SmapMslLstmForecastBaselineRun,
    SmapMslRobustThresholdOperatingPoint,
    SmapMslRobustThresholdSweepAggregate,
    SmapMslRobustThresholdSweepRun,
    aggregate_smap_msl_robust_threshold_sweep,
    run_smap_msl_classical_baselines,
    run_smap_msl_lstm_forecast_baseline,
    run_smap_msl_robust_threshold_sweep,
    select_smap_msl_robust_threshold_operating_points,
    select_smap_msl_robust_threshold_policy_runs,
    write_smap_msl_classical_baselines_csv,
    write_smap_msl_classical_baselines_json,
    write_smap_msl_lstm_forecast_baseline_csv,
    write_smap_msl_lstm_forecast_baseline_json,
    write_smap_msl_robust_threshold_operating_points_csv,
    write_smap_msl_robust_threshold_operating_points_json,
    write_smap_msl_robust_threshold_policy_csv,
    write_smap_msl_robust_threshold_policy_json,
    write_smap_msl_robust_threshold_sweep_aggregate_csv,
    write_smap_msl_robust_threshold_sweep_aggregate_json,
    write_smap_msl_robust_threshold_sweep_csv,
    write_smap_msl_robust_threshold_sweep_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aerospace-prognostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_app_commands(subparsers)
    register_cmapss_data_commands(subparsers)
    register_cmapss_baseline_commands(subparsers)
    register_cmapss_validation_commands(subparsers)

    register_smap_msl_download_command(subparsers)
    register_workflow_commands(
        subparsers,
        cmapss_subsets=CMAPSS_SUBSETS,
        cmapss_deep_models=CMAPSS_DEEP_COMPARISON_MODELS,
        cmapss_training_losses=CMAPSS_DEEP_TRAINING_LOSSES,
        classical_anomaly_methods=CLASSICAL_ANOMALY_BASELINE_METHODS,
    )
    register_cmapss_deep_commands(subparsers)

    register_cmapss_report_commands(subparsers)
    register_anomaly_commands(subparsers)

    register_smap_msl_data_commands(subparsers)

    smap_msl_classical = subparsers.add_parser(
        "smap-msl-classical-baselines",
        help="Run classical anomaly baselines across raw Telemanom SMAP/MSL channels",
    )
    smap_msl_classical.add_argument("--data-dir", type=Path, required=True)
    smap_msl_classical.add_argument("--channels", nargs="+")
    smap_msl_classical.add_argument("--max-channels", type=int)
    smap_msl_classical.add_argument(
        "--methods",
        nargs="+",
        choices=CLASSICAL_ANOMALY_BASELINE_METHODS,
        default=list(CLASSICAL_ANOMALY_BASELINE_METHODS),
    )
    smap_msl_classical.add_argument("--robust-threshold", type=float, default=3.5)
    smap_msl_classical.add_argument("--pca-components", type=int)
    smap_msl_classical.add_argument("--pca-threshold-quantile", type=float, default=0.99)
    smap_msl_classical.add_argument("--isolation-contamination", type=float, default=0.05)
    smap_msl_classical.add_argument("--random-state", type=int, default=42)
    smap_msl_classical.add_argument("--output-json", type=Path)
    smap_msl_classical.add_argument("--output-csv", type=Path)

    smap_msl_robust_sweep = subparsers.add_parser(
        "smap-msl-robust-threshold-sweep",
        help="Sweep robust z-score thresholds across raw Telemanom SMAP/MSL channels",
    )
    smap_msl_robust_sweep.add_argument("--data-dir", type=Path, required=True)
    smap_msl_robust_sweep.add_argument("--channels", nargs="+")
    smap_msl_robust_sweep.add_argument("--max-channels", type=int)
    smap_msl_robust_sweep.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[3.5, 5.0, 7.0, 10.0],
    )
    smap_msl_robust_sweep.add_argument("--output-json", type=Path)
    smap_msl_robust_sweep.add_argument("--output-csv", type=Path)
    smap_msl_robust_sweep.add_argument("--aggregate-json", type=Path)
    smap_msl_robust_sweep.add_argument("--aggregate-csv", type=Path)
    smap_msl_robust_sweep.add_argument("--false-alarm-budget", type=float)
    smap_msl_robust_sweep.add_argument(
        "--selection-group",
        choices=["spacecraft", "global"],
        default="spacecraft",
    )
    smap_msl_robust_sweep.add_argument("--operating-point-json", type=Path)
    smap_msl_robust_sweep.add_argument("--operating-point-csv", type=Path)
    smap_msl_robust_sweep.add_argument("--policy-json", type=Path)
    smap_msl_robust_sweep.add_argument("--policy-csv", type=Path)

    smap_msl_lstm = subparsers.add_parser(
        "smap-msl-lstm-forecast-baseline",
        help="Run an LSTM one-step forecast anomaly baseline across SMAP/MSL channels",
    )
    smap_msl_lstm.add_argument("--data-dir", type=Path, required=True)
    smap_msl_lstm.add_argument("--channels", nargs="+")
    smap_msl_lstm.add_argument("--max-channels", type=int)
    smap_msl_lstm.add_argument("--window-size", type=int, default=30)
    smap_msl_lstm.add_argument("--hidden-size", type=int, default=32)
    smap_msl_lstm.add_argument("--num-layers", type=int, default=1)
    smap_msl_lstm.add_argument("--dropout", type=float, default=0.0)
    smap_msl_lstm.add_argument("--epochs", type=int, default=10)
    smap_msl_lstm.add_argument("--batch-size", type=int, default=64)
    smap_msl_lstm.add_argument("--learning-rate", type=float, default=1e-3)
    smap_msl_lstm.add_argument("--threshold-sigma", type=float, default=3.0)
    smap_msl_lstm.add_argument(
        "--threshold-method",
        choices=LSTM_FORECAST_THRESHOLD_METHODS,
        default="robust",
    )
    smap_msl_lstm.add_argument("--dynamic-batch-size", type=int, default=70)
    smap_msl_lstm.add_argument("--dynamic-window-batches", type=int, default=30)
    smap_msl_lstm.add_argument("--dynamic-smoothing-fraction", type=float, default=0.05)
    smap_msl_lstm.add_argument("--dynamic-z-start", type=float, default=2.5)
    smap_msl_lstm.add_argument("--dynamic-z-stop", type=float, default=12.0)
    smap_msl_lstm.add_argument("--dynamic-z-step", type=float, default=0.5)
    smap_msl_lstm.add_argument("--dynamic-error-buffer", type=int, default=100)
    smap_msl_lstm.add_argument("--dynamic-prune-p", type=float, default=0.13)
    smap_msl_lstm.add_argument("--random-state", type=int, default=42)
    smap_msl_lstm.add_argument("--device", default="cpu")
    smap_msl_lstm.add_argument("--output-json", type=Path)
    smap_msl_lstm.add_argument("--output-csv", type=Path)

    register_smap_msl_report_commands(subparsers)

    register_deployment_artifact_commands(subparsers)

    register_dashboard_commands(subparsers)

    register_deployment_release_commands(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    workflow_result = handle_workflow_command(args, runner=main)
    if workflow_result is not None:
        return workflow_result

    app_result = handle_app_command(args)
    if app_result is not None:
        return app_result

    deployment_result = handle_deployment_command(args)
    if deployment_result is not None:
        return deployment_result

    report_result = handle_report_command(args)
    if report_result is not None:
        return report_result

    smap_msl_data_result = handle_smap_msl_data_command(args)
    if smap_msl_data_result is not None:
        return smap_msl_data_result

    cmapss_data_result = handle_cmapss_data_command(args)
    if cmapss_data_result is not None:
        return cmapss_data_result

    cmapss_baseline_result = handle_cmapss_baseline_command(args)
    if cmapss_baseline_result is not None:
        return cmapss_baseline_result

    cmapss_validation_result = handle_cmapss_validation_command(args)
    if cmapss_validation_result is not None:
        return cmapss_validation_result

    anomaly_result = handle_anomaly_command(args)
    if anomaly_result is not None:
        return anomaly_result

    cmapss_deep_result = handle_cmapss_deep_command(args)
    if cmapss_deep_result is not None:
        return cmapss_deep_result

    if args.command == "smap-msl-classical-baselines":
        runs = run_smap_msl_classical_baselines(
            args.data_dir,
            channels=tuple(args.channels) if args.channels is not None else None,
            max_channels=args.max_channels,
            methods=tuple(args.methods),
            robust_threshold=args.robust_threshold,
            pca_components=args.pca_components,
            pca_threshold_quantile=args.pca_threshold_quantile,
            isolation_contamination=args.isolation_contamination,
            random_state=args.random_state,
        )
        print(f"channels={len(_smap_msl_result_channels(runs))}")
        print(f"runs={len(runs)}")
        _print_smap_msl_classical_table(runs)
        if args.output_json is not None:
            write_smap_msl_classical_baselines_json(runs, args.output_json)
        if args.output_csv is not None:
            write_smap_msl_classical_baselines_csv(runs, args.output_csv)
        return 0

    if args.command == "smap-msl-robust-threshold-sweep":
        runs = run_smap_msl_robust_threshold_sweep(
            args.data_dir,
            thresholds=tuple(args.thresholds),
            channels=tuple(args.channels) if args.channels is not None else None,
            max_channels=args.max_channels,
        )
        aggregates = aggregate_smap_msl_robust_threshold_sweep(runs)
        print(f"channels={len(_smap_msl_robust_sweep_channels(runs))}")
        print(f"thresholds={','.join(_format_cli_float(value) for value in args.thresholds)}")
        print(f"runs={len(runs)}")
        _print_smap_msl_robust_threshold_aggregate_table(aggregates)
        operating_point_outputs_requested = (
            args.operating_point_json is not None
            or args.operating_point_csv is not None
            or args.policy_json is not None
            or args.policy_csv is not None
        )
        if args.false_alarm_budget is not None or operating_point_outputs_requested:
            if args.false_alarm_budget is None:
                raise ValueError(
                    "--false-alarm-budget is required when writing threshold-policy outputs"
                )
            operating_points = select_smap_msl_robust_threshold_operating_points(
                runs,
                false_alarm_budget=args.false_alarm_budget,
                group_by=args.selection_group,
            )
            print(f"false_alarm_budget={_format_cli_float(args.false_alarm_budget)}")
            print(f"selection_group={args.selection_group}")
            _print_smap_msl_robust_threshold_operating_point_table(operating_points)
            policy_runs = select_smap_msl_robust_threshold_policy_runs(
                runs,
                operating_points,
            )
            print(f"policy_runs={len(policy_runs)}")
            if args.operating_point_json is not None:
                write_smap_msl_robust_threshold_operating_points_json(
                    operating_points,
                    args.operating_point_json,
                )
            if args.operating_point_csv is not None:
                write_smap_msl_robust_threshold_operating_points_csv(
                    operating_points,
                    args.operating_point_csv,
                )
            if args.policy_json is not None:
                write_smap_msl_robust_threshold_policy_json(
                    policy_runs,
                    operating_points,
                    args.policy_json,
                )
            if args.policy_csv is not None:
                write_smap_msl_robust_threshold_policy_csv(
                    policy_runs,
                    operating_points,
                    args.policy_csv,
                )
        if args.output_json is not None:
            write_smap_msl_robust_threshold_sweep_json(runs, args.output_json)
        if args.output_csv is not None:
            write_smap_msl_robust_threshold_sweep_csv(runs, args.output_csv)
        if args.aggregate_json is not None:
            write_smap_msl_robust_threshold_sweep_aggregate_json(
                aggregates,
                args.aggregate_json,
            )
        if args.aggregate_csv is not None:
            write_smap_msl_robust_threshold_sweep_aggregate_csv(
                aggregates,
                args.aggregate_csv,
            )
        return 0

    if args.command == "smap-msl-lstm-forecast-baseline":
        from aerospace_prognostics.anomaly.forecasting import DynamicThresholdConfig

        runs = run_smap_msl_lstm_forecast_baseline(
            args.data_dir,
            channels=tuple(args.channels) if args.channels is not None else None,
            max_channels=args.max_channels,
            window_size=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold_sigma=args.threshold_sigma,
            threshold_method=args.threshold_method,
            dynamic_threshold_config=DynamicThresholdConfig(
                batch_size=args.dynamic_batch_size,
                window_batches=args.dynamic_window_batches,
                smoothing_fraction=args.dynamic_smoothing_fraction,
                z_start=args.dynamic_z_start,
                z_stop=args.dynamic_z_stop,
                z_step=args.dynamic_z_step,
                error_buffer=args.dynamic_error_buffer,
                p=args.dynamic_prune_p,
            ),
            random_state=args.random_state,
            device=args.device,
        )
        print(f"channels={len(_smap_msl_result_channels(runs))}")
        print(f"runs={len(runs)}")
        _print_smap_msl_forecast_table(runs)
        if args.output_json is not None:
            write_smap_msl_lstm_forecast_baseline_json(runs, args.output_json)
        if args.output_csv is not None:
            write_smap_msl_lstm_forecast_baseline_csv(runs, args.output_csv)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _print_smap_msl_classical_table(runs: Iterable[SmapMslClassicalBaselineRun]) -> None:
    print("channel_id,spacecraft,model,precision,recall,f1,point_adjusted_f1,false_alarm_rate")
    for run in runs:
        print(
            f"{run.channel_id},"
            f"{run.spacecraft},"
            f"{run.model_name},"
            f"{run.metrics.precision:.6f},"
            f"{run.metrics.recall:.6f},"
            f"{run.metrics.f1:.6f},"
            f"{run.point_adjusted_metrics.f1:.6f},"
            f"{run.metrics.false_alarm_rate:.6f}"
        )


def _print_smap_msl_forecast_table(runs: Iterable[SmapMslLstmForecastBaselineRun]) -> None:
    print(
        "channel_id,spacecraft,model,epochs,final_train_loss,"
        "precision,recall,f1,point_adjusted_f1,false_alarm_rate"
    )
    for run in runs:
        final_train_loss = run.history[-1].train_loss if run.history else 0.0
        print(
            f"{run.channel_id},"
            f"{run.spacecraft},"
            f"{run.model_name},"
            f"{len(run.history)},"
            f"{final_train_loss:.6f},"
            f"{run.metrics.precision:.6f},"
            f"{run.metrics.recall:.6f},"
            f"{run.metrics.f1:.6f},"
            f"{run.point_adjusted_metrics.f1:.6f},"
            f"{run.metrics.false_alarm_rate:.6f}"
        )


def _print_smap_msl_robust_threshold_aggregate_table(
    aggregates: Iterable[SmapMslRobustThresholdSweepAggregate],
) -> None:
    print(
        "threshold,channels,wins_by_f1,mean_precision,mean_recall,mean_f1,"
        "mean_point_adjusted_f1,mean_false_alarm_rate,mean_miss_rate"
    )
    for aggregate in aggregates:
        print(
            f"{aggregate.threshold:g},"
            f"{aggregate.channels},"
            f"{aggregate.wins_by_f1},"
            f"{aggregate.mean_precision:.6f},"
            f"{aggregate.mean_recall:.6f},"
            f"{aggregate.mean_f1:.6f},"
            f"{aggregate.mean_point_adjusted_f1:.6f},"
            f"{aggregate.mean_false_alarm_rate:.6f},"
            f"{aggregate.mean_miss_rate:.6f}"
        )


def _print_smap_msl_robust_threshold_operating_point_table(
    operating_points: Iterable[SmapMslRobustThresholdOperatingPoint],
) -> None:
    print(
        "scope,group,false_alarm_budget,selected_threshold,feasible,channels,"
        "mean_precision,mean_recall,mean_f1,mean_point_adjusted_f1,"
        "mean_false_alarm_rate,mean_miss_rate"
    )
    for operating_point in operating_points:
        print(
            f"{operating_point.scope},"
            f"{operating_point.group},"
            f"{operating_point.false_alarm_budget:g},"
            f"{operating_point.selected_threshold:g},"
            f"{operating_point.feasible},"
            f"{operating_point.channels},"
            f"{operating_point.mean_precision:.6f},"
            f"{operating_point.mean_recall:.6f},"
            f"{operating_point.mean_f1:.6f},"
            f"{operating_point.mean_point_adjusted_f1:.6f},"
            f"{operating_point.mean_false_alarm_rate:.6f},"
            f"{operating_point.mean_miss_rate:.6f}"
        )


def _smap_msl_result_channels(
    runs: Iterable[SmapMslClassicalBaselineRun | SmapMslLstmForecastBaselineRun],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(run.channel_id for run in runs))


def _smap_msl_robust_sweep_channels(
    runs: Iterable[SmapMslRobustThresholdSweepRun],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(run.channel_id for run in runs))


def _format_cli_float(value: float) -> str:
    return f"{value:g}"


def _write_json_payload(payload: object, path: Path) -> None:
    output_path = _prepare_output_path(path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
