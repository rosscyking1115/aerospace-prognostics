"""C-MAPSS sequence export and deep-baseline command handlers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
from aerospace_prognostics.evaluation import (
    RegressionRunResult,
    write_results_csv,
    write_results_json,
)
from aerospace_prognostics.sequence_exports import export_cmapss_sequence_splits

CMAPSS_DEEP_COMPARISON_MODELS = ("cnn", "rescnn", "lstm", "bilstm", "tcn", "transformer")
CMAPSS_DEEP_TRAINING_LOSSES = (
    "mse",
    "nasa_surrogate",
    "mse_nasa_blend_w0p001",
    "mse_nasa_blend_w0p0001",
    "asymmetric_mse_late_w1p5",
    "asymmetric_mse_late_w2",
    "asymmetric_mse_late_w3",
    "target_weighted_mse_high_w2",
    "target_weighted_mse_mid_high_w1p5",
    "mse_monotonic_w0p1",
    "asymmetric_mse_late_w1p5_monotonic_w0p1",
    "mse_unit_monotonic_w0p1",
    "asymmetric_mse_late_w1p5_unit_monotonic_w0p1",
)


def register_cmapss_deep_commands(subparsers: Any) -> None:
    sequence_export = subparsers.add_parser(
        "cmapss-export-sequences",
        help="Export C-MAPSS train/validation/test sequence tensors for Phase 2 models",
    )
    sequence_export.add_argument("--data-dir", type=Path, required=True)
    sequence_export.add_argument("--output-dir", type=Path, required=True)
    sequence_export.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    sequence_export.add_argument("--window-size", type=int, default=30)
    sequence_export.add_argument("--stride", type=int, default=1)
    sequence_export.add_argument("--rul-cap", type=int, default=125)
    sequence_export.add_argument("--random-state", type=int, default=42)
    sequence_export.add_argument("--validation-fraction", type=float, default=0.2)
    sequence_export.add_argument("--validation-horizon", type=int, default=30)
    sequence_export.add_argument("--no-standardize", action="store_true")

    cnn_baseline = subparsers.add_parser(
        "cmapss-cnn-baseline",
        help="Train a 1D-CNN RUL baseline from exported C-MAPSS sequence tensors",
    )
    cnn_baseline.add_argument("--sequence-dir", type=Path, required=True)
    cnn_baseline.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    cnn_baseline.add_argument("--epochs", type=int, default=5)
    cnn_baseline.add_argument("--batch-size", type=int, default=256)
    cnn_baseline.add_argument("--learning-rate", type=float, default=1e-3)
    cnn_baseline.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    cnn_baseline.add_argument("--hidden-channels", type=int, default=32)
    cnn_baseline.add_argument("--kernel-size", type=int, default=3)
    cnn_baseline.add_argument("--dropout", type=float, default=0.1)
    cnn_baseline.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    cnn_baseline.add_argument("--random-state", type=int, default=42)
    cnn_baseline.add_argument("--device", default="cpu")
    cnn_baseline.add_argument("--output-json", type=Path)
    cnn_baseline.add_argument("--output-csv", type=Path)
    cnn_baseline.add_argument("--history-json", type=Path)

    lstm_baseline = subparsers.add_parser(
        "cmapss-lstm-baseline",
        help="Train an LSTM/BiLSTM RUL baseline from exported C-MAPSS sequence tensors",
    )
    lstm_baseline.add_argument("--sequence-dir", type=Path, required=True)
    lstm_baseline.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    lstm_baseline.add_argument("--epochs", type=int, default=5)
    lstm_baseline.add_argument("--batch-size", type=int, default=256)
    lstm_baseline.add_argument("--learning-rate", type=float, default=1e-3)
    lstm_baseline.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    lstm_baseline.add_argument("--hidden-size", type=int, default=32)
    lstm_baseline.add_argument("--num-layers", type=int, default=1)
    lstm_baseline.add_argument("--dropout", type=float, default=0.1)
    lstm_baseline.add_argument("--bidirectional", action="store_true")
    lstm_baseline.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    lstm_baseline.add_argument("--random-state", type=int, default=42)
    lstm_baseline.add_argument("--device", default="cpu")
    lstm_baseline.add_argument("--output-json", type=Path)
    lstm_baseline.add_argument("--output-csv", type=Path)
    lstm_baseline.add_argument("--history-json", type=Path)

    tcn_baseline = subparsers.add_parser(
        "cmapss-tcn-baseline",
        help="Train a TCN RUL baseline from exported C-MAPSS sequence tensors",
    )
    tcn_baseline.add_argument("--sequence-dir", type=Path, required=True)
    tcn_baseline.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    tcn_baseline.add_argument("--epochs", type=int, default=5)
    tcn_baseline.add_argument("--batch-size", type=int, default=256)
    tcn_baseline.add_argument("--learning-rate", type=float, default=1e-3)
    tcn_baseline.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    tcn_baseline.add_argument("--hidden-channels", type=int, default=32)
    tcn_baseline.add_argument("--num-levels", type=int, default=3)
    tcn_baseline.add_argument("--kernel-size", type=int, default=3)
    tcn_baseline.add_argument("--dropout", type=float, default=0.1)
    tcn_baseline.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    tcn_baseline.add_argument("--random-state", type=int, default=42)
    tcn_baseline.add_argument("--device", default="cpu")
    tcn_baseline.add_argument("--output-json", type=Path)
    tcn_baseline.add_argument("--output-csv", type=Path)
    tcn_baseline.add_argument("--history-json", type=Path)

    transformer_baseline = subparsers.add_parser(
        "cmapss-transformer-baseline",
        help="Train a Transformer encoder RUL baseline from exported C-MAPSS sequence tensors",
    )
    transformer_baseline.add_argument("--sequence-dir", type=Path, required=True)
    transformer_baseline.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    transformer_baseline.add_argument("--epochs", type=int, default=5)
    transformer_baseline.add_argument("--batch-size", type=int, default=256)
    transformer_baseline.add_argument("--learning-rate", type=float, default=1e-3)
    transformer_baseline.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    transformer_baseline.add_argument("--d-model", type=int, default=32)
    transformer_baseline.add_argument("--num-heads", type=int, default=4)
    transformer_baseline.add_argument("--num-layers", type=int, default=2)
    transformer_baseline.add_argument("--dim-feedforward", type=int, default=64)
    transformer_baseline.add_argument("--dropout", type=float, default=0.1)
    transformer_baseline.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    transformer_baseline.add_argument("--random-state", type=int, default=42)
    transformer_baseline.add_argument("--device", default="cpu")
    transformer_baseline.add_argument("--output-json", type=Path)
    transformer_baseline.add_argument("--output-csv", type=Path)
    transformer_baseline.add_argument("--history-json", type=Path)

    deep_compare = subparsers.add_parser(
        "cmapss-deep-baseline-compare",
        help="Compare Phase 2 deep RUL baselines across compact hyperparameter grids",
    )
    deep_compare.add_argument("--sequence-dir", type=Path, required=True)
    deep_compare.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    deep_compare.add_argument(
        "--models",
        nargs="+",
        choices=CMAPSS_DEEP_COMPARISON_MODELS,
        default=["cnn", "bilstm", "tcn"],
    )
    deep_compare.add_argument("--epochs", type=int, default=5)
    deep_compare.add_argument("--batch-size", type=int, default=256)
    deep_compare.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3])
    deep_compare.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    deep_compare.add_argument("--hidden-sizes", nargs="+", type=int, default=[32])
    deep_compare.add_argument("--num-layers", type=int, default=1)
    deep_compare.add_argument("--tcn-levels", type=int, default=3)
    deep_compare.add_argument(
        "--tcn-normalization",
        choices=["none", "layer_norm"],
        default="none",
    )
    deep_compare.add_argument("--tcn-weight-norm", action="store_true")
    deep_compare.add_argument("--tcn-pooling", choices=["last", "mean"], default="last")
    deep_compare.add_argument("--transformer-heads", type=int, default=4)
    deep_compare.add_argument("--transformer-dim-feedforward", type=int)
    deep_compare.add_argument("--kernel-size", type=int, default=3)
    deep_compare.add_argument("--dropout", type=float, default=0.1)
    deep_compare.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    deep_compare.add_argument("--random-state", type=int, default=42)
    deep_compare.add_argument("--device", default="cpu")
    deep_compare.add_argument("--output-json", type=Path)
    deep_compare.add_argument("--output-csv", type=Path)


def handle_cmapss_deep_command(args: argparse.Namespace) -> int | None:
    if args.command == "cmapss-export-sequences":
        results = [
            export_cmapss_sequence_splits(
                args.data_dir,
                args.output_dir,
                subset,
                window_size=args.window_size,
                stride=args.stride,
                rul_cap=args.rul_cap,
                random_state=args.random_state,
                validation_fraction=args.validation_fraction,
                validation_horizon=args.validation_horizon,
                standardize=not args.no_standardize,
            )
            for subset in args.subsets
        ]
        print(f"window_size={args.window_size}")
        print(f"stride={args.stride}")
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizon={args.validation_horizon}")
        print(f"standardize={not args.no_standardize}")
        for result in results:
            print(
                f"{result.subset},"
                f"train_windows={result.train_windows},"
                f"validation_windows={result.validation_windows},"
                f"validation_selection_windows={result.validation_selection_windows},"
                f"test_windows={result.test_windows},"
                f"features={len(result.feature_columns)},"
                f"metadata={result.metadata_path}"
            )
        return 0

    if args.command == "cmapss-cnn-baseline":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_all_cmapss_cnn_baseline_runs,
        )

        runs = run_all_cmapss_cnn_baseline_runs(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            training_loss=args.training_loss,
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        results = [run.result for run in runs]
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(f"learning_rate={args.learning_rate}")
        print(f"training_loss={args.training_loss}")
        print(f"hidden_channels={args.hidden_channels}")
        print(f"kernel_size={args.kernel_size}")
        print(f"dropout={args.dropout}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        print(
            "selected_epochs="
            + ",".join(f"{run.result.subset}:{run.selected_epoch}" for run in runs)
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        if args.history_json is not None:
            _write_deep_history_json(runs, args.history_json)
        return 0

    if args.command == "cmapss-lstm-baseline":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_all_cmapss_lstm_baseline_runs,
        )

        runs = run_all_cmapss_lstm_baseline_runs(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            training_loss=args.training_loss,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        results = [run.result for run in runs]
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(f"learning_rate={args.learning_rate}")
        print(f"training_loss={args.training_loss}")
        print(f"hidden_size={args.hidden_size}")
        print(f"num_layers={args.num_layers}")
        print(f"dropout={args.dropout}")
        print(f"bidirectional={args.bidirectional}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        print(
            "selected_epochs="
            + ",".join(f"{run.result.subset}:{run.selected_epoch}" for run in runs)
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        if args.history_json is not None:
            _write_deep_history_json(runs, args.history_json)
        return 0

    if args.command == "cmapss-tcn-baseline":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_all_cmapss_tcn_baseline_runs,
        )

        runs = run_all_cmapss_tcn_baseline_runs(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            training_loss=args.training_loss,
            hidden_channels=args.hidden_channels,
            num_levels=args.num_levels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        results = [run.result for run in runs]
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(f"learning_rate={args.learning_rate}")
        print(f"training_loss={args.training_loss}")
        print(f"hidden_channels={args.hidden_channels}")
        print(f"num_levels={args.num_levels}")
        print(f"kernel_size={args.kernel_size}")
        print(f"dropout={args.dropout}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        print(
            "selected_epochs="
            + ",".join(f"{run.result.subset}:{run.selected_epoch}" for run in runs)
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        if args.history_json is not None:
            _write_deep_history_json(runs, args.history_json)
        return 0

    if args.command == "cmapss-transformer-baseline":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_all_cmapss_transformer_baseline_runs,
        )

        runs = run_all_cmapss_transformer_baseline_runs(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            training_loss=args.training_loss,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        results = [run.result for run in runs]
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(f"learning_rate={args.learning_rate}")
        print(f"training_loss={args.training_loss}")
        print(f"d_model={args.d_model}")
        print(f"num_heads={args.num_heads}")
        print(f"num_layers={args.num_layers}")
        print(f"dim_feedforward={args.dim_feedforward}")
        print(f"dropout={args.dropout}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        print(
            "selected_epochs="
            + ",".join(f"{run.result.subset}:{run.selected_epoch}" for run in runs)
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        if args.history_json is not None:
            _write_deep_history_json(runs, args.history_json)
        return 0

    if args.command == "cmapss-deep-baseline-compare":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_cmapss_deep_baseline_comparison,
        )

        results = run_cmapss_deep_baseline_comparison(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            models=tuple(args.models),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rates=tuple(args.learning_rates),
            training_loss=args.training_loss,
            hidden_sizes=tuple(args.hidden_sizes),
            num_layers=args.num_layers,
            tcn_levels=args.tcn_levels,
            tcn_normalization=args.tcn_normalization,
            tcn_weight_norm=args.tcn_weight_norm,
            tcn_pooling=args.tcn_pooling,
            transformer_heads=args.transformer_heads,
            transformer_dim_feedforward=args.transformer_dim_feedforward,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        print(f"models={','.join(args.models)}")
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(
            "learning_rates="
            + ",".join(_format_cli_float(value) for value in args.learning_rates)
        )
        print(f"training_loss={args.training_loss}")
        print(f"hidden_sizes={','.join(str(value) for value in args.hidden_sizes)}")
        print(f"num_layers={args.num_layers}")
        print(f"tcn_levels={args.tcn_levels}")
        print(f"tcn_normalization={args.tcn_normalization}")
        print(f"tcn_weight_norm={args.tcn_weight_norm}")
        print(f"tcn_pooling={args.tcn_pooling}")
        print(f"transformer_heads={args.transformer_heads}")
        print(f"transformer_dim_feedforward={args.transformer_dim_feedforward}")
        print(f"kernel_size={args.kernel_size}")
        print(f"dropout={args.dropout}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        _print_results_table(results)
        print(
            "selected_by_nasa="
            + ",".join(
                f"{subset}:{_best_result_for_subset(results, subset).model_name}"
                for subset in args.subsets
            )
        )
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    return None


def _print_results_table(results: Iterable[RegressionRunResult]) -> None:
    print("subset,model,standardize,rmse,nasa_score")
    for result in results:
        print(
            f"{result.subset},"
            f"{result.model_name},"
            f"{result.standardize},"
            f"{result.rmse:.6f},"
            f"{result.nasa_score:.6f}"
        )


def _best_result_for_subset(
    results: Iterable[RegressionRunResult],
    subset: str,
) -> RegressionRunResult:
    subset_results = [result for result in results if result.subset == subset]
    if not subset_results:
        raise ValueError(f"no results for subset: {subset}")
    return min(subset_results, key=lambda result: result.nasa_score)


def _format_cli_float(value: float) -> str:
    return f"{value:g}"


def _write_deep_history_json(
    results: list[object],
    path: Path,
) -> None:
    output_path = _prepare_output_path(path)
    payload = [result.to_dict() for result in results]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
