"""C-MAPSS classical baseline command handlers for the project CLI."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
from aerospace_prognostics.evaluation import (
    RegressionRunResult,
    write_results_csv,
    write_results_json,
)
from aerospace_prognostics.experiments.cmapss_baseline import (
    CMAPSS_ENGINEERED_DEFAULT_WINDOWS,
    run_all_cmapss_engineered_default_windows,
    run_all_cmapss_engineered_hist_gradient_boosting,
    run_all_cmapss_hist_gradient_boosting,
    run_cmapss_engineered_hist_gradient_boosting,
    run_cmapss_engineered_window_sweep,
    run_cmapss_hist_gradient_boosting,
)


def register_cmapss_baseline_commands(subparsers: Any) -> None:
    baseline = subparsers.add_parser(
        "cmapss-baseline",
        help="Train a first-pass C-MAPSS gradient-boosting baseline",
    )
    baseline.add_argument("--data-dir", type=Path, required=True)
    baseline.add_argument("--subset", choices=CMAPSS_SUBSETS, required=True)
    baseline.add_argument("--rul-cap", type=int, default=125)
    baseline.add_argument("--random-state", type=int, default=42)
    baseline.add_argument("--output-json", type=Path)
    baseline.add_argument("--standardize", action="store_true")

    baseline_all = subparsers.add_parser(
        "cmapss-baseline-all",
        help="Train the first-pass C-MAPSS baseline on multiple subsets",
    )
    baseline_all.add_argument("--data-dir", type=Path, required=True)
    baseline_all.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    baseline_all.add_argument("--rul-cap", type=int, default=125)
    baseline_all.add_argument("--random-state", type=int, default=42)
    baseline_all.add_argument("--output-json", type=Path)
    baseline_all.add_argument("--output-csv", type=Path)
    baseline_all.add_argument("--standardize", action="store_true")

    engineered = subparsers.add_parser(
        "cmapss-engineered-baseline",
        help="Train a feature-engineered C-MAPSS gradient-boosting baseline",
    )
    engineered.add_argument("--data-dir", type=Path, required=True)
    engineered.add_argument("--subset", choices=CMAPSS_SUBSETS, required=True)
    engineered.add_argument("--rul-cap", type=int, default=125)
    engineered.add_argument("--random-state", type=int, default=42)
    engineered.add_argument("--rolling-window", type=int, default=5)
    engineered.add_argument("--output-json", type=Path)
    engineered.add_argument("--no-standardize", action="store_true")

    engineered_all = subparsers.add_parser(
        "cmapss-engineered-baseline-all",
        help="Train the feature-engineered C-MAPSS baseline on multiple subsets",
    )
    engineered_all.add_argument("--data-dir", type=Path, required=True)
    engineered_all.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    engineered_all.add_argument("--rul-cap", type=int, default=125)
    engineered_all.add_argument("--random-state", type=int, default=42)
    engineered_all.add_argument("--rolling-window", type=int, default=5)
    engineered_all.add_argument("--output-json", type=Path)
    engineered_all.add_argument("--output-csv", type=Path)
    engineered_all.add_argument("--no-standardize", action="store_true")

    engineered_sweep = subparsers.add_parser(
        "cmapss-engineered-window-sweep",
        help="Compare engineered C-MAPSS baselines across rolling-window sizes",
    )
    engineered_sweep.add_argument("--data-dir", type=Path, required=True)
    engineered_sweep.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    engineered_sweep.add_argument("--rolling-windows", nargs="+", type=int, default=[3, 5, 10])
    engineered_sweep.add_argument("--rul-cap", type=int, default=125)
    engineered_sweep.add_argument("--random-state", type=int, default=42)
    engineered_sweep.add_argument("--output-json", type=Path)
    engineered_sweep.add_argument("--output-csv", type=Path)
    engineered_sweep.add_argument("--no-standardize", action="store_true")

    engineered_best = subparsers.add_parser(
        "cmapss-engineered-best-baseline-all",
        help="Train engineered baselines using current per-subset rolling-window defaults",
    )
    engineered_best.add_argument("--data-dir", type=Path, required=True)
    engineered_best.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    engineered_best.add_argument("--rul-cap", type=int, default=125)
    engineered_best.add_argument("--random-state", type=int, default=42)
    engineered_best.add_argument("--output-json", type=Path)
    engineered_best.add_argument("--output-csv", type=Path)
    engineered_best.add_argument("--no-standardize", action="store_true")


def handle_cmapss_baseline_command(args: argparse.Namespace) -> int | None:
    if args.command == "cmapss-baseline":
        result = run_cmapss_hist_gradient_boosting(
            args.data_dir,
            args.subset,
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=args.standardize,
        )
        _print_single_result(result)
        if args.output_json is not None:
            result.write_json(args.output_json)
        return 0

    if args.command == "cmapss-baseline-all":
        results = run_all_cmapss_hist_gradient_boosting(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=args.standardize,
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-engineered-baseline":
        result = run_cmapss_engineered_hist_gradient_boosting(
            args.data_dir,
            args.subset,
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            rolling_window=args.rolling_window,
            standardize=not args.no_standardize,
        )
        _print_single_result(result)
        if args.output_json is not None:
            result.write_json(args.output_json)
        return 0

    if args.command == "cmapss-engineered-baseline-all":
        results = run_all_cmapss_engineered_hist_gradient_boosting(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            rolling_window=args.rolling_window,
            standardize=not args.no_standardize,
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-engineered-window-sweep":
        results = run_cmapss_engineered_window_sweep(
            args.data_dir,
            subsets=tuple(args.subsets),
            rolling_windows=tuple(args.rolling_windows),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=not args.no_standardize,
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-engineered-best-baseline-all":
        results = run_all_cmapss_engineered_default_windows(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=not args.no_standardize,
        )
        _print_default_windows(args.subsets)
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    return None


def _print_single_result(result: RegressionRunResult) -> None:
    print(f"dataset={result.dataset}")
    print(f"subset={result.subset}")
    print(f"model={result.model_name}")
    print(f"standardize={result.standardize}")
    print(f"rmse={result.rmse:.6f}")
    print(f"nasa_score={result.nasa_score:.6f}")


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


def _print_default_windows(subsets: Iterable[str]) -> None:
    print(
        "rolling_windows="
        + ",".join(f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}" for subset in subsets)
    )
