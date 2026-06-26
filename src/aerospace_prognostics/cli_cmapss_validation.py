"""C-MAPSS validation and policy command handlers for the project CLI."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
from aerospace_prognostics.evaluation import (
    RegressionRunResult,
    write_results_csv,
    write_results_json,
)
from aerospace_prognostics.experiments.cmapss_baseline import (
    CMAPSS_ENGINEERED_DEFAULT_WINDOWS,
    CMAPSS_HGB_PARAM_GRID,
    CMAPSS_SENSOR_FILTER_CANDIDATES,
    CMAPSS_VALIDATION_SELECTED_FEATURES,
    CMAPSS_VALIDATION_SELECTED_HGB_PARAMS,
    CmapssValidationAggregateResult,
    run_all_cmapss_regime_aware_engineered_default_windows,
    run_all_cmapss_validation_selected_default_windows,
    run_all_cmapss_validation_selected_hgb_policy_default_windows,
    run_cmapss_repeated_validation_feature_comparison,
    run_cmapss_validation_feature_comparison,
    run_cmapss_validation_selected_hgb_grid,
    run_cmapss_validation_sensor_filter_comparison,
)


def register_cmapss_validation_commands(subparsers: Any) -> None:
    regime_engineered = subparsers.add_parser(
        "cmapss-regime-engineered-best-baseline-all",
        help="Train regime-aware engineered baselines with per-subset windows",
    )
    _add_common_official_test_args(regime_engineered)

    validation_selected = subparsers.add_parser(
        "cmapss-validation-selected-baseline-all",
        help="Train official-test baselines using the repeated-validation feature policy",
    )
    _add_common_official_test_args(validation_selected)

    hgb_policy = subparsers.add_parser(
        "cmapss-hgb-policy-baseline-all",
        help="Train official-test baselines using validation-selected features and HGB params",
    )
    _add_common_official_test_args(hgb_policy)

    validation_candidates = subparsers.add_parser(
        "cmapss-validate-feature-candidates",
        help="Compare engineered and regime-aware candidates on temporal validation splits",
    )
    _add_common_validation_args(validation_candidates)

    repeated_validation = subparsers.add_parser(
        "cmapss-validate-feature-candidates-repeated",
        help="Aggregate candidate validation across multiple seeds and horizons",
    )
    repeated_validation.add_argument("--data-dir", type=Path, required=True)
    repeated_validation.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    repeated_validation.add_argument("--rul-cap", type=int, default=125)
    repeated_validation.add_argument("--random-states", nargs="+", type=int, default=[11, 42])
    repeated_validation.add_argument("--n-regimes", type=int, default=6)
    repeated_validation.add_argument("--validation-fraction", type=float, default=0.2)
    repeated_validation.add_argument("--validation-horizons", nargs="+", type=int, default=[20, 30])
    repeated_validation.add_argument("--output-json", type=Path)
    repeated_validation.add_argument("--output-csv", type=Path)
    repeated_validation.add_argument("--no-standardize", action="store_true")

    hgb_grid = subparsers.add_parser(
        "cmapss-validate-hgb-grid",
        help="Validate compact HGB parameter candidates for the current feature policy",
    )
    _add_common_validation_args(hgb_grid)

    sensor_filters = subparsers.add_parser(
        "cmapss-validate-sensor-filters",
        help="Validate full versus EDA-filtered sensor sets for the current policy",
    )
    _add_common_validation_args(sensor_filters)
    sensor_filters.add_argument("--min-abs-rul-correlation", type=float, default=0.05)
    sensor_filters.add_argument("--min-abs-standardized-drift", type=float, default=0.2)


def handle_cmapss_validation_command(args: argparse.Namespace) -> int | None:
    if args.command == "cmapss-regime-engineered-best-baseline-all":
        results = run_all_cmapss_regime_aware_engineered_default_windows(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
        )
        _print_default_windows(args.subsets)
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        _write_regression_outputs(results, args.output_json, args.output_csv)
        return 0

    if args.command == "cmapss-validation-selected-baseline-all":
        results = run_all_cmapss_validation_selected_default_windows(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
        )
        _print_default_windows(args.subsets)
        _print_feature_policy(args.subsets)
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        _write_regression_outputs(results, args.output_json, args.output_csv)
        return 0

    if args.command == "cmapss-hgb-policy-baseline-all":
        results = run_all_cmapss_validation_selected_hgb_policy_default_windows(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
        )
        _print_default_windows(args.subsets)
        _print_feature_policy(args.subsets)
        _print_hgb_policy(args.subsets)
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        _write_regression_outputs(results, args.output_json, args.output_csv)
        return 0

    if args.command == "cmapss-validate-feature-candidates":
        results = run_cmapss_validation_feature_comparison(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            validation_fraction=args.validation_fraction,
            validation_horizon=args.validation_horizon,
            standardize=not args.no_standardize,
        )
        _print_default_windows(args.subsets)
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizon={args.validation_horizon}")
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        _print_selected_by_nasa(results, args.subsets)
        _write_regression_outputs(results, args.output_json, args.output_csv)
        return 0

    if args.command == "cmapss-validate-feature-candidates-repeated":
        results = run_cmapss_repeated_validation_feature_comparison(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_states=tuple(args.random_states),
            n_regimes=args.n_regimes,
            validation_fraction=args.validation_fraction,
            validation_horizons=tuple(args.validation_horizons),
            standardize=not args.no_standardize,
        )
        _print_default_windows(args.subsets)
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizons={','.join(str(value) for value in args.validation_horizons)}")
        print(f"random_states={','.join(str(value) for value in args.random_states)}")
        print(f"max_regimes={args.n_regimes}")
        _print_validation_aggregate_table(results)
        print(
            "selected_by_mean_nasa="
            + ",".join(
                f"{subset}:{_best_aggregate_for_subset(results, subset).model_name}"
                for subset in args.subsets
            )
        )
        if args.output_json is not None:
            _write_validation_aggregate_json(results, args.output_json)
        if args.output_csv is not None:
            _write_validation_aggregate_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-validate-hgb-grid":
        results = run_cmapss_validation_selected_hgb_grid(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            validation_fraction=args.validation_fraction,
            validation_horizon=args.validation_horizon,
            standardize=not args.no_standardize,
        )
        _print_default_windows(args.subsets)
        _print_feature_policy(args.subsets)
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizon={args.validation_horizon}")
        print(f"param_grid={','.join(str(params['label']) for params in CMAPSS_HGB_PARAM_GRID)}")
        _print_results_table(results)
        _print_selected_by_nasa(results, args.subsets)
        _write_regression_outputs(results, args.output_json, args.output_csv)
        return 0

    if args.command == "cmapss-validate-sensor-filters":
        results = run_cmapss_validation_sensor_filter_comparison(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            validation_fraction=args.validation_fraction,
            validation_horizon=args.validation_horizon,
            min_abs_rul_correlation=args.min_abs_rul_correlation,
            min_abs_standardized_drift=args.min_abs_standardized_drift,
            standardize=not args.no_standardize,
        )
        _print_default_windows(args.subsets)
        _print_feature_policy(args.subsets)
        _print_hgb_policy(args.subsets)
        print(f"sensor_filter_candidates={','.join(CMAPSS_SENSOR_FILTER_CANDIDATES)}")
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizon={args.validation_horizon}")
        print(f"min_abs_rul_correlation={args.min_abs_rul_correlation}")
        print(f"min_abs_standardized_drift={args.min_abs_standardized_drift}")
        _print_results_table(results)
        _print_selected_by_nasa(results, args.subsets)
        _write_regression_outputs(results, args.output_json, args.output_csv)
        return 0

    return None


def _add_common_official_test_args(command: Any) -> None:
    command.add_argument("--data-dir", type=Path, required=True)
    command.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    command.add_argument("--rul-cap", type=int, default=125)
    command.add_argument("--random-state", type=int, default=42)
    command.add_argument("--n-regimes", type=int, default=6)
    command.add_argument("--output-json", type=Path)
    command.add_argument("--output-csv", type=Path)
    command.add_argument("--no-standardize", action="store_true")


def _add_common_validation_args(command: Any) -> None:
    command.add_argument("--data-dir", type=Path, required=True)
    command.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    command.add_argument("--rul-cap", type=int, default=125)
    command.add_argument("--random-state", type=int, default=42)
    command.add_argument("--n-regimes", type=int, default=6)
    command.add_argument("--validation-fraction", type=float, default=0.2)
    command.add_argument("--validation-horizon", type=int, default=30)
    command.add_argument("--output-json", type=Path)
    command.add_argument("--output-csv", type=Path)
    command.add_argument("--no-standardize", action="store_true")


def _print_default_windows(subsets: Iterable[str]) -> None:
    print(
        "rolling_windows="
        + ",".join(f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}" for subset in subsets)
    )


def _print_feature_policy(subsets: Iterable[str]) -> None:
    print(
        "feature_policy="
        + ",".join(f"{subset}:{CMAPSS_VALIDATION_SELECTED_FEATURES[subset]}" for subset in subsets)
    )


def _print_hgb_policy(subsets: Iterable[str]) -> None:
    print(
        "hgb_policy="
        + ",".join(
            f"{subset}:{CMAPSS_VALIDATION_SELECTED_HGB_PARAMS[subset]}" for subset in subsets
        )
    )


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


def _print_validation_aggregate_table(
    results: Iterable[CmapssValidationAggregateResult],
) -> None:
    print("subset,model,standardize,runs,wins_by_nasa,mean_rmse,mean_nasa_score")
    for result in results:
        print(
            f"{result.subset},"
            f"{result.model_name},"
            f"{result.standardize},"
            f"{result.runs},"
            f"{result.wins_by_nasa},"
            f"{result.mean_rmse:.6f},"
            f"{result.mean_nasa_score:.6f}"
        )


def _print_selected_by_nasa(results: Iterable[RegressionRunResult], subsets: Iterable[str]) -> None:
    print(
        "selected_by_nasa="
        + ",".join(
            f"{subset}:{_best_result_for_subset(results, subset).model_name}" for subset in subsets
        )
    )


def _best_result_for_subset(
    results: Iterable[RegressionRunResult],
    subset: str,
) -> RegressionRunResult:
    subset_results = [result for result in results if result.subset == subset]
    if not subset_results:
        raise ValueError(f"no results for subset: {subset}")
    return min(subset_results, key=lambda result: result.nasa_score)


def _best_aggregate_for_subset(
    results: Iterable[CmapssValidationAggregateResult],
    subset: str,
) -> CmapssValidationAggregateResult:
    subset_results = [result for result in results if result.subset == subset]
    if not subset_results:
        raise ValueError(f"no aggregate results for subset: {subset}")
    return min(subset_results, key=lambda result: result.mean_nasa_score)


def _write_regression_outputs(
    results: list[RegressionRunResult],
    output_json: Path | None,
    output_csv: Path | None,
) -> None:
    if output_json is not None:
        write_results_json(results, output_json)
    if output_csv is not None:
        write_results_csv(results, output_csv)


def _write_validation_aggregate_json(
    results: list[CmapssValidationAggregateResult],
    path: Path,
) -> None:
    payload = [result.to_dict() for result in results]
    write_json_payload(payload, path)


def _write_validation_aggregate_csv(
    results: list[CmapssValidationAggregateResult],
    path: Path,
) -> None:
    if not results:
        raise ValueError("results must contain at least one item")
    output_path = prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].to_dict()))
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)
