"""Command-line utilities for local dataset checks."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

from aerospace_prognostics.analysis.cmapss_eda import build_cmapss_eda_report
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS, load_cmapss_subset
from aerospace_prognostics.data.downloads import NASA_CMAPSS_URL, download_cmapss_dataset
from aerospace_prognostics.data.manifest import (
    build_cmapss_manifest,
    read_manifest,
    verify_manifest,
)
from aerospace_prognostics.data.summary import summarise_cmapss_frame
from aerospace_prognostics.evaluation import (
    RegressionRunResult,
    write_results_csv,
    write_results_json,
)
from aerospace_prognostics.experiments.cmapss_baseline import (
    CMAPSS_ENGINEERED_DEFAULT_WINDOWS,
    CMAPSS_VALIDATION_SELECTED_FEATURES,
    CmapssValidationAggregateResult,
    run_all_cmapss_engineered_default_windows,
    run_all_cmapss_engineered_hist_gradient_boosting,
    run_all_cmapss_hist_gradient_boosting,
    run_all_cmapss_regime_aware_engineered_default_windows,
    run_all_cmapss_validation_selected_default_windows,
    run_cmapss_engineered_hist_gradient_boosting,
    run_cmapss_engineered_window_sweep,
    run_cmapss_hist_gradient_boosting,
    run_cmapss_repeated_validation_feature_comparison,
    run_cmapss_validation_feature_comparison,
)
from aerospace_prognostics.workflows.phase1 import run_phase1_cmapss_workflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aerospace-prognostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("cmapss-summary", help="Summarise a local C-MAPSS subset")
    summary.add_argument("--data-dir", type=Path, required=True)
    summary.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], required=True)
    summary.add_argument("--rul-cap", type=int, default=125)

    baseline = subparsers.add_parser(
        "cmapss-baseline",
        help="Train a first-pass C-MAPSS gradient-boosting baseline",
    )
    baseline.add_argument("--data-dir", type=Path, required=True)
    baseline.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], required=True)
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

    regime_engineered = subparsers.add_parser(
        "cmapss-regime-engineered-best-baseline-all",
        help="Train regime-aware engineered baselines with per-subset windows",
    )
    regime_engineered.add_argument("--data-dir", type=Path, required=True)
    regime_engineered.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    regime_engineered.add_argument("--rul-cap", type=int, default=125)
    regime_engineered.add_argument("--random-state", type=int, default=42)
    regime_engineered.add_argument("--n-regimes", type=int, default=6)
    regime_engineered.add_argument("--output-json", type=Path)
    regime_engineered.add_argument("--output-csv", type=Path)
    regime_engineered.add_argument("--no-standardize", action="store_true")

    validation_selected = subparsers.add_parser(
        "cmapss-validation-selected-baseline-all",
        help="Train official-test baselines using the repeated-validation feature policy",
    )
    validation_selected.add_argument("--data-dir", type=Path, required=True)
    validation_selected.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    validation_selected.add_argument("--rul-cap", type=int, default=125)
    validation_selected.add_argument("--random-state", type=int, default=42)
    validation_selected.add_argument("--n-regimes", type=int, default=6)
    validation_selected.add_argument("--output-json", type=Path)
    validation_selected.add_argument("--output-csv", type=Path)
    validation_selected.add_argument("--no-standardize", action="store_true")

    validation_candidates = subparsers.add_parser(
        "cmapss-validate-feature-candidates",
        help="Compare engineered and regime-aware candidates on temporal validation splits",
    )
    validation_candidates.add_argument("--data-dir", type=Path, required=True)
    validation_candidates.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    validation_candidates.add_argument("--rul-cap", type=int, default=125)
    validation_candidates.add_argument("--random-state", type=int, default=42)
    validation_candidates.add_argument("--n-regimes", type=int, default=6)
    validation_candidates.add_argument("--validation-fraction", type=float, default=0.2)
    validation_candidates.add_argument("--validation-horizon", type=int, default=30)
    validation_candidates.add_argument("--output-json", type=Path)
    validation_candidates.add_argument("--output-csv", type=Path)
    validation_candidates.add_argument("--no-standardize", action="store_true")

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

    eda = subparsers.add_parser("cmapss-eda", help="Build a C-MAPSS EDA summary report")
    eda.add_argument("--data-dir", type=Path, required=True)
    eda.add_argument("--subset", choices=CMAPSS_SUBSETS, required=True)
    eda.add_argument("--rul-cap", type=int, default=125)
    eda.add_argument("--output-json", type=Path)

    manifest = subparsers.add_parser("cmapss-manifest", help="Write a C-MAPSS file manifest")
    manifest.add_argument("--data-dir", type=Path, required=True)
    manifest.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    manifest.add_argument("--output-json", type=Path, required=True)

    verify = subparsers.add_parser("cmapss-verify", help="Verify C-MAPSS files against a manifest")
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    download = subparsers.add_parser(
        "cmapss-download",
        help="Download and extract the official NASA C-MAPSS raw text files",
    )
    download.add_argument("--output-dir", type=Path, default=Path("data/raw/cmapss"))
    download.add_argument(
        "--archive-path",
        type=Path,
        default=Path("data/raw/downloads/cmapss_nasa.zip"),
    )
    download.add_argument("--source-url", default=NASA_CMAPSS_URL)
    download.add_argument("--force", action="store_true")

    phase1 = subparsers.add_parser(
        "phase1-cmapss",
        help="Run the Phase 1 C-MAPSS provenance, EDA, and baseline workflow",
    )
    phase1.add_argument("--data-dir", type=Path, required=True)
    phase1.add_argument("--artifact-dir", type=Path, default=Path("artifacts/phase1"))
    phase1.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    phase1.add_argument("--rul-cap", type=int, default=125)
    phase1.add_argument("--random-state", type=int, default=42)
    phase1.add_argument("--no-standardize", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "cmapss-summary":
        bundle = load_cmapss_subset(args.data_dir, args.subset, rul_cap=args.rul_cap)
        train_summary = summarise_cmapss_frame(bundle.train)
        test_summary = summarise_cmapss_frame(bundle.test)
        print(f"subset={bundle.subset}")
        print(
            "train_rows="
            f"{train_summary.rows} train_units={train_summary.units} "
            f"train_cycle_range={train_summary.min_cycle}-{train_summary.max_cycle} "
            f"train_unit_cycle_range={train_summary.min_unit_cycles}-{train_summary.max_unit_cycles}"
        )
        print(
            "test_rows="
            f"{test_summary.rows} test_units={test_summary.units} "
            f"test_cycle_range={test_summary.min_cycle}-{test_summary.max_cycle} "
            f"test_unit_cycle_range={test_summary.min_unit_cycles}-{test_summary.max_unit_cycles}"
        )
        print(f"test_rul_values={len(bundle.test_rul)}")
        return 0

    if args.command == "cmapss-baseline":
        result = run_cmapss_hist_gradient_boosting(
            args.data_dir,
            args.subset,
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=args.standardize,
        )
        print(f"dataset={result.dataset}")
        print(f"subset={result.subset}")
        print(f"model={result.model_name}")
        print(f"standardize={result.standardize}")
        print(f"rmse={result.rmse:.6f}")
        print(f"nasa_score={result.nasa_score:.6f}")
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
        print(f"dataset={result.dataset}")
        print(f"subset={result.subset}")
        print(f"model={result.model_name}")
        print(f"standardize={result.standardize}")
        print(f"rmse={result.rmse:.6f}")
        print(f"nasa_score={result.nasa_score:.6f}")
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
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-regime-engineered-best-baseline-all":
        results = run_all_cmapss_regime_aware_engineered_default_windows(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
        )
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
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
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(
            "feature_policy="
            + ",".join(
                f"{subset}:{CMAPSS_VALIDATION_SELECTED_FEATURES[subset]}"
                for subset in args.subsets
            )
        )
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
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
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizon={args.validation_horizon}")
        print(f"max_regimes={args.n_regimes}")
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
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
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

    if args.command == "cmapss-eda":
        bundle = load_cmapss_subset(args.data_dir, args.subset, rul_cap=args.rul_cap)
        report = build_cmapss_eda_report(bundle)
        flat_sensors = [
            summary.sensor for summary in report.sensor_summaries if summary.is_near_constant
        ]
        largest_drift = max(report.sensor_summaries, key=lambda summary: abs(summary.drift))
        print(f"subset={report.subset}")
        print(f"train_rows={report.train_rows} train_units={report.train_units}")
        print(f"test_rows={report.test_rows} test_units={report.test_units}")
        print(f"near_constant_sensors={','.join(flat_sensors) if flat_sensors else 'none'}")
        print(f"largest_abs_drift_sensor={largest_drift.sensor} drift={largest_drift.drift:.6f}")
        if args.output_json is not None:
            report.write_json(args.output_json)
        return 0

    if args.command == "cmapss-manifest":
        manifest = build_cmapss_manifest(args.data_dir, subsets=tuple(args.subsets))
        manifest.write_json(args.output_json)
        print(f"dataset={manifest.dataset}")
        print(f"files={len(manifest.entries)}")
        print(f"manifest={args.output_json}")
        return 0

    if args.command == "cmapss-verify":
        manifest = read_manifest(args.manifest)
        problems = verify_manifest(manifest, root=args.data_dir)
        if problems:
            print("status=failed")
            for problem in problems:
                print(f"problem={problem}")
            return 1
        print("status=ok")
        print(f"files={len(manifest.entries)}")
        return 0

    if args.command == "cmapss-download":
        result = download_cmapss_dataset(
            args.output_dir,
            source_url=args.source_url,
            archive_path=args.archive_path,
            force=args.force,
        )
        print(f"source_url={result.source_url}")
        print(f"archive={result.archive_path}")
        print(f"output_dir={result.output_dir}")
        print(f"metadata={result.metadata_path}")
        print(f"files={len(result.extracted_files)}")
        return 0

    if args.command == "phase1-cmapss":
        result = run_phase1_cmapss_workflow(
            args.data_dir,
            args.artifact_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=not args.no_standardize,
        )
        print(f"artifact_dir={result.artifact_dir}")
        print(f"manifest={result.manifest_path}")
        print(f"baseline_json={result.baseline_json_path}")
        print(f"baseline_csv={result.baseline_csv_path}")
        print(f"summary={result.summary_markdown_path}")
        print(f"eda_reports={len(result.eda_paths)}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


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


def _best_aggregate_for_subset(
    results: Iterable[CmapssValidationAggregateResult],
    subset: str,
) -> CmapssValidationAggregateResult:
    subset_results = [result for result in results if result.subset == subset]
    if not subset_results:
        raise ValueError(f"no aggregate results for subset: {subset}")
    return min(subset_results, key=lambda result: result.mean_nasa_score)


def _write_validation_aggregate_json(
    results: list[CmapssValidationAggregateResult],
    path: Path,
) -> None:
    output_path = _prepare_output_path(path)
    payload = [result.to_dict() for result in results]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_validation_aggregate_csv(
    results: list[CmapssValidationAggregateResult],
    path: Path,
) -> None:
    if not results:
        raise ValueError("results must contain at least one item")
    output_path = _prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].to_dict()))
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)


def _prepare_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
