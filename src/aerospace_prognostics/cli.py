"""Command-line utilities for local dataset checks."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from aerospace_prognostics.analysis.cmapss_eda import build_cmapss_eda_report
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS, load_cmapss_subset
from aerospace_prognostics.data.summary import summarise_cmapss_frame
from aerospace_prognostics.evaluation import (
    RegressionRunResult,
    write_results_csv,
    write_results_json,
)
from aerospace_prognostics.experiments.cmapss_baseline import (
    run_all_cmapss_hist_gradient_boosting,
    run_cmapss_hist_gradient_boosting,
)


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

    eda = subparsers.add_parser("cmapss-eda", help="Build a C-MAPSS EDA summary report")
    eda.add_argument("--data-dir", type=Path, required=True)
    eda.add_argument("--subset", choices=CMAPSS_SUBSETS, required=True)
    eda.add_argument("--rul-cap", type=int, default=125)
    eda.add_argument("--output-json", type=Path)

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


if __name__ == "__main__":
    raise SystemExit(main())
