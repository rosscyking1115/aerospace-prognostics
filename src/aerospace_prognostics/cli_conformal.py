"""Conformal prediction command handlers for the project CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
from aerospace_prognostics.experiments.cmapss_conformal import (
    build_attainability_table,
    run_cmapss_conformal_seed_sweep,
    run_cmapss_conformal_study,
)
from aerospace_prognostics.reports.cmapss_conformal import (
    CONFORMAL_ARTIFACT_DIRECTORY,
    write_cmapss_conformal_evidence,
)

CMAPSS_TRAINING_UNITS = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}
"""Training-unit counts per subset, used only to derive the attainability table.

These are properties of the published NASA files. The study asserts them against
the loaded data at run time rather than trusting this constant.
"""


def register_conformal_commands(subparsers: Any) -> None:
    """Register the conformal prediction commands."""
    conformal = subparsers.add_parser(
        "cmapss-conformal",
        help="Fit and evaluate unit-grouped split conformal RUL intervals",
    )
    conformal.add_argument("--data-dir", type=Path, required=True)
    conformal.add_argument("--subset", choices=CMAPSS_SUBSETS, default="FD001")
    conformal.add_argument("--alpha", type=float, default=0.10)
    conformal.add_argument("--calibration-units", type=int, default=30)
    conformal.add_argument("--random-state", type=int, default=42)
    conformal.add_argument("--rul-cap", type=int, default=125)
    conformal.add_argument("--min-calibration-history", type=int, default=20)
    conformal.add_argument(
        "--seed-sweep",
        nargs="*",
        type=int,
        help="Re-run across these unit splits; omit values for the default ten seeds",
    )
    conformal.add_argument(
        "--output-directory",
        type=Path,
        default=CONFORMAL_ARTIFACT_DIRECTORY,
        help="Directory for the committed evidence bundle",
    )
    conformal.add_argument("--stem", type=str, default=None)


def handle_conformal_command(args: argparse.Namespace) -> int | None:
    """Run a conformal command, or return None if this is not one."""
    if getattr(args, "command", None) != "cmapss-conformal":
        return None

    study = run_cmapss_conformal_study(
        args.data_dir,
        args.subset,
        alpha=args.alpha,
        calibration_unit_count=args.calibration_units,
        random_state=args.random_state,
        rul_cap=args.rul_cap,
        min_calibration_history=args.min_calibration_history,
    )

    seed_sweep = None
    if args.seed_sweep is not None:
        sweep_kwargs = {"seeds": tuple(args.seed_sweep)} if args.seed_sweep else {}
        seed_sweep = run_cmapss_conformal_seed_sweep(
            args.data_dir,
            args.subset,
            alpha=args.alpha,
            calibration_unit_count=args.calibration_units,
            rul_cap=args.rul_cap,
            min_calibration_history=args.min_calibration_history,
            **sweep_kwargs,
        )

    evidence = write_cmapss_conformal_evidence(
        study,
        output_directory=args.output_directory,
        seed_sweep=seed_sweep,
        attainability_rows=build_attainability_table(
            available_units_by_subset=CMAPSS_TRAINING_UNITS,
        ),
        stem=args.stem,
    )

    primary = study.variants[0]
    print(
        f"{study.population.subset} {primary.design}: "
        f"coverage {primary.evaluation.empirical_coverage:.6f} "
        f"at nominal {primary.interval.nominal_coverage:.6f}, "
        f"width {primary.evaluation.mean_interval_width:.6f}, "
        f"{study.population.calibration_units} calibration units, "
        f"{study.population.evaluation_units} test units"
    )
    print(f"wrote {evidence.json_path}")
    print(f"wrote {evidence.variants_csv_path}")
    print(f"wrote {evidence.markdown_path}")
    if evidence.seed_sweep_csv_path is not None:
        print(f"wrote {evidence.seed_sweep_csv_path}")
    if evidence.attainability_csv_path is not None:
        print(f"wrote {evidence.attainability_csv_path}")
    return 0
