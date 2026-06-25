"""C-MAPSS data command handlers for the project CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerospace_prognostics.analysis.cmapss_eda import build_cmapss_eda_report
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS, load_cmapss_subset
from aerospace_prognostics.data.downloads import NASA_CMAPSS_URL, download_cmapss_dataset
from aerospace_prognostics.data.manifest import (
    build_cmapss_manifest,
    read_manifest,
    verify_manifest,
)
from aerospace_prognostics.data.summary import summarise_cmapss_frame


def register_cmapss_data_commands(subparsers: Any) -> None:
    summary = subparsers.add_parser("cmapss-summary", help="Summarise a local C-MAPSS subset")
    summary.add_argument("--data-dir", type=Path, required=True)
    summary.add_argument("--subset", choices=CMAPSS_SUBSETS, required=True)
    summary.add_argument("--rul-cap", type=int, default=125)

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


def handle_cmapss_data_command(args: argparse.Namespace) -> int | None:
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

    return None
