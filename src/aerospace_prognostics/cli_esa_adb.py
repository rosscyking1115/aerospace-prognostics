"""ESA-ADB source manifest, archive validation, and mission scoring CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from aerospace_prognostics.data.esa_adb import (
    build_esa_adb_source_manifest,
    read_esa_adb_evaluator_labels,
    read_esa_adb_source_manifest,
    verify_esa_adb_archives,
    write_esa_adb_archive_validation,
    write_esa_adb_source_manifest,
)
from aerospace_prognostics.data.esa_adb_scoring import (
    ESA_ADB_LIGHTWEIGHT_CHANNELS,
    score_esa_adb_mission_from_predictions,
    write_esa_adb_event_wise_evidence,
)


def register_esa_adb_commands(subparsers: Any) -> None:
    source_manifest = subparsers.add_parser(
        "esa-adb-source-manifest",
        help="Write the tracked ESA-ADB Zenodo v2 source archive manifest",
    )
    source_manifest.add_argument("--output-json", type=Path, required=True)

    verify_archives = subparsers.add_parser(
        "esa-adb-verify-archives",
        help="Verify locally supplied ESA-ADB archives without downloading them",
    )
    verify_archives.add_argument("--archive-dir", type=Path, default=Path("data/raw/esa_adb"))
    verify_archives.add_argument("--manifest", type=Path)
    verify_archives.add_argument("--missions", nargs="+")
    verify_archives.add_argument("--output-json", type=Path)

    mission_score = subparsers.add_parser(
        "esa-adb-mission-score",
        help=(
            "Score event-wise detection from prepared ESA-ADB labels and "
            "per-channel binary predictions (no download)"
        ),
    )
    mission_score.add_argument("--mission", required=True)
    mission_score.add_argument("--labels-csv", type=Path, required=True)
    mission_score.add_argument("--anomaly-types-csv", type=Path, required=True)
    mission_score.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory of <channel>.csv files with Timestamp,Score columns",
    )
    mission_score.add_argument("--beta", type=float, default=0.5)
    mission_score.add_argument("--lightweight", action="store_true")
    mission_score.add_argument("--exclude-categories", nargs="+", default=[])
    mission_score.add_argument("--output-json", type=Path)
    mission_score.add_argument("--output-markdown", type=Path)


def handle_esa_adb_command(args: argparse.Namespace) -> int | None:
    if args.command == "esa-adb-source-manifest":
        manifest = write_esa_adb_source_manifest(args.output_json)
        print(f"dataset={manifest['dataset']}")
        print(f"dataset_version={manifest['dataset_version']}")
        print(f"dataset_doi={manifest['dataset_doi']}")
        print(f"files={len(manifest['files'])}")
        print("benchmark_missions=" + ",".join(manifest["benchmark_missions"]))
        print(f"manifest={args.output_json}")
        return 0

    if args.command == "esa-adb-verify-archives":
        manifest = (
            read_esa_adb_source_manifest(args.manifest)
            if args.manifest is not None
            else build_esa_adb_source_manifest()
        )
        missions = tuple(args.missions) if args.missions is not None else None
        if args.output_json is not None:
            result = write_esa_adb_archive_validation(
                args.output_json,
                args.archive_dir,
                manifest=manifest,
                missions=missions,
            )
        else:
            result = verify_esa_adb_archives(
                args.archive_dir,
                manifest=manifest,
                missions=missions,
            )

        print(f"status={result['status']}")
        print(f"archive_dir={result['archive_dir']}")
        print(f"missions={','.join(result['missions'])}")
        print(f"files_checked={result['files_checked']}")
        print(f"files_missing={result['files_missing']}")
        print(f"files_with_mismatches={result['files_with_mismatches']}")
        if args.output_json is not None:
            print(f"output_json={args.output_json}")
        for problem in result["problems"]:
            print(f"problem={problem}")
        return 0 if result["status"] == "ok" else 1

    if args.command == "esa-adb-mission-score":
        labels = read_esa_adb_evaluator_labels(args.labels_csv, args.anomaly_types_csv)
        predictions_by_channel = _load_channel_predictions(args.predictions_dir)
        evidence = score_esa_adb_mission_from_predictions(
            labels,
            predictions_by_channel,
            mission=args.mission,
            lightweight=args.lightweight,
            beta=args.beta,
            exclude_categories=tuple(args.exclude_categories),
        )
        write_esa_adb_event_wise_evidence(
            evidence,
            json_path=args.output_json,
            markdown_path=args.output_markdown,
        )

        print(f"mission={evidence['mission']}")
        print(f"lightweight={evidence['lightweight_subset']}")
        print(f"target_channels={len(evidence['target_channels'])}")
        print(f"total_events={evidence['total_events']}")
        print(f"detected_events={evidence['detected_events']}")
        print(f"false_alarms={evidence['false_alarms']}")
        print(f"event_wise_precision={evidence['event_wise_precision']:.6f}")
        print(f"event_wise_recall={evidence['event_wise_recall']:.6f}")
        print(f"event_wise_fbeta={evidence['event_wise_fbeta']:.6f}")
        if args.output_json is not None:
            print(f"output_json={args.output_json}")
        if args.output_markdown is not None:
            print(f"output_markdown={args.output_markdown}")
        return 0

    return None


def _load_channel_predictions(predictions_dir: Path) -> dict[str, pd.DataFrame]:
    csv_paths = sorted(predictions_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"no per-channel prediction CSV files found in {predictions_dir}")
    return {path.stem: pd.read_csv(path) for path in csv_paths}


# Re-exported for command discovery/tests.
LIGHTWEIGHT_MISSIONS = tuple(ESA_ADB_LIGHTWEIGHT_CHANNELS)
