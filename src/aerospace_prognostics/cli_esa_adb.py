"""ESA-ADB source manifest and archive validation CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerospace_prognostics.data.esa_adb import (
    build_esa_adb_source_manifest,
    read_esa_adb_source_manifest,
    verify_esa_adb_archives,
    write_esa_adb_archive_validation,
    write_esa_adb_source_manifest,
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

    return None
