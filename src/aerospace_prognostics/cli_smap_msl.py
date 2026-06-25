"""SMAP/MSL data command handlers for the project CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aerospace_prognostics.data.downloads import (
    TELEMANOM_SMAP_MSL_DATA_URL,
    TELEMANOM_SMAP_MSL_LABELS_URL,
    download_smap_msl_dataset,
)
from aerospace_prognostics.data.smap_msl import (
    SmapMslChannelSelection,
    export_smap_msl_channel_csv,
    load_smap_msl_channel,
    read_smap_msl_labels,
    select_smap_msl_channels,
    write_smap_msl_channel_selection_csv,
    write_smap_msl_channel_selection_json,
)


def register_smap_msl_download_command(subparsers: Any) -> None:
    smap_msl_download = subparsers.add_parser(
        "smap-msl-download",
        help="Download and extract the Telemanom SMAP/MSL raw arrays and labels",
    )
    smap_msl_download.add_argument("--output-dir", type=Path, default=Path("data/raw/smap_msl"))
    smap_msl_download.add_argument(
        "--archive-path",
        type=Path,
        default=Path("data/raw/downloads/smap_msl_telemanom.zip"),
        help="Destination for the downloaded archive, or an existing Kaggle archive to import",
    )
    smap_msl_download.add_argument("--source-url", default=TELEMANOM_SMAP_MSL_DATA_URL)
    smap_msl_download.add_argument("--labels-url", default=TELEMANOM_SMAP_MSL_LABELS_URL)
    smap_msl_download.add_argument("--force", action="store_true")


def register_smap_msl_data_commands(subparsers: Any) -> None:
    smap_msl_summary = subparsers.add_parser(
        "smap-msl-summary",
        help="Summarise local Telemanom SMAP/MSL labels and optional channel arrays",
    )
    smap_msl_summary.add_argument("--data-dir", type=Path, required=True)
    smap_msl_summary.add_argument("--channel-id")

    smap_msl_select = subparsers.add_parser(
        "smap-msl-select-channels",
        help="Select deterministic SMAP/MSL channels for bounded benchmark sweeps",
    )
    smap_msl_select.add_argument("--data-dir", type=Path, required=True)
    smap_msl_select.add_argument("--count", type=int, default=20)
    smap_msl_select.add_argument(
        "--strategy",
        choices=["balanced", "label_order"],
        default="balanced",
    )
    smap_msl_select.add_argument("--spacecraft", nargs="+")
    smap_msl_select.add_argument("--min-anomaly-sequences", type=int, default=1)
    smap_msl_select.add_argument("--output-json", type=Path)
    smap_msl_select.add_argument("--output-csv", type=Path)

    smap_msl_export = subparsers.add_parser(
        "smap-msl-export-channel-csv",
        help="Export one Telemanom SMAP/MSL channel to train/test CSVs",
    )
    smap_msl_export.add_argument("--data-dir", type=Path, required=True)
    smap_msl_export.add_argument("--channel-id", required=True)
    smap_msl_export.add_argument("--output-dir", type=Path, required=True)
    smap_msl_export.add_argument("--metadata-json", type=Path)


def handle_smap_msl_data_command(args: argparse.Namespace) -> int | None:
    if args.command == "smap-msl-download":
        try:
            result = download_smap_msl_dataset(
                args.output_dir,
                source_url=args.source_url,
                labels_url=args.labels_url,
                archive_path=args.archive_path,
                force=args.force,
            )
        except RuntimeError as exc:
            print("status=failed")
            for line in str(exc).splitlines():
                print(f"problem={line}")
            return 1
        print(f"source_url={result.source_url}")
        print(f"labels_url={result.labels_url}")
        print(f"archive={result.archive_path}")
        print(f"output_dir={result.output_dir}")
        print(f"labels={result.labels_path}")
        print(f"metadata={result.metadata_path}")
        print(f"arrays={len(result.extracted_arrays)}")
        return 0

    if args.command == "smap-msl-summary":
        if args.channel_id is None:
            labels = read_smap_msl_labels(args.data_dir)
            spacecraft_counts: dict[str, int] = {}
            for metadata in labels:
                spacecraft_counts[metadata.spacecraft] = (
                    spacecraft_counts.get(metadata.spacecraft, 0) + 1
                )
            print(f"channels={len(labels)}")
            print(f"anomaly_sequences={sum(len(item.anomaly_sequences) for item in labels)}")
            print(
                "spacecraft="
                + ",".join(
                    f"{spacecraft}:{count}"
                    for spacecraft, count in sorted(spacecraft_counts.items())
                )
            )
            return 0

        channel = load_smap_msl_channel(args.data_dir, args.channel_id)
        print(f"channel_id={channel.metadata.channel_id}")
        print(f"spacecraft={channel.metadata.spacecraft}")
        print(f"train_shape={channel.train_values.shape[0]}x{channel.train_values.shape[1]}")
        print(f"test_shape={channel.test_values.shape[0]}x{channel.test_values.shape[1]}")
        print(f"anomaly_sequences={len(channel.metadata.anomaly_sequences)}")
        print(f"labelled_anomaly_points={int(channel.test_labels.sum())}")
        return 0

    if args.command == "smap-msl-select-channels":
        selections = select_smap_msl_channels(
            args.data_dir,
            count=args.count,
            strategy=args.strategy,
            spacecraft=tuple(args.spacecraft) if args.spacecraft is not None else None,
            min_anomaly_sequences=args.min_anomaly_sequences,
        )
        print(f"selected_channels={len(selections)}")
        print("channels=" + " ".join(selection.channel_id for selection in selections))
        _print_smap_msl_channel_selection_table(selections)
        if args.output_json is not None:
            write_smap_msl_channel_selection_json(selections, args.output_json)
        if args.output_csv is not None:
            write_smap_msl_channel_selection_csv(selections, args.output_csv)
        return 0

    if args.command == "smap-msl-export-channel-csv":
        export = export_smap_msl_channel_csv(args.data_dir, args.channel_id, args.output_dir)
        print(f"channel_id={export.channel_id}")
        print(f"train_csv={export.train_csv}")
        print(f"test_csv={export.test_csv}")
        print(f"train_rows={export.train_rows}")
        print(f"test_rows={export.test_rows}")
        print(f"features={len(export.feature_names)}")
        if args.metadata_json is not None:
            _write_json_payload(export.to_dict(), args.metadata_json)
        return 0

    return None


def _print_smap_msl_channel_selection_table(
    selections: Iterable[SmapMslChannelSelection],
) -> None:
    print("rank,channel_id,spacecraft,anomaly_sequences,anomaly_points,num_values")
    for selection in selections:
        print(
            f"{selection.rank},"
            f"{selection.channel_id},"
            f"{selection.spacecraft},"
            f"{selection.anomaly_sequences},"
            f"{selection.anomaly_points},"
            f"{selection.num_values if selection.num_values is not None else ''}"
        )


def _write_json_payload(payload: object, path: Path) -> None:
    output_path = _prepare_output_path(path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
