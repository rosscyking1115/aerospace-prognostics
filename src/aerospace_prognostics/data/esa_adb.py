"""ESA-ADB source manifest and local archive validation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import md5
from pathlib import Path
from typing import Any

import pandas as pd

from aerospace_prognostics.artifact_io import write_json_payload

ESA_ADB_SOURCE_MANIFEST_SCHEMA = "aerospace-prognostics/esa-adb-source-manifest/v1"
ESA_ADB_ARCHIVE_VALIDATION_SCHEMA = "aerospace-prognostics/esa-adb-archive-validation/v1"
ESA_ADB_LABEL_COLUMNS = ("ID", "Channel", "StartTime", "EndTime")
ESA_ADB_ANOMALY_TYPE_COLUMNS = ("Category", "Dimensionality", "Locality", "Length")
ESA_ADB_PREDICTION_COLUMNS = ("Timestamp", "Score")

ESA_ADB_DATASET = "ESA Anomaly Dataset"
ESA_ADB_DATASET_VERSION = "v2"
ESA_ADB_DATASET_DOI = "10.5281/zenodo.15237121"
ESA_ADB_ORIGINAL_DATASET_DOI = "10.5281/zenodo.12528696"
ESA_ADB_DATASET_LICENSE = "CC BY 3.0 IGO"
ESA_ADB_SOURCE_RECORD_URL = "https://zenodo.org/records/15237121"
ESA_ADB_OFFICIAL_REPOSITORY_URL = "https://github.com/kplabs-pl/ESA-ADB"
ESA_ADB_OFFICIAL_REPOSITORY_REF = "main"
ESA_ADB_DEFAULT_ARCHIVE_DIR = "data/raw/esa_adb"
ESA_ADB_BENCHMARK_MISSIONS = ["Mission1", "Mission2"]

_ESA_ADB_SOURCE_FILES: tuple[dict[str, str], ...] = (
    {
        "mission": "Mission1",
        "file_name": "ESA-Mission1.zip",
        "default_local_path": "data/raw/esa_adb/ESA-Mission1.zip",
        "md5": "9770ad12ed730238f37c42d5c27ab436",
        "size_label": "3.8 GB",
        "zenodo_record_url": ESA_ADB_SOURCE_RECORD_URL,
        "benchmark_scope": "benchmark",
    },
    {
        "mission": "Mission2",
        "file_name": "ESA-Mission2.zip",
        "default_local_path": "data/raw/esa_adb/ESA-Mission2.zip",
        "md5": "bfc72012691427d9327eb41f726ce45e",
        "size_label": "4.1 GB",
        "zenodo_record_url": ESA_ADB_SOURCE_RECORD_URL,
        "benchmark_scope": "benchmark",
    },
    {
        "mission": "Mission3",
        "file_name": "ESA-Mission3.zip",
        "default_local_path": "data/raw/esa_adb/ESA-Mission3.zip",
        "md5": "d63943f09c81378acd9fc5e565ecc66e",
        "size_label": "3.7 GB",
        "zenodo_record_url": ESA_ADB_SOURCE_RECORD_URL,
        "benchmark_scope": "exploration_only",
    },
)


def build_esa_adb_source_manifest() -> dict[str, Any]:
    """Return the tracked source manifest for ESA Anomaly Dataset v2 archives."""

    return {
        "schema_version": ESA_ADB_SOURCE_MANIFEST_SCHEMA,
        "dataset": ESA_ADB_DATASET,
        "dataset_version": ESA_ADB_DATASET_VERSION,
        "dataset_doi": ESA_ADB_DATASET_DOI,
        "original_paper_dataset_doi": ESA_ADB_ORIGINAL_DATASET_DOI,
        "dataset_license": ESA_ADB_DATASET_LICENSE,
        "source_record_url": ESA_ADB_SOURCE_RECORD_URL,
        "official_repository_url": ESA_ADB_OFFICIAL_REPOSITORY_URL,
        "official_repository_ref": ESA_ADB_OFFICIAL_REPOSITORY_REF,
        "official_repository_commit": None,
        "default_archive_dir": ESA_ADB_DEFAULT_ARCHIVE_DIR,
        "benchmark_missions": list(ESA_ADB_BENCHMARK_MISSIONS),
        "files": [dict(entry) for entry in _ESA_ADB_SOURCE_FILES],
    }


def write_esa_adb_source_manifest(path: str | Path) -> dict[str, Any]:
    """Write the ESA-ADB source manifest as deterministic JSON."""

    manifest = build_esa_adb_source_manifest()
    write_json_payload(manifest, path)
    return manifest


def read_esa_adb_source_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate an ESA-ADB source manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != ESA_ADB_SOURCE_MANIFEST_SCHEMA:
        raise ValueError(
            "unexpected ESA-ADB source manifest schema: "
            f"{payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("files"), list):
        raise ValueError("ESA-ADB source manifest files must be a list")
    return payload


def read_esa_adb_evaluator_labels(
    labels_csv_path: str | Path,
    anomaly_types_csv_path: str | Path,
) -> pd.DataFrame:
    """Read ESA-ADB labels in the shape expected by the official evaluator.

    The official evaluator reads ``labels.csv`` with ``StartTime`` and
    ``EndTime`` parsed as datetimes, then joins the last four columns from
    ``anomaly_types.csv`` by ``ID``. In the published benchmark files those
    columns are ``Category``, ``Dimensionality``, ``Locality``, and ``Length``.
    """

    labels = pd.read_csv(labels_csv_path)
    anomaly_types = pd.read_csv(anomaly_types_csv_path)
    return build_esa_adb_evaluator_labels(labels, anomaly_types)


def build_esa_adb_evaluator_labels(
    labels: pd.DataFrame,
    anomaly_types: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ESA-ADB label rows with official anomaly type metadata."""

    _require_columns(labels, ESA_ADB_LABEL_COLUMNS, "labels.csv")
    _require_columns(anomaly_types, ("ID", *ESA_ADB_ANOMALY_TYPE_COLUMNS), "anomaly_types.csv")

    labels = labels.loc[:, list(ESA_ADB_LABEL_COLUMNS)].copy()
    labels["ID"] = labels["ID"].astype(str)
    labels["Channel"] = labels["Channel"].astype(str)
    labels["StartTime"] = _parse_esa_timestamp_column(labels["StartTime"], "StartTime")
    labels["EndTime"] = _parse_esa_timestamp_column(labels["EndTime"], "EndTime")
    _validate_esa_adb_label_intervals(labels)

    anomaly_types = anomaly_types.loc[:, ("ID", *ESA_ADB_ANOMALY_TYPE_COLUMNS)].copy()
    anomaly_types["ID"] = anomaly_types["ID"].astype(str)

    duplicate_type_ids = sorted(anomaly_types.loc[anomaly_types["ID"].duplicated(), "ID"].unique())
    if duplicate_type_ids:
        raise ValueError(f"anomaly_types.csv contains duplicate ID rows: {duplicate_type_ids}")

    missing_type_ids = sorted(set(labels["ID"]) - set(anomaly_types["ID"]))
    if missing_type_ids:
        raise ValueError(f"missing anomaly_types.csv rows for label IDs: {missing_type_ids}")

    merged = labels.merge(anomaly_types, on="ID", how="left", validate="many_to_one")
    return merged


def group_esa_adb_binary_events(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Group a binary ``Timestamp, Score`` series using ESA-ADB TimeEval semantics.

    The official helper treats each positive sample as covering the interval
    from that timestamp to the next timestamp. Runs that reach the final sample
    end at the final timestamp and are closed at both ends.
    """

    predictions = _normalise_esa_adb_prediction_frame(predictions, channel_name="global")
    scores = predictions["Score"].to_list()
    timestamps = predictions["Timestamp"].to_list()
    events: list[dict[str, Any]] = []
    index = 0
    n_rows = len(predictions)

    while index < n_rows:
        if scores[index] <= 0:
            index += 1
            continue

        start_index = index
        while index < n_rows and scores[index] > 0:
            index += 1

        reaches_final_sample = index == n_rows
        end_index = index - 1 if reaches_final_sample else index
        events.append(
            {
                "start_time": timestamps[start_index],
                "end_time": timestamps[end_index],
                "end_inclusive": reaches_final_sample,
            }
        )

    return events


def build_esa_adb_metric_inputs(
    labels: pd.DataFrame,
    predictions_by_channel: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Build official-compatible ESA-ADB global and per-channel metric inputs."""

    labels = _normalise_esa_adb_evaluator_labels(labels)
    if not predictions_by_channel:
        raise ValueError("ESA-ADB metric inputs require at least one prediction channel")

    channel_predictions = {
        str(channel): _normalise_esa_adb_prediction_frame(frame, channel_name=str(channel))
        for channel, frame in predictions_by_channel.items()
    }
    _validate_prediction_timestamp_alignment(channel_predictions)

    target_channels = tuple(channel_predictions)
    metric_labels = labels[labels["Channel"].isin(target_channels)].copy()
    global_labels = metric_labels.drop(columns=["Channel"])

    first_channel = target_channels[0]
    global_predictions = channel_predictions[first_channel][["Timestamp"]].copy()
    score_frame = pd.DataFrame(
        {channel: frame["Score"].to_numpy() for channel, frame in channel_predictions.items()}
    )
    global_predictions["Score"] = score_frame.max(axis=1).astype("uint8")

    return {
        "global_labels": global_labels.reset_index(drop=True),
        "global_predictions": global_predictions.reset_index(drop=True),
        "channel_labels": metric_labels.reset_index(drop=True),
        "channel_predictions": {
            channel: frame.reset_index(drop=True) for channel, frame in channel_predictions.items()
        },
        "target_channels": list(target_channels),
    }


def verify_esa_adb_archives(
    archive_dir: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
    missions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Verify locally supplied ESA-ADB archives against manifest MD5 values.

    This intentionally performs no network I/O. It is the safety gate before any
    extraction, preprocessing, or benchmark work touches large raw archives.
    """

    source_manifest = manifest if manifest is not None else build_esa_adb_source_manifest()
    selected_missions = _normalise_requested_missions(source_manifest, missions)
    root = Path(archive_dir)
    problems: list[str] = []
    verified_files: list[dict[str, Any]] = []

    for entry in _files_for_missions(source_manifest, selected_missions):
        file_name = _required_string(entry, "file_name")
        expected_md5 = _required_string(entry, "md5").lower()
        path = root / file_name
        record: dict[str, Any] = {
            "mission": _required_string(entry, "mission"),
            "file_name": file_name,
            "path": str(path),
            "expected_md5": expected_md5,
            "exists": path.exists(),
            "size_bytes": None,
            "actual_md5": None,
            "md5_verified": False,
        }

        if not path.exists():
            problems.append(f"{file_name} is missing")
            verified_files.append(record)
            continue

        actual_md5 = file_md5(path)
        record["size_bytes"] = path.stat().st_size
        record["actual_md5"] = actual_md5
        record["md5_verified"] = actual_md5 == expected_md5
        if actual_md5 != expected_md5:
            problems.append(f"{file_name} has unexpected md5")
        verified_files.append(record)

    return {
        "schema_version": ESA_ADB_ARCHIVE_VALIDATION_SCHEMA,
        "dataset": source_manifest["dataset"],
        "dataset_version": source_manifest["dataset_version"],
        "dataset_doi": source_manifest["dataset_doi"],
        "archive_dir": str(root),
        "missions": list(selected_missions),
        "status": "ok" if not problems else "failed",
        "files_checked": len(verified_files),
        "files_missing": sum(1 for record in verified_files if not record["exists"]),
        "files_with_mismatches": sum(
            1 for record in verified_files if record["exists"] and not record["md5_verified"]
        ),
        "problems": problems,
        "verified_files": verified_files,
    }


def write_esa_adb_archive_validation(
    path: str | Path,
    archive_dir: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
    missions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Verify local archives and write the validation result as JSON."""

    result = verify_esa_adb_archives(archive_dir, manifest=manifest, missions=missions)
    write_json_payload(result, path)
    return result


def file_md5(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's MD5 digest for source records that publish MD5 checksums."""

    digest = md5(usedforsecurity=False)
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_requested_missions(
    manifest: dict[str, Any],
    missions: tuple[str, ...] | None,
) -> tuple[str, ...]:
    available = {
        _required_string(entry, "mission")
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
    }
    if missions is None:
        return tuple(_required_string(entry, "mission") for entry in manifest["files"])

    normalised = tuple(mission.strip() for mission in missions)
    unknown = [mission for mission in normalised if mission not in available]
    if unknown:
        raise ValueError(f"unknown ESA-ADB mission(s): {unknown}")
    return normalised


def _files_for_missions(
    manifest: dict[str, Any],
    missions: tuple[str, ...],
) -> list[dict[str, Any]]:
    mission_set = set(missions)
    return [
        entry
        for entry in manifest["files"]
        if isinstance(entry, dict) and _required_string(entry, "mission") in mission_set
    ]


def _required_string(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"ESA-ADB manifest entry field {key!r} is missing or invalid")
    return value


def _normalise_esa_adb_evaluator_labels(labels: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        labels,
        (*ESA_ADB_LABEL_COLUMNS, *ESA_ADB_ANOMALY_TYPE_COLUMNS),
        "ESA-ADB evaluator labels",
    )
    labels = labels.loc[:, [*ESA_ADB_LABEL_COLUMNS, *ESA_ADB_ANOMALY_TYPE_COLUMNS]].copy()
    labels["ID"] = labels["ID"].astype(str)
    labels["Channel"] = labels["Channel"].astype(str)
    labels["StartTime"] = _parse_esa_timestamp_column(labels["StartTime"], "StartTime")
    labels["EndTime"] = _parse_esa_timestamp_column(labels["EndTime"], "EndTime")
    _validate_esa_adb_label_intervals(labels)
    return labels


def _normalise_esa_adb_prediction_frame(
    predictions: pd.DataFrame,
    *,
    channel_name: str,
) -> pd.DataFrame:
    _require_columns(
        predictions,
        ESA_ADB_PREDICTION_COLUMNS,
        f"ESA-ADB predictions for {channel_name}",
    )
    predictions = predictions.loc[:, list(ESA_ADB_PREDICTION_COLUMNS)].copy()
    if predictions.empty:
        raise ValueError(f"ESA-ADB predictions for {channel_name} must contain at least one row")
    predictions["Timestamp"] = _parse_esa_timestamp_column(predictions["Timestamp"], "Timestamp")
    if predictions["Timestamp"].duplicated().any():
        raise ValueError(f"ESA-ADB predictions for {channel_name} contain duplicate timestamps")
    if not predictions["Timestamp"].is_monotonic_increasing:
        raise ValueError(f"ESA-ADB predictions for {channel_name} timestamps must be increasing")

    if not predictions["Score"].isin([0, 1, False, True]).all():
        raise ValueError(f"ESA-ADB predictions for {channel_name} must be binary")
    predictions["Score"] = predictions["Score"].astype("uint8")
    return predictions


def _validate_prediction_timestamp_alignment(
    predictions_by_channel: Mapping[str, pd.DataFrame],
) -> None:
    timestamps: pd.Series | None = None
    for channel, predictions in predictions_by_channel.items():
        current = predictions["Timestamp"]
        if timestamps is None:
            timestamps = current
            continue
        if len(current) != len(timestamps) or not current.reset_index(drop=True).equals(
            timestamps.reset_index(drop=True)
        ):
            raise ValueError(
                "ESA-ADB prediction timestamps must align across channels; "
                f"channel {channel} differs from the first channel"
            )


def _parse_esa_timestamp_column(values: pd.Series, column_name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise", utc=True).dt.tz_convert(None)
    if parsed.isna().any():
        raise ValueError(f"ESA-ADB timestamp column {column_name} contains missing values")
    return parsed


def _validate_esa_adb_label_intervals(labels: pd.DataFrame) -> None:
    if (labels["StartTime"] > labels["EndTime"]).any():
        raise ValueError("ESA-ADB labels contain StartTime values after EndTime")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], source_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required ESA-ADB column(s): {missing}")
