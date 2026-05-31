"""Load NASA/JPL Telemanom SMAP/MSL anomaly-detection data."""

from __future__ import annotations

import ast
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SMAP_MSL_LABEL_FILENAME = "labeled_anomalies.csv"


@dataclass(frozen=True)
class SmapMslChannelMetadata:
    """Metadata and anomaly intervals for one SMAP/MSL telemetry channel."""

    channel_id: str
    spacecraft: str
    anomaly_sequences: tuple[tuple[int, int], ...]
    anomaly_classes: tuple[str, ...]
    num_values: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SmapMslChannelData:
    """Raw train/test arrays and generated binary test labels for one channel."""

    metadata: SmapMslChannelMetadata
    train_values: np.ndarray
    test_values: np.ndarray
    test_labels: np.ndarray

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"feature_{index}" for index in range(self.train_values.shape[1]))


@dataclass(frozen=True)
class SmapMslChannelCsvExport:
    """CSV export paths for one channel."""

    channel_id: str
    train_csv: Path
    test_csv: Path
    train_rows: int
    test_rows: int
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["train_csv"] = str(self.train_csv)
        payload["test_csv"] = str(self.test_csv)
        return payload


def read_smap_msl_labels(data_dir: Path) -> tuple[SmapMslChannelMetadata, ...]:
    """Read Telemanom `labeled_anomalies.csv` metadata."""

    labels_path = data_dir / SMAP_MSL_LABEL_FILENAME
    if not labels_path.exists():
        raise FileNotFoundError(f"missing SMAP/MSL labels file: {labels_path}")

    with labels_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    metadata = tuple(_metadata_from_row(row) for row in rows)
    if not metadata:
        raise ValueError(f"SMAP/MSL labels file is empty: {labels_path}")
    return metadata


def load_smap_msl_channel(data_dir: Path, channel_id: str) -> SmapMslChannelData:
    """Load train/test arrays and binary test labels for a Telemanom channel."""

    labels_by_channel = {
        metadata.channel_id: metadata for metadata in read_smap_msl_labels(data_dir)
    }
    if channel_id not in labels_by_channel:
        raise ValueError(f"channel_id not found in labels: {channel_id}")

    train_path = _smap_msl_split_path(data_dir, "train", channel_id)
    test_path = _smap_msl_split_path(data_dir, "test", channel_id)
    train_values = _load_2d_npy(train_path, name=f"{channel_id} train values")
    test_values = _load_2d_npy(test_path, name=f"{channel_id} test values")
    if train_values.shape[1] != test_values.shape[1]:
        raise ValueError(
            f"{channel_id} train/test feature counts differ: "
            f"{train_values.shape[1]} != {test_values.shape[1]}"
        )

    metadata = labels_by_channel[channel_id]
    test_labels = build_smap_msl_label_vector(
        len(test_values),
        metadata.anomaly_sequences,
        channel_id=channel_id,
    )
    return SmapMslChannelData(
        metadata=metadata,
        train_values=train_values,
        test_values=test_values,
        test_labels=test_labels,
    )


def build_smap_msl_label_vector(
    length: int,
    anomaly_sequences: tuple[tuple[int, int], ...],
    *,
    channel_id: str = "channel",
) -> np.ndarray:
    """Build binary labels from inclusive Telemanom anomaly intervals."""

    if length <= 0:
        raise ValueError("length must be positive")
    labels = np.zeros(length, dtype=np.int8)
    for start, end in anomaly_sequences:
        if start < 0 or end < start or end >= length:
            raise ValueError(
                f"{channel_id} anomaly interval [{start}, {end}] is outside test length {length}"
            )
        labels[start : end + 1] = 1
    return labels


def export_smap_msl_channel_csv(
    data_dir: Path,
    channel_id: str,
    output_dir: Path,
) -> SmapMslChannelCsvExport:
    """Export one SMAP/MSL channel to train/test CSVs for generic anomaly CLIs."""

    channel = load_smap_msl_channel(data_dir, channel_id)
    channel_dir = output_dir / channel_id
    channel_dir.mkdir(parents=True, exist_ok=True)
    feature_names = channel.feature_names
    train_csv = channel_dir / "train.csv"
    test_csv = channel_dir / "test.csv"
    _write_values_csv(train_csv, channel.train_values, feature_names)
    _write_values_csv(test_csv, channel.test_values, feature_names, labels=channel.test_labels)
    return SmapMslChannelCsvExport(
        channel_id=channel_id,
        train_csv=train_csv,
        test_csv=test_csv,
        train_rows=len(channel.train_values),
        test_rows=len(channel.test_values),
        feature_names=feature_names,
    )


def _metadata_from_row(row: dict[str, str]) -> SmapMslChannelMetadata:
    normalized = {_normalize_column_name(key): value for key, value in row.items()}
    channel_id = normalized.get("chan_id") or normalized.get("channel_id")
    if not channel_id:
        raise ValueError("SMAP/MSL labels row is missing chan_id/channel_id")
    sequences = _parse_anomaly_sequences(normalized.get("anomaly_sequences", "[]"))
    return SmapMslChannelMetadata(
        channel_id=channel_id,
        spacecraft=normalized.get("spacecraft", ""),
        anomaly_sequences=sequences,
        anomaly_classes=_parse_anomaly_classes(normalized.get("class", ""), len(sequences)),
        num_values=_parse_optional_int(normalized.get("num_values")),
    )


def _parse_anomaly_sequences(value: str) -> tuple[tuple[int, int], ...]:
    if value is None or not value.strip():
        return ()
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("anomaly_sequences must be a list of [start, end] intervals")
    intervals: list[tuple[int, int]] = []
    for interval in parsed:
        if not isinstance(interval, list | tuple) or len(interval) != 2:
            raise ValueError("anomaly_sequences must contain [start, end] intervals")
        intervals.append((int(interval[0]), int(interval[1])))
    return tuple(intervals)


def _parse_anomaly_classes(value: str, expected_length: int) -> tuple[str, ...]:
    if value is None or not value.strip():
        return tuple("unknown" for _ in range(expected_length))
    stripped = value.strip()
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        parsed = [item.strip().strip("'\"") for item in stripped.strip("[]").split(",") if item]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list | tuple):
        raise ValueError("class must be a list-like value")
    classes = tuple(str(item) for item in parsed)
    if expected_length and len(classes) != expected_length:
        return classes + tuple("unknown" for _ in range(expected_length - len(classes)))
    return classes


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _smap_msl_split_path(data_dir: Path, split: str, channel_id: str) -> Path:
    direct_path = data_dir / split / f"{channel_id}.npy"
    if direct_path.exists():
        return direct_path
    nested_path = data_dir / "data" / split / f"{channel_id}.npy"
    if nested_path.exists():
        return nested_path
    raise FileNotFoundError(
        f"missing SMAP/MSL {split} array for {channel_id}; "
        f"checked {direct_path} and {nested_path}"
    )


def _load_2d_npy(path: Path, *, name: str) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape (n_timesteps, n_inputs)")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one feature")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _write_values_csv(
    path: Path,
    values: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    labels: np.ndarray | None = None,
) -> None:
    fieldnames = ("timestep", *feature_names, *(("label",) if labels is not None else ()))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for timestep, row in enumerate(values):
            payload = {"timestep": timestep}
            payload.update(
                {
                    feature_name: f"{value:.12g}"
                    for feature_name, value in zip(feature_names, row, strict=True)
                }
            )
            if labels is not None:
                payload["label"] = int(labels[timestep])
            writer.writerow(payload)


def _normalize_column_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")
