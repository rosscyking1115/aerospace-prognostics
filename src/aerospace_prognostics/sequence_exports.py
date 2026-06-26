"""C-MAPSS sequence export utilities for deep-learning baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aerospace_prognostics.artifact_io import write_json_payload
from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.experiments.cmapss_baseline import (
    make_cmapss_temporal_validation_split,
)
from aerospace_prognostics.features import cmapss_feature_columns
from aerospace_prognostics.preprocessing import FeatureStandardizer
from aerospace_prognostics.sequences import (
    SequenceWindowDataset,
    make_last_sequence_windows,
    make_sequence_windows,
)


@dataclass(frozen=True)
class CmapssSequenceExportResult:
    """Paths and counts for one exported C-MAPSS sequence split bundle."""

    subset: str
    output_dir: Path
    metadata_path: Path
    train_npz_path: Path
    validation_npz_path: Path
    validation_selection_npz_path: Path
    test_npz_path: Path
    train_windows: int
    validation_windows: int
    validation_selection_windows: int
    test_windows: int
    window_size: int
    stride: int
    feature_columns: tuple[str, ...]
    standardize: bool


def export_cmapss_sequence_splits(
    data_dir: str | Path,
    output_dir: str | Path,
    subset: str,
    *,
    window_size: int = 30,
    stride: int = 1,
    feature_columns: list[str] | None = None,
    rul_cap: int = 125,
    random_state: int = 42,
    validation_fraction: float = 0.2,
    validation_horizon: int = 30,
    standardize: bool = True,
) -> CmapssSequenceExportResult:
    """Export train, validation, and official-test C-MAPSS sequence tensors."""

    columns = tuple(feature_columns or cmapss_feature_columns())
    bundle = load_cmapss_subset(data_dir, subset, rul_cap=rul_cap)
    split = make_cmapss_temporal_validation_split(
        bundle.train,
        validation_fraction=validation_fraction,
        validation_horizon=validation_horizon,
        random_state=random_state,
    )

    train_frame = split.train
    validation_frame = split.validation
    test_frame = bundle.test
    if standardize:
        standardizer = FeatureStandardizer.fit(
            train_frame,
            feature_columns=list(columns),
        )
        train_frame = standardizer.transform_frame(train_frame)
        validation_frame = standardizer.transform_frame(validation_frame)
        test_frame = standardizer.transform_frame(test_frame)

    train_dataset = make_sequence_windows(
        train_frame,
        window_size=window_size,
        stride=stride,
        feature_columns=list(columns),
        target_column="rul_capped",
    )
    validation_dataset = _with_targets(
        make_last_sequence_windows(
            validation_frame,
            window_size=window_size,
            feature_columns=list(columns),
        ),
        split.validation_rul.to_numpy(dtype=np.float32),
    )
    validation_selection_dataset = make_sequence_windows(
        validation_frame,
        window_size=window_size,
        stride=stride,
        feature_columns=list(columns),
        target_column="rul_capped",
    )
    test_dataset = _with_targets(
        make_last_sequence_windows(
            test_frame,
            window_size=window_size,
            feature_columns=list(columns),
        ),
        bundle.test_rul.to_numpy(dtype=np.float32),
    )

    subset_dir = Path(output_dir) / bundle.subset.lower()
    subset_dir.mkdir(parents=True, exist_ok=True)
    train_npz_path = subset_dir / "train_sequences.npz"
    validation_npz_path = subset_dir / "validation_sequences.npz"
    validation_selection_npz_path = subset_dir / "validation_selection_sequences.npz"
    test_npz_path = subset_dir / "test_sequences.npz"
    metadata_path = subset_dir / "metadata.json"

    _write_sequence_npz(train_dataset, train_npz_path)
    _write_sequence_npz(validation_dataset, validation_npz_path)
    _write_sequence_npz(validation_selection_dataset, validation_selection_npz_path)
    _write_sequence_npz(test_dataset, test_npz_path)

    result = CmapssSequenceExportResult(
        subset=bundle.subset,
        output_dir=subset_dir,
        metadata_path=metadata_path,
        train_npz_path=train_npz_path,
        validation_npz_path=validation_npz_path,
        validation_selection_npz_path=validation_selection_npz_path,
        test_npz_path=test_npz_path,
        train_windows=len(train_dataset.windows),
        validation_windows=len(validation_dataset.windows),
        validation_selection_windows=len(validation_selection_dataset.windows),
        test_windows=len(test_dataset.windows),
        window_size=window_size,
        stride=stride,
        feature_columns=columns,
        standardize=standardize,
    )
    _write_metadata(
        result,
        metadata_path,
        rul_cap=rul_cap,
        random_state=random_state,
        validation_fraction=validation_fraction,
        validation_horizon=validation_horizon,
    )
    return result


def _with_targets(dataset: SequenceWindowDataset, targets: np.ndarray) -> SequenceWindowDataset:
    if len(dataset.windows) != len(targets):
        raise ValueError("sequence windows and targets must have the same length")
    return SequenceWindowDataset(
        windows=dataset.windows,
        targets=targets,
        unit_numbers=dataset.unit_numbers,
        end_cycles=dataset.end_cycles,
        feature_columns=dataset.feature_columns,
    )


def _write_sequence_npz(dataset: SequenceWindowDataset, path: Path) -> None:
    targets = (
        dataset.targets
        if dataset.targets is not None
        else np.empty((0,), dtype=np.float32)
    )
    np.savez_compressed(
        path,
        windows=dataset.windows,
        targets=targets,
        unit_numbers=dataset.unit_numbers,
        end_cycles=dataset.end_cycles,
        feature_columns=np.asarray(dataset.feature_columns),
    )


def _write_metadata(
    result: CmapssSequenceExportResult,
    path: Path,
    *,
    rul_cap: int,
    random_state: int,
    validation_fraction: float,
    validation_horizon: int,
) -> None:
    payload: dict[str, Any] = {
        **asdict(result),
        "output_dir": result.output_dir.as_posix(),
        "metadata_path": result.metadata_path.as_posix(),
        "train_npz_path": result.train_npz_path.as_posix(),
        "validation_npz_path": result.validation_npz_path.as_posix(),
        "validation_selection_npz_path": result.validation_selection_npz_path.as_posix(),
        "test_npz_path": result.test_npz_path.as_posix(),
        "rul_cap": rul_cap,
        "random_state": random_state,
        "validation_fraction": validation_fraction,
        "validation_horizon": validation_horizon,
        "target": (
            "rul_capped for train and rolling validation-selection windows; "
            "uncapped remaining cycles for validation/test final windows"
        ),
    }
    write_json_payload(payload, path)
