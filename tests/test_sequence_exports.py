from __future__ import annotations

import json

import numpy as np

from aerospace_prognostics.sequence_exports import export_cmapss_sequence_splits
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_export_cmapss_sequence_splits_writes_npz_and_metadata(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    output_dir = tmp_path / "sequences"

    result = export_cmapss_sequence_splits(
        tmp_path,
        output_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    assert result.train_npz_path.exists()
    assert result.validation_npz_path.exists()
    assert result.test_npz_path.exists()
    assert result.metadata_path.exists()
    assert result.train_windows == 2
    assert result.validation_windows == 1
    assert result.test_windows == 2

    train_npz = np.load(result.train_npz_path)
    validation_npz = np.load(result.validation_npz_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert train_npz["windows"].shape == (2, 2, 24)
    assert train_npz["targets"].shape == (2,)
    assert validation_npz["windows"].shape == (1, 2, 24)
    assert validation_npz["targets"].tolist() == [1.0]
    assert metadata["subset"] == "FD001"
    assert metadata["window_size"] == 2
    assert metadata["standardize"] is True
