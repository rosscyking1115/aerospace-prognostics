from __future__ import annotations

import pandas as pd
import pytest

from aerospace_prognostics.data.cmapss import (
    CMAPSS_COLUMNS,
    add_train_rul_targets,
    load_cmapss_subset,
)


def test_add_train_rul_targets_per_unit() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_in_cycles": [1, 2, 3, 1, 2],
        }
    )

    result = add_train_rul_targets(frame, cap=1)

    assert result["rul"].tolist() == [2, 1, 0, 1, 0]
    assert result["rul_capped"].tolist() == [1, 1, 0, 1, 0]


def test_add_train_rul_targets_requires_core_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        add_train_rul_targets(pd.DataFrame({"unit_number": [1]}))


def test_load_cmapss_subset_reads_expected_files(tmp_path) -> None:
    train_row = [1, 1, *([0.0] * 3), *([10.0] * 21)]
    test_row = [1, 1, *([0.0] * 3), *([11.0] * 21)]

    (tmp_path / "train_FD001.txt").write_text(
        " ".join(str(value) for value in train_row) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "test_FD001.txt").write_text(
        " ".join(str(value) for value in test_row) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "RUL_FD001.txt").write_text("42\n", encoding="utf-8")

    bundle = load_cmapss_subset(tmp_path, "fd001", rul_cap=125)

    assert bundle.subset == "FD001"
    assert bundle.train.columns.tolist() == [*CMAPSS_COLUMNS, "rul", "rul_capped"]
    assert bundle.test.columns.tolist() == CMAPSS_COLUMNS
    assert bundle.test_rul.tolist() == [42]


def test_load_cmapss_subset_validates_subset_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="subset must be one of"):
        load_cmapss_subset(tmp_path, "FD999")
