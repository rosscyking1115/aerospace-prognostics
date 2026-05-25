from __future__ import annotations

import pandas as pd
import pytest

from aerospace_prognostics.data.cmapss import add_train_rul_targets


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

