from __future__ import annotations

import pandas as pd
import pytest

from aerospace_prognostics.data.summary import summarise_cmapss_frame


def test_summarise_cmapss_frame_counts_rows_units_and_cycles() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_in_cycles": [1, 2, 3, 1, 2],
        }
    )

    summary = summarise_cmapss_frame(frame)

    assert summary.to_dict() == {
        "rows": 5,
        "units": 2,
        "min_cycle": 1,
        "max_cycle": 3,
        "min_unit_cycles": 2,
        "max_unit_cycles": 3,
    }


def test_summarise_cmapss_frame_validates_input() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        summarise_cmapss_frame(pd.DataFrame({"unit_number": [], "time_in_cycles": []}))

    with pytest.raises(ValueError, match="missing required columns"):
        summarise_cmapss_frame(pd.DataFrame({"unit_number": [1]}))

