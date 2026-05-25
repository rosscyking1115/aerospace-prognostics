from __future__ import annotations

import pandas as pd
import pytest

from aerospace_prognostics.features import (
    cmapss_feature_columns,
    cycle_feature_table,
    last_cycle_feature_table,
)


def test_cmapss_feature_columns_can_exclude_settings() -> None:
    columns = cmapss_feature_columns(include_settings=False)

    assert columns[0] == "sensor_1"
    assert columns[-1] == "sensor_21"
    assert "op_setting_1" not in columns


def test_cycle_feature_table_returns_features_and_target() -> None:
    frame = pd.DataFrame(
        {
            "op_setting_1": [0.0, 1.0],
            "sensor_1": [10.0, 11.0],
            "rul_capped": [30, 29],
        }
    )

    features, target = cycle_feature_table(
        frame,
        feature_columns=["op_setting_1", "sensor_1"],
    )

    assert features.to_dict(orient="list") == {"op_setting_1": [0.0, 1.0], "sensor_1": [10.0, 11.0]}
    assert target.tolist() == [30, 29]


def test_last_cycle_feature_table_returns_one_row_per_unit() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [2, 1, 1, 2],
            "time_in_cycles": [1, 2, 1, 2],
            "sensor_1": [20.0, 11.0, 10.0, 21.0],
        }
    )

    features = last_cycle_feature_table(frame, feature_columns=["sensor_1"])

    assert features["sensor_1"].tolist() == [11.0, 21.0]


def test_feature_tables_validate_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        cycle_feature_table(pd.DataFrame({"sensor_1": [1]}), feature_columns=["sensor_1"])

    with pytest.raises(ValueError, match="missing columns"):
        last_cycle_feature_table(pd.DataFrame({"sensor_1": [1]}), feature_columns=["sensor_1"])

