"""Feature-table construction for first-pass C-MAPSS baselines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aerospace_prognostics.data.cmapss import OPERATIONAL_SETTING_COLUMNS, SENSOR_COLUMNS

if TYPE_CHECKING:
    import pandas as pd


def cmapss_feature_columns(*, include_settings: bool = True) -> list[str]:
    """Return canonical C-MAPSS model input columns."""

    if include_settings:
        return [*OPERATIONAL_SETTING_COLUMNS, *SENSOR_COLUMNS]
    return list(SENSOR_COLUMNS)


def cycle_feature_table(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_column: str = "rul_capped",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return per-cycle features and target for a classical baseline."""

    columns = feature_columns or cmapss_feature_columns()
    missing = [column for column in [*columns, target_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")

    return frame.loc[:, columns].copy(), frame.loc[:, target_column].copy()


def last_cycle_feature_table(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return one feature row per unit using each unit's final observed cycle."""

    columns = feature_columns or cmapss_feature_columns()
    required = ["unit_number", "time_in_cycles", *columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")

    last_rows = (
        frame.sort_values(["unit_number", "time_in_cycles"])
        .groupby("unit_number", as_index=False)
        .tail(1)
        .sort_values("unit_number")
    )
    return last_rows.loc[:, columns].reset_index(drop=True)
