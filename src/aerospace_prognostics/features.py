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


def engineered_cycle_feature_table(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_column: str = "rul_capped",
    rolling_window: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return per-cycle engineered features and target for a stronger classical baseline."""

    features = engineered_feature_frame(
        frame,
        feature_columns=feature_columns,
        rolling_window=rolling_window,
    )
    missing = [target_column] if target_column not in frame.columns else []
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    return features, frame.loc[:, target_column].copy()


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


def engineered_last_cycle_feature_table(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Return engineered final-observation features for each unit."""

    features = engineered_feature_frame(
        frame,
        feature_columns=feature_columns,
        rolling_window=rolling_window,
    )
    keyed_features = features.copy()
    keyed_features["_unit_number"] = frame["unit_number"]
    keyed_features["_time_in_cycles"] = frame["time_in_cycles"]
    last_rows = (
        keyed_features.sort_values(["_unit_number", "_time_in_cycles"])
        .groupby("_unit_number", as_index=False)
        .tail(1)
        .sort_values("_unit_number")
    )
    return last_rows.loc[:, features.columns].reset_index(drop=True)


def engineered_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Build leakage-safe rolling and degradation-delta features for C-MAPSS rows."""

    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")

    columns = feature_columns or cmapss_feature_columns()
    required = ["unit_number", "time_in_cycles", *columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")

    sorted_frame = frame.sort_values(["unit_number", "time_in_cycles"]).copy()
    features = sorted_frame.loc[:, columns].copy()
    features.insert(0, "time_in_cycles", sorted_frame["time_in_cycles"].astype(float))

    grouped = sorted_frame.groupby("unit_number", sort=False)
    sensor_columns = [column for column in columns if column in SENSOR_COLUMNS]
    for column in sensor_columns:
        rolling = grouped[column].rolling(rolling_window, min_periods=1)
        rolling_mean = rolling.mean().reset_index(level=0, drop=True)
        rolling_min = rolling.min().reset_index(level=0, drop=True)
        rolling_max = rolling.max().reset_index(level=0, drop=True)
        initial_value = grouped[column].transform("first")

        features[f"{column}_rolling_mean_{rolling_window}"] = rolling_mean
        features[f"{column}_rolling_range_{rolling_window}"] = rolling_max - rolling_min
        features[f"{column}_delta_from_initial"] = sorted_frame[column] - initial_value
        features[f"{column}_rolling_slope_{rolling_window}"] = (
            sorted_frame[column] - grouped[column].shift(rolling_window - 1)
        ).fillna(0.0) / (rolling_window - 1)

    return features.sort_index()
