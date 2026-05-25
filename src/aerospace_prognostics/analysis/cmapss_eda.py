"""C-MAPSS exploratory analysis summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aerospace_prognostics.data.cmapss import (
    OPERATIONAL_SETTING_COLUMNS,
    SENSOR_COLUMNS,
    CmapssSubset,
)


@dataclass(frozen=True)
class SensorEdaSummary:
    """Compact per-sensor summary for C-MAPSS training data."""

    sensor: str
    mean: float
    std: float
    min_value: float
    max_value: float
    missing_fraction: float
    rul_correlation: float | None
    early_life_mean: float
    late_life_mean: float
    drift: float
    is_near_constant: bool


@dataclass(frozen=True)
class CmapssEdaReport:
    """Structured EDA report for one C-MAPSS subset."""

    subset: str
    train_rows: int
    train_units: int
    test_rows: int
    test_units: int
    sensor_summaries: list[SensorEdaSummary]
    operating_setting_ranges: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""

        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        """Write this report as pretty JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def build_cmapss_eda_report(
    bundle: CmapssSubset,
    *,
    early_fraction: float = 0.25,
    late_fraction: float = 0.25,
    near_constant_std: float = 1e-8,
) -> CmapssEdaReport:
    """Build an EDA report from a loaded C-MAPSS subset."""

    if not 0 < early_fraction <= 1:
        raise ValueError("early_fraction must be in the interval (0, 1]")
    if not 0 < late_fraction <= 1:
        raise ValueError("late_fraction must be in the interval (0, 1]")

    train = bundle.train
    sensor_summaries = [
        _summarise_sensor(
            train,
            sensor,
            early_fraction=early_fraction,
            late_fraction=late_fraction,
            near_constant_std=near_constant_std,
        )
        for sensor in SENSOR_COLUMNS
    ]

    return CmapssEdaReport(
        subset=bundle.subset,
        train_rows=len(bundle.train),
        train_units=bundle.train["unit_number"].nunique(),
        test_rows=len(bundle.test),
        test_units=bundle.test["unit_number"].nunique(),
        sensor_summaries=sensor_summaries,
        operating_setting_ranges=_operating_setting_ranges(train),
    )


def _summarise_sensor(
    frame,
    sensor: str,
    *,
    early_fraction: float,
    late_fraction: float,
    near_constant_std: float,
) -> SensorEdaSummary:
    values = frame[sensor]
    std = float(values.std(ddof=0))
    early_life_mean = _lifecycle_mean(frame, sensor, fraction=early_fraction, from_start=True)
    late_life_mean = _lifecycle_mean(frame, sensor, fraction=late_fraction, from_start=False)
    correlation = frame[[sensor, "rul"]].corr().iloc[0, 1]
    return SensorEdaSummary(
        sensor=sensor,
        mean=float(values.mean()),
        std=std,
        min_value=float(values.min()),
        max_value=float(values.max()),
        missing_fraction=float(values.isna().mean()),
        rul_correlation=None if _is_nan(correlation) else float(correlation),
        early_life_mean=early_life_mean,
        late_life_mean=late_life_mean,
        drift=late_life_mean - early_life_mean,
        is_near_constant=std <= near_constant_std,
    )


def _lifecycle_mean(frame, sensor: str, *, fraction: float, from_start: bool) -> float:
    selected_values = []
    sorted_frame = frame.sort_values(["unit_number", "time_in_cycles"])
    for _, unit_frame in sorted_frame.groupby("unit_number"):
        take_count = max(1, int(round(len(unit_frame) * fraction)))
        selected = unit_frame.head(take_count) if from_start else unit_frame.tail(take_count)
        selected_values.append(selected[sensor])
    total = sum(series.sum() for series in selected_values)
    count = sum(len(series) for series in selected_values)
    return float(total / count)


def _operating_setting_ranges(frame) -> dict[str, dict[str, float]]:
    return {
        column: {"min": float(frame[column].min()), "max": float(frame[column].max())}
        for column in OPERATIONAL_SETTING_COLUMNS
    }


def _is_nan(value: float) -> bool:
    return value != value
