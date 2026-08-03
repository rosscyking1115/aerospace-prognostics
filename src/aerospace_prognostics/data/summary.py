"""Dataset summary helpers for C-MAPSS inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class CmapssFrameSummary:
    """Compact summary of one C-MAPSS frame."""

    rows: int
    units: int
    min_cycle: int
    max_cycle: int
    min_unit_cycles: int
    max_unit_cycles: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


def summarise_cmapss_frame(frame: pd.DataFrame) -> CmapssFrameSummary:
    """Return unit/cycle counts for a C-MAPSS train or test frame."""
    required = {"unit_number", "time_in_cycles"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("frame must contain at least one row")

    unit_cycle_counts = frame.groupby("unit_number")["time_in_cycles"].count()
    return CmapssFrameSummary(
        rows=len(frame),
        units=frame["unit_number"].nunique(),
        min_cycle=int(frame["time_in_cycles"].min()),
        max_cycle=int(frame["time_in_cycles"].max()),
        min_unit_cycles=int(unit_cycle_counts.min()),
        max_unit_cycles=int(unit_cycle_counts.max()),
    )

