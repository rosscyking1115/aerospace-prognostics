"""Reusable evaluation result containers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegressionRunResult:
    """Structured result for one RUL regression run."""

    dataset: str
    subset: str
    model_name: str
    rmse: float
    nasa_score: float
    train_rows: int
    train_units: int
    test_rows: int
    test_units: int
    test_rul_values: int
    rul_cap: int
    random_state: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""

        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        """Write this result as pretty JSON."""

        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

