"""Reusable evaluation result containers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload


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
    standardize: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        """Write this result as pretty JSON."""
        write_json_payload(self.to_dict(), path)


def write_results_json(results: list[RegressionRunResult], path: str | Path) -> None:
    """Write multiple run results as pretty JSON."""
    payload = [result.to_dict() for result in results]
    write_json_payload(payload, path)


def write_results_csv(results: list[RegressionRunResult], path: str | Path) -> None:
    """Write multiple run results as a flat CSV table."""
    if not results:
        raise ValueError("results must contain at least one item")

    output_path = prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].to_dict()))
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)
