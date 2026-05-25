from __future__ import annotations

from pathlib import Path


def cmapss_row(unit: int, cycle: int, sensor_value: float) -> str:
    values = [unit, cycle, *([0.0] * 3), *([sensor_value] * 21)]
    return " ".join(str(value) for value in values)


def write_tiny_cmapss_subset(path: Path, subset: str = "FD001") -> None:
    train_rows = [
        cmapss_row(1, 1, 10.0),
        cmapss_row(1, 2, 11.0),
        cmapss_row(1, 3, 12.0),
        cmapss_row(2, 1, 20.0),
        cmapss_row(2, 2, 21.0),
        cmapss_row(2, 3, 22.0),
    ]
    test_rows = [
        cmapss_row(1, 1, 10.5),
        cmapss_row(1, 2, 11.5),
        cmapss_row(2, 1, 20.5),
        cmapss_row(2, 2, 21.5),
    ]

    (path / f"train_{subset}.txt").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (path / f"test_{subset}.txt").write_text("\n".join(test_rows) + "\n", encoding="utf-8")
    (path / f"RUL_{subset}.txt").write_text("1\n1\n", encoding="utf-8")

