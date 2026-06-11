"""Tiny C-MAPSS-compatible fixtures for no-download demos and tests."""

from __future__ import annotations

from pathlib import Path


def cmapss_row(unit: int, cycle: int, sensor_value: float) -> str:
    """Return one whitespace-delimited C-MAPSS row with repeated sensor values."""

    values = [unit, cycle, *([0.0] * 3), *([sensor_value] * 21)]
    return " ".join(str(value) for value in values)


def write_tiny_cmapss_subset(path: str | Path, subset: str = "FD001") -> None:
    """Write a tiny C-MAPSS-compatible subset for demos, CI, and tests."""

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    (output_dir / f"train_{subset}.txt").write_text(
        "\n".join(train_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"test_{subset}.txt").write_text(
        "\n".join(test_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"RUL_{subset}.txt").write_text("1\n1\n", encoding="utf-8")


def write_all_tiny_cmapss_subsets(path: str | Path) -> None:
    """Write tiny fixtures for all four standard C-MAPSS subset names."""

    for subset in ("FD001", "FD002", "FD003", "FD004"):
        write_tiny_cmapss_subset(path, subset=subset)
