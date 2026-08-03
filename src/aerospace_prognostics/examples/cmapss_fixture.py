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


# Training engines, as (unit number, total life in cycles). Lives vary so the
# pooled RUL distribution is broad and no constant sits near every test answer.
_DISCRIMINATING_TRAIN_LIVES = (
    (1, 42),
    (2, 55),
    (3, 61),
    (4, 48),
    (5, 73),
    (6, 66),
    (7, 51),
    (8, 88),
    (9, 79),
    (10, 44),
    (11, 94),
    (12, 58),
)

# Test engines, as (unit number, true remaining useful life at truncation).
# Deliberately spread across the range: a single constant is badly wrong on at
# least one end no matter which constant is chosen.
_DISCRIMINATING_TEST_RULS = (
    (1, 5),
    (2, 20),
    (3, 45),
    (4, 70),
    (5, 90),
)

_DISCRIMINATING_HEALTHY_SENSOR = 100.0


def _discriminating_sensor_value(unit: int, cycle: int, rul: int) -> float:
    """Return a sensor reading that falls as remaining life falls.

    The signal is ``100 - rul`` plus a small deterministic wobble, so a model
    can recover RUL from the sensor level while the data still looks noisy
    rather than analytically perfect. The wobble is derived from the unit and
    cycle numbers rather than an RNG, so the fixture is byte-identical on every
    run and the tests built on it cannot flake.
    """
    wobble = (((unit * 7 + cycle * 13) % 5) - 2) * 0.3
    return _DISCRIMINATING_HEALTHY_SENSOR - float(rul) + wobble


def write_discriminating_cmapss_subset(path: str | Path, subset: str = "FD001") -> None:
    """Write a fixture on which a real model must beat a constant predictor.

    The tiny fixture in :func:`write_tiny_cmapss_subset` is deliberately
    minimal, and one consequence is that it cannot tell a working model from a
    broken one: its test RUL values coincide with its train median, so a
    constant predictor scores a perfect RMSE of 0.0 and nothing can do better.

    This fixture removes both halves of that coincidence:

    - test units are truncated at five different remaining lifetimes, so no
      single constant fits them all and the naive floor is forced above zero;
    - sensor readings carry a monotone degradation signal that a real
      estimator can invert, so a working model scores far below that floor.

    It is still small enough to generate per test (759 train rows and 200 test
    rows) and ships no real NASA data, staying inside the repository's
    no-redistribution posture.
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        cmapss_row(unit, cycle, _discriminating_sensor_value(unit, cycle, life - cycle))
        for unit, life in _DISCRIMINATING_TRAIN_LIVES
        for cycle in range(1, life + 1)
    ]

    # Each test unit is observed up to truncation; its recorded RUL is what the
    # model must predict from the final cycle, exactly as in the official files.
    test_rows = [
        cmapss_row(
            unit,
            cycle,
            _discriminating_sensor_value(unit, cycle, observed_cycles - cycle + rul),
        )
        for unit, rul in _DISCRIMINATING_TEST_RULS
        for observed_cycles in (40,)
        for cycle in range(1, observed_cycles + 1)
    ]

    (output_dir / f"train_{subset}.txt").write_text(
        "\n".join(train_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"test_{subset}.txt").write_text(
        "\n".join(test_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"RUL_{subset}.txt").write_text(
        "\n".join(str(rul) for _unit, rul in _DISCRIMINATING_TEST_RULS) + "\n",
        encoding="utf-8",
    )


def write_all_tiny_cmapss_subsets(path: str | Path) -> None:
    """Write tiny fixtures for all four standard C-MAPSS subset names."""
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        write_tiny_cmapss_subset(path, subset=subset)
