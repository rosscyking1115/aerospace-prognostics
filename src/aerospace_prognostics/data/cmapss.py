"""NASA C-MAPSS data loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


OPERATIONAL_SETTING_COLUMNS = [f"op_setting_{index}" for index in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
CMAPSS_COLUMNS = ["unit_number", "time_in_cycles", *OPERATIONAL_SETTING_COLUMNS, *SENSOR_COLUMNS]
CMAPSS_SUBSETS = ("FD001", "FD002", "FD003", "FD004")


@dataclass(frozen=True)
class CmapssSubset:
    """Loaded C-MAPSS train/test files for one subset."""

    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series


def read_cmapss_frame(path: str | Path) -> pd.DataFrame:
    """Read a C-MAPSS train/test text file with canonical column names."""

    import pandas as pd

    return pd.read_csv(
        Path(path),
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
        usecols=range(len(CMAPSS_COLUMNS)),
        engine="python",
    )


def add_train_rul_targets(frame: pd.DataFrame, *, cap: int | None = None) -> pd.DataFrame:
    """Add uncapped and optional capped RUL targets to a C-MAPSS training frame."""

    required = {"unit_number", "time_in_cycles"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    result = frame.copy()
    max_cycles = result.groupby("unit_number")["time_in_cycles"].transform("max")
    result["rul"] = max_cycles - result["time_in_cycles"]
    if cap is not None:
        if cap <= 0:
            raise ValueError("cap must be positive")
        result["rul_capped"] = result["rul"].clip(upper=cap)
    return result


def read_test_rul(path: str | Path) -> pd.Series:
    """Read the C-MAPSS `RUL_FD*.txt` file as one value per test unit."""

    import pandas as pd

    return pd.read_csv(Path(path), sep=r"\s+", header=None, usecols=[0], engine="python")[0]


def load_cmapss_subset(root: str | Path, subset: str, *, rul_cap: int = 125) -> CmapssSubset:
    """Load train, test, and held-out RUL labels for a C-MAPSS subset."""

    normalised_subset = subset.upper()
    if normalised_subset not in CMAPSS_SUBSETS:
        raise ValueError(f"subset must be one of {CMAPSS_SUBSETS}")

    root_path = Path(root)
    train_path = root_path / f"train_{normalised_subset}.txt"
    test_path = root_path / f"test_{normalised_subset}.txt"
    rul_path = root_path / f"RUL_{normalised_subset}.txt"

    missing = [path for path in [train_path, test_path, rul_path] if not path.exists()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing C-MAPSS files: {missing_names}")

    return CmapssSubset(
        subset=normalised_subset,
        train=add_train_rul_targets(read_cmapss_frame(train_path), cap=rul_cap),
        test=read_cmapss_frame(test_path),
        test_rul=read_test_rul(rul_path),
    )
