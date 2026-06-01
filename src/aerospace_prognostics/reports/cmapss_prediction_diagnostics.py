"""Prediction diagnostics for C-MAPSS deep RUL model outputs."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CmapssPredictionDiagnosticRow:
    """Aggregate prediction diagnostics for one subset/model pair."""

    subset: str
    model_name: str
    prediction_count: int
    mean_actual_rul: float
    mean_predicted_rul: float
    mean_error: float
    mean_absolute_error: float
    max_absolute_error: float
    mean_late_error: float
    late_prediction_rate: float
    mean_early_error: float
    early_prediction_rate: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a flat serialisable row."""

        return asdict(self)


@dataclass(frozen=True)
class CmapssPredictionOutlierRow:
    """High-error official-test prediction row."""

    rank_by_absolute_error: int
    subset: str
    model_name: str
    unit_number: int
    actual_rul: float
    predicted_rul: float
    error: float
    absolute_error: float
    late_error: float
    early_error: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a flat serialisable row."""

        return asdict(self)


def build_cmapss_prediction_diagnostics(
    predictions_csv: str | Path,
) -> list[CmapssPredictionDiagnosticRow]:
    """Build aggregate diagnostics from a deep prediction CSV."""

    rows = _read_prediction_rows(predictions_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["subset"], row["model_name"]), []).append(row)

    diagnostics: list[CmapssPredictionDiagnosticRow] = []
    for subset, model_name in sorted(grouped):
        group = grouped[(subset, model_name)]
        absolute_errors = [_float(row["absolute_error"]) for row in group]
        late_errors = [_float(row["late_error"]) for row in group]
        early_errors = [_float(row["early_error"]) for row in group]
        diagnostics.append(
            CmapssPredictionDiagnosticRow(
                subset=subset,
                model_name=model_name,
                prediction_count=len(group),
                mean_actual_rul=_mean(_float(row["actual_rul"]) for row in group),
                mean_predicted_rul=_mean(
                    _float(row["predicted_rul"]) for row in group
                ),
                mean_error=_mean(_float(row["error"]) for row in group),
                mean_absolute_error=_mean(absolute_errors),
                max_absolute_error=max(absolute_errors),
                mean_late_error=_mean(late_errors),
                late_prediction_rate=_rate(value > 0 for value in late_errors),
                mean_early_error=_mean(early_errors),
                early_prediction_rate=_rate(value > 0 for value in early_errors),
            )
        )
    return diagnostics


def select_cmapss_high_error_predictions(
    predictions_csv: str | Path,
    *,
    top_n: int = 10,
) -> list[CmapssPredictionOutlierRow]:
    """Select the highest-absolute-error prediction rows."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    rows = sorted(
        _read_prediction_rows(predictions_csv),
        key=lambda row: (
            -_float(row["absolute_error"]),
            row["subset"],
            row["model_name"],
            int(row["unit_number"]),
        ),
    )[:top_n]
    return [
        CmapssPredictionOutlierRow(
            rank_by_absolute_error=rank,
            subset=row["subset"],
            model_name=row["model_name"],
            unit_number=int(row["unit_number"]),
            actual_rul=_float(row["actual_rul"]),
            predicted_rul=_float(row["predicted_rul"]),
            error=_float(row["error"]),
            absolute_error=_float(row["absolute_error"]),
            late_error=_float(row["late_error"]),
            early_error=_float(row["early_error"]),
        )
        for rank, row in enumerate(rows, start=1)
    ]


def write_cmapss_prediction_diagnostics_csv(
    rows: list[CmapssPredictionDiagnosticRow],
    path: str | Path,
) -> None:
    """Write aggregate diagnostics rows as CSV."""

    if not rows:
        raise ValueError("rows must contain at least one item")
    output_path = _prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def write_cmapss_prediction_diagnostics_markdown(
    diagnostics: list[CmapssPredictionDiagnosticRow],
    outliers: list[CmapssPredictionOutlierRow],
    path: str | Path,
) -> None:
    """Write prediction diagnostics as a compact Markdown report."""

    output_path = _prepare_output_path(path)
    output_path.write_text(
        render_cmapss_prediction_diagnostics_markdown(diagnostics, outliers),
        encoding="utf-8",
    )


def render_cmapss_prediction_diagnostics_markdown(
    diagnostics: list[CmapssPredictionDiagnosticRow],
    outliers: list[CmapssPredictionOutlierRow],
) -> str:
    """Render prediction diagnostics as Markdown tables."""

    if not diagnostics:
        raise ValueError("diagnostics must contain at least one item")
    lines = [
        "# C-MAPSS Deep Prediction Diagnostics",
        "",
        "## Model Error Summary",
        "",
        (
            "| Subset | Model | Rows | Mean Error | Mean Abs Error | "
            "Max Abs Error | Mean Late Error | Late Rate | Mean Early Error | Early Rate |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(
            "| "
            f"{row.subset} | "
            f"`{row.model_name}` | "
            f"{row.prediction_count} | "
            f"{row.mean_error:.6f} | "
            f"{row.mean_absolute_error:.6f} | "
            f"{row.max_absolute_error:.6f} | "
            f"{row.mean_late_error:.6f} | "
            f"{row.late_prediction_rate:.6f} | "
            f"{row.mean_early_error:.6f} | "
            f"{row.early_prediction_rate:.6f} |"
        )
    if outliers:
        lines.extend(
            [
                "",
                "## Highest Absolute Errors",
                "",
                (
                    "| Rank | Subset | Model | Unit | Actual RUL | Predicted RUL | "
                    "Error | Abs Error | Late Error | Early Error |"
                ),
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in outliers:
            lines.append(
                "| "
                f"{row.rank_by_absolute_error} | "
                f"{row.subset} | "
                f"`{row.model_name}` | "
                f"{row.unit_number} | "
                f"{row.actual_rul:.6f} | "
                f"{row.predicted_rul:.6f} | "
                f"{row.error:.6f} | "
                f"{row.absolute_error:.6f} | "
                f"{row.late_error:.6f} | "
                f"{row.early_error:.6f} |"
            )
    return "\n".join(lines) + "\n"


def _read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"missing prediction CSV: {prediction_path}")
    with prediction_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"prediction CSV has no rows: {prediction_path}")
    required_columns = {
        "subset",
        "model_name",
        "unit_number",
        "actual_rul",
        "predicted_rul",
        "error",
        "absolute_error",
        "late_error",
        "early_error",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"prediction CSV is missing required columns: {missing}")
    return rows


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("cannot average an empty metric sequence")
    return sum(sequence) / len(sequence)


def _rate(values: Iterable[bool]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("cannot compute a rate for an empty sequence")
    return sum(1 for value in sequence if value) / len(sequence)


def _float(value: Any) -> float:
    return float(value)


def _prepare_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
