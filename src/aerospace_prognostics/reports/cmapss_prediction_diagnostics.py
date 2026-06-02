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


@dataclass(frozen=True)
class CmapssPredictionRulBinDiagnosticRow:
    """Aggregate prediction diagnostics for one actual-RUL range."""

    subset: str
    model_name: str
    actual_rul_bin: str
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
class CmapssPredictionMonotonicityDiagnosticRow:
    """Temporal consistency diagnostics for one subset/model pair."""

    subset: str
    model_name: str
    unit_count: int
    transition_count: int
    violation_count: int
    violation_rate: float
    mean_step_change: float
    mean_violation_magnitude: float
    max_violation_magnitude: float

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
        diagnostics.append(_prediction_diagnostic_row(subset, model_name, group))
    return diagnostics


def build_cmapss_prediction_rul_bin_diagnostics(
    predictions_csv: str | Path,
) -> list[CmapssPredictionRulBinDiagnosticRow]:
    """Build aggregate diagnostics grouped by actual-RUL ranges."""

    rows = _read_prediction_rows(predictions_csv)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        bin_label = _actual_rul_bin(_float(row["actual_rul"]))
        grouped.setdefault((row["subset"], row["model_name"], bin_label), []).append(row)

    diagnostics: list[CmapssPredictionRulBinDiagnosticRow] = []
    for subset, model_name, bin_label in sorted(
        grouped,
        key=lambda key: (key[0], key[1], _actual_rul_bin_sort_key(key[2])),
    ):
        group = grouped[(subset, model_name, bin_label)]
        row = _prediction_diagnostic_row(subset, model_name, group)
        diagnostics.append(
            CmapssPredictionRulBinDiagnosticRow(
                subset=row.subset,
                model_name=row.model_name,
                actual_rul_bin=bin_label,
                prediction_count=row.prediction_count,
                mean_actual_rul=row.mean_actual_rul,
                mean_predicted_rul=row.mean_predicted_rul,
                mean_error=row.mean_error,
                mean_absolute_error=row.mean_absolute_error,
                max_absolute_error=row.max_absolute_error,
                mean_late_error=row.mean_late_error,
                late_prediction_rate=row.late_prediction_rate,
                mean_early_error=row.mean_early_error,
                early_prediction_rate=row.early_prediction_rate,
            )
        )
    return diagnostics


def build_cmapss_prediction_monotonicity_diagnostics(
    predictions_csv: str | Path,
) -> list[CmapssPredictionMonotonicityDiagnosticRow]:
    """Build temporal monotonicity diagnostics from a deep prediction CSV."""

    rows = _read_prediction_rows(predictions_csv, required_columns=("end_cycle",))
    grouped_units: dict[tuple[str, str, int], list[tuple[int, dict[str, str]]]] = {}
    for index, row in enumerate(rows):
        key = (row["subset"], row["model_name"], int(row["unit_number"]))
        grouped_units.setdefault(key, []).append((index, row))

    unit_counts: dict[tuple[str, str], int] = {}
    step_changes_by_model: dict[tuple[str, str], list[float]] = {}
    violations_by_model: dict[tuple[str, str], list[float]] = {}
    for subset, model_name, _unit_number in sorted(grouped_units):
        model_key = (subset, model_name)
        unit_counts[model_key] = unit_counts.get(model_key, 0) + 1
        step_changes = step_changes_by_model.setdefault(model_key, [])
        violations = violations_by_model.setdefault(model_key, [])
        unit_rows = sorted(
            grouped_units[(subset, model_name, _unit_number)],
            key=lambda item: (int(item[1]["end_cycle"]), item[0]),
        )
        for (_previous_index, previous_row), (_next_index, next_row) in zip(
            unit_rows,
            unit_rows[1:],
            strict=False,
        ):
            step_change = _float(next_row["predicted_rul"]) - _float(
                previous_row["predicted_rul"]
            )
            step_changes.append(step_change)
            if step_change > 0.0:
                violations.append(step_change)

    diagnostics: list[CmapssPredictionMonotonicityDiagnosticRow] = []
    for subset, model_name in sorted(unit_counts):
        model_key = (subset, model_name)
        step_changes = step_changes_by_model.get(model_key, [])
        violations = violations_by_model.get(model_key, [])
        transition_count = len(step_changes)
        violation_count = len(violations)
        diagnostics.append(
            CmapssPredictionMonotonicityDiagnosticRow(
                subset=subset,
                model_name=model_name,
                unit_count=unit_counts[model_key],
                transition_count=transition_count,
                violation_count=violation_count,
                violation_rate=violation_count / transition_count
                if transition_count
                else 0.0,
                mean_step_change=_mean(step_changes) if step_changes else 0.0,
                mean_violation_magnitude=_mean(violations) if violations else 0.0,
                max_violation_magnitude=max(violations) if violations else 0.0,
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


def write_cmapss_prediction_rul_bin_diagnostics_csv(
    rows: list[CmapssPredictionRulBinDiagnosticRow],
    path: str | Path,
) -> None:
    """Write actual-RUL bin diagnostics rows as CSV."""

    if not rows:
        raise ValueError("rows must contain at least one item")
    output_path = _prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def write_cmapss_prediction_monotonicity_diagnostics_csv(
    rows: list[CmapssPredictionMonotonicityDiagnosticRow],
    path: str | Path,
) -> None:
    """Write temporal monotonicity diagnostics rows as CSV."""

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
    *,
    rul_bin_diagnostics: list[CmapssPredictionRulBinDiagnosticRow] | None = None,
    monotonicity_diagnostics: list[CmapssPredictionMonotonicityDiagnosticRow] | None = None,
) -> None:
    """Write prediction diagnostics as a compact Markdown report."""

    output_path = _prepare_output_path(path)
    output_path.write_text(
        render_cmapss_prediction_diagnostics_markdown(
            diagnostics,
            outliers,
            rul_bin_diagnostics=rul_bin_diagnostics,
            monotonicity_diagnostics=monotonicity_diagnostics,
        ),
        encoding="utf-8",
    )


def render_cmapss_prediction_diagnostics_markdown(
    diagnostics: list[CmapssPredictionDiagnosticRow],
    outliers: list[CmapssPredictionOutlierRow],
    *,
    rul_bin_diagnostics: list[CmapssPredictionRulBinDiagnosticRow] | None = None,
    monotonicity_diagnostics: list[CmapssPredictionMonotonicityDiagnosticRow] | None = None,
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
    if rul_bin_diagnostics:
        lines.extend(
            [
                "",
                "## Error By Actual RUL Bin",
                "",
                (
                    "| Subset | Model | Actual RUL Bin | Rows | Mean Error | "
                    "Mean Abs Error | Max Abs Error | Late Rate | Early Rate |"
                ),
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rul_bin_diagnostics:
            lines.append(
                "| "
                f"{row.subset} | "
                f"`{row.model_name}` | "
                f"{row.actual_rul_bin} | "
                f"{row.prediction_count} | "
                f"{row.mean_error:.6f} | "
                f"{row.mean_absolute_error:.6f} | "
                f"{row.max_absolute_error:.6f} | "
                f"{row.late_prediction_rate:.6f} | "
                f"{row.early_prediction_rate:.6f} |"
            )
    if monotonicity_diagnostics:
        lines.extend(
            [
                "",
                "## Prediction Monotonicity",
                "",
                (
                    "| Subset | Model | Units | Transitions | Violations | Violation Rate | "
                    "Mean Step Change | Mean Violation | Max Violation |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in monotonicity_diagnostics:
            lines.append(
                "| "
                f"{row.subset} | "
                f"`{row.model_name}` | "
                f"{row.unit_count} | "
                f"{row.transition_count} | "
                f"{row.violation_count} | "
                f"{row.violation_rate:.6f} | "
                f"{row.mean_step_change:.6f} | "
                f"{row.mean_violation_magnitude:.6f} | "
                f"{row.max_violation_magnitude:.6f} |"
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


def _read_prediction_rows(
    path: str | Path,
    *,
    required_columns: Iterable[str] = (),
) -> list[dict[str, str]]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"missing prediction CSV: {prediction_path}")
    with prediction_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"prediction CSV has no rows: {prediction_path}")
    base_required_columns = {
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
    missing_columns = (base_required_columns | set(required_columns)) - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"prediction CSV is missing required columns: {missing}")
    return rows


def _prediction_diagnostic_row(
    subset: str,
    model_name: str,
    rows: list[dict[str, str]],
) -> CmapssPredictionDiagnosticRow:
    absolute_errors = [_float(row["absolute_error"]) for row in rows]
    late_errors = [_float(row["late_error"]) for row in rows]
    early_errors = [_float(row["early_error"]) for row in rows]
    return CmapssPredictionDiagnosticRow(
        subset=subset,
        model_name=model_name,
        prediction_count=len(rows),
        mean_actual_rul=_mean(_float(row["actual_rul"]) for row in rows),
        mean_predicted_rul=_mean(_float(row["predicted_rul"]) for row in rows),
        mean_error=_mean(_float(row["error"]) for row in rows),
        mean_absolute_error=_mean(absolute_errors),
        max_absolute_error=max(absolute_errors),
        mean_late_error=_mean(late_errors),
        late_prediction_rate=_rate(value > 0 for value in late_errors),
        mean_early_error=_mean(early_errors),
        early_prediction_rate=_rate(value > 0 for value in early_errors),
    )


def _actual_rul_bin(actual_rul: float) -> str:
    if actual_rul <= 30:
        return "0-30"
    if actual_rul <= 60:
        return "31-60"
    if actual_rul <= 90:
        return "61-90"
    if actual_rul <= 120:
        return "91-120"
    return "121+"


def _actual_rul_bin_sort_key(label: str) -> int:
    order = {"0-30": 0, "31-60": 1, "61-90": 2, "91-120": 3, "121+": 4}
    return order[label]


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
