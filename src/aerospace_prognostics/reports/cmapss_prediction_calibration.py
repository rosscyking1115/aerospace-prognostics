"""Post-hoc calibration utilities for C-MAPSS deep prediction CSVs."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aerospace_prognostics.artifact_io import prepare_output_path
from aerospace_prognostics.metrics import nasa_rul_score
from aerospace_prognostics.reports.cmapss_prediction_diagnostics import (
    CmapssPredictionDiagnosticRow,
    CmapssPredictionRulBinDiagnosticRow,
    CmapssPredictionUnitDiagnosticRow,
    build_cmapss_prediction_diagnostics,
    build_cmapss_prediction_rul_bin_diagnostics,
    build_cmapss_prediction_unit_diagnostics,
    select_cmapss_high_error_predictions,
    write_cmapss_prediction_diagnostics_csv,
    write_cmapss_prediction_diagnostics_markdown,
    write_cmapss_prediction_rul_bin_diagnostics_csv,
    write_cmapss_prediction_unit_diagnostics_csv,
)


@dataclass(frozen=True)
class CmapssAffinePredictionCalibration:
    """Affine calibration fitted for one subset/model pair."""

    subset: str
    model_name: str
    calibration_count: int
    intercept: float
    slope: float
    mean_actual_rul: float
    mean_raw_predicted_rul: float
    raw_predicted_rul_variance: float
    clip_min: float
    method: str = "validation_affine"

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a flat serialisable row."""
        return asdict(self)


@dataclass(frozen=True)
class CmapssPredictedRulBinResidualCalibration:
    """Residual calibration fitted for one raw-predicted-RUL bin."""

    subset: str
    model_name: str
    predicted_rul_bin: str
    lower_bound: float
    upper_bound: float | str
    calibration_count: int
    mean_actual_rul: float
    mean_raw_predicted_rul: float
    mean_error: float
    shrinkage_weight: float
    correction: float
    clip_min: float
    method: str = "validation_predicted_bin_residual"

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a flat serialisable row."""
        return asdict(self)


CmapssPredictionCalibration = (
    CmapssAffinePredictionCalibration | CmapssPredictedRulBinResidualCalibration
)


@dataclass(frozen=True)
class CmapssPredictionCalibrationResult:
    """Paths and summaries produced by a calibrated prediction run."""

    calibrated_predictions_csv_path: Path
    calibrations: tuple[CmapssPredictionCalibration, ...]
    calibrated_prediction_count: int
    calibration_csv_path: Path | None = None
    diagnostics_csv_path: Path | None = None
    rul_bins_csv_path: Path | None = None
    unit_diagnostics_csv_path: Path | None = None
    diagnostics_markdown_path: Path | None = None
    diagnostics: tuple[CmapssPredictionDiagnosticRow, ...] = ()
    rul_bin_diagnostics: tuple[CmapssPredictionRulBinDiagnosticRow, ...] = ()
    unit_diagnostics: tuple[CmapssPredictionUnitDiagnosticRow, ...] = ()


def fit_cmapss_affine_prediction_calibrations(
    calibration_csv: str | Path,
    *,
    clip_min: float = 0.0,
) -> tuple[CmapssAffinePredictionCalibration, ...]:
    """Fit one affine RUL calibration per subset/model from validation predictions."""
    rows = _read_prediction_rows(calibration_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["subset"], row["model_name"]), []).append(row)

    calibrations: list[CmapssAffinePredictionCalibration] = []
    for subset, model_name in sorted(grouped):
        group = grouped[(subset, model_name)]
        calibrations.append(
            _fit_group_affine_calibration(
                subset,
                model_name,
                group,
                clip_min=clip_min,
            )
        )
    return tuple(calibrations)


def fit_cmapss_predicted_rul_bin_residual_calibrations(
    calibration_csv: str | Path,
    *,
    shrinkage_strength: float = 100.0,
    clip_min: float = 0.0,
) -> tuple[CmapssPredictedRulBinResidualCalibration, ...]:
    """Fit residual corrections by raw predicted-RUL bin."""
    if shrinkage_strength < 0.0:
        raise ValueError("shrinkage_strength must be non-negative")
    rows = _read_prediction_rows(calibration_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["subset"], row["model_name"]), []).append(row)

    calibrations: list[CmapssPredictedRulBinResidualCalibration] = []
    for subset, model_name in sorted(grouped):
        group = grouped[(subset, model_name)]
        calibrations.append(
            _fit_predicted_rul_bin_residual_calibration(
                subset,
                model_name,
                "all",
                group,
                shrinkage_strength=shrinkage_strength,
                clip_min=clip_min,
            )
        )
        rows_by_bin: dict[str, list[dict[str, str]]] = {}
        for row in group:
            bin_label = _predicted_rul_bin(_float(row["predicted_rul"]))
            rows_by_bin.setdefault(bin_label, []).append(row)
        for bin_label in _PREDICTED_RUL_BIN_ORDER:
            if bin_label in rows_by_bin:
                calibrations.append(
                    _fit_predicted_rul_bin_residual_calibration(
                        subset,
                        model_name,
                        bin_label,
                        rows_by_bin[bin_label],
                        shrinkage_strength=shrinkage_strength,
                        clip_min=clip_min,
                    )
                )
    return tuple(calibrations)


def fit_cmapss_predicted_rul_bin_nasa_shift_calibrations(
    calibration_csv: str | Path,
    *,
    shrinkage_strength: float = 100.0,
    max_shift: float = 30.0,
    shift_step: float = 1.0,
    clip_min: float = 0.0,
) -> tuple[CmapssPredictedRulBinResidualCalibration, ...]:
    """Fit raw-predicted-RUL bin shifts by validation NASA score."""
    if shrinkage_strength < 0.0:
        raise ValueError("shrinkage_strength must be non-negative")
    if max_shift < 0.0:
        raise ValueError("max_shift must be non-negative")
    if shift_step <= 0.0:
        raise ValueError("shift_step must be positive")
    rows = _read_prediction_rows(calibration_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["subset"], row["model_name"]), []).append(row)

    calibrations: list[CmapssPredictedRulBinResidualCalibration] = []
    for subset, model_name in sorted(grouped):
        group = grouped[(subset, model_name)]
        calibrations.append(
            _fit_predicted_rul_bin_nasa_shift_calibration(
                subset,
                model_name,
                "all",
                group,
                shrinkage_strength=shrinkage_strength,
                max_shift=max_shift,
                shift_step=shift_step,
                clip_min=clip_min,
            )
        )
        rows_by_bin: dict[str, list[dict[str, str]]] = {}
        for row in group:
            bin_label = _predicted_rul_bin(_float(row["predicted_rul"]))
            rows_by_bin.setdefault(bin_label, []).append(row)
        for bin_label in _PREDICTED_RUL_BIN_ORDER:
            if bin_label in rows_by_bin:
                calibrations.append(
                    _fit_predicted_rul_bin_nasa_shift_calibration(
                        subset,
                        model_name,
                        bin_label,
                        rows_by_bin[bin_label],
                        shrinkage_strength=shrinkage_strength,
                        max_shift=max_shift,
                        shift_step=shift_step,
                        clip_min=clip_min,
                    )
                )
    return tuple(calibrations)


def write_cmapss_affine_prediction_calibration_csv(
    calibrations: Iterable[CmapssPredictionCalibration],
    path: str | Path,
) -> Path:
    """Write fitted calibration parameters as CSV."""
    calibration_rows = tuple(calibrations)
    if not calibration_rows:
        raise ValueError("calibrations must contain at least one item")
    output_path = prepare_output_path(path)
    fieldnames = _calibration_fieldnames(calibration_rows)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(row.to_dict() for row in calibration_rows)
    return output_path


def write_cmapss_affine_calibrated_predictions_csv(
    predictions_csv: str | Path,
    output_csv: str | Path,
    calibrations: Iterable[CmapssPredictionCalibration],
) -> int:
    """Apply fitted calibrations and write a recalculated prediction CSV."""
    prediction_path = Path(predictions_csv)
    rows, fieldnames = _read_prediction_rows_with_fieldnames(prediction_path)
    calibration_rows = tuple(calibrations)
    affine_by_key = {
        (calibration.subset, calibration.model_name): calibration
        for calibration in calibration_rows
        if isinstance(calibration, CmapssAffinePredictionCalibration)
    }
    residual_by_key = {
        (calibration.subset, calibration.model_name, calibration.predicted_rul_bin): (
            calibration
        )
        for calibration in calibration_rows
        if isinstance(calibration, CmapssPredictedRulBinResidualCalibration)
    }
    output_path = prepare_output_path(output_csv)
    output_fieldnames = _calibrated_prediction_fieldnames(fieldnames)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in rows:
            calibration_key = (row["subset"], row["model_name"])
            calibration = _select_prediction_calibration(
                row,
                affine_by_key,
                residual_by_key,
            )
            if calibration is None:
                subset, model_name = calibration_key
                raise ValueError(
                    "missing calibration for prediction row: "
                    f"subset={subset}, model_name={model_name}"
                )
            writer.writerow(_calibrated_prediction_row(row, calibration))

    return len(rows)


def calibrate_cmapss_deep_predictions(
    *,
    calibration_csv: str | Path,
    predictions_csv: str | Path,
    output_csv: str | Path,
    method: str = "affine",
    output_calibration_csv: str | Path | None = None,
    output_diagnostics_csv: str | Path | None = None,
    output_rul_bins_csv: str | Path | None = None,
    output_unit_diagnostics_csv: str | Path | None = None,
    output_markdown: str | Path | None = None,
    top_n: int = 10,
    clip_min: float = 0.0,
    shrinkage_strength: float = 100.0,
    nasa_shift_max: float = 30.0,
    nasa_shift_step: float = 1.0,
) -> CmapssPredictionCalibrationResult:
    """Fit validation calibration, write calibrated predictions, and reports."""
    if method == "affine":
        calibrations: tuple[CmapssPredictionCalibration, ...] = (
            fit_cmapss_affine_prediction_calibrations(
                calibration_csv,
                clip_min=clip_min,
            )
        )
    elif method == "predicted_bin_residual":
        calibrations = fit_cmapss_predicted_rul_bin_residual_calibrations(
            calibration_csv,
            shrinkage_strength=shrinkage_strength,
            clip_min=clip_min,
        )
    elif method == "predicted_bin_nasa_shift":
        calibrations = fit_cmapss_predicted_rul_bin_nasa_shift_calibrations(
            calibration_csv,
            shrinkage_strength=shrinkage_strength,
            max_shift=nasa_shift_max,
            shift_step=nasa_shift_step,
            clip_min=clip_min,
        )
    else:
        raise ValueError(
            "method must be 'affine', 'predicted_bin_residual', "
            "or 'predicted_bin_nasa_shift'"
        )
    calibrated_prediction_count = write_cmapss_affine_calibrated_predictions_csv(
        predictions_csv,
        output_csv,
        calibrations,
    )

    calibration_csv_path = Path(output_calibration_csv) if output_calibration_csv else None
    if calibration_csv_path is not None:
        write_cmapss_affine_prediction_calibration_csv(
            calibrations,
            calibration_csv_path,
        )

    diagnostics: tuple[CmapssPredictionDiagnosticRow, ...] = ()
    rul_bin_diagnostics: tuple[CmapssPredictionRulBinDiagnosticRow, ...] = ()
    unit_diagnostics: tuple[CmapssPredictionUnitDiagnosticRow, ...] = ()
    if output_diagnostics_csv is not None or output_markdown is not None:
        diagnostics = tuple(build_cmapss_prediction_diagnostics(output_csv))
    if output_rul_bins_csv is not None or output_markdown is not None:
        rul_bin_diagnostics = tuple(
            build_cmapss_prediction_rul_bin_diagnostics(output_csv)
        )
    if output_unit_diagnostics_csv is not None or output_markdown is not None:
        unit_diagnostics = tuple(build_cmapss_prediction_unit_diagnostics(output_csv))

    diagnostics_csv_path = Path(output_diagnostics_csv) if output_diagnostics_csv else None
    if diagnostics_csv_path is not None:
        write_cmapss_prediction_diagnostics_csv(list(diagnostics), diagnostics_csv_path)

    rul_bins_csv_path = Path(output_rul_bins_csv) if output_rul_bins_csv else None
    if rul_bins_csv_path is not None:
        write_cmapss_prediction_rul_bin_diagnostics_csv(
            list(rul_bin_diagnostics),
            rul_bins_csv_path,
        )

    unit_diagnostics_csv_path = (
        Path(output_unit_diagnostics_csv) if output_unit_diagnostics_csv else None
    )
    if unit_diagnostics_csv_path is not None:
        write_cmapss_prediction_unit_diagnostics_csv(
            list(unit_diagnostics),
            unit_diagnostics_csv_path,
        )

    diagnostics_markdown_path = Path(output_markdown) if output_markdown else None
    if diagnostics_markdown_path is not None:
        write_cmapss_prediction_diagnostics_markdown(
            list(diagnostics),
            select_cmapss_high_error_predictions(output_csv, top_n=top_n),
            diagnostics_markdown_path,
            rul_bin_diagnostics=list(rul_bin_diagnostics),
            unit_diagnostics=list(unit_diagnostics),
        )

    return CmapssPredictionCalibrationResult(
        calibrated_predictions_csv_path=Path(output_csv),
        calibrations=calibrations,
        calibrated_prediction_count=calibrated_prediction_count,
        calibration_csv_path=calibration_csv_path,
        diagnostics_csv_path=diagnostics_csv_path,
        rul_bins_csv_path=rul_bins_csv_path,
        unit_diagnostics_csv_path=unit_diagnostics_csv_path,
        diagnostics_markdown_path=diagnostics_markdown_path,
        diagnostics=diagnostics,
        rul_bin_diagnostics=rul_bin_diagnostics,
        unit_diagnostics=unit_diagnostics,
    )


def _fit_group_affine_calibration(
    subset: str,
    model_name: str,
    rows: list[dict[str, str]],
    *,
    clip_min: float,
) -> CmapssAffinePredictionCalibration:
    raw_predictions = [_float(row["predicted_rul"]) for row in rows]
    actual_values = [_float(row["actual_rul"]) for row in rows]
    mean_raw_prediction = _mean(raw_predictions)
    mean_actual = _mean(actual_values)
    variance = _mean(
        (raw_prediction - mean_raw_prediction) ** 2
        for raw_prediction in raw_predictions
    )
    if variance <= 1e-12:
        slope = 1.0
        intercept = _mean(
            actual - raw_prediction
            for actual, raw_prediction in zip(
                actual_values,
                raw_predictions,
                strict=True,
            )
        )
    else:
        covariance = _mean(
            (raw_prediction - mean_raw_prediction) * (actual - mean_actual)
            for raw_prediction, actual in zip(
                raw_predictions,
                actual_values,
                strict=True,
            )
        )
        slope = covariance / variance
        intercept = mean_actual - slope * mean_raw_prediction

    return CmapssAffinePredictionCalibration(
        subset=subset,
        model_name=model_name,
        calibration_count=len(rows),
        intercept=intercept,
        slope=slope,
        mean_actual_rul=mean_actual,
        mean_raw_predicted_rul=mean_raw_prediction,
        raw_predicted_rul_variance=variance,
        clip_min=clip_min,
    )


def _calibrated_prediction_row(
    row: dict[str, str],
    calibration: CmapssPredictionCalibration,
) -> dict[str, Any]:
    raw_prediction = _float(row.get("raw_predicted_rul") or row["predicted_rul"])
    if isinstance(calibration, CmapssAffinePredictionCalibration):
        calibrated_prediction = calibration.intercept + calibration.slope * raw_prediction
        calibration_payload: dict[str, Any] = {
            "calibration_count": calibration.calibration_count,
            "calibration_intercept": calibration.intercept,
            "calibration_slope": calibration.slope,
            "calibration_predicted_rul_bin": "",
            "calibration_correction": "",
            "calibration_shrinkage_weight": "",
        }
    else:
        calibrated_prediction = raw_prediction + calibration.correction
        calibration_payload = {
            "calibration_count": calibration.calibration_count,
            "calibration_intercept": "",
            "calibration_slope": "",
            "calibration_predicted_rul_bin": calibration.predicted_rul_bin,
            "calibration_correction": calibration.correction,
            "calibration_shrinkage_weight": calibration.shrinkage_weight,
        }
    calibrated_prediction = max(calibration.clip_min, calibrated_prediction)
    actual_rul = _float(row["actual_rul"])
    error = calibrated_prediction - actual_rul

    output_row: dict[str, Any] = dict(row)
    output_row.update(
        {
            "raw_predicted_rul": raw_prediction,
            "predicted_rul": calibrated_prediction,
            "error": error,
            "absolute_error": abs(error),
            "late_error": max(error, 0.0),
            "early_error": max(-error, 0.0),
            "calibration_method": calibration.method,
            "calibration_clip_min": calibration.clip_min,
            **calibration_payload,
        }
    )
    return output_row


def _fit_predicted_rul_bin_residual_calibration(
    subset: str,
    model_name: str,
    bin_label: str,
    rows: list[dict[str, str]],
    *,
    shrinkage_strength: float,
    clip_min: float,
) -> CmapssPredictedRulBinResidualCalibration:
    mean_error = _mean(_float(row["error"]) for row in rows)
    shrinkage_weight = len(rows) / (len(rows) + shrinkage_strength)
    lower_bound, upper_bound = _predicted_rul_bin_bounds(bin_label)
    return CmapssPredictedRulBinResidualCalibration(
        subset=subset,
        model_name=model_name,
        predicted_rul_bin=bin_label,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        calibration_count=len(rows),
        mean_actual_rul=_mean(_float(row["actual_rul"]) for row in rows),
        mean_raw_predicted_rul=_mean(_float(row["predicted_rul"]) for row in rows),
        mean_error=mean_error,
        shrinkage_weight=shrinkage_weight,
        correction=-mean_error * shrinkage_weight,
        clip_min=clip_min,
    )


def _fit_predicted_rul_bin_nasa_shift_calibration(
    subset: str,
    model_name: str,
    bin_label: str,
    rows: list[dict[str, str]],
    *,
    shrinkage_strength: float,
    max_shift: float,
    shift_step: float,
    clip_min: float,
) -> CmapssPredictedRulBinResidualCalibration:
    raw_predictions = [_float(row["predicted_rul"]) for row in rows]
    actual_values = [_float(row["actual_rul"]) for row in rows]
    best_shift = _best_nasa_shift(
        actual_values,
        raw_predictions,
        max_shift=max_shift,
        shift_step=shift_step,
        clip_min=clip_min,
    )
    shrinkage_weight = len(rows) / (len(rows) + shrinkage_strength)
    lower_bound, upper_bound = _predicted_rul_bin_bounds(bin_label)
    return CmapssPredictedRulBinResidualCalibration(
        subset=subset,
        model_name=model_name,
        predicted_rul_bin=bin_label,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        calibration_count=len(rows),
        mean_actual_rul=_mean(actual_values),
        mean_raw_predicted_rul=_mean(raw_predictions),
        mean_error=_mean(_float(row["error"]) for row in rows),
        shrinkage_weight=shrinkage_weight,
        correction=best_shift * shrinkage_weight,
        clip_min=clip_min,
        method="validation_predicted_bin_nasa_shift",
    )


def _best_nasa_shift(
    actual_values: list[float],
    raw_predictions: list[float],
    *,
    max_shift: float,
    shift_step: float,
    clip_min: float,
) -> float:
    candidates = _candidate_shifts(max_shift=max_shift, shift_step=shift_step)
    return min(
        candidates,
        key=lambda shift: (
            nasa_rul_score(
                actual_values,
                [max(clip_min, prediction + shift) for prediction in raw_predictions],
            ),
            abs(shift),
            shift,
        ),
    )


def _candidate_shifts(*, max_shift: float, shift_step: float) -> tuple[float, ...]:
    shifts: list[float] = []
    shift = -max_shift
    while shift <= max_shift + (shift_step * 0.5):
        shifts.append(round(shift, 12))
        shift += shift_step
    if 0.0 not in shifts:
        shifts.append(0.0)
    return tuple(sorted(set(shifts)))


def _select_prediction_calibration(
    row: dict[str, str],
    affine_by_key: dict[tuple[str, str], CmapssAffinePredictionCalibration],
    residual_by_key: dict[
        tuple[str, str, str],
        CmapssPredictedRulBinResidualCalibration,
    ],
) -> CmapssPredictionCalibration | None:
    subset = row["subset"]
    model_name = row["model_name"]
    if affine_by_key:
        return affine_by_key.get((subset, model_name))
    raw_prediction = _float(row.get("raw_predicted_rul") or row["predicted_rul"])
    bin_label = _predicted_rul_bin(raw_prediction)
    return residual_by_key.get(
        (subset, model_name, bin_label),
        residual_by_key.get((subset, model_name, "all")),
    )


def _read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    rows, _ = _read_prediction_rows_with_fieldnames(path)
    return rows


def _read_prediction_rows_with_fieldnames(
    path: str | Path,
) -> tuple[list[dict[str, str]], list[str]]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"missing prediction CSV: {prediction_path}")
    with prediction_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"prediction CSV has no rows: {prediction_path}")
    missing_columns = _REQUIRED_PREDICTION_COLUMNS - set(fieldnames)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"prediction CSV is missing required columns: {missing}")
    return rows, fieldnames


def _calibrated_prediction_fieldnames(fieldnames: list[str]) -> list[str]:
    output_fieldnames = list(fieldnames)
    for fieldname in (
        "raw_predicted_rul",
        "calibration_method",
        "calibration_count",
        "calibration_intercept",
        "calibration_slope",
        "calibration_predicted_rul_bin",
        "calibration_correction",
        "calibration_shrinkage_weight",
        "calibration_clip_min",
    ):
        if fieldname not in output_fieldnames:
            output_fieldnames.append(fieldname)
    return output_fieldnames


def _calibration_fieldnames(
    calibrations: tuple[CmapssPredictionCalibration, ...],
) -> list[str]:
    fieldnames: list[str] = []
    for calibration in calibrations:
        for fieldname in calibration.to_dict():
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    return fieldnames


def _predicted_rul_bin(predicted_rul: float) -> str:
    if predicted_rul <= 30:
        return "0-30"
    if predicted_rul <= 60:
        return "31-60"
    if predicted_rul <= 90:
        return "61-90"
    if predicted_rul <= 120:
        return "91-120"
    return "121+"


def _predicted_rul_bin_bounds(label: str) -> tuple[float, float | str]:
    bounds: dict[str, tuple[float, float | str]] = {
        "all": (0.0, "inf"),
        "0-30": (0.0, 30.0),
        "31-60": (30.0, 60.0),
        "61-90": (60.0, 90.0),
        "91-120": (90.0, 120.0),
        "121+": (120.0, "inf"),
    }
    return bounds[label]


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("cannot average an empty metric sequence")
    return sum(sequence) / len(sequence)


def _float(value: Any) -> float:
    return float(value)


_REQUIRED_PREDICTION_COLUMNS = {
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

_PREDICTED_RUL_BIN_ORDER = ("0-30", "31-60", "61-90", "91-120", "121+")
