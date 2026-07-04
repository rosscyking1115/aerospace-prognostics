"""Phase 3 C-MAPSS uncertainty and monotonicity audit reports."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload
from aerospace_prognostics.reports.cmapss_prediction_diagnostics import (
    CmapssPredictionMonotonicityDiagnosticRow,
    CmapssPredictionUnitDiagnosticRow,
    build_cmapss_prediction_monotonicity_diagnostics,
    build_cmapss_prediction_unit_diagnostics,
)


@dataclass(frozen=True)
class CmapssIntervalCalibrationRow:
    """Validation-fitted interval calibration for one subset/model pair."""

    subset: str
    model_name: str
    method: str
    confidence: float
    calibration_count: int
    interval_radius: float
    mean_absolute_residual: float
    max_absolute_residual: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssPredictedBinIntervalCalibrationRow:
    """Validation-fitted interval calibration for one predicted-RUL bin."""

    subset: str
    model_name: str
    predicted_rul_bin: str
    method: str
    confidence: float
    calibration_count: int
    interval_radius: float
    mean_absolute_residual: float
    max_absolute_residual: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssUncertaintySummaryRow:
    """Official-test interval coverage summary for one subset/model pair."""

    subset: str
    model_name: str
    method: str
    confidence: float
    prediction_count: int
    covered_count: int
    coverage: float
    interval_radius: float
    mean_interval_width: float
    median_interval_width: float
    mean_absolute_error: float
    late_prediction_count: int
    late_prediction_coverage: float
    uncovered_late_prediction_count: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssUncertaintyBinRow:
    """Official-test interval coverage by actual-RUL bin."""

    subset: str
    model_name: str
    actual_rul_bin: str
    prediction_count: int
    covered_count: int
    coverage: float
    mean_interval_width: float
    mean_absolute_error: float
    late_prediction_rate: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssUncertaintyFailureRow:
    """Highest-risk uncovered interval case."""

    rank: int
    subset: str
    model_name: str
    unit_number: int
    actual_rul: float
    predicted_rul: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    error: float
    absolute_error: float
    failure_type: str

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssFailureNoteRow:
    """Unit-level interval failure note across audit strategies."""

    rank: int
    subset: str
    model_name: str
    unit_number: int
    actual_rul: float
    predicted_rul: float
    error: float
    absolute_error: float
    actual_rul_bin: str
    predicted_rul_bin: str
    failure_type: str
    global_covered: bool
    predicted_bin_covered: bool
    predicted_bin_floor_covered: bool
    uncovered_strategy_count: int
    uncovered_strategies: str
    note: str

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssTailFallbackCalibrationRow:
    """Validation-fitted tail fallback interval calibration."""

    subset: str
    model_name: str
    method: str
    tail_threshold: float
    global_confidence: float
    tail_confidence: float
    tail_calibration_count: int
    global_interval_radius: float
    tail_interval_radius: float
    fallback_interval_radius: float
    tail_mean_absolute_residual: float
    tail_max_absolute_residual: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssIntervalComparisonRow:
    """Global-vs-predicted-bin interval coverage comparison."""

    subset: str
    model_name: str
    prediction_count: int
    global_coverage: float
    predicted_bin_coverage: float
    coverage_delta: float
    global_mean_interval_width: float
    predicted_bin_mean_interval_width: float
    mean_interval_width_delta: float
    predicted_bin_floor_coverage: float
    predicted_bin_floor_mean_interval_width: float
    predicted_bin_floor_mean_interval_width_delta: float
    global_uncovered_late_prediction_count: int
    predicted_bin_uncovered_late_prediction_count: int
    predicted_bin_floor_uncovered_late_prediction_count: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssTailFallbackComparisonRow:
    """Global-vs-tail-fallback interval coverage comparison."""

    subset: str
    model_name: str
    prediction_count: int
    global_coverage: float
    tail_fallback_coverage: float
    coverage_delta: float
    global_mean_interval_width: float
    tail_fallback_mean_interval_width: float
    mean_interval_width_delta: float
    global_median_interval_width: float
    tail_fallback_median_interval_width: float
    median_interval_width_delta: float
    global_uncovered_late_prediction_count: int
    tail_fallback_uncovered_late_prediction_count: int
    uncovered_late_prediction_count_delta: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssTailFallbackFailureNoteRow:
    """Unit-level failure note comparing global and tail fallback intervals."""

    rank: int
    subset: str
    model_name: str
    unit_number: int
    actual_rul: float
    predicted_rul: float
    error: float
    absolute_error: float
    actual_rul_bin: str
    predicted_rul_bin: str
    failure_type: str
    global_covered: bool
    tail_fallback_covered: bool
    note: str

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssMonotonicityComparisonRow:
    """Raw-vs-calibrated monotonicity comparison for one subset/model pair."""

    subset: str
    model_name: str
    raw_violation_rate: float
    calibrated_violation_rate: float
    violation_rate_delta: float
    raw_max_violation_magnitude: float
    calibrated_max_violation_magnitude: float
    max_violation_delta: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True)
class CmapssPhase3AuditResult:
    """Phase 3 evidence bundle for C-MAPSS predictions."""

    interval_calibrations: tuple[CmapssIntervalCalibrationRow, ...]
    uncertainty_summaries: tuple[CmapssUncertaintySummaryRow, ...]
    uncertainty_bins: tuple[CmapssUncertaintyBinRow, ...]
    failure_cases: tuple[CmapssUncertaintyFailureRow, ...]
    failure_notes: tuple[CmapssFailureNoteRow, ...]
    predicted_bin_interval_calibrations: tuple[
        CmapssPredictedBinIntervalCalibrationRow, ...
    ]
    predicted_bin_uncertainty_summaries: tuple[CmapssUncertaintySummaryRow, ...]
    predicted_bin_uncertainty_bins: tuple[CmapssUncertaintyBinRow, ...]
    predicted_bin_failure_cases: tuple[CmapssUncertaintyFailureRow, ...]
    predicted_bin_floor_uncertainty_summaries: tuple[
        CmapssUncertaintySummaryRow, ...
    ]
    predicted_bin_floor_uncertainty_bins: tuple[CmapssUncertaintyBinRow, ...]
    predicted_bin_floor_failure_cases: tuple[CmapssUncertaintyFailureRow, ...]
    interval_comparisons: tuple[CmapssIntervalComparisonRow, ...]
    tail_fallback_calibrations: tuple[CmapssTailFallbackCalibrationRow, ...]
    tail_fallback_summaries: tuple[CmapssUncertaintySummaryRow, ...]
    tail_fallback_bins: tuple[CmapssUncertaintyBinRow, ...]
    tail_fallback_failure_cases: tuple[CmapssUncertaintyFailureRow, ...]
    tail_fallback_comparisons: tuple[CmapssTailFallbackComparisonRow, ...]
    tail_fallback_failure_notes: tuple[CmapssTailFallbackFailureNoteRow, ...]
    monotonicity_diagnostics: tuple[CmapssPredictionMonotonicityDiagnosticRow, ...]
    unit_diagnostics: tuple[CmapssPredictionUnitDiagnosticRow, ...]
    monotonicity_comparisons: tuple[CmapssMonotonicityComparisonRow, ...]
    training_recommendation: str
    output_json_path: Path | None = None
    output_markdown_path: Path | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "aerospace-prognostics/cmapss-phase3-audit/v1",
            "uncertainty": {
                "calibrations": [
                    row.to_dict() for row in self.interval_calibrations
                ],
                "summaries": [row.to_dict() for row in self.uncertainty_summaries],
                "bins": [row.to_dict() for row in self.uncertainty_bins],
                "failure_cases": [row.to_dict() for row in self.failure_cases],
                "failure_notes": [row.to_dict() for row in self.failure_notes],
                "predicted_bin_calibrations": [
                    row.to_dict()
                    for row in self.predicted_bin_interval_calibrations
                ],
                "predicted_bin_summaries": [
                    row.to_dict()
                    for row in self.predicted_bin_uncertainty_summaries
                ],
                "predicted_bin_bins": [
                    row.to_dict() for row in self.predicted_bin_uncertainty_bins
                ],
                "predicted_bin_failure_cases": [
                    row.to_dict() for row in self.predicted_bin_failure_cases
                ],
                "predicted_bin_floor_summaries": [
                    row.to_dict()
                    for row in self.predicted_bin_floor_uncertainty_summaries
                ],
                "predicted_bin_floor_bins": [
                    row.to_dict()
                    for row in self.predicted_bin_floor_uncertainty_bins
                ],
                "predicted_bin_floor_failure_cases": [
                    row.to_dict()
                    for row in self.predicted_bin_floor_failure_cases
                ],
                "global_vs_predicted_bin_comparison": [
                    row.to_dict() for row in self.interval_comparisons
                ],
                "tail_fallback_calibrations": [
                    row.to_dict() for row in self.tail_fallback_calibrations
                ],
                "tail_fallback_summaries": [
                    row.to_dict() for row in self.tail_fallback_summaries
                ],
                "tail_fallback_bins": [
                    row.to_dict() for row in self.tail_fallback_bins
                ],
                "tail_fallback_failure_cases": [
                    row.to_dict() for row in self.tail_fallback_failure_cases
                ],
                "global_vs_tail_fallback_comparison": [
                    row.to_dict() for row in self.tail_fallback_comparisons
                ],
                "tail_fallback_failure_notes": [
                    row.to_dict() for row in self.tail_fallback_failure_notes
                ],
            },
            "monotonicity": {
                "diagnostics": [
                    row.to_dict() for row in self.monotonicity_diagnostics
                ],
                "unit_diagnostics": [
                    row.to_dict() for row in self.unit_diagnostics
                ],
                "calibrated_comparisons": [
                    row.to_dict() for row in self.monotonicity_comparisons
                ],
            },
            "training_recommendation": self.training_recommendation,
        }


def run_cmapss_phase3_audit(
    *,
    calibration_csv: str | Path,
    predictions_csv: str | Path,
    calibrated_predictions_csv: str | Path | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    confidence: float = 0.9,
    top_n: int = 10,
    clip_min: float = 0.0,
) -> CmapssPhase3AuditResult:
    """Build Phase 3 interval and monotonicity evidence from prediction CSVs."""

    calibrations = fit_cmapss_interval_calibrations(
        calibration_csv,
        confidence=confidence,
    )
    predicted_bin_calibrations = fit_cmapss_predicted_bin_interval_calibrations(
        calibration_csv,
        confidence=confidence,
    )
    tail_fallback_calibrations = fit_cmapss_tail_fallback_calibrations(
        calibration_csv,
        calibrations,
    )
    prediction_rows = _read_prediction_rows(predictions_csv)
    summary_rows, bin_rows, failure_rows = build_cmapss_uncertainty_audit(
        prediction_rows,
        calibrations,
        top_n=top_n,
        clip_min=clip_min,
    )
    (
        predicted_bin_summary_rows,
        predicted_bin_rows,
        predicted_bin_failure_rows,
    ) = build_cmapss_predicted_bin_uncertainty_audit(
        prediction_rows,
        predicted_bin_calibrations,
        top_n=top_n,
        clip_min=clip_min,
    )
    (
        predicted_bin_floor_summary_rows,
        predicted_bin_floor_rows,
        predicted_bin_floor_failure_rows,
    ) = build_cmapss_predicted_bin_uncertainty_audit(
        prediction_rows,
        predicted_bin_calibrations,
        top_n=top_n,
        clip_min=clip_min,
        floor_to_global=True,
    )
    interval_comparisons = compare_cmapss_interval_strategies(
        summary_rows,
        predicted_bin_summary_rows,
        predicted_bin_floor_summary_rows,
    )
    (
        tail_fallback_summary_rows,
        tail_fallback_bin_rows,
        tail_fallback_failure_rows,
    ) = build_cmapss_tail_fallback_uncertainty_audit(
        prediction_rows,
        tail_fallback_calibrations,
        top_n=top_n,
        clip_min=clip_min,
    )
    tail_fallback_comparisons = compare_cmapss_tail_fallback_strategies(
        summary_rows,
        tail_fallback_summary_rows,
    )
    failure_notes = build_cmapss_interval_failure_notes(
        prediction_rows,
        calibrations,
        predicted_bin_calibrations,
        top_n=top_n,
        clip_min=clip_min,
    )
    tail_fallback_failure_notes = build_cmapss_tail_fallback_failure_notes(
        prediction_rows,
        calibrations,
        tail_fallback_calibrations,
        top_n=top_n,
        clip_min=clip_min,
    )
    monotonicity_rows = tuple(
        build_cmapss_prediction_monotonicity_diagnostics(predictions_csv)
    )
    unit_rows = tuple(build_cmapss_prediction_unit_diagnostics(predictions_csv))
    monotonicity_comparisons = (
        compare_cmapss_monotonicity(predictions_csv, calibrated_predictions_csv)
        if calibrated_predictions_csv is not None
        else ()
    )
    recommendation = _training_recommendation(
        summary_rows,
        monotonicity_rows,
        monotonicity_comparisons,
    )
    result = CmapssPhase3AuditResult(
        interval_calibrations=calibrations,
        uncertainty_summaries=summary_rows,
        uncertainty_bins=bin_rows,
        failure_cases=failure_rows,
        failure_notes=failure_notes,
        predicted_bin_interval_calibrations=predicted_bin_calibrations,
        predicted_bin_uncertainty_summaries=predicted_bin_summary_rows,
        predicted_bin_uncertainty_bins=predicted_bin_rows,
        predicted_bin_failure_cases=predicted_bin_failure_rows,
        predicted_bin_floor_uncertainty_summaries=predicted_bin_floor_summary_rows,
        predicted_bin_floor_uncertainty_bins=predicted_bin_floor_rows,
        predicted_bin_floor_failure_cases=predicted_bin_floor_failure_rows,
        interval_comparisons=interval_comparisons,
        tail_fallback_calibrations=tail_fallback_calibrations,
        tail_fallback_summaries=tail_fallback_summary_rows,
        tail_fallback_bins=tail_fallback_bin_rows,
        tail_fallback_failure_cases=tail_fallback_failure_rows,
        tail_fallback_comparisons=tail_fallback_comparisons,
        tail_fallback_failure_notes=tail_fallback_failure_notes,
        monotonicity_diagnostics=monotonicity_rows,
        unit_diagnostics=unit_rows,
        monotonicity_comparisons=monotonicity_comparisons,
        training_recommendation=recommendation,
        output_json_path=Path(output_json) if output_json else None,
        output_markdown_path=Path(output_markdown) if output_markdown else None,
    )
    if output_json is not None:
        write_json_payload(result.to_payload(), output_json)
    if output_markdown is not None:
        write_cmapss_phase3_audit_markdown(result, output_markdown)
    return result


def fit_cmapss_interval_calibrations(
    calibration_csv: str | Path,
    *,
    confidence: float = 0.9,
) -> tuple[CmapssIntervalCalibrationRow, ...]:
    """Fit symmetric absolute-residual interval radii from validation predictions."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rows = _read_prediction_rows(calibration_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["subset"], row["model_name"]), []).append(row)

    calibrations: list[CmapssIntervalCalibrationRow] = []
    for subset, model_name in sorted(grouped):
        group = grouped[(subset, model_name)]
        residuals = sorted(_float(row["absolute_error"]) for row in group)
        radius = _nearest_rank_quantile(residuals, confidence)
        calibrations.append(
            CmapssIntervalCalibrationRow(
                subset=subset,
                model_name=model_name,
                method="validation_absolute_residual_quantile",
                confidence=confidence,
                calibration_count=len(group),
                interval_radius=radius,
                mean_absolute_residual=_mean(residuals),
                max_absolute_residual=max(residuals),
            )
        )
    return tuple(calibrations)


def fit_cmapss_predicted_bin_interval_calibrations(
    calibration_csv: str | Path,
    *,
    confidence: float = 0.9,
) -> tuple[CmapssPredictedBinIntervalCalibrationRow, ...]:
    """Fit interval radii by inference-safe predicted-RUL bin."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rows = _read_prediction_rows(calibration_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    grouped_bins: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        model_key = (row["subset"], row["model_name"])
        grouped.setdefault(model_key, []).append(row)
        bin_label = _predicted_rul_bin(_float(row["predicted_rul"]))
        grouped_bins.setdefault((*model_key, bin_label), []).append(row)

    calibrations: list[CmapssPredictedBinIntervalCalibrationRow] = []
    for subset, model_name in sorted(grouped):
        calibrations.append(
            _predicted_bin_interval_calibration_row(
                subset,
                model_name,
                "all",
                grouped[(subset, model_name)],
                confidence=confidence,
            )
        )
        for bin_label in _PREDICTED_RUL_BIN_ORDER:
            bin_key = (subset, model_name, bin_label)
            if bin_key in grouped_bins:
                calibrations.append(
                    _predicted_bin_interval_calibration_row(
                        subset,
                        model_name,
                        bin_label,
                        grouped_bins[bin_key],
                        confidence=confidence,
                    )
                )
    return tuple(calibrations)


def fit_cmapss_tail_fallback_calibrations(
    calibration_csv: str | Path,
    global_calibrations: Iterable[CmapssIntervalCalibrationRow],
    *,
    tail_threshold: float = 91.0,
    tail_confidence: float = 0.95,
) -> tuple[CmapssTailFallbackCalibrationRow, ...]:
    """Fit inference-safe tail fallback radii from validation predictions."""

    if not 0.0 < tail_confidence < 1.0:
        raise ValueError("tail_confidence must be between 0 and 1")
    global_by_key = {
        (row.subset, row.model_name): row for row in tuple(global_calibrations)
    }
    rows = _read_prediction_rows(calibration_csv)
    grouped_tail_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if _float(row["predicted_rul"]) >= tail_threshold:
            grouped_tail_rows.setdefault((row["subset"], row["model_name"]), []).append(
                row
            )

    calibrations: list[CmapssTailFallbackCalibrationRow] = []
    for subset, model_name in sorted(global_by_key):
        global_row = global_by_key[(subset, model_name)]
        tail_rows = grouped_tail_rows.get((subset, model_name), [])
        residuals = sorted(_float(row["absolute_error"]) for row in tail_rows)
        tail_radius = (
            _nearest_rank_quantile(residuals, tail_confidence)
            if residuals
            else global_row.interval_radius
        )
        calibrations.append(
            CmapssTailFallbackCalibrationRow(
                subset=subset,
                model_name=model_name,
                method="global_tail_fallback",
                tail_threshold=tail_threshold,
                global_confidence=global_row.confidence,
                tail_confidence=tail_confidence,
                tail_calibration_count=len(tail_rows),
                global_interval_radius=global_row.interval_radius,
                tail_interval_radius=tail_radius,
                fallback_interval_radius=max(global_row.interval_radius, tail_radius),
                tail_mean_absolute_residual=_mean(residuals) if residuals else 0.0,
                tail_max_absolute_residual=max(residuals) if residuals else 0.0,
            )
        )
    return tuple(calibrations)


def build_cmapss_uncertainty_audit(
    prediction_rows: Iterable[dict[str, str]],
    calibrations: Iterable[CmapssIntervalCalibrationRow],
    *,
    top_n: int = 10,
    clip_min: float = 0.0,
) -> tuple[
    tuple[CmapssUncertaintySummaryRow, ...],
    tuple[CmapssUncertaintyBinRow, ...],
    tuple[CmapssUncertaintyFailureRow, ...],
]:
    """Summarize interval coverage and uncovered cases for prediction rows."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    calibration_by_key = {
        (row.subset, row.model_name): row for row in tuple(calibrations)
    }
    annotated_rows = [
        _annotated_interval_row(row, calibration_by_key, clip_min=clip_min)
        for row in prediction_rows
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    grouped_bins: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in annotated_rows:
        model_key = (row["subset"], row["model_name"])
        grouped.setdefault(model_key, []).append(row)
        bin_label = _actual_rul_bin(row["actual_rul"])
        grouped_bins.setdefault((*model_key, bin_label), []).append(row)

    summary_rows = tuple(
        _summary_row(subset, model_name, rows)
        for subset, model_name in sorted(grouped)
        for rows in (grouped[(subset, model_name)],)
    )
    bin_rows = tuple(
        _bin_row(subset, model_name, bin_label, rows)
        for subset, model_name, bin_label in sorted(
            grouped_bins,
            key=lambda key: (key[0], key[1], _actual_rul_bin_sort_key(key[2])),
        )
        for rows in (grouped_bins[(subset, model_name, bin_label)],)
    )
    failure_rows = tuple(
        CmapssUncertaintyFailureRow(
            rank=rank,
            subset=row["subset"],
            model_name=row["model_name"],
            unit_number=row["unit_number"],
            actual_rul=row["actual_rul"],
            predicted_rul=row["predicted_rul"],
            lower_bound=row["lower_bound"],
            upper_bound=row["upper_bound"],
            interval_width=row["interval_width"],
            error=row["error"],
            absolute_error=row["absolute_error"],
            failure_type=row["failure_type"],
        )
        for rank, row in enumerate(
            sorted(
                (row for row in annotated_rows if not row["covered"]),
                key=lambda item: (
                    -item["absolute_error"],
                    item["subset"],
                    item["model_name"],
                    item["unit_number"],
                ),
            )[:top_n],
            start=1,
        )
    )
    return summary_rows, bin_rows, failure_rows


def build_cmapss_predicted_bin_uncertainty_audit(
    prediction_rows: Iterable[dict[str, str]],
    calibrations: Iterable[CmapssPredictedBinIntervalCalibrationRow],
    *,
    top_n: int = 10,
    clip_min: float = 0.0,
    floor_to_global: bool = False,
) -> tuple[
    tuple[CmapssUncertaintySummaryRow, ...],
    tuple[CmapssUncertaintyBinRow, ...],
    tuple[CmapssUncertaintyFailureRow, ...],
]:
    """Summarize coverage using predicted-RUL-bin interval radii."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    calibration_by_key = {
        (row.subset, row.model_name, row.predicted_rul_bin): row
        for row in tuple(calibrations)
    }
    annotated_rows = [
        _annotated_predicted_bin_interval_row(
            row,
            calibration_by_key,
            clip_min=clip_min,
            floor_to_global=floor_to_global,
        )
        for row in prediction_rows
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    grouped_bins: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in annotated_rows:
        model_key = (row["subset"], row["model_name"])
        grouped.setdefault(model_key, []).append(row)
        bin_label = _actual_rul_bin(row["actual_rul"])
        grouped_bins.setdefault((*model_key, bin_label), []).append(row)

    summary_rows = tuple(
        _summary_row(subset, model_name, rows)
        for subset, model_name in sorted(grouped)
        for rows in (grouped[(subset, model_name)],)
    )
    bin_rows = tuple(
        _bin_row(subset, model_name, bin_label, rows)
        for subset, model_name, bin_label in sorted(
            grouped_bins,
            key=lambda key: (key[0], key[1], _actual_rul_bin_sort_key(key[2])),
        )
        for rows in (grouped_bins[(subset, model_name, bin_label)],)
    )
    failure_rows = tuple(
        CmapssUncertaintyFailureRow(
            rank=rank,
            subset=row["subset"],
            model_name=row["model_name"],
            unit_number=row["unit_number"],
            actual_rul=row["actual_rul"],
            predicted_rul=row["predicted_rul"],
            lower_bound=row["lower_bound"],
            upper_bound=row["upper_bound"],
            interval_width=row["interval_width"],
            error=row["error"],
            absolute_error=row["absolute_error"],
            failure_type=row["failure_type"],
        )
        for rank, row in enumerate(
            sorted(
                (row for row in annotated_rows if not row["covered"]),
                key=lambda item: (
                    -item["absolute_error"],
                    item["subset"],
                    item["model_name"],
                    item["unit_number"],
                ),
            )[:top_n],
            start=1,
        )
    )
    return summary_rows, bin_rows, failure_rows


def build_cmapss_tail_fallback_uncertainty_audit(
    prediction_rows: Iterable[dict[str, str]],
    calibrations: Iterable[CmapssTailFallbackCalibrationRow],
    *,
    top_n: int = 10,
    clip_min: float = 0.0,
) -> tuple[
    tuple[CmapssUncertaintySummaryRow, ...],
    tuple[CmapssUncertaintyBinRow, ...],
    tuple[CmapssUncertaintyFailureRow, ...],
]:
    """Summarize coverage using global intervals plus a predicted-RUL tail fallback."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    calibration_by_key = {
        (row.subset, row.model_name): row for row in tuple(calibrations)
    }
    annotated_rows = [
        _annotated_tail_fallback_interval_row(
            row,
            calibration_by_key,
            clip_min=clip_min,
        )
        for row in prediction_rows
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    grouped_bins: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in annotated_rows:
        model_key = (row["subset"], row["model_name"])
        grouped.setdefault(model_key, []).append(row)
        bin_label = _actual_rul_bin(row["actual_rul"])
        grouped_bins.setdefault((*model_key, bin_label), []).append(row)

    summary_rows = tuple(
        _summary_row(subset, model_name, rows)
        for subset, model_name in sorted(grouped)
        for rows in (grouped[(subset, model_name)],)
    )
    bin_rows = tuple(
        _bin_row(subset, model_name, bin_label, rows)
        for subset, model_name, bin_label in sorted(
            grouped_bins,
            key=lambda key: (key[0], key[1], _actual_rul_bin_sort_key(key[2])),
        )
        for rows in (grouped_bins[(subset, model_name, bin_label)],)
    )
    failure_rows = tuple(
        CmapssUncertaintyFailureRow(
            rank=rank,
            subset=row["subset"],
            model_name=row["model_name"],
            unit_number=row["unit_number"],
            actual_rul=row["actual_rul"],
            predicted_rul=row["predicted_rul"],
            lower_bound=row["lower_bound"],
            upper_bound=row["upper_bound"],
            interval_width=row["interval_width"],
            error=row["error"],
            absolute_error=row["absolute_error"],
            failure_type=row["failure_type"],
        )
        for rank, row in enumerate(
            sorted(
                (row for row in annotated_rows if not row["covered"]),
                key=lambda item: (
                    -item["absolute_error"],
                    item["subset"],
                    item["model_name"],
                    item["unit_number"],
                ),
            )[:top_n],
            start=1,
        )
    )
    return summary_rows, bin_rows, failure_rows


def compare_cmapss_interval_strategies(
    global_summaries: Iterable[CmapssUncertaintySummaryRow],
    predicted_bin_summaries: Iterable[CmapssUncertaintySummaryRow],
    predicted_bin_floor_summaries: Iterable[CmapssUncertaintySummaryRow],
) -> tuple[CmapssIntervalComparisonRow, ...]:
    """Compare global, predicted-bin, and global-floor interval summaries."""

    global_by_key = {
        (row.subset, row.model_name): row for row in tuple(global_summaries)
    }
    predicted_bin_by_key = {
        (row.subset, row.model_name): row for row in tuple(predicted_bin_summaries)
    }
    predicted_bin_floor_by_key = {
        (row.subset, row.model_name): row
        for row in tuple(predicted_bin_floor_summaries)
    }
    comparisons: list[CmapssIntervalComparisonRow] = []
    comparable_keys = (
        global_by_key.keys()
        & predicted_bin_by_key.keys()
        & predicted_bin_floor_by_key.keys()
    )
    for subset, model_name in sorted(comparable_keys):
        global_row = global_by_key[(subset, model_name)]
        predicted_bin_row = predicted_bin_by_key[(subset, model_name)]
        predicted_bin_floor_row = predicted_bin_floor_by_key[(subset, model_name)]
        comparisons.append(
            CmapssIntervalComparisonRow(
                subset=subset,
                model_name=model_name,
                prediction_count=global_row.prediction_count,
                global_coverage=global_row.coverage,
                predicted_bin_coverage=predicted_bin_row.coverage,
                coverage_delta=predicted_bin_row.coverage - global_row.coverage,
                global_mean_interval_width=global_row.mean_interval_width,
                predicted_bin_mean_interval_width=(
                    predicted_bin_row.mean_interval_width
                ),
                mean_interval_width_delta=(
                    predicted_bin_row.mean_interval_width
                    - global_row.mean_interval_width
                ),
                predicted_bin_floor_coverage=predicted_bin_floor_row.coverage,
                predicted_bin_floor_mean_interval_width=(
                    predicted_bin_floor_row.mean_interval_width
                ),
                predicted_bin_floor_mean_interval_width_delta=(
                    predicted_bin_floor_row.mean_interval_width
                    - global_row.mean_interval_width
                ),
                global_uncovered_late_prediction_count=(
                    global_row.uncovered_late_prediction_count
                ),
                predicted_bin_uncovered_late_prediction_count=(
                    predicted_bin_row.uncovered_late_prediction_count
                ),
                predicted_bin_floor_uncovered_late_prediction_count=(
                    predicted_bin_floor_row.uncovered_late_prediction_count
                ),
            )
        )
    return tuple(comparisons)


def compare_cmapss_tail_fallback_strategies(
    global_summaries: Iterable[CmapssUncertaintySummaryRow],
    tail_fallback_summaries: Iterable[CmapssUncertaintySummaryRow],
) -> tuple[CmapssTailFallbackComparisonRow, ...]:
    """Compare global and global-tail-fallback interval summaries."""

    global_by_key = {
        (row.subset, row.model_name): row for row in tuple(global_summaries)
    }
    tail_fallback_by_key = {
        (row.subset, row.model_name): row for row in tuple(tail_fallback_summaries)
    }
    comparisons: list[CmapssTailFallbackComparisonRow] = []
    for subset, model_name in sorted(global_by_key.keys() & tail_fallback_by_key.keys()):
        global_row = global_by_key[(subset, model_name)]
        tail_row = tail_fallback_by_key[(subset, model_name)]
        comparisons.append(
            CmapssTailFallbackComparisonRow(
                subset=subset,
                model_name=model_name,
                prediction_count=global_row.prediction_count,
                global_coverage=global_row.coverage,
                tail_fallback_coverage=tail_row.coverage,
                coverage_delta=tail_row.coverage - global_row.coverage,
                global_mean_interval_width=global_row.mean_interval_width,
                tail_fallback_mean_interval_width=tail_row.mean_interval_width,
                mean_interval_width_delta=(
                    tail_row.mean_interval_width - global_row.mean_interval_width
                ),
                global_median_interval_width=global_row.median_interval_width,
                tail_fallback_median_interval_width=tail_row.median_interval_width,
                median_interval_width_delta=(
                    tail_row.median_interval_width - global_row.median_interval_width
                ),
                global_uncovered_late_prediction_count=(
                    global_row.uncovered_late_prediction_count
                ),
                tail_fallback_uncovered_late_prediction_count=(
                    tail_row.uncovered_late_prediction_count
                ),
                uncovered_late_prediction_count_delta=(
                    tail_row.uncovered_late_prediction_count
                    - global_row.uncovered_late_prediction_count
                ),
            )
        )
    return tuple(comparisons)


def build_cmapss_interval_failure_notes(
    prediction_rows: Iterable[dict[str, str]],
    calibrations: Iterable[CmapssIntervalCalibrationRow],
    predicted_bin_calibrations: Iterable[CmapssPredictedBinIntervalCalibrationRow],
    *,
    top_n: int = 10,
    clip_min: float = 0.0,
) -> tuple[CmapssFailureNoteRow, ...]:
    """Build unit-level notes for rows uncovered by any interval strategy."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    global_by_key = {
        (row.subset, row.model_name): row for row in tuple(calibrations)
    }
    predicted_bin_by_key = {
        (row.subset, row.model_name, row.predicted_rul_bin): row
        for row in tuple(predicted_bin_calibrations)
    }
    candidate_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        global_row = _annotated_interval_row(
            row,
            global_by_key,
            clip_min=clip_min,
        )
        predicted_bin_row = _annotated_predicted_bin_interval_row(
            row,
            predicted_bin_by_key,
            clip_min=clip_min,
        )
        predicted_bin_floor_row = _annotated_predicted_bin_interval_row(
            row,
            predicted_bin_by_key,
            clip_min=clip_min,
            floor_to_global=True,
        )
        uncovered_strategies = tuple(
            strategy
            for strategy, covered in (
                ("global", global_row["covered"]),
                ("predicted_bin", predicted_bin_row["covered"]),
                ("predicted_bin_global_floor", predicted_bin_floor_row["covered"]),
            )
            if not covered
        )
        if uncovered_strategies:
            candidate_rows.append(
                {
                    "global": global_row,
                    "predicted_bin": predicted_bin_row,
                    "predicted_bin_floor": predicted_bin_floor_row,
                    "uncovered_strategies": uncovered_strategies,
                }
            )

    sorted_rows = sorted(
        candidate_rows,
        key=lambda item: (
            -len(item["uncovered_strategies"]),
            0 if item["global"]["late_prediction"] else 1,
            -item["global"]["absolute_error"],
            item["global"]["subset"],
            item["global"]["model_name"],
            item["global"]["unit_number"],
        ),
    )
    return tuple(
        _failure_note_row(rank, row)
        for rank, row in enumerate(sorted_rows[:top_n], start=1)
    )


def build_cmapss_tail_fallback_failure_notes(
    prediction_rows: Iterable[dict[str, str]],
    global_calibrations: Iterable[CmapssIntervalCalibrationRow],
    tail_fallback_calibrations: Iterable[CmapssTailFallbackCalibrationRow],
    *,
    top_n: int = 10,
    clip_min: float = 0.0,
) -> tuple[CmapssTailFallbackFailureNoteRow, ...]:
    """Build unit notes for rows missed by global or tail fallback intervals."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    global_by_key = {
        (row.subset, row.model_name): row for row in tuple(global_calibrations)
    }
    tail_by_key = {
        (row.subset, row.model_name): row for row in tuple(tail_fallback_calibrations)
    }
    candidate_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        global_row = _annotated_interval_row(row, global_by_key, clip_min=clip_min)
        tail_row = _annotated_tail_fallback_interval_row(
            row,
            tail_by_key,
            clip_min=clip_min,
        )
        if not global_row["covered"] or not tail_row["covered"]:
            candidate_rows.append({"global": global_row, "tail_fallback": tail_row})

    sorted_rows = sorted(
        candidate_rows,
        key=lambda item: (
            0 if item["global"]["late_prediction"] else 1,
            -item["global"]["absolute_error"],
            item["global"]["subset"],
            item["global"]["model_name"],
            item["global"]["unit_number"],
        ),
    )
    return tuple(
        _tail_fallback_failure_note_row(rank, row)
        for rank, row in enumerate(sorted_rows[:top_n], start=1)
    )


def compare_cmapss_monotonicity(
    raw_predictions_csv: str | Path,
    calibrated_predictions_csv: str | Path,
) -> tuple[CmapssMonotonicityComparisonRow, ...]:
    """Compare raw and calibrated monotonicity diagnostics by subset/model."""

    raw_rows = {
        (row.subset, row.model_name): row
        for row in build_cmapss_prediction_monotonicity_diagnostics(raw_predictions_csv)
    }
    calibrated_rows = {
        (row.subset, row.model_name): row
        for row in build_cmapss_prediction_monotonicity_diagnostics(
            calibrated_predictions_csv
        )
    }
    comparisons: list[CmapssMonotonicityComparisonRow] = []
    for subset, model_name in sorted(raw_rows.keys() & calibrated_rows.keys()):
        raw = raw_rows[(subset, model_name)]
        calibrated = calibrated_rows[(subset, model_name)]
        comparisons.append(
            CmapssMonotonicityComparisonRow(
                subset=subset,
                model_name=model_name,
                raw_violation_rate=raw.violation_rate,
                calibrated_violation_rate=calibrated.violation_rate,
                violation_rate_delta=calibrated.violation_rate - raw.violation_rate,
                raw_max_violation_magnitude=raw.max_violation_magnitude,
                calibrated_max_violation_magnitude=(
                    calibrated.max_violation_magnitude
                ),
                max_violation_delta=(
                    calibrated.max_violation_magnitude
                    - raw.max_violation_magnitude
                ),
            )
        )
    return tuple(comparisons)


def write_cmapss_phase3_audit_markdown(
    result: CmapssPhase3AuditResult,
    path: str | Path,
) -> Path:
    """Write a compact Phase 3 audit Markdown report."""

    output_path = prepare_output_path(path)
    output_path.write_text(render_cmapss_phase3_audit_markdown(result), encoding="utf-8")
    return output_path


def render_cmapss_phase3_audit_markdown(result: CmapssPhase3AuditResult) -> str:
    """Render Phase 3 uncertainty and monotonicity evidence as Markdown."""

    lines = [
        "# C-MAPSS Phase 3 Audit",
        "",
        "## Uncertainty Coverage",
        "",
        (
            "| Subset | Model | Rows | Confidence | Coverage | Radius | "
            "Mean Width | Late Rows | Late Coverage | Uncovered Late |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.uncertainty_summaries:
        lines.append(
            "| "
            f"{row.subset} | "
            f"`{row.model_name}` | "
            f"{row.prediction_count} | "
            f"{row.confidence:.6f} | "
            f"{row.coverage:.6f} | "
            f"{row.interval_radius:.6f} | "
            f"{row.mean_interval_width:.6f} | "
            f"{row.late_prediction_count} | "
            f"{row.late_prediction_coverage:.6f} | "
            f"{row.uncovered_late_prediction_count} |"
        )
    lines.extend(
        [
            "",
            "## Coverage By Actual RUL Bin",
            "",
            (
                "| Subset | Model | Actual RUL Bin | Rows | Coverage | "
                "Mean Width | Mean Abs Error | Late Rate |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result.uncertainty_bins:
        lines.append(
            "| "
            f"{row.subset} | "
            f"`{row.model_name}` | "
            f"{row.actual_rul_bin} | "
            f"{row.prediction_count} | "
            f"{row.coverage:.6f} | "
            f"{row.mean_interval_width:.6f} | "
            f"{row.mean_absolute_error:.6f} | "
            f"{row.late_prediction_rate:.6f} |"
        )
    if result.failure_cases:
        lines.extend(
            [
                "",
                "## Uncovered Interval Cases",
                "",
                (
                    "| Rank | Subset | Model | Unit | Actual | Predicted | "
                    "Interval | Abs Error | Failure |"
                ),
                "|---:|---|---|---:|---:|---:|---|---:|---|",
            ]
        )
        for row in result.failure_cases:
            lines.append(
                "| "
                f"{row.rank} | "
                f"{row.subset} | "
                f"`{row.model_name}` | "
                f"{row.unit_number} | "
                f"{row.actual_rul:.6f} | "
                f"{row.predicted_rul:.6f} | "
                f"{row.lower_bound:.6f}-{row.upper_bound:.6f} | "
                f"{row.absolute_error:.6f} | "
                f"{row.failure_type} |"
            )
    if result.failure_notes:
        lines.extend(
            [
                "",
                "## Unit Failure Notes",
                "",
                (
                    "| Rank | Subset | Model | Unit | Actual Bin | Predicted Bin | "
                    "Global | Predicted-Bin | Floor | Failure | Note |"
                ),
                "|---:|---|---|---:|---|---|---:|---:|---:|---|---|",
            ]
        )
        for row in result.failure_notes:
            lines.append(
                "| "
                f"{row.rank} | "
                f"{row.subset} | "
                f"`{row.model_name}` | "
                f"{row.unit_number} | "
                f"{row.actual_rul_bin} | "
                f"{row.predicted_rul_bin} | "
                f"{_covered_label(row.global_covered)} | "
                f"{_covered_label(row.predicted_bin_covered)} | "
                f"{_covered_label(row.predicted_bin_floor_covered)} | "
                f"{row.failure_type} | "
                f"{row.note} |"
            )
    lines.extend(
        [
            "",
            "## Global Vs Predicted-Bin Intervals",
            "",
            (
                "| Subset | Model | Rows | Global Coverage | Predicted-Bin Coverage | "
                "Floor Coverage | Coverage Delta | Global Mean Width | "
                "Predicted-Bin Mean Width | Floor Mean Width | "
                "Late Failures Delta | Floor Late Failures Delta |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result.interval_comparisons:
        late_failure_delta = (
            row.predicted_bin_uncovered_late_prediction_count
            - row.global_uncovered_late_prediction_count
        )
        floor_late_failure_delta = (
            row.predicted_bin_floor_uncovered_late_prediction_count
            - row.global_uncovered_late_prediction_count
        )
        lines.append(
            "| "
            f"{row.subset} | "
            f"`{row.model_name}` | "
            f"{row.prediction_count} | "
            f"{row.global_coverage:.6f} | "
            f"{row.predicted_bin_coverage:.6f} | "
            f"{row.predicted_bin_floor_coverage:.6f} | "
            f"{row.coverage_delta:.6f} | "
            f"{row.global_mean_interval_width:.6f} | "
            f"{row.predicted_bin_mean_interval_width:.6f} | "
            f"{row.predicted_bin_floor_mean_interval_width:.6f} | "
            f"{late_failure_delta} | "
            f"{floor_late_failure_delta} |"
        )
    lines.extend(
        [
            "",
            "## Predicted-Bin Interval Calibration",
            "",
            (
                "| Subset | Model | Predicted RUL Bin | Rows | Radius | "
                "Mean Abs Residual | Max Abs Residual |"
            ),
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in result.predicted_bin_interval_calibrations:
        lines.append(
            "| "
            f"{row.subset} | "
            f"`{row.model_name}` | "
            f"{row.predicted_rul_bin} | "
            f"{row.calibration_count} | "
            f"{row.interval_radius:.6f} | "
            f"{row.mean_absolute_residual:.6f} | "
            f"{row.max_absolute_residual:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Global Vs Tail Fallback Intervals",
            "",
            (
                "| Subset | Model | Rows | Global Coverage | Tail Fallback Coverage | "
                "Coverage Delta | Global Mean Width | Tail Mean Width | "
                "Mean Width Delta | Global Median Width | Tail Median Width | "
                "Late Failures Delta |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result.tail_fallback_comparisons:
        lines.append(
            "| "
            f"{row.subset} | "
            f"`{row.model_name}` | "
            f"{row.prediction_count} | "
            f"{row.global_coverage:.6f} | "
            f"{row.tail_fallback_coverage:.6f} | "
            f"{row.coverage_delta:.6f} | "
            f"{row.global_mean_interval_width:.6f} | "
            f"{row.tail_fallback_mean_interval_width:.6f} | "
            f"{row.mean_interval_width_delta:.6f} | "
            f"{row.global_median_interval_width:.6f} | "
            f"{row.tail_fallback_median_interval_width:.6f} | "
            f"{row.uncovered_late_prediction_count_delta} |"
        )
    lines.extend(
        [
            "",
            "## Tail Fallback Calibration",
            "",
            (
                "| Subset | Model | Tail Threshold | Tail Confidence | Tail Rows | "
                "Global Radius | Tail Radius | Fallback Radius |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result.tail_fallback_calibrations:
        lines.append(
            "| "
            f"{row.subset} | "
            f"`{row.model_name}` | "
            f"{row.tail_threshold:.6f} | "
            f"{row.tail_confidence:.6f} | "
            f"{row.tail_calibration_count} | "
            f"{row.global_interval_radius:.6f} | "
            f"{row.tail_interval_radius:.6f} | "
            f"{row.fallback_interval_radius:.6f} |"
        )
    if result.tail_fallback_failure_notes:
        lines.extend(
            [
                "",
                "## Tail Fallback Unit Notes",
                "",
                (
                    "| Rank | Subset | Model | Unit | Actual Bin | Predicted Bin | "
                    "Global | Tail Fallback | Failure | Note |"
                ),
                "|---:|---|---|---:|---|---|---:|---:|---|---|",
            ]
        )
        for row in result.tail_fallback_failure_notes:
            lines.append(
                "| "
                f"{row.rank} | "
                f"{row.subset} | "
                f"`{row.model_name}` | "
                f"{row.unit_number} | "
                f"{row.actual_rul_bin} | "
                f"{row.predicted_rul_bin} | "
                f"{_covered_label(row.global_covered)} | "
                f"{_covered_label(row.tail_fallback_covered)} | "
                f"{row.failure_type} | "
                f"{row.note} |"
            )
    lines.extend(
        [
            "",
            "## Monotonicity Diagnostics",
            "",
            (
                "| Subset | Model | Units | Transitions | Violations | "
                "Violation Rate | Mean Violation | Max Violation |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result.monotonicity_diagnostics:
        lines.append(
            "| "
            f"{row.subset} | "
            f"`{row.model_name}` | "
            f"{row.unit_count} | "
            f"{row.transition_count} | "
            f"{row.violation_count} | "
            f"{row.violation_rate:.6f} | "
            f"{row.mean_violation_magnitude:.6f} | "
            f"{row.max_violation_magnitude:.6f} |"
        )
    if result.monotonicity_comparisons:
        lines.extend(
            [
                "",
                "## Raw Vs Calibrated Monotonicity",
                "",
                (
                    "| Subset | Model | Raw Violation Rate | Calibrated Violation Rate | "
                    "Delta | Raw Max Violation | Calibrated Max Violation |"
                ),
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result.monotonicity_comparisons:
            lines.append(
                "| "
                f"{row.subset} | "
                f"`{row.model_name}` | "
                f"{row.raw_violation_rate:.6f} | "
                f"{row.calibrated_violation_rate:.6f} | "
                f"{row.violation_rate_delta:.6f} | "
                f"{row.raw_max_violation_magnitude:.6f} | "
                f"{row.calibrated_max_violation_magnitude:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Training Recommendation",
            "",
            result.training_recommendation,
        ]
    )
    return "\n".join(lines) + "\n"


def _annotated_interval_row(
    row: dict[str, str],
    calibration_by_key: dict[tuple[str, str], CmapssIntervalCalibrationRow],
    *,
    clip_min: float,
) -> dict[str, Any]:
    subset = row["subset"]
    model_name = row["model_name"]
    calibration = calibration_by_key.get((subset, model_name))
    if calibration is None:
        raise ValueError(
            "missing interval calibration for prediction row: "
            f"subset={subset}, model_name={model_name}"
        )
    predicted_rul = _float(row["predicted_rul"])
    actual_rul = _float(row["actual_rul"])
    lower_bound = max(clip_min, predicted_rul - calibration.interval_radius)
    upper_bound = predicted_rul + calibration.interval_radius
    covered = lower_bound <= actual_rul <= upper_bound
    error = _float(row["error"])
    return {
        "subset": subset,
        "model_name": model_name,
        "unit_number": int(row["unit_number"]),
        "actual_rul": actual_rul,
        "predicted_rul": predicted_rul,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "interval_width": upper_bound - lower_bound,
        "covered": covered,
        "error": error,
        "absolute_error": _float(row["absolute_error"]),
        "late_prediction": error > 0.0,
        "failure_type": _failure_type(error, covered),
        "method": calibration.method,
        "confidence": calibration.confidence,
        "interval_radius": calibration.interval_radius,
    }


def _annotated_predicted_bin_interval_row(
    row: dict[str, str],
    calibration_by_key: dict[
        tuple[str, str, str],
        CmapssPredictedBinIntervalCalibrationRow,
    ],
    *,
    clip_min: float,
    floor_to_global: bool = False,
) -> dict[str, Any]:
    subset = row["subset"]
    model_name = row["model_name"]
    bin_label = _predicted_rul_bin(_float(row["predicted_rul"]))
    global_calibration = calibration_by_key.get((subset, model_name, "all"))
    calibration = calibration_by_key.get(
        (subset, model_name, bin_label),
        global_calibration,
    )
    if calibration is None:
        raise ValueError(
            "missing predicted-bin interval calibration for prediction row: "
            f"subset={subset}, model_name={model_name}, predicted_rul_bin={bin_label}"
        )
    interval_radius = calibration.interval_radius
    method = calibration.method
    if floor_to_global:
        if global_calibration is None:
            raise ValueError(
                "missing global interval calibration for predicted-bin floor: "
                f"subset={subset}, model_name={model_name}"
            )
        interval_radius = max(interval_radius, global_calibration.interval_radius)
        method = f"{calibration.method}_global_floor"
    predicted_rul = _float(row["predicted_rul"])
    actual_rul = _float(row["actual_rul"])
    lower_bound = max(clip_min, predicted_rul - interval_radius)
    upper_bound = predicted_rul + interval_radius
    covered = lower_bound <= actual_rul <= upper_bound
    error = _float(row["error"])
    return {
        "subset": subset,
        "model_name": model_name,
        "unit_number": int(row["unit_number"]),
        "actual_rul": actual_rul,
        "predicted_rul": predicted_rul,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "interval_width": upper_bound - lower_bound,
        "covered": covered,
        "error": error,
        "absolute_error": _float(row["absolute_error"]),
        "late_prediction": error > 0.0,
        "failure_type": _failure_type(error, covered),
        "method": method,
        "confidence": calibration.confidence,
        "interval_radius": interval_radius,
    }


def _annotated_tail_fallback_interval_row(
    row: dict[str, str],
    calibration_by_key: dict[tuple[str, str], CmapssTailFallbackCalibrationRow],
    *,
    clip_min: float,
) -> dict[str, Any]:
    subset = row["subset"]
    model_name = row["model_name"]
    calibration = calibration_by_key.get((subset, model_name))
    if calibration is None:
        raise ValueError(
            "missing tail fallback calibration for prediction row: "
            f"subset={subset}, model_name={model_name}"
        )
    predicted_rul = _float(row["predicted_rul"])
    actual_rul = _float(row["actual_rul"])
    interval_radius = (
        calibration.fallback_interval_radius
        if predicted_rul >= calibration.tail_threshold
        else calibration.global_interval_radius
    )
    lower_bound = max(clip_min, predicted_rul - interval_radius)
    upper_bound = predicted_rul + interval_radius
    covered = lower_bound <= actual_rul <= upper_bound
    error = _float(row["error"])
    return {
        "subset": subset,
        "model_name": model_name,
        "unit_number": int(row["unit_number"]),
        "actual_rul": actual_rul,
        "predicted_rul": predicted_rul,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "interval_width": upper_bound - lower_bound,
        "covered": covered,
        "error": error,
        "absolute_error": _float(row["absolute_error"]),
        "late_prediction": error > 0.0,
        "failure_type": _failure_type(error, covered),
        "method": calibration.method,
        "confidence": calibration.global_confidence,
        "interval_radius": interval_radius,
    }


def _summary_row(
    subset: str,
    model_name: str,
    rows: list[dict[str, Any]],
) -> CmapssUncertaintySummaryRow:
    late_rows = [row for row in rows if row["late_prediction"]]
    uncovered_late_rows = [row for row in late_rows if not row["covered"]]
    return CmapssUncertaintySummaryRow(
        subset=subset,
        model_name=model_name,
        method=rows[0]["method"],
        confidence=rows[0]["confidence"],
        prediction_count=len(rows),
        covered_count=sum(1 for row in rows if row["covered"]),
        coverage=_rate(row["covered"] for row in rows),
        interval_radius=_mean(row["interval_radius"] for row in rows),
        mean_interval_width=_mean(row["interval_width"] for row in rows),
        median_interval_width=_median(row["interval_width"] for row in rows),
        mean_absolute_error=_mean(row["absolute_error"] for row in rows),
        late_prediction_count=len(late_rows),
        late_prediction_coverage=(
            _rate(row["covered"] for row in late_rows) if late_rows else 0.0
        ),
        uncovered_late_prediction_count=len(uncovered_late_rows),
    )


def _bin_row(
    subset: str,
    model_name: str,
    bin_label: str,
    rows: list[dict[str, Any]],
) -> CmapssUncertaintyBinRow:
    return CmapssUncertaintyBinRow(
        subset=subset,
        model_name=model_name,
        actual_rul_bin=bin_label,
        prediction_count=len(rows),
        covered_count=sum(1 for row in rows if row["covered"]),
        coverage=_rate(row["covered"] for row in rows),
        mean_interval_width=_mean(row["interval_width"] for row in rows),
        mean_absolute_error=_mean(row["absolute_error"] for row in rows),
        late_prediction_rate=_rate(row["late_prediction"] for row in rows),
    )


def _failure_note_row(
    rank: int,
    row: dict[str, Any],
) -> CmapssFailureNoteRow:
    global_row = row["global"]
    predicted_bin_row = row["predicted_bin"]
    predicted_bin_floor_row = row["predicted_bin_floor"]
    uncovered_strategies = tuple(row["uncovered_strategies"])
    failure_type = _failure_type(global_row["error"], True)
    if not global_row["covered"]:
        failure_type = global_row["failure_type"]
    elif not predicted_bin_row["covered"]:
        failure_type = predicted_bin_row["failure_type"]
    elif not predicted_bin_floor_row["covered"]:
        failure_type = predicted_bin_floor_row["failure_type"]
    return CmapssFailureNoteRow(
        rank=rank,
        subset=global_row["subset"],
        model_name=global_row["model_name"],
        unit_number=global_row["unit_number"],
        actual_rul=global_row["actual_rul"],
        predicted_rul=global_row["predicted_rul"],
        error=global_row["error"],
        absolute_error=global_row["absolute_error"],
        actual_rul_bin=_actual_rul_bin(global_row["actual_rul"]),
        predicted_rul_bin=_predicted_rul_bin(global_row["predicted_rul"]),
        failure_type=failure_type,
        global_covered=global_row["covered"],
        predicted_bin_covered=predicted_bin_row["covered"],
        predicted_bin_floor_covered=predicted_bin_floor_row["covered"],
        uncovered_strategy_count=len(uncovered_strategies),
        uncovered_strategies=", ".join(uncovered_strategies),
        note=_failure_note_text(global_row, uncovered_strategies),
    )


def _tail_fallback_failure_note_row(
    rank: int,
    row: dict[str, Any],
) -> CmapssTailFallbackFailureNoteRow:
    global_row = row["global"]
    tail_row = row["tail_fallback"]
    failure_type = global_row["failure_type"]
    if global_row["covered"] and not tail_row["covered"]:
        failure_type = tail_row["failure_type"]
    note = _tail_fallback_note_text(global_row, tail_row)
    return CmapssTailFallbackFailureNoteRow(
        rank=rank,
        subset=global_row["subset"],
        model_name=global_row["model_name"],
        unit_number=global_row["unit_number"],
        actual_rul=global_row["actual_rul"],
        predicted_rul=global_row["predicted_rul"],
        error=global_row["error"],
        absolute_error=global_row["absolute_error"],
        actual_rul_bin=_actual_rul_bin(global_row["actual_rul"]),
        predicted_rul_bin=_predicted_rul_bin(global_row["predicted_rul"]),
        failure_type=failure_type,
        global_covered=global_row["covered"],
        tail_fallback_covered=tail_row["covered"],
        note=note,
    )


def _failure_note_text(
    row: dict[str, Any],
    uncovered_strategies: tuple[str, ...],
) -> str:
    direction = "late overestimate" if row["error"] > 0 else "early underestimate"
    return (
        f"{direction}; actual bin {_actual_rul_bin(row['actual_rul'])}; "
        f"predicted bin {_predicted_rul_bin(row['predicted_rul'])}; "
        f"uncovered by {', '.join(uncovered_strategies)}"
    )


def _tail_fallback_note_text(
    global_row: dict[str, Any],
    tail_row: dict[str, Any],
) -> str:
    direction = "late overestimate" if global_row["error"] > 0 else "early underestimate"
    global_state = "covered" if global_row["covered"] else "uncovered"
    tail_state = "covered" if tail_row["covered"] else "uncovered"
    return (
        f"{direction}; actual bin {_actual_rul_bin(global_row['actual_rul'])}; "
        f"predicted bin {_predicted_rul_bin(global_row['predicted_rul'])}; "
        f"global {global_state}; tail fallback {tail_state}"
    )


def _predicted_bin_interval_calibration_row(
    subset: str,
    model_name: str,
    predicted_rul_bin: str,
    rows: list[dict[str, str]],
    *,
    confidence: float,
) -> CmapssPredictedBinIntervalCalibrationRow:
    residuals = sorted(_float(row["absolute_error"]) for row in rows)
    return CmapssPredictedBinIntervalCalibrationRow(
        subset=subset,
        model_name=model_name,
        predicted_rul_bin=predicted_rul_bin,
        method="validation_predicted_rul_bin_absolute_residual_quantile",
        confidence=confidence,
        calibration_count=len(rows),
        interval_radius=_nearest_rank_quantile(residuals, confidence),
        mean_absolute_residual=_mean(residuals),
        max_absolute_residual=max(residuals),
    )


def _training_recommendation(
    summaries: tuple[CmapssUncertaintySummaryRow, ...],
    monotonicity_rows: tuple[CmapssPredictionMonotonicityDiagnosticRow, ...],
    comparisons: tuple[CmapssMonotonicityComparisonRow, ...],
) -> str:
    low_coverage = [
        row
        for row in summaries
        if row.coverage + 1e-12 < row.confidence
        or row.uncovered_late_prediction_count > 0
    ]
    monotonic_issues = [row for row in monotonicity_rows if row.violation_count > 0]
    improved_by_calibration = [
        row for row in comparisons if row.violation_rate_delta < 0.0
    ]
    if low_coverage or monotonic_issues:
        details: list[str] = []
        if low_coverage:
            details.append("interval coverage or late-risk coverage is below target")
        if monotonic_issues:
            details.append("prediction trajectories still contain RUL increases")
        if improved_by_calibration:
            details.append("calibration changes monotonicity, so compare raw and calibrated traces")
        return (
            "diagnostic_first: do not add new constrained losses yet; "
            + "; ".join(details)
            + "."
        )
    return (
        "diagnostic_first: current evidence does not justify a new constrained "
        "loss before broader validation."
    )


def _failure_type(error: float, covered: bool) -> str:
    if covered:
        return "covered"
    if error > 0.0:
        return "late_uncovered"
    if error < 0.0:
        return "early_uncovered"
    return "uncovered"


def _covered_label(covered: bool) -> str:
    return "covered" if covered else "uncovered"


def _nearest_rank_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile for an empty sequence")
    rank = max(1, math.ceil(quantile * len(values)))
    return values[min(rank - 1, len(values) - 1)]


def _read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"missing prediction CSV: {prediction_path}")
    with prediction_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"prediction CSV has no rows: {prediction_path}")
    missing_columns = _REQUIRED_PREDICTION_COLUMNS - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"prediction CSV is missing required columns: {missing}")
    return rows


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


def _actual_rul_bin_sort_key(label: str) -> int:
    return {"0-30": 0, "31-60": 1, "61-90": 2, "91-120": 3, "121+": 4}[
        label
    ]


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("cannot average an empty metric sequence")
    return sum(sequence) / len(sequence)


def _median(values: Iterable[float]) -> float:
    sequence = sorted(values)
    if not sequence:
        raise ValueError("cannot compute median for an empty sequence")
    midpoint = len(sequence) // 2
    if len(sequence) % 2:
        return sequence[midpoint]
    return (sequence[midpoint - 1] + sequence[midpoint]) / 2


def _rate(values: Iterable[bool]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("cannot compute a rate for an empty sequence")
    return sum(1 for value in sequence if value) / len(sequence)


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
