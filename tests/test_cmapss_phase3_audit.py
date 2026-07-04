from __future__ import annotations

import csv
import json

import pytest

from aerospace_prognostics.reports.cmapss_phase3 import run_cmapss_phase3_audit


def test_cmapss_phase3_audit_reports_interval_coverage_and_failures(tmp_path) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    predictions_csv = tmp_path / "official_predictions.csv"
    output_json = tmp_path / "reports" / "phase3_audit.json"
    output_markdown = tmp_path / "reports" / "phase3_audit.md"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "transformer", 1, 50.0, 48.0),
            _prediction("FD001", "transformer", 2, 60.0, 56.0),
            _prediction("FD001", "transformer", 3, 70.0, 76.0),
        ],
    )
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "transformer", 10, 50.0, 45.0),
            _prediction("FD001", "transformer", 11, 40.0, 52.0),
            _prediction("FD001", "transformer", 12, 80.0, 77.0),
        ],
    )

    result = run_cmapss_phase3_audit(
        calibration_csv=calibration_csv,
        predictions_csv=predictions_csv,
        output_json=output_json,
        output_markdown=output_markdown,
        confidence=0.67,
        top_n=2,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = result.uncertainty_summaries[0]
    assert summary.interval_radius == pytest.approx(6.0)
    assert summary.coverage == pytest.approx(2 / 3)
    assert summary.mean_interval_width == pytest.approx(12.0)
    assert summary.late_prediction_count == 1
    assert summary.late_prediction_coverage == pytest.approx(0.0)
    assert result.failure_cases[0].unit_number == 11
    assert result.failure_cases[0].failure_type == "late_uncovered"
    assert payload["schema_version"] == "aerospace-prognostics/cmapss-phase3-audit/v1"
    assert payload["uncertainty"]["summaries"][0]["coverage"] == pytest.approx(2 / 3)
    assert "failure_notes" in payload["uncertainty"]
    assert "## Uncertainty Coverage" in output_markdown.read_text(encoding="utf-8")


def test_cmapss_phase3_audit_compares_predicted_bin_interval_coverage(
    tmp_path,
) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    predictions_csv = tmp_path / "official_predictions.csv"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "transformer", 1, 48.0, 50.0),
            _prediction("FD001", "transformer", 2, 51.0, 55.0),
            _prediction("FD001", "transformer", 3, 100.0, 130.0),
            _prediction("FD001", "transformer", 4, 120.0, 140.0),
        ],
    )
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "transformer", 10, 51.0, 55.0),
            _prediction("FD001", "transformer", 11, 100.0, 130.0),
            _prediction("FD001", "transformer", 12, 180.0, 150.0),
        ],
    )

    result = run_cmapss_phase3_audit(
        calibration_csv=calibration_csv,
        predictions_csv=predictions_csv,
        confidence=0.75,
    )

    high_rul_calibration = [
        row
        for row in result.predicted_bin_interval_calibrations
        if row.predicted_rul_bin == "121+"
    ][0]
    comparison = result.interval_comparisons[0]
    assert high_rul_calibration.interval_radius == pytest.approx(30.0)
    assert comparison.global_coverage == pytest.approx(1 / 3)
    assert comparison.predicted_bin_coverage == pytest.approx(1.0)
    assert comparison.coverage_delta == pytest.approx(2 / 3)
    assert comparison.global_uncovered_late_prediction_count == 1
    assert comparison.predicted_bin_uncovered_late_prediction_count == 0
    assert comparison.predicted_bin_floor_coverage == pytest.approx(1.0)


def test_cmapss_phase3_audit_reports_predicted_bin_global_floor(
    tmp_path,
) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    predictions_csv = tmp_path / "official_predictions.csv"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "transformer", 1, 50.0, 51.0),
            _prediction("FD001", "transformer", 2, 53.0, 55.0),
            _prediction("FD001", "transformer", 3, 100.0, 120.0),
            _prediction("FD001", "transformer", 4, 100.0, 130.0),
        ],
    )
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "transformer", 10, 70.0, 55.0),
        ],
    )

    result = run_cmapss_phase3_audit(
        calibration_csv=calibration_csv,
        predictions_csv=predictions_csv,
        confidence=0.75,
    )

    comparison = result.interval_comparisons[0]
    assert comparison.global_coverage == pytest.approx(1.0)
    assert comparison.predicted_bin_coverage == pytest.approx(0.0)
    assert comparison.predicted_bin_floor_coverage == pytest.approx(1.0)
    assert comparison.predicted_bin_mean_interval_width == pytest.approx(4.0)
    assert comparison.predicted_bin_floor_mean_interval_width == pytest.approx(40.0)
    assert result.predicted_bin_floor_uncertainty_summaries[0].coverage == pytest.approx(
        1.0
    )


def test_cmapss_phase3_audit_reports_tail_fallback_experiment(tmp_path) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    predictions_csv = tmp_path / "official_predictions.csv"
    output_markdown = tmp_path / "phase3_audit.md"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "transformer", 1, 20.0, 21.0),
            _prediction("FD001", "transformer", 2, 50.0, 52.0),
            _prediction("FD001", "transformer", 3, 100.0, 120.0),
            _prediction("FD001", "transformer", 4, 100.0, 135.0),
        ],
    )
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "transformer", 10, 70.0, 95.0),
            _prediction("FD001", "transformer", 11, 70.0, 75.0),
        ],
    )

    result = run_cmapss_phase3_audit(
        calibration_csv=calibration_csv,
        predictions_csv=predictions_csv,
        output_markdown=output_markdown,
        confidence=0.75,
    )

    calibration = result.tail_fallback_calibrations[0]
    comparison = result.tail_fallback_comparisons[0]
    notes = {row.unit_number: row for row in result.tail_fallback_failure_notes}
    assert calibration.tail_calibration_count == 2
    assert calibration.tail_interval_radius == pytest.approx(35.0)
    assert calibration.fallback_interval_radius == pytest.approx(35.0)
    assert comparison.global_coverage == pytest.approx(0.5)
    assert comparison.tail_fallback_coverage == pytest.approx(1.0)
    assert comparison.tail_fallback_mean_interval_width == pytest.approx(55.0)
    assert notes[10].global_covered is False
    assert notes[10].tail_fallback_covered is True
    assert 11 not in notes
    assert "tail_fallback_calibrations" in result.to_payload()["uncertainty"]
    assert "## Global Vs Tail Fallback Intervals" in output_markdown.read_text(
        encoding="utf-8"
    )


def test_cmapss_phase3_audit_adds_unit_failure_notes(tmp_path) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    predictions_csv = tmp_path / "official_predictions.csv"
    output_markdown = tmp_path / "phase3_audit.md"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "transformer", 1, 50.0, 51.0),
            _prediction("FD001", "transformer", 2, 53.0, 55.0),
            _prediction("FD001", "transformer", 3, 100.0, 120.0),
            _prediction("FD001", "transformer", 4, 100.0, 130.0),
        ],
    )
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "transformer", 10, 70.0, 55.0),
            _prediction("FD001", "transformer", 11, 70.0, 95.0),
        ],
    )

    result = run_cmapss_phase3_audit(
        calibration_csv=calibration_csv,
        predictions_csv=predictions_csv,
        output_markdown=output_markdown,
        confidence=0.75,
    )

    notes = {row.unit_number: row for row in result.failure_notes}
    assert notes[10].actual_rul_bin == "61-90"
    assert notes[10].predicted_rul_bin == "31-60"
    assert notes[10].global_covered is True
    assert notes[10].predicted_bin_covered is False
    assert notes[10].predicted_bin_floor_covered is True
    assert notes[10].uncovered_strategy_count == 1
    assert "predicted_bin" in notes[10].uncovered_strategies
    assert notes[11].failure_type == "late_uncovered"
    assert notes[11].uncovered_strategy_count == 3
    assert "## Unit Failure Notes" in output_markdown.read_text(encoding="utf-8")


def test_cmapss_phase3_audit_compares_raw_and_calibrated_monotonicity(
    tmp_path,
) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    predictions_csv = tmp_path / "official_predictions.csv"
    calibrated_predictions_csv = tmp_path / "calibrated_predictions.csv"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "transformer", 1, 100.0, 95.0),
            _prediction("FD001", "transformer", 2, 90.0, 86.0),
        ],
    )
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "transformer", 1, 100.0, 80.0, end_cycle=10),
            _prediction("FD001", "transformer", 1, 99.0, 84.0, end_cycle=11),
            _prediction("FD001", "transformer", 1, 98.0, 83.0, end_cycle=12),
        ],
    )
    _write_predictions(
        calibrated_predictions_csv,
        [
            _prediction("FD001", "transformer", 1, 100.0, 80.0, end_cycle=10),
            _prediction("FD001", "transformer", 1, 99.0, 79.0, end_cycle=11),
            _prediction("FD001", "transformer", 1, 98.0, 78.0, end_cycle=12),
        ],
    )

    result = run_cmapss_phase3_audit(
        calibration_csv=calibration_csv,
        predictions_csv=predictions_csv,
        calibrated_predictions_csv=calibrated_predictions_csv,
    )

    comparison = result.monotonicity_comparisons[0]
    assert comparison.raw_violation_rate == pytest.approx(0.5)
    assert comparison.calibrated_violation_rate == pytest.approx(0.0)
    assert comparison.violation_rate_delta == pytest.approx(-0.5)
    assert "diagnostic_first" in result.training_recommendation


def _prediction(
    subset: str,
    model_name: str,
    unit_number: int,
    actual_rul: float,
    predicted_rul: float,
    *,
    end_cycle: int | None = None,
) -> dict[str, str | int | float]:
    error = predicted_rul - actual_rul
    return {
        "dataset": "C-MAPSS-sequence",
        "prediction_split": "validation_selection",
        "subset": subset,
        "model_name": model_name,
        "selected_epoch": 1,
        "unit_number": unit_number,
        "end_cycle": end_cycle if end_cycle is not None else 10 + unit_number,
        "actual_rul": actual_rul,
        "predicted_rul": predicted_rul,
        "error": error,
        "absolute_error": abs(error),
        "late_error": max(error, 0.0),
        "early_error": max(-error, 0.0),
    }


def _write_predictions(
    path,
    rows: list[dict[str, str | int | float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
