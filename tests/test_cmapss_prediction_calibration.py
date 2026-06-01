from __future__ import annotations

import csv

import pytest

from aerospace_prognostics.reports.cmapss_prediction_calibration import (
    calibrate_cmapss_deep_predictions,
    fit_cmapss_affine_prediction_calibrations,
    fit_cmapss_predicted_rul_bin_nasa_shift_calibrations,
    fit_cmapss_predicted_rul_bin_residual_calibrations,
)


def test_fit_cmapss_affine_prediction_calibrations_by_subset_and_model(
    tmp_path,
) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "cnn", 1, 10.0, 5.0),
            _prediction("FD001", "cnn", 2, 30.0, 15.0),
            _prediction("FD001", "tcn", 1, 18.0, 20.0),
            _prediction("FD001", "tcn", 2, 22.0, 20.0),
        ],
    )

    calibrations = fit_cmapss_affine_prediction_calibrations(calibration_csv)

    assert [(row.subset, row.model_name) for row in calibrations] == [
        ("FD001", "cnn"),
        ("FD001", "tcn"),
    ]
    assert calibrations[0].calibration_count == 2
    assert calibrations[0].intercept == pytest.approx(0.0)
    assert calibrations[0].slope == pytest.approx(2.0)
    assert calibrations[1].raw_predicted_rul_variance == pytest.approx(0.0)
    assert calibrations[1].intercept == pytest.approx(0.0)
    assert calibrations[1].slope == pytest.approx(1.0)


def test_calibrate_cmapss_deep_predictions_writes_recomputed_reports(tmp_path) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    predictions_csv = tmp_path / "official_predictions.csv"
    output_csv = tmp_path / "reports" / "official_predictions_calibrated.csv"
    output_calibration_csv = tmp_path / "reports" / "calibration.csv"
    output_diagnostics_csv = tmp_path / "reports" / "diagnostics.csv"
    output_rul_bins_csv = tmp_path / "reports" / "diagnostics_by_rul_bin.csv"
    output_markdown = tmp_path / "reports" / "diagnostics.md"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "cnn", 1, 10.0, 5.0),
            _prediction("FD001", "cnn", 2, 30.0, 15.0),
        ],
    )
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "cnn", 3, 20.0, 8.0),
        ],
    )

    result = calibrate_cmapss_deep_predictions(
        calibration_csv=calibration_csv,
        predictions_csv=predictions_csv,
        output_csv=output_csv,
        output_calibration_csv=output_calibration_csv,
        output_diagnostics_csv=output_diagnostics_csv,
        output_rul_bins_csv=output_rul_bins_csv,
        output_markdown=output_markdown,
    )

    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))
    assert result.calibrated_prediction_count == 1
    assert result.calibration_csv_path == output_calibration_csv
    assert result.diagnostics_csv_path == output_diagnostics_csv
    assert rows[0]["calibration_method"] == "validation_affine"
    assert float(rows[0]["raw_predicted_rul"]) == pytest.approx(8.0)
    assert float(rows[0]["predicted_rul"]) == pytest.approx(16.0)
    assert float(rows[0]["error"]) == pytest.approx(-4.0)
    assert float(rows[0]["absolute_error"]) == pytest.approx(4.0)
    assert float(rows[0]["late_error"]) == pytest.approx(0.0)
    assert float(rows[0]["early_error"]) == pytest.approx(4.0)
    assert output_calibration_csv.exists()
    assert output_diagnostics_csv.exists()
    assert output_rul_bins_csv.exists()
    assert "# C-MAPSS Deep Prediction Diagnostics" in output_markdown.read_text(
        encoding="utf-8"
    )


def test_predicted_rul_bin_residual_calibration_applies_inference_safe_bins(
    tmp_path,
) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    predictions_csv = tmp_path / "official_predictions.csv"
    output_csv = tmp_path / "reports" / "official_predictions_calibrated.csv"
    output_calibration_csv = tmp_path / "reports" / "calibration.csv"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "cnn", 1, 10.0, 20.0),
            _prediction("FD001", "cnn", 2, 20.0, 40.0),
            _prediction("FD001", "cnn", 3, 55.0, 45.0),
            _prediction("FD001", "cnn", 4, 65.0, 80.0),
        ],
    )
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "cnn", 5, 30.0, 45.0),
        ],
    )

    calibrations = fit_cmapss_predicted_rul_bin_residual_calibrations(
        calibration_csv,
        shrinkage_strength=0.0,
    )
    result = calibrate_cmapss_deep_predictions(
        calibration_csv=calibration_csv,
        predictions_csv=predictions_csv,
        output_csv=output_csv,
        output_calibration_csv=output_calibration_csv,
        method="predicted_bin_residual",
        shrinkage_strength=0.0,
    )

    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))
    calibration_rows = list(
        csv.DictReader(output_calibration_csv.open("r", encoding="utf-8", newline=""))
    )
    bin_calibration = [
        row for row in calibrations if row.predicted_rul_bin == "31-60"
    ][0]
    assert bin_calibration.correction == pytest.approx(-5.0)
    assert result.calibrated_prediction_count == 1
    assert rows[0]["calibration_method"] == "validation_predicted_bin_residual"
    assert rows[0]["calibration_predicted_rul_bin"] == "31-60"
    assert float(rows[0]["raw_predicted_rul"]) == pytest.approx(45.0)
    assert float(rows[0]["predicted_rul"]) == pytest.approx(40.0)
    assert float(rows[0]["error"]) == pytest.approx(10.0)
    assert {row["predicted_rul_bin"] for row in calibration_rows} >= {
        "all",
        "0-30",
        "31-60",
        "61-90",
    }


def test_predicted_rul_bin_nasa_shift_prefers_validation_nasa_score(
    tmp_path,
) -> None:
    calibration_csv = tmp_path / "validation_predictions.csv"
    predictions_csv = tmp_path / "official_predictions.csv"
    output_csv = tmp_path / "reports" / "official_predictions_calibrated.csv"
    _write_predictions(
        calibration_csv,
        [
            _prediction("FD001", "cnn", 1, 30.0, 45.0),
            _prediction("FD001", "cnn", 2, 35.0, 45.0),
            _prediction("FD001", "cnn", 3, 40.0, 45.0),
        ],
    )
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "cnn", 4, 30.0, 45.0),
        ],
    )

    calibrations = fit_cmapss_predicted_rul_bin_nasa_shift_calibrations(
        calibration_csv,
        shrinkage_strength=0.0,
        max_shift=20.0,
        shift_step=5.0,
    )
    result = calibrate_cmapss_deep_predictions(
        calibration_csv=calibration_csv,
        predictions_csv=predictions_csv,
        output_csv=output_csv,
        method="predicted_bin_nasa_shift",
        shrinkage_strength=0.0,
    )

    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))
    bin_calibration = [
        row for row in calibrations if row.predicted_rul_bin == "31-60"
    ][0]
    assert bin_calibration.method == "validation_predicted_bin_nasa_shift"
    assert bin_calibration.correction < 0.0
    assert result.calibrated_prediction_count == 1
    assert rows[0]["calibration_method"] == "validation_predicted_bin_nasa_shift"
    assert rows[0]["calibration_predicted_rul_bin"] == "31-60"
    assert float(rows[0]["predicted_rul"]) < 45.0


def _prediction(
    subset: str,
    model_name: str,
    unit_number: int,
    actual_rul: float,
    predicted_rul: float,
) -> dict[str, str | int | float]:
    error = predicted_rul - actual_rul
    return {
        "dataset": "C-MAPSS-sequence",
        "prediction_split": "validation_selection",
        "subset": subset,
        "model_name": model_name,
        "selected_epoch": 1,
        "unit_number": unit_number,
        "end_cycle": 10 + unit_number,
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
