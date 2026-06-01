from __future__ import annotations

import csv

from aerospace_prognostics.reports.cmapss_prediction_diagnostics import (
    build_cmapss_prediction_diagnostics,
    build_cmapss_prediction_rul_bin_diagnostics,
    render_cmapss_prediction_diagnostics_markdown,
    select_cmapss_high_error_predictions,
    write_cmapss_prediction_diagnostics_csv,
    write_cmapss_prediction_diagnostics_markdown,
    write_cmapss_prediction_rul_bin_diagnostics_csv,
)


def test_build_cmapss_prediction_diagnostics_summarizes_model_errors(
    tmp_path,
) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "cnn", 1, 100.0, 110.0),
            _prediction("FD001", "cnn", 2, 80.0, 70.0),
            _prediction("FD001", "tcn", 1, 100.0, 130.0),
        ],
    )

    rows = build_cmapss_prediction_diagnostics(predictions_csv)

    assert [(row.subset, row.model_name) for row in rows] == [
        ("FD001", "cnn"),
        ("FD001", "tcn"),
    ]
    cnn_row = rows[0]
    assert cnn_row.prediction_count == 2
    assert cnn_row.mean_error == 0.0
    assert cnn_row.mean_absolute_error == 10.0
    assert cnn_row.max_absolute_error == 10.0
    assert cnn_row.mean_late_error == 5.0
    assert cnn_row.late_prediction_rate == 0.5
    assert cnn_row.mean_early_error == 5.0
    assert cnn_row.early_prediction_rate == 0.5


def test_cmapss_prediction_diagnostics_outputs_csv_and_markdown(tmp_path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    output_csv = tmp_path / "reports" / "diagnostics.csv"
    output_bins_csv = tmp_path / "reports" / "diagnostics_by_rul_bin.csv"
    output_markdown = tmp_path / "reports" / "diagnostics.md"
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "cnn", 1, 100.0, 110.0),
            _prediction("FD001", "tcn", 2, 80.0, 55.0),
        ],
    )
    diagnostics = build_cmapss_prediction_diagnostics(predictions_csv)
    rul_bin_diagnostics = build_cmapss_prediction_rul_bin_diagnostics(predictions_csv)
    outliers = select_cmapss_high_error_predictions(predictions_csv, top_n=1)

    write_cmapss_prediction_diagnostics_csv(diagnostics, output_csv)
    write_cmapss_prediction_rul_bin_diagnostics_csv(rul_bin_diagnostics, output_bins_csv)
    write_cmapss_prediction_diagnostics_markdown(
        diagnostics,
        outliers,
        output_markdown,
        rul_bin_diagnostics=rul_bin_diagnostics,
    )

    csv_rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))
    bin_rows = list(
        csv.DictReader(output_bins_csv.open("r", encoding="utf-8", newline=""))
    )
    markdown = output_markdown.read_text(encoding="utf-8")
    assert csv_rows[0]["model_name"] == "cnn"
    assert bin_rows[0]["actual_rul_bin"] == "91-120"
    assert "# C-MAPSS Deep Prediction Diagnostics" in markdown
    assert "## Error By Actual RUL Bin" in markdown
    assert "## Highest Absolute Errors" in markdown
    assert "`tcn`" in markdown


def test_build_cmapss_prediction_rul_bin_diagnostics_groups_actual_rul_ranges(
    tmp_path,
) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "cnn", 1, 20.0, 25.0),
            _prediction("FD001", "cnn", 2, 55.0, 45.0),
            _prediction("FD001", "cnn", 3, 75.0, 85.0),
            _prediction("FD001", "cnn", 4, 110.0, 80.0),
            _prediction("FD001", "cnn", 5, 135.0, 130.0),
        ],
    )

    rows = build_cmapss_prediction_rul_bin_diagnostics(predictions_csv)

    assert [row.actual_rul_bin for row in rows] == [
        "0-30",
        "31-60",
        "61-90",
        "91-120",
        "121+",
    ]
    assert rows[0].prediction_count == 1
    assert rows[0].mean_error == 5.0
    assert rows[3].mean_absolute_error == 30.0


def test_select_cmapss_high_error_predictions_ranks_by_absolute_error(tmp_path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions(
        predictions_csv,
        [
            _prediction("FD001", "cnn", 1, 100.0, 110.0),
            _prediction("FD001", "cnn", 2, 80.0, 50.0),
            _prediction("FD001", "tcn", 3, 80.0, 60.0),
        ],
    )

    outliers = select_cmapss_high_error_predictions(predictions_csv, top_n=2)

    assert [(row.rank_by_absolute_error, row.unit_number) for row in outliers] == [
        (1, 2),
        (2, 3),
    ]
    assert outliers[0].early_error == 30.0


def test_render_cmapss_prediction_diagnostics_markdown_rejects_empty_rows() -> None:
    try:
        render_cmapss_prediction_diagnostics_markdown([], [])
    except ValueError as error:
        assert "diagnostics must contain at least one item" in str(error)
    else:
        raise AssertionError("expected empty diagnostics to fail")


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
        "subset": subset,
        "model_name": model_name,
        "selected_epoch": 1,
        "unit_number": unit_number,
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
