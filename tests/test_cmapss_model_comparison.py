from __future__ import annotations

import csv

from aerospace_prognostics.evaluation import RegressionRunResult, write_results_csv
from aerospace_prognostics.reports.cmapss_model_comparison import (
    build_cmapss_model_comparison,
    render_cmapss_model_comparison_markdown,
    write_cmapss_model_comparison_csv,
    write_cmapss_model_comparison_markdown,
)


def test_build_cmapss_model_comparison_ranks_candidates_against_baseline(
    tmp_path,
) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"
    write_results_csv(
        [
            _result("FD001", "hgb", rmse=10.0, nasa_score=100.0),
            _result("FD002", "hgb", rmse=20.0, nasa_score=200.0),
        ],
        baseline_csv,
    )
    write_results_csv(
        [
            _result("FD001", "cnn", rmse=12.0, nasa_score=120.0),
            _result("FD001", "tcn", rmse=9.0, nasa_score=80.0),
            _result("FD002", "cnn", rmse=22.0, nasa_score=250.0),
        ],
        candidate_csv,
    )

    rows = build_cmapss_model_comparison(baseline_csv, (candidate_csv,))

    assert [(row.subset, row.rank_by_nasa, row.model_name) for row in rows] == [
        ("FD001", 1, "tcn"),
        ("FD001", 2, "hgb"),
        ("FD001", 3, "cnn"),
        ("FD002", 1, "hgb"),
        ("FD002", 2, "cnn"),
    ]
    tcn_row = rows[0]
    assert tcn_row.phase == "phase2_deep"
    assert tcn_row.rmse_delta == -1.0
    assert tcn_row.nasa_score_delta == -20.0
    assert tcn_row.nasa_score_ratio == 0.8


def test_build_cmapss_model_comparison_summarizes_prediction_csvs(tmp_path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    prediction_csv = tmp_path / "predictions.csv"
    write_results_csv([_result("FD001", "hgb", rmse=10.0, nasa_score=100.0)], baseline_csv)
    _write_prediction_csv(
        prediction_csv,
        [
            {
                "subset": "FD001",
                "model_name": "transformer",
                "actual_rul": "100",
                "predicted_rul": "90",
            },
            {
                "subset": "FD001",
                "model_name": "transformer",
                "actual_rul": "120",
                "predicted_rul": "125",
            },
        ],
    )

    rows = build_cmapss_model_comparison(
        baseline_csv,
        prediction_csvs=(prediction_csv,),
        prediction_label="phase2_calibrated",
        prediction_model_suffixes=("nasa_shift",),
    )

    prediction_row = next(row for row in rows if row.phase == "phase2_calibrated")
    assert prediction_row.model_name == "transformer_nasa_shift"
    assert prediction_row.rmse == 7.905694150420948
    assert prediction_row.nasa_score > 0.0


def test_write_cmapss_model_comparison_outputs_csv_and_markdown(tmp_path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"
    output_csv = tmp_path / "reports" / "comparison.csv"
    output_markdown = tmp_path / "reports" / "comparison.md"
    write_results_csv([_result("FD001", "hgb", rmse=10.0, nasa_score=100.0)], baseline_csv)
    write_results_csv(
        [_result("FD001", "transformer", rmse=8.0, nasa_score=75.0)],
        candidate_csv,
    )
    rows = build_cmapss_model_comparison(baseline_csv, (candidate_csv,))

    write_cmapss_model_comparison_csv(rows, output_csv)
    write_cmapss_model_comparison_markdown(rows, output_markdown)

    csv_rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))
    markdown = output_markdown.read_text(encoding="utf-8")
    assert csv_rows[0]["model_name"] == "transformer"
    assert "# C-MAPSS Model Comparison" in markdown
    assert "`transformer`" in markdown
    assert markdown.endswith("\n")


def test_render_cmapss_model_comparison_markdown_rejects_empty_rows() -> None:
    try:
        render_cmapss_model_comparison_markdown([])
    except ValueError as error:
        assert "rows must contain at least one item" in str(error)
    else:
        raise AssertionError("expected empty comparison rows to fail")


def _result(
    subset: str,
    model_name: str,
    *,
    rmse: float,
    nasa_score: float,
) -> RegressionRunResult:
    return RegressionRunResult(
        dataset="C-MAPSS",
        subset=subset,
        model_name=model_name,
        rmse=rmse,
        nasa_score=nasa_score,
        train_rows=10,
        train_units=2,
        test_rows=4,
        test_units=2,
        test_rul_values=2,
        rul_cap=125,
        random_state=42,
        standardize=True,
    )


def _write_prediction_csv(path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["subset", "model_name", "actual_rul", "predicted_rul"],
        )
        writer.writeheader()
        writer.writerows(rows)
