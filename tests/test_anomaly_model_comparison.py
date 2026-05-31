from __future__ import annotations

import csv

from aerospace_prognostics.reports.anomaly_model_comparison import (
    build_anomaly_model_comparison,
    render_anomaly_model_comparison_markdown,
    write_anomaly_model_comparison_csv,
    write_anomaly_model_comparison_markdown,
)


def test_build_anomaly_model_comparison_ranks_by_pointwise_f1(tmp_path) -> None:
    classical_csv = tmp_path / "classical.csv"
    forecast_csv = tmp_path / "forecast.csv"
    _write_result_csv(
        classical_csv,
        [
            _row("P-1", "SMAP", "robust_zscore", f1=0.40, point_adjusted_f1=0.90),
            _row("P-1", "SMAP", "pca_reconstruction", f1=0.60, point_adjusted_f1=0.70),
        ],
    )
    _write_result_csv(
        forecast_csv,
        [_row("P-1", "SMAP", "lstm_forecast_dynamic_threshold", f1=0.55)],
    )

    rows = build_anomaly_model_comparison(
        (classical_csv, forecast_csv),
        source_labels=("classical", "lstm"),
    )

    assert [(row.rank_by_f1, row.source, row.model_name) for row in rows] == [
        (1, "classical", "pca_reconstruction"),
        (2, "lstm", "lstm_forecast_dynamic_threshold"),
        (3, "classical", "robust_zscore"),
    ]
    assert rows[0].point_adjusted_f1 == 0.70
    assert rows[0].train_rows == 5
    assert rows[0].test_rows == 6
    assert rows[0].anomaly_points == 2


def test_write_anomaly_model_comparison_outputs_csv_and_markdown(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "reports" / "comparison.csv"
    output_markdown = tmp_path / "reports" / "comparison.md"
    _write_result_csv(
        input_csv,
        [
            _row("P-1", "SMAP", "robust_zscore", f1=0.40),
            _row("P-1", "SMAP", "isolation_forest", f1=0.60),
            _row("E-1", "SMAP", "robust_zscore", f1=0.50),
        ],
    )
    rows = build_anomaly_model_comparison((input_csv,), source_labels=("classical",))

    write_anomaly_model_comparison_csv(rows, output_csv)
    write_anomaly_model_comparison_markdown(rows, output_markdown)

    csv_rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))
    markdown = output_markdown.read_text(encoding="utf-8")
    assert csv_rows[0]["model_name"] == "robust_zscore"
    assert "# Telemetry Anomaly Model Comparison" in markdown
    assert "`robust_zscore`" in markdown
    assert "## Winner Counts" in markdown
    assert "| classical | `isolation_forest` | 1 |" in markdown
    assert "| classical | `robust_zscore` | 1 |" in markdown
    assert "## Average Metrics By Source And Model" in markdown


def test_render_anomaly_model_comparison_markdown_rejects_empty_rows() -> None:
    try:
        render_anomaly_model_comparison_markdown([])
    except ValueError as error:
        assert "rows must contain at least one item" in str(error)
    else:
        raise AssertionError("expected empty comparison rows to fail")


def _write_result_csv(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(
    channel_id: str,
    spacecraft: str,
    model_name: str,
    *,
    f1: float,
    point_adjusted_f1: float = 0.50,
) -> dict[str, object]:
    return {
        "channel_id": channel_id,
        "spacecraft": spacecraft,
        "model_name": model_name,
        "train_rows": 5,
        "test_rows": 6,
        "feature_count": 2,
        "anomaly_sequences": 1,
        "anomaly_points": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": f1,
        "point_adjusted_f1": point_adjusted_f1,
        "false_alarm_rate": 0.1,
        "miss_rate": 0.5,
        "support": 2,
        "predicted_positives": 2,
    }
