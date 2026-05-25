from __future__ import annotations

import json

from aerospace_prognostics.cli import main
from tests.cmapss_fixtures import write_all_tiny_cmapss_subsets, write_tiny_cmapss_subset


def test_cmapss_summary_command_prints_dataset_shape(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)

    exit_code = main(["cmapss-summary", "--data-dir", str(tmp_path), "--subset", "FD001"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "subset=FD001" in output
    assert "train_rows=6 train_units=2" in output
    assert "test_rows=4 test_units=2" in output
    assert "test_rul_values=2" in output


def test_cmapss_baseline_command_writes_json_result(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)
    output_path = tmp_path / "result.json"

    exit_code = main(
        [
            "cmapss-baseline",
            "--data-dir",
            str(tmp_path),
            "--subset",
            "FD001",
            "--output-json",
            str(output_path),
            "--standardize",
        ]
    )

    terminal_output = capsys.readouterr().out
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "model=hist_gradient_boosting" in terminal_output
    assert "standardize=True" in terminal_output
    assert result["dataset"] == "C-MAPSS"
    assert result["subset"] == "FD001"
    assert result["train_rows"] == 6
    assert result["test_units"] == 2
    assert result["standardize"] is True


def test_cmapss_baseline_all_command_writes_result_tables(tmp_path, capsys) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    json_path = tmp_path / "nested" / "results.json"
    csv_path = tmp_path / "nested" / "results.csv"

    exit_code = main(
        [
            "cmapss-baseline-all",
            "--data-dir",
            str(tmp_path),
            "--standardize",
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "subset,model,standardize,rmse,nasa_score" in terminal_output
    assert "FD004,hist_gradient_boosting,True" in terminal_output
    assert json_path.exists()
    assert csv_path.exists()


def test_cmapss_eda_command_writes_json_report(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)
    output_path = tmp_path / "eda" / "fd001.json"

    exit_code = main(
        [
            "cmapss-eda",
            "--data-dir",
            str(tmp_path),
            "--subset",
            "FD001",
            "--output-json",
            str(output_path),
        ]
    )

    terminal_output = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "subset=FD001" in terminal_output
    assert "largest_abs_drift_sensor=sensor_1" in terminal_output
    assert report["train_rows"] == 6
