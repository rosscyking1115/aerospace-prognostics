from __future__ import annotations

import json

from aerospace_prognostics.cli import main
from tests.cmapss_fixtures import write_tiny_cmapss_subset


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
        ]
    )

    terminal_output = capsys.readouterr().out
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "model=hist_gradient_boosting" in terminal_output
    assert result["dataset"] == "C-MAPSS"
    assert result["subset"] == "FD001"
    assert result["train_rows"] == 6
    assert result["test_units"] == 2
