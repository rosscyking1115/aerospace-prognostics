from __future__ import annotations

import json

from aerospace_prognostics.evaluation import RegressionRunResult


def test_regression_run_result_writes_json(tmp_path) -> None:
    result = RegressionRunResult(
        dataset="C-MAPSS",
        subset="FD001",
        model_name="toy",
        rmse=1.0,
        nasa_score=2.0,
        train_rows=3,
        train_units=1,
        test_rows=2,
        test_units=1,
        test_rul_values=1,
        rul_cap=125,
        random_state=42,
        standardize=True,
    )
    output_path = tmp_path / "result.json"

    result.write_json(output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == result.to_dict()
