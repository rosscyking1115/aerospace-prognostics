from __future__ import annotations

from aerospace_prognostics.experiments.cmapss_baseline import run_cmapss_hist_gradient_boosting
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_run_cmapss_hist_gradient_boosting_returns_structured_result(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)

    result = run_cmapss_hist_gradient_boosting(
        tmp_path,
        "FD001",
        random_state=11,
        standardize=True,
    )

    assert result.dataset == "C-MAPSS"
    assert result.model_name == "hist_gradient_boosting"
    assert result.random_state == 11
    assert result.standardize is True
    assert result.train_units == 2
    assert result.test_rul_values == 2
    assert result.rmse >= 0
    assert result.nasa_score >= 0
