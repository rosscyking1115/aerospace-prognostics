from __future__ import annotations

from aerospace_prognostics.experiments.cmapss_baseline import (
    CMAPSS_ENGINEERED_DEFAULT_WINDOWS,
    run_all_cmapss_engineered_default_windows,
    run_all_cmapss_engineered_hist_gradient_boosting,
    run_all_cmapss_hist_gradient_boosting,
    run_cmapss_engineered_hist_gradient_boosting,
    run_cmapss_engineered_window_sweep,
    run_cmapss_hist_gradient_boosting,
)
from tests.cmapss_fixtures import write_all_tiny_cmapss_subsets, write_tiny_cmapss_subset


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


def test_run_all_cmapss_hist_gradient_boosting_returns_each_subset(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_all_cmapss_hist_gradient_boosting(
        tmp_path,
        standardize=True,
    )

    assert [result.subset for result in results] == ["FD001", "FD002", "FD003", "FD004"]
    assert all(result.standardize for result in results)


def test_run_cmapss_engineered_hist_gradient_boosting_returns_structured_result(
    tmp_path,
) -> None:
    write_tiny_cmapss_subset(tmp_path)

    result = run_cmapss_engineered_hist_gradient_boosting(
        tmp_path,
        "FD001",
        rolling_window=2,
    )

    assert result.model_name == "hist_gradient_boosting_engineered_w2"
    assert result.standardize is True
    assert result.rmse >= 0


def test_run_all_cmapss_engineered_hist_gradient_boosting_returns_each_subset(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_all_cmapss_engineered_hist_gradient_boosting(
        tmp_path,
        rolling_window=2,
    )

    assert [result.subset for result in results] == ["FD001", "FD002", "FD003", "FD004"]
    assert all("engineered_w2" in result.model_name for result in results)


def test_run_cmapss_engineered_window_sweep_returns_window_subset_grid(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_cmapss_engineered_window_sweep(
        tmp_path,
        rolling_windows=(2, 3),
    )

    assert len(results) == 8
    assert [result.model_name for result in results[:4]] == [
        "hist_gradient_boosting_engineered_w2",
        "hist_gradient_boosting_engineered_w2",
        "hist_gradient_boosting_engineered_w2",
        "hist_gradient_boosting_engineered_w2",
    ]
    assert all(result.standardize for result in results)


def test_run_all_cmapss_engineered_default_windows_uses_subset_defaults(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_all_cmapss_engineered_default_windows(tmp_path)

    assert [result.subset for result in results] == ["FD001", "FD002", "FD003", "FD004"]
    assert [result.model_name for result in results] == [
        f"hist_gradient_boosting_engineered_w{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
        for subset in ("FD001", "FD002", "FD003", "FD004")
    ]


def test_run_all_cmapss_engineered_default_windows_validates_missing_subset(
    tmp_path,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    try:
        run_all_cmapss_engineered_default_windows(
            tmp_path,
            window_by_subset={"FD001": 2},
        )
    except ValueError as exc:
        assert "missing rolling-window defaults" in str(exc)
    else:
        raise AssertionError("expected missing default windows to fail")
