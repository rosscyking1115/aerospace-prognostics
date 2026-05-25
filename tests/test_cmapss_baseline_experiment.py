from __future__ import annotations

from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.experiments.cmapss_baseline import (
    CMAPSS_ENGINEERED_DEFAULT_WINDOWS,
    CMAPSS_VALIDATION_SELECTED_FEATURES,
    make_cmapss_temporal_validation_split,
    run_all_cmapss_engineered_default_windows,
    run_all_cmapss_engineered_hist_gradient_boosting,
    run_all_cmapss_hist_gradient_boosting,
    run_all_cmapss_regime_aware_engineered_default_windows,
    run_all_cmapss_validation_selected_default_windows,
    run_cmapss_engineered_hist_gradient_boosting,
    run_cmapss_engineered_window_sweep,
    run_cmapss_hist_gradient_boosting,
    run_cmapss_regime_aware_engineered_hist_gradient_boosting,
    run_cmapss_repeated_validation_feature_comparison,
    run_cmapss_validation_feature_comparison,
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


def test_run_cmapss_regime_aware_engineered_hist_gradient_boosting_returns_result(
    tmp_path,
) -> None:
    write_tiny_cmapss_subset(tmp_path)

    result = run_cmapss_regime_aware_engineered_hist_gradient_boosting(
        tmp_path,
        "FD001",
        rolling_window=2,
        n_regimes=2,
    )

    assert result.model_name == "hist_gradient_boosting_regime_engineered_w2_r1"
    assert result.standardize is True
    assert result.rmse >= 0


def test_run_all_cmapss_regime_aware_engineered_default_windows_returns_each_subset(
    tmp_path,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_all_cmapss_regime_aware_engineered_default_windows(tmp_path)

    assert [result.subset for result in results] == ["FD001", "FD002", "FD003", "FD004"]
    assert all("regime_engineered" in result.model_name for result in results)


def test_run_all_cmapss_validation_selected_default_windows_uses_policy(
    tmp_path,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_all_cmapss_validation_selected_default_windows(
        tmp_path,
        n_regimes=2,
    )

    assert CMAPSS_VALIDATION_SELECTED_FEATURES["FD002"] == "engineered"
    assert [result.subset for result in results] == ["FD001", "FD002", "FD003", "FD004"]
    assert [result.model_name for result in results] == [
        "hist_gradient_boosting_regime_engineered_w10_r1",
        "hist_gradient_boosting_engineered_w3",
        "hist_gradient_boosting_regime_engineered_w5_r1",
        "hist_gradient_boosting_regime_engineered_w3_r1",
    ]


def test_make_cmapss_temporal_validation_split_holds_out_truncated_units(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    bundle = load_cmapss_subset(tmp_path, "FD001")

    split = make_cmapss_temporal_validation_split(
        bundle.train,
        validation_fraction=0.5,
        validation_horizon=1,
        random_state=1,
    )

    assert split.train["unit_number"].nunique() == 1
    assert split.validation["unit_number"].nunique() == 1
    assert split.validation["time_in_cycles"].max() == 2
    assert split.validation_rul.tolist() == [1.0]


def test_run_cmapss_validation_feature_comparison_returns_two_candidates_per_subset(
    tmp_path,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_cmapss_validation_feature_comparison(
        tmp_path,
        subsets=("FD001",),
        validation_horizon=1,
        n_regimes=2,
    )

    assert [result.dataset for result in results] == [
        "C-MAPSS-validation",
        "C-MAPSS-validation",
    ]
    assert [result.subset for result in results] == ["FD001", "FD001"]
    assert "regime_engineered" not in results[0].model_name
    assert "regime_engineered" in results[1].model_name


def test_run_cmapss_repeated_validation_feature_comparison_aggregates_candidates(
    tmp_path,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_cmapss_repeated_validation_feature_comparison(
        tmp_path,
        subsets=("FD001",),
        random_states=(1, 2),
        validation_horizons=(1,),
        n_regimes=2,
    )

    assert [result.subset for result in results] == ["FD001", "FD001"]
    assert all(result.dataset == "C-MAPSS-validation-aggregate" for result in results)
    assert all(result.runs == 2 for result in results)
    assert sum(result.wins_by_nasa for result in results) == 2
