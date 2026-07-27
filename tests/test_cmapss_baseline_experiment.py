from __future__ import annotations

import pytest

from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.experiments.cmapss_baseline import (
    CMAPSS_ENGINEERED_DEFAULT_WINDOWS,
    CMAPSS_HGB_PARAM_GRID,
    CMAPSS_NAIVE_STRATEGIES,
    CMAPSS_SENSOR_FILTER_CANDIDATES,
    CMAPSS_VALIDATION_SELECTED_FEATURES,
    CMAPSS_VALIDATION_SELECTED_HGB_PARAMS,
    make_cmapss_temporal_validation_split,
    run_all_cmapss_engineered_default_windows,
    run_all_cmapss_engineered_hist_gradient_boosting,
    run_all_cmapss_hist_gradient_boosting,
    run_all_cmapss_regime_aware_engineered_default_windows,
    run_all_cmapss_validation_selected_default_windows,
    run_all_cmapss_validation_selected_hgb_policy_default_windows,
    run_cmapss_engineered_hist_gradient_boosting,
    run_cmapss_engineered_window_sweep,
    run_cmapss_hist_gradient_boosting,
    run_cmapss_naive_baseline,
    run_cmapss_regime_aware_engineered_hist_gradient_boosting,
    run_cmapss_repeated_validation_feature_comparison,
    run_cmapss_validation_feature_comparison,
    run_cmapss_validation_selected_hgb_grid,
    run_cmapss_validation_sensor_filter_comparison,
)
from tests.cmapss_fixtures import (
    write_all_tiny_cmapss_subsets,
    write_discriminating_cmapss_subset,
    write_tiny_cmapss_subset,
)


def test_run_cmapss_naive_baseline_returns_structured_result(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)

    result = run_cmapss_naive_baseline(tmp_path, "FD001")

    assert result.dataset == "C-MAPSS"
    assert result.model_name == "naive_train_median"
    assert result.test_rul_values == 2
    assert result.rmse >= 0
    assert result.nasa_score >= 0


def test_run_cmapss_naive_baseline_supports_every_strategy(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)

    for strategy in CMAPSS_NAIVE_STRATEGIES:
        result = run_cmapss_naive_baseline(tmp_path, "FD001", strategy=strategy)
        assert result.model_name == f"naive_{strategy}"
        assert result.rmse >= 0


def test_run_cmapss_naive_baseline_rejects_unknown_strategy(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)

    with pytest.raises(ValueError, match="strategy must be one of"):
        run_cmapss_naive_baseline(tmp_path, "FD001", strategy="oracle")


def test_naive_baseline_predicts_a_single_constant(tmp_path) -> None:
    """The floor must ignore the sensors entirely.

    The tiny fixture is degenerate for skill purposes (its test RUL values
    coincide with the train median, so a constant scores RMSE 0.0). It proves
    the constant-prediction *contract* only. The skill comparison lives in
    ``test_learned_baseline_beats_the_naive_floor``, which uses the
    discriminating fixture instead.
    """

    write_tiny_cmapss_subset(tmp_path)

    capped = run_cmapss_naive_baseline(tmp_path, "FD001", strategy="rul_cap")
    median_based = run_cmapss_naive_baseline(tmp_path, "FD001", strategy="train_median")

    # Same data, same model family, different constant => different error.
    assert capped.rmse != median_based.rmse
    assert capped.rul_cap == 125


def test_learned_baseline_beats_the_naive_floor(tmp_path) -> None:
    """A model that predicts nonsense must fail this test.

    This is the check the suite was missing. Every other test proves plumbing:
    that commands run, that artifacts are written, that shapes match. None of
    them would notice if the estimator lost all predictive power, because the
    tiny fixture's test RUL happens to equal its train median, so a constant
    predictor scores a perfect RMSE 0.0 on it.

    The discriminating fixture removes that coincidence: its test units are
    truncated at different points, so no single constant can be right for all
    of them, and its sensor values carry a genuine monotone degradation signal
    that a real estimator can recover. Break the model and this test goes red.
    """

    write_discriminating_cmapss_subset(tmp_path)

    naive = run_cmapss_naive_baseline(tmp_path, "FD001", strategy="train_median")
    learned = run_cmapss_hist_gradient_boosting(tmp_path, "FD001", standardize=True)

    # The floor must be a real floor: a constant cannot fit a varied test set.
    assert naive.rmse > 0.0
    # And the learned model must clear it decisively. The gate is naive/5
    # (~6.9 on this fixture) against a healthy score of ~1.9, so there is ample
    # headroom for library and platform drift while still catching an estimator
    # that has degraded badly rather than only one that has died completely.
    assert learned.rmse < naive.rmse / 5


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


def test_run_all_cmapss_validation_selected_hgb_policy_default_windows_uses_policy(
    tmp_path,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_all_cmapss_validation_selected_hgb_policy_default_windows(
        tmp_path,
        n_regimes=2,
    )

    assert CMAPSS_VALIDATION_SELECTED_HGB_PARAMS["FD002"] == "slow_regularized"
    assert [result.subset for result in results] == ["FD001", "FD002", "FD003", "FD004"]
    assert [result.model_name for result in results] == [
        "hist_gradient_boosting_regime_engineered_w10_r1_default",
        "hist_gradient_boosting_engineered_w3_slow_regularized",
        "hist_gradient_boosting_regime_engineered_w5_r1_slow_regularized",
        "hist_gradient_boosting_regime_engineered_w3_r1_default",
    ]


def test_run_cmapss_validation_sensor_filter_comparison_scores_candidates(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_cmapss_validation_sensor_filter_comparison(
        tmp_path,
        subsets=("FD001",),
        n_regimes=2,
        validation_horizon=1,
    )

    assert CMAPSS_SENSOR_FILTER_CANDIDATES == ("all_sensors", "eda_filtered")
    assert all(result.dataset == "C-MAPSS-validation-sensor-filter" for result in results)
    assert [result.subset for result in results] == ["FD001", "FD001"]
    assert [result.model_name for result in results] == [
        "hist_gradient_boosting_regime_engineered_w10_r1_default",
        "hist_gradient_boosting_regime_engineered_w10_r1_default_eda_filtered",
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


def test_run_cmapss_validation_selected_hgb_grid_returns_param_candidates(
    tmp_path,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    results = run_cmapss_validation_selected_hgb_grid(
        tmp_path,
        subsets=("FD001",),
        validation_horizon=1,
        n_regimes=2,
    )

    assert len(results) == len(CMAPSS_HGB_PARAM_GRID)
    assert all(result.dataset == "C-MAPSS-validation-hgb-grid" for result in results)
    assert all(result.subset == "FD001" for result in results)
    assert [result.model_name for result in results] == [
        "hist_gradient_boosting_regime_engineered_w10_r1_default",
        "hist_gradient_boosting_regime_engineered_w10_r1_slow_regularized",
        "hist_gradient_boosting_regime_engineered_w10_r1_shallow_fast",
    ]
