"""First-pass C-MAPSS baseline experiment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS, load_cmapss_subset
from aerospace_prognostics.evaluation import RegressionRunResult
from aerospace_prognostics.features import (
    OperatingRegimeFeatureTransformer,
    cycle_feature_table,
    engineered_cycle_feature_table,
    engineered_last_cycle_feature_table,
    last_cycle_feature_table,
)
from aerospace_prognostics.metrics import nasa_rul_score, rmse
from aerospace_prognostics.models.baselines import hist_gradient_boosting_rul
from aerospace_prognostics.preprocessing import FeatureStandardizer

if TYPE_CHECKING:
    import pandas as pd

CMAPSS_ENGINEERED_DEFAULT_WINDOWS = {
    "FD001": 10,
    "FD002": 3,
    "FD003": 5,
    "FD004": 3,
}

CMAPSS_VALIDATION_SELECTED_FEATURES = {
    "FD001": "regime_engineered",
    "FD002": "engineered",
    "FD003": "regime_engineered",
    "FD004": "regime_engineered",
}

CMAPSS_HGB_PARAM_GRID = (
    {
        "label": "default",
        "learning_rate": 0.05,
        "max_iter": 200,
        "l2_regularization": 0.01,
    },
    {
        "label": "slow_regularized",
        "learning_rate": 0.03,
        "max_iter": 350,
        "l2_regularization": 0.05,
    },
    {
        "label": "shallow_fast",
        "learning_rate": 0.08,
        "max_iter": 160,
        "l2_regularization": 0.02,
        "max_leaf_nodes": 15,
    },
)


@dataclass(frozen=True)
class CmapssTemporalValidationSplit:
    """Unit-held-out C-MAPSS split with truncated validation histories."""

    train: pd.DataFrame
    validation: pd.DataFrame
    validation_rul: pd.Series
    validation_units: tuple[int, ...]
    validation_horizon: int


@dataclass(frozen=True)
class CmapssValidationAggregateResult:
    """Aggregate repeated temporal-validation metrics for one candidate."""

    dataset: str
    subset: str
    model_name: str
    runs: int
    wins_by_nasa: int
    mean_rmse: float
    mean_nasa_score: float
    standardize: bool

    def to_dict(self) -> dict[str, bool | float | int | str]:
        """Return a JSON-serialisable dictionary."""

        return {
            "dataset": self.dataset,
            "subset": self.subset,
            "model_name": self.model_name,
            "runs": self.runs,
            "wins_by_nasa": self.wins_by_nasa,
            "mean_rmse": self.mean_rmse,
            "mean_nasa_score": self.mean_nasa_score,
            "standardize": self.standardize,
        }


def make_cmapss_temporal_validation_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    validation_horizon: int = 30,
    random_state: int = 42,
) -> CmapssTemporalValidationSplit:
    """Hold out units and truncate their histories for validation scoring."""

    import pandas as pd

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if validation_horizon < 1:
        raise ValueError("validation_horizon must be at least 1")

    units = sorted(int(unit) for unit in frame["unit_number"].unique())
    if len(units) < 2:
        raise ValueError("at least two units are required for validation")

    shuffled_units = units.copy()
    random.Random(random_state).shuffle(shuffled_units)
    validation_count = min(len(units) - 1, max(1, round(len(units) * validation_fraction)))
    validation_units = tuple(sorted(shuffled_units[:validation_count]))
    train = frame.loc[~frame["unit_number"].isin(validation_units)].copy()

    validation_frames = []
    validation_rul_values = []
    for unit in validation_units:
        unit_frame = frame.loc[frame["unit_number"] == unit].sort_values("time_in_cycles")
        max_cycle = int(unit_frame["time_in_cycles"].max())
        holdout_cycles = min(validation_horizon, max_cycle - 1)
        if holdout_cycles < 1:
            raise ValueError("validation units must contain at least two cycles")
        cutoff_cycle = max_cycle - holdout_cycles
        observed = unit_frame.loc[unit_frame["time_in_cycles"] <= cutoff_cycle].copy()
        validation_frames.append(observed)
        cutoff_row = unit_frame.loc[unit_frame["time_in_cycles"] == cutoff_cycle].iloc[-1]
        validation_rul_values.append(float(cutoff_row["rul"]))

    return CmapssTemporalValidationSplit(
        train=train,
        validation=pd.concat(validation_frames, axis=0),
        validation_rul=pd.Series(validation_rul_values, name="rul"),
        validation_units=validation_units,
        validation_horizon=validation_horizon,
    )


def run_cmapss_hist_gradient_boosting(
    data_dir: str | Path,
    subset: str,
    *,
    rul_cap: int = 125,
    random_state: int = 42,
    standardize: bool = False,
) -> RegressionRunResult:
    """Train and evaluate the first-pass C-MAPSS gradient-boosting baseline."""

    bundle = load_cmapss_subset(data_dir, subset, rul_cap=rul_cap)
    train_frame = bundle.train
    test_frame = bundle.test
    if standardize:
        standardizer = FeatureStandardizer.fit(train_frame)
        train_frame = standardizer.transform_frame(train_frame)
        test_frame = standardizer.transform_frame(test_frame)

    train_features, train_target = cycle_feature_table(train_frame)
    test_features = last_cycle_feature_table(test_frame)

    model = hist_gradient_boosting_rul(random_state=random_state)
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)

    return RegressionRunResult(
        dataset="C-MAPSS",
        subset=bundle.subset,
        model_name="hist_gradient_boosting",
        rmse=rmse(bundle.test_rul, predictions),
        nasa_score=nasa_rul_score(bundle.test_rul, predictions),
        train_rows=len(bundle.train),
        train_units=bundle.train["unit_number"].nunique(),
        test_rows=len(bundle.test),
        test_units=bundle.test["unit_number"].nunique(),
        test_rul_values=len(bundle.test_rul),
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
    )


def run_cmapss_engineered_hist_gradient_boosting(
    data_dir: str | Path,
    subset: str,
    *,
    rul_cap: int = 125,
    random_state: int = 42,
    rolling_window: int = 5,
    standardize: bool = True,
) -> RegressionRunResult:
    """Train and evaluate a feature-engineered C-MAPSS gradient-boosting baseline."""

    bundle = load_cmapss_subset(data_dir, subset, rul_cap=rul_cap)
    train_features, train_target = engineered_cycle_feature_table(
        bundle.train,
        rolling_window=rolling_window,
    )
    test_features = engineered_last_cycle_feature_table(
        bundle.test,
        rolling_window=rolling_window,
    )

    if standardize:
        standardizer = FeatureStandardizer.fit(
            train_features,
            feature_columns=list(train_features.columns),
        )
        train_features = standardizer.transform_features(train_features)
        test_features = standardizer.transform_features(test_features)

    model = hist_gradient_boosting_rul(random_state=random_state)
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)

    return RegressionRunResult(
        dataset="C-MAPSS",
        subset=bundle.subset,
        model_name=f"hist_gradient_boosting_engineered_w{rolling_window}",
        rmse=rmse(bundle.test_rul, predictions),
        nasa_score=nasa_rul_score(bundle.test_rul, predictions),
        train_rows=len(bundle.train),
        train_units=bundle.train["unit_number"].nunique(),
        test_rows=len(bundle.test),
        test_units=bundle.test["unit_number"].nunique(),
        test_rul_values=len(bundle.test_rul),
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
    )


def run_all_cmapss_hist_gradient_boosting(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    rul_cap: int = 125,
    random_state: int = 42,
    standardize: bool = False,
) -> list[RegressionRunResult]:
    """Train and evaluate the baseline for each requested C-MAPSS subset."""

    return [
        run_cmapss_hist_gradient_boosting(
            data_dir,
            subset,
            rul_cap=rul_cap,
            random_state=random_state,
            standardize=standardize,
        )
        for subset in subsets
    ]


def run_all_cmapss_engineered_hist_gradient_boosting(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    rul_cap: int = 125,
    random_state: int = 42,
    rolling_window: int = 5,
    standardize: bool = True,
) -> list[RegressionRunResult]:
    """Train and evaluate the engineered baseline for each requested C-MAPSS subset."""

    return [
        run_cmapss_engineered_hist_gradient_boosting(
            data_dir,
            subset,
            rul_cap=rul_cap,
            random_state=random_state,
            rolling_window=rolling_window,
            standardize=standardize,
        )
        for subset in subsets
    ]


def run_cmapss_engineered_window_sweep(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    rolling_windows: tuple[int, ...] = (3, 5, 10),
    rul_cap: int = 125,
    random_state: int = 42,
    standardize: bool = True,
) -> list[RegressionRunResult]:
    """Train the engineered baseline across rolling-window sizes."""

    return [
        result
        for rolling_window in rolling_windows
        for result in run_all_cmapss_engineered_hist_gradient_boosting(
            data_dir,
            subsets=subsets,
            rul_cap=rul_cap,
            random_state=random_state,
            rolling_window=rolling_window,
            standardize=standardize,
        )
    ]


def run_all_cmapss_engineered_default_windows(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    window_by_subset: dict[str, int] | None = None,
    rul_cap: int = 125,
    random_state: int = 42,
    standardize: bool = True,
) -> list[RegressionRunResult]:
    """Train the engineered baseline with per-subset rolling-window defaults."""

    windows = window_by_subset or CMAPSS_ENGINEERED_DEFAULT_WINDOWS
    missing = [subset for subset in subsets if subset not in windows]
    if missing:
        raise ValueError(f"missing rolling-window defaults for subsets: {missing}")

    return [
        run_cmapss_engineered_hist_gradient_boosting(
            data_dir,
            subset,
            rul_cap=rul_cap,
            random_state=random_state,
            rolling_window=windows[subset],
            standardize=standardize,
        )
        for subset in subsets
    ]


def run_cmapss_regime_aware_engineered_hist_gradient_boosting(
    data_dir: str | Path,
    subset: str,
    *,
    rul_cap: int = 125,
    random_state: int = 42,
    rolling_window: int = 5,
    n_regimes: int = 6,
    standardize: bool = True,
) -> RegressionRunResult:
    """Train a regime-aware engineered C-MAPSS gradient-boosting baseline."""

    bundle = load_cmapss_subset(data_dir, subset, rul_cap=rul_cap)
    transformer = OperatingRegimeFeatureTransformer.fit(
        bundle.train,
        n_regimes=n_regimes,
        random_state=random_state,
    )
    train_features = transformer.transform_engineered_frame(
        bundle.train,
        rolling_window=rolling_window,
    )
    train_target = bundle.train.loc[train_features.index, "rul_capped"].copy()
    test_features = transformer.transform_engineered_last_cycle_frame(
        bundle.test,
        rolling_window=rolling_window,
    )

    if standardize:
        standardizer = FeatureStandardizer.fit(
            train_features,
            feature_columns=list(train_features.columns),
        )
        train_features = standardizer.transform_features(train_features)
        test_features = standardizer.transform_features(test_features)

    model = hist_gradient_boosting_rul(random_state=random_state)
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)

    return RegressionRunResult(
        dataset="C-MAPSS",
        subset=bundle.subset,
        model_name=(
            f"hist_gradient_boosting_regime_engineered_w{rolling_window}"
            f"_r{transformer.n_regimes}"
        ),
        rmse=rmse(bundle.test_rul, predictions),
        nasa_score=nasa_rul_score(bundle.test_rul, predictions),
        train_rows=len(bundle.train),
        train_units=bundle.train["unit_number"].nunique(),
        test_rows=len(bundle.test),
        test_units=bundle.test["unit_number"].nunique(),
        test_rul_values=len(bundle.test_rul),
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
    )


def run_all_cmapss_regime_aware_engineered_default_windows(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    window_by_subset: dict[str, int] | None = None,
    rul_cap: int = 125,
    random_state: int = 42,
    n_regimes: int = 6,
    standardize: bool = True,
) -> list[RegressionRunResult]:
    """Train regime-aware engineered baselines with per-subset window defaults."""

    windows = window_by_subset or CMAPSS_ENGINEERED_DEFAULT_WINDOWS
    missing = [subset for subset in subsets if subset not in windows]
    if missing:
        raise ValueError(f"missing rolling-window defaults for subsets: {missing}")

    return [
        run_cmapss_regime_aware_engineered_hist_gradient_boosting(
            data_dir,
            subset,
            rul_cap=rul_cap,
            random_state=random_state,
            rolling_window=windows[subset],
            n_regimes=n_regimes,
            standardize=standardize,
        )
        for subset in subsets
    ]


def run_all_cmapss_validation_selected_default_windows(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    window_by_subset: dict[str, int] | None = None,
    feature_policy_by_subset: dict[str, str] | None = None,
    rul_cap: int = 125,
    random_state: int = 42,
    n_regimes: int = 6,
    standardize: bool = True,
) -> list[RegressionRunResult]:
    """Train official-test baselines using the repeated-validation feature policy."""

    windows = window_by_subset or CMAPSS_ENGINEERED_DEFAULT_WINDOWS
    feature_policy = feature_policy_by_subset or CMAPSS_VALIDATION_SELECTED_FEATURES
    missing_windows = [subset for subset in subsets if subset not in windows]
    if missing_windows:
        raise ValueError(f"missing rolling-window defaults for subsets: {missing_windows}")
    missing_policy = [subset for subset in subsets if subset not in feature_policy]
    if missing_policy:
        raise ValueError(f"missing feature-policy defaults for subsets: {missing_policy}")

    results = []
    for subset in subsets:
        feature_candidate = feature_policy[subset]
        if feature_candidate == "engineered":
            results.append(
                run_cmapss_engineered_hist_gradient_boosting(
                    data_dir,
                    subset,
                    rul_cap=rul_cap,
                    random_state=random_state,
                    rolling_window=windows[subset],
                    standardize=standardize,
                )
            )
        elif feature_candidate == "regime_engineered":
            results.append(
                run_cmapss_regime_aware_engineered_hist_gradient_boosting(
                    data_dir,
                    subset,
                    rul_cap=rul_cap,
                    random_state=random_state,
                    rolling_window=windows[subset],
                    n_regimes=n_regimes,
                    standardize=standardize,
                )
            )
        else:
            raise ValueError(
                "feature policy values must be 'engineered' or 'regime_engineered'"
            )
    return results


def run_cmapss_engineered_validation_hist_gradient_boosting(
    data_dir: str | Path,
    subset: str,
    *,
    rul_cap: int = 125,
    random_state: int = 42,
    rolling_window: int = 5,
    validation_fraction: float = 0.2,
    validation_horizon: int = 30,
    standardize: bool = True,
) -> RegressionRunResult:
    """Score the engineered baseline on a unit-held-out temporal validation split."""

    bundle = load_cmapss_subset(data_dir, subset, rul_cap=rul_cap)
    split = make_cmapss_temporal_validation_split(
        bundle.train,
        validation_fraction=validation_fraction,
        validation_horizon=validation_horizon,
        random_state=random_state,
    )
    train_features, train_target = engineered_cycle_feature_table(
        split.train,
        rolling_window=rolling_window,
    )
    validation_features = engineered_last_cycle_feature_table(
        split.validation,
        rolling_window=rolling_window,
    )

    if standardize:
        standardizer = FeatureStandardizer.fit(
            train_features,
            feature_columns=list(train_features.columns),
        )
        train_features = standardizer.transform_features(train_features)
        validation_features = standardizer.transform_features(validation_features)

    model = hist_gradient_boosting_rul(random_state=random_state)
    model.fit(train_features, train_target)
    predictions = model.predict(validation_features)

    return RegressionRunResult(
        dataset="C-MAPSS-validation",
        subset=bundle.subset,
        model_name=f"hist_gradient_boosting_engineered_w{rolling_window}",
        rmse=rmse(split.validation_rul, predictions),
        nasa_score=nasa_rul_score(split.validation_rul, predictions),
        train_rows=len(split.train),
        train_units=split.train["unit_number"].nunique(),
        test_rows=len(split.validation),
        test_units=len(split.validation_units),
        test_rul_values=len(split.validation_rul),
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
    )


def run_cmapss_regime_aware_engineered_validation_hist_gradient_boosting(
    data_dir: str | Path,
    subset: str,
    *,
    rul_cap: int = 125,
    random_state: int = 42,
    rolling_window: int = 5,
    n_regimes: int = 6,
    validation_fraction: float = 0.2,
    validation_horizon: int = 30,
    standardize: bool = True,
) -> RegressionRunResult:
    """Score the regime-aware baseline on a unit-held-out temporal validation split."""

    bundle = load_cmapss_subset(data_dir, subset, rul_cap=rul_cap)
    split = make_cmapss_temporal_validation_split(
        bundle.train,
        validation_fraction=validation_fraction,
        validation_horizon=validation_horizon,
        random_state=random_state,
    )
    transformer = OperatingRegimeFeatureTransformer.fit(
        split.train,
        n_regimes=n_regimes,
        random_state=random_state,
    )
    train_features = transformer.transform_engineered_frame(
        split.train,
        rolling_window=rolling_window,
    )
    train_target = split.train.loc[train_features.index, "rul_capped"].copy()
    validation_features = transformer.transform_engineered_last_cycle_frame(
        split.validation,
        rolling_window=rolling_window,
    )

    if standardize:
        standardizer = FeatureStandardizer.fit(
            train_features,
            feature_columns=list(train_features.columns),
        )
        train_features = standardizer.transform_features(train_features)
        validation_features = standardizer.transform_features(validation_features)

    model = hist_gradient_boosting_rul(random_state=random_state)
    model.fit(train_features, train_target)
    predictions = model.predict(validation_features)

    return RegressionRunResult(
        dataset="C-MAPSS-validation",
        subset=bundle.subset,
        model_name=(
            f"hist_gradient_boosting_regime_engineered_w{rolling_window}"
            f"_r{transformer.n_regimes}"
        ),
        rmse=rmse(split.validation_rul, predictions),
        nasa_score=nasa_rul_score(split.validation_rul, predictions),
        train_rows=len(split.train),
        train_units=split.train["unit_number"].nunique(),
        test_rows=len(split.validation),
        test_units=len(split.validation_units),
        test_rul_values=len(split.validation_rul),
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
    )


def run_cmapss_validation_feature_comparison(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    window_by_subset: dict[str, int] | None = None,
    rul_cap: int = 125,
    random_state: int = 42,
    n_regimes: int = 6,
    validation_fraction: float = 0.2,
    validation_horizon: int = 30,
    standardize: bool = True,
) -> list[RegressionRunResult]:
    """Compare engineered and regime-aware candidates on temporal validation splits."""

    windows = window_by_subset or CMAPSS_ENGINEERED_DEFAULT_WINDOWS
    missing = [subset for subset in subsets if subset not in windows]
    if missing:
        raise ValueError(f"missing rolling-window defaults for subsets: {missing}")

    results = []
    for subset in subsets:
        results.append(
            run_cmapss_engineered_validation_hist_gradient_boosting(
                data_dir,
                subset,
                rul_cap=rul_cap,
                random_state=random_state,
                rolling_window=windows[subset],
                validation_fraction=validation_fraction,
                validation_horizon=validation_horizon,
                standardize=standardize,
            )
        )
        results.append(
            run_cmapss_regime_aware_engineered_validation_hist_gradient_boosting(
                data_dir,
                subset,
                rul_cap=rul_cap,
                random_state=random_state,
                rolling_window=windows[subset],
                n_regimes=n_regimes,
                validation_fraction=validation_fraction,
                validation_horizon=validation_horizon,
                standardize=standardize,
            )
        )
    return results


def run_cmapss_repeated_validation_feature_comparison(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    window_by_subset: dict[str, int] | None = None,
    rul_cap: int = 125,
    random_states: tuple[int, ...] = (11, 42),
    n_regimes: int = 6,
    validation_fraction: float = 0.2,
    validation_horizons: tuple[int, ...] = (20, 30),
    standardize: bool = True,
) -> list[CmapssValidationAggregateResult]:
    """Aggregate feature-candidate validation across seeds and truncation horizons."""

    if not random_states:
        raise ValueError("random_states must contain at least one seed")
    if not validation_horizons:
        raise ValueError("validation_horizons must contain at least one horizon")

    grouped_results: dict[tuple[str, str], list[RegressionRunResult]] = {}
    wins_by_candidate: dict[tuple[str, str], int] = {}

    for random_state in random_states:
        for validation_horizon in validation_horizons:
            comparison_results = run_cmapss_validation_feature_comparison(
                data_dir,
                subsets=subsets,
                window_by_subset=window_by_subset,
                rul_cap=rul_cap,
                random_state=random_state,
                n_regimes=n_regimes,
                validation_fraction=validation_fraction,
                validation_horizon=validation_horizon,
                standardize=standardize,
            )
            for result in comparison_results:
                key = (result.subset, result.model_name)
                grouped_results.setdefault(key, []).append(result)
            for subset in subsets:
                subset_results = [
                    result for result in comparison_results if result.subset == subset
                ]
                best_result = min(subset_results, key=lambda result: result.nasa_score)
                wins_by_candidate[(best_result.subset, best_result.model_name)] = (
                    wins_by_candidate.get((best_result.subset, best_result.model_name), 0) + 1
                )

    return [
        CmapssValidationAggregateResult(
            dataset="C-MAPSS-validation-aggregate",
            subset=subset,
            model_name=model_name,
            runs=len(results),
            wins_by_nasa=wins_by_candidate.get((subset, model_name), 0),
            mean_rmse=mean(result.rmse for result in results),
            mean_nasa_score=mean(result.nasa_score for result in results),
            standardize=standardize,
        )
        for (subset, model_name), results in sorted(grouped_results.items())
    ]


def run_cmapss_validation_selected_hgb_grid(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    window_by_subset: dict[str, int] | None = None,
    feature_policy_by_subset: dict[str, str] | None = None,
    param_grid: tuple[dict[str, float | int | str], ...] = CMAPSS_HGB_PARAM_GRID,
    rul_cap: int = 125,
    random_state: int = 42,
    n_regimes: int = 6,
    validation_fraction: float = 0.2,
    validation_horizon: int = 30,
    standardize: bool = True,
) -> list[RegressionRunResult]:
    """Score HGB parameter candidates for the validation-selected feature policy."""

    if not param_grid:
        raise ValueError("param_grid must contain at least one candidate")

    windows = window_by_subset or CMAPSS_ENGINEERED_DEFAULT_WINDOWS
    feature_policy = feature_policy_by_subset or CMAPSS_VALIDATION_SELECTED_FEATURES
    missing_windows = [subset for subset in subsets if subset not in windows]
    if missing_windows:
        raise ValueError(f"missing rolling-window defaults for subsets: {missing_windows}")
    missing_policy = [subset for subset in subsets if subset not in feature_policy]
    if missing_policy:
        raise ValueError(f"missing feature-policy defaults for subsets: {missing_policy}")

    return [
        _run_cmapss_validation_selected_hgb_candidate(
            data_dir,
            subset,
            feature_candidate=feature_policy[subset],
            rolling_window=windows[subset],
            hgb_params=params,
            rul_cap=rul_cap,
            random_state=random_state,
            n_regimes=n_regimes,
            validation_fraction=validation_fraction,
            validation_horizon=validation_horizon,
            standardize=standardize,
        )
        for subset in subsets
        for params in param_grid
    ]


def _run_cmapss_validation_selected_hgb_candidate(
    data_dir: str | Path,
    subset: str,
    *,
    feature_candidate: str,
    rolling_window: int,
    hgb_params: dict[str, float | int | str],
    rul_cap: int,
    random_state: int,
    n_regimes: int,
    validation_fraction: float,
    validation_horizon: int,
    standardize: bool,
) -> RegressionRunResult:
    bundle = load_cmapss_subset(data_dir, subset, rul_cap=rul_cap)
    split = make_cmapss_temporal_validation_split(
        bundle.train,
        validation_fraction=validation_fraction,
        validation_horizon=validation_horizon,
        random_state=random_state,
    )

    if feature_candidate == "engineered":
        train_features, train_target = engineered_cycle_feature_table(
            split.train,
            rolling_window=rolling_window,
        )
        validation_features = engineered_last_cycle_feature_table(
            split.validation,
            rolling_window=rolling_window,
        )
        model_prefix = f"hist_gradient_boosting_engineered_w{rolling_window}"
    elif feature_candidate == "regime_engineered":
        transformer = OperatingRegimeFeatureTransformer.fit(
            split.train,
            n_regimes=n_regimes,
            random_state=random_state,
        )
        train_features = transformer.transform_engineered_frame(
            split.train,
            rolling_window=rolling_window,
        )
        train_target = split.train.loc[train_features.index, "rul_capped"].copy()
        validation_features = transformer.transform_engineered_last_cycle_frame(
            split.validation,
            rolling_window=rolling_window,
        )
        model_prefix = (
            f"hist_gradient_boosting_regime_engineered_w{rolling_window}"
            f"_r{transformer.n_regimes}"
        )
    else:
        raise ValueError("feature_candidate must be 'engineered' or 'regime_engineered'")

    if standardize:
        standardizer = FeatureStandardizer.fit(
            train_features,
            feature_columns=list(train_features.columns),
        )
        train_features = standardizer.transform_features(train_features)
        validation_features = standardizer.transform_features(validation_features)

    params = {key: value for key, value in hgb_params.items() if key != "label"}
    model = hist_gradient_boosting_rul(random_state=random_state, **params)
    model.fit(train_features, train_target)
    predictions = model.predict(validation_features)
    label = hgb_params.get("label", "candidate")

    return RegressionRunResult(
        dataset="C-MAPSS-validation-hgb-grid",
        subset=bundle.subset,
        model_name=f"{model_prefix}_{label}",
        rmse=rmse(split.validation_rul, predictions),
        nasa_score=nasa_rul_score(split.validation_rul, predictions),
        train_rows=len(split.train),
        train_units=split.train["unit_number"].nunique(),
        test_rows=len(split.validation),
        test_units=len(split.validation_units),
        test_rul_values=len(split.validation_rul),
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
    )
