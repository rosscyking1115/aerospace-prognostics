"""First-pass C-MAPSS baseline experiment."""

from __future__ import annotations

from pathlib import Path

from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS, load_cmapss_subset
from aerospace_prognostics.evaluation import RegressionRunResult
from aerospace_prognostics.features import (
    cycle_feature_table,
    engineered_cycle_feature_table,
    engineered_last_cycle_feature_table,
    last_cycle_feature_table,
)
from aerospace_prognostics.metrics import nasa_rul_score, rmse
from aerospace_prognostics.models.baselines import hist_gradient_boosting_rul
from aerospace_prognostics.preprocessing import FeatureStandardizer


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
