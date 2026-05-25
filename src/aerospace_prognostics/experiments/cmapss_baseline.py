"""First-pass C-MAPSS baseline experiment."""

from __future__ import annotations

from pathlib import Path

from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.evaluation import RegressionRunResult
from aerospace_prognostics.features import cycle_feature_table, last_cycle_feature_table
from aerospace_prognostics.metrics import nasa_rul_score, rmse
from aerospace_prognostics.models.baselines import hist_gradient_boosting_rul


def run_cmapss_hist_gradient_boosting(
    data_dir: str | Path,
    subset: str,
    *,
    rul_cap: int = 125,
    random_state: int = 42,
) -> RegressionRunResult:
    """Train and evaluate the first-pass C-MAPSS gradient-boosting baseline."""

    bundle = load_cmapss_subset(data_dir, subset, rul_cap=rul_cap)
    train_features, train_target = cycle_feature_table(bundle.train)
    test_features = last_cycle_feature_table(bundle.test)

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
    )

