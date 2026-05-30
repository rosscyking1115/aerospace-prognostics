from __future__ import annotations

import numpy as np
import pytest

from aerospace_prognostics.anomaly.baselines import (
    CLASSICAL_ANOMALY_BASELINE_METHODS,
    fit_robust_zscore_model,
    robust_zscore_scores,
    run_classical_anomaly_baselines,
    run_robust_zscore_baseline,
)


def test_robust_zscore_baseline_flags_large_deviations() -> None:
    train_values = np.array(
        [
            [0.0, 10.0],
            [0.1, 10.1],
            [-0.1, 9.9],
            [0.0, 10.0],
        ]
    )
    test_values = np.array(
        [
            [0.0, 10.0],
            [8.0, 10.0],
            [0.0, 25.0],
            [0.1, 10.1],
        ]
    )

    result = run_robust_zscore_baseline(
        train_values,
        test_values,
        labels=[0, 1, 1, 0],
        feature_names=["bus_voltage", "thermal_zone"],
        threshold=3.5,
    )

    assert result.model.feature_names == ("bus_voltage", "thermal_zone")
    assert result.predictions == (0, 1, 1, 0)
    assert result.metrics.f1 == 1.0
    assert result.point_adjusted_metrics.f1 == 1.0
    assert result.scores[1] > result.model.threshold
    assert result.to_dict()["model"]["feature_names"] == ("bus_voltage", "thermal_zone")


def test_robust_zscore_model_falls_back_for_constant_features() -> None:
    model = fit_robust_zscore_model(
        [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
        feature_names=["constant_a", "constant_b"],
    )

    assert model.scales == (1.0, 1.0)
    assert robust_zscore_scores(model, [[1.0, 7.0]]).tolist() == [2.0]


def test_robust_zscore_model_rejects_feature_name_mismatch() -> None:
    with pytest.raises(ValueError, match="feature_names length"):
        fit_robust_zscore_model([[1.0, 2.0]], feature_names=["only_one"])


def test_classical_anomaly_baselines_share_result_contract() -> None:
    train_values = np.array(
        [
            [-0.2, -0.2],
            [-0.1, -0.1],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
        ]
    )
    test_values = np.array(
        [
            [0.0, 0.0],
            [8.0, -8.0],
            [0.1, 0.1],
            [-7.0, 7.0],
        ]
    )

    results = run_classical_anomaly_baselines(
        train_values,
        test_values,
        labels=[0, 1, 0, 1],
        feature_names=["bus_voltage", "thermal_zone"],
        pca_components=1,
        pca_threshold_quantile=0.95,
        isolation_contamination=0.2,
    )

    assert [result.model_name for result in results] == list(CLASSICAL_ANOMALY_BASELINE_METHODS)
    assert all(len(result.scores) == 4 for result in results)
    assert all(len(result.predictions) == 4 for result in results)
    assert all(result.metrics.support == 2 for result in results)
    assert results[1].model_config["n_components"] == 1
    assert results[1].predictions[1] == 1
    assert results[2].model_config["random_state"] == 42


def test_classical_anomaly_baselines_reject_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown anomaly baseline methods"):
        run_classical_anomaly_baselines(
            [[0.0, 0.0]],
            [[1.0, 1.0]],
            labels=[1],
            feature_names=["a", "b"],
            methods=["not_a_model"],
        )
