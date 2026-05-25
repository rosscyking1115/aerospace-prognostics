from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerospace_prognostics.preprocessing import FeatureStandardizer
from aerospace_prognostics.sequences import make_sequence_windows


def test_feature_standardizer_uses_train_statistics_for_frames() -> None:
    train = pd.DataFrame({"sensor_1": [0.0, 2.0], "rul_capped": [1, 0]})
    test = pd.DataFrame({"sensor_1": [4.0], "rul_capped": [0]})

    standardizer = FeatureStandardizer.fit(train, feature_columns=["sensor_1"])
    transformed_train = standardizer.transform_frame(train)
    transformed_test = standardizer.transform_frame(test)

    assert transformed_train["sensor_1"].tolist() == [-1.0, 1.0]
    assert transformed_train["rul_capped"].tolist() == [1, 0]
    assert transformed_test["sensor_1"].tolist() == [3.0]


def test_feature_standardizer_preserves_feature_table_shape() -> None:
    features = pd.DataFrame({"sensor_1": [0.0, 2.0], "sensor_2": [10.0, 14.0]})
    standardizer = FeatureStandardizer.fit(features, feature_columns=["sensor_1", "sensor_2"])

    transformed = standardizer.transform_features(features)

    assert transformed.columns.tolist() == ["sensor_1", "sensor_2"]
    np.testing.assert_allclose(transformed.mean().to_numpy(), np.array([0.0, 0.0]), atol=1e-7)


def test_feature_standardizer_transforms_sequence_dataset_metadata_safely() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [1, 1, 1],
            "time_in_cycles": [1, 2, 3],
            "sensor_1": [0.0, 1.0, 2.0],
            "rul_capped": [2, 1, 0],
        }
    )
    dataset = make_sequence_windows(frame, window_size=2, feature_columns=["sensor_1"])
    standardizer = FeatureStandardizer.fit(frame, feature_columns=["sensor_1"])

    transformed = standardizer.transform_sequence_dataset(dataset)

    assert transformed.feature_columns == dataset.feature_columns
    assert transformed.unit_numbers.tolist() == dataset.unit_numbers.tolist()
    assert transformed.end_cycles.tolist() == dataset.end_cycles.tolist()
    assert transformed.targets is not dataset.targets
    assert transformed.windows.shape == dataset.windows.shape


def test_feature_standardizer_validates_columns_and_shapes() -> None:
    frame = pd.DataFrame({"sensor_1": [1.0]})
    standardizer = FeatureStandardizer.fit(frame, feature_columns=["sensor_1"])

    with pytest.raises(ValueError, match="missing columns"):
        FeatureStandardizer.fit(frame, feature_columns=["sensor_2"])

    with pytest.raises(ValueError, match="shape"):
        standardizer.transform_windows(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="feature dimension"):
        standardizer.transform_windows(np.zeros((1, 2, 2), dtype=np.float32))

