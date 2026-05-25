from __future__ import annotations

import pandas as pd

from aerospace_prognostics.models.baselines import hist_gradient_boosting_rul


def test_hist_gradient_boosting_rul_can_fit_toy_data() -> None:
    features = pd.DataFrame({"sensor_1": [0.0, 1.0, 2.0, 3.0], "sensor_2": [3.0, 2.0, 1.0, 0.0]})
    target = pd.Series([30.0, 20.0, 10.0, 0.0])

    model = hist_gradient_boosting_rul(random_state=7)
    model.fit(features, target)

    predictions = model.predict(features)

    assert predictions.shape == (4,)
