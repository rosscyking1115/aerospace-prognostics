from __future__ import annotations

from math import exp, isclose, sqrt

import pytest

from aerospace_prognostics.metrics import nasa_rul_score, piecewise_rul, rmse


def test_rmse_matches_definition() -> None:
    assert isclose(rmse([10, 20, 30], [10, 18, 34]), sqrt((0**2 + 2**2 + 4**2) / 3))


def test_nasa_score_penalises_late_predictions_more_than_early_predictions() -> None:
    actual = [100]
    early = nasa_rul_score(actual, [90])
    late = nasa_rul_score(actual, [110])

    assert isclose(early, exp(10 / 13) - 1)
    assert isclose(late, exp(10 / 10) - 1)
    assert late > early


def test_piecewise_rul_caps_values() -> None:
    assert piecewise_rul([10, 125, 150], cap=125) == [10, 125, 125]


def test_metric_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="same length"):
        rmse([1, 2], [1])

    with pytest.raises(ValueError, match="at least one"):
        nasa_rul_score([], [])

