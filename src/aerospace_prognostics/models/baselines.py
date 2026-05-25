"""Classical baseline model factories."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor


def hist_gradient_boosting_rul(random_state: int = 42) -> HistGradientBoostingRegressor:
    """Return a deterministic first-pass gradient-boosting RUL baseline."""

    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        l2_regularization=0.01,
        random_state=random_state,
    )

