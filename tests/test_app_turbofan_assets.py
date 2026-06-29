from __future__ import annotations

import pytest

from aerospace_prognostics.app.turbofan_assets import (
    turbofan_asset_attention_reasons,
    turbofan_asset_risk_level,
    turbofan_asset_status,
)


@pytest.mark.parametrize(
    ("predicted_rul", "predicted_rul_lower", "expected"),
    [
        (20.0, None, "critical"),
        (25.0, 19.0, "critical"),
        (50.0, None, "watch"),
        (55.0, 45.0, "watch"),
        (51.0, None, "nominal"),
    ],
)
def test_turbofan_asset_risk_level_uses_prediction_floor(
    predicted_rul: float,
    predicted_rul_lower: float | None,
    expected: str,
) -> None:
    assert (
        turbofan_asset_risk_level(
            predicted_rul=predicted_rul,
            predicted_rul_lower=predicted_rul_lower,
        )
        == expected
    )


def test_turbofan_asset_attention_reasons_explain_critical_interval_crossing() -> None:
    reasons = turbofan_asset_attention_reasons(
        predicted_rul=24.0,
        predicted_rul_lower=18.0,
        predicted_rul_upper=55.0,
        risk_level="critical",
    )

    assert reasons == [
        "RUL at or below critical threshold",
        "Interval lower bound crosses critical threshold",
        "Wide RUL interval",
    ]


def test_turbofan_asset_attention_reasons_include_critical_boundary_crossing() -> None:
    assert turbofan_asset_attention_reasons(
        predicted_rul=24.0,
        predicted_rul_lower=20.0,
        predicted_rul_upper=40.0,
        risk_level="critical",
    ) == [
        "RUL at or below critical threshold",
        "Interval lower bound crosses critical threshold",
    ]


def test_turbofan_asset_attention_reasons_explain_watch_band() -> None:
    assert turbofan_asset_attention_reasons(
        predicted_rul=45.0,
        predicted_rul_lower=None,
        predicted_rul_upper=None,
        risk_level="watch",
    ) == ["RUL inside watch threshold"]


@pytest.mark.parametrize(
    ("risk_level", "expected"),
    [
        ("critical", "maintenance_review"),
        ("watch", "monitor"),
        ("nominal", "nominal"),
        ("other", "unknown"),
    ],
)
def test_turbofan_asset_status_maps_risk_to_registry_status(
    risk_level: str,
    expected: str,
) -> None:
    assert turbofan_asset_status(risk_level) == expected
