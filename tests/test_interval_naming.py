"""No served surface may call the interval quantile level a confidence.

The band the serving artifact returns is the 0.90 quantile of the model's
absolute residuals on its own training rows. It is in-sample, and no coverage
guarantee attaches to it. Until 2026-08-09 the field was called
`interval_confidence`, which states the one thing the number does not support --
and a field name travels with the data into every consumer, so six documents
disclosing the caveat could not undo it.

The rename is only worth as much as its enforcement. These tests fail if the
word reappears on the prediction contract, the database schema, the API
response schema, or the fitted calibration payload, so a future edit that
reintroduces it turns CI red instead of quietly shipping a false name.

Scope: the *served* surface. `reports/cmapss_phase3.py` keeps a `confidence`
parameter for the research audit, which is a report input rather than a data
contract that reaches a consumer, and its documents carry the correction.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from aerospace_prognostics.app.database import initialize_app_database
from aerospace_prognostics.deployment.artifacts import (
    CmapssPrediction,
    _fit_rul_interval_calibration,
    _rul_prediction_interval,
)

FORBIDDEN = "confidence"


def _offending(names: object) -> list[str]:
    return sorted(name for name in names if FORBIDDEN in str(name).lower())  # type: ignore[union-attr]


def test_the_prediction_contract_has_no_confidence_field() -> None:
    names = [field.name for field in dataclasses.fields(CmapssPrediction)]
    assert "interval_quantile_level" in names
    assert not _offending(names)


def test_the_serialised_prediction_has_no_confidence_key() -> None:
    payload = CmapssPrediction(
        unit_number=1,
        predicted_rul=42.0,
        predicted_rul_lower=20.0,
        predicted_rul_upper=64.0,
        interval_method="train_residual_absolute_quantile",
        interval_quantile_level=0.9,
    ).to_dict()
    assert payload["interval_quantile_level"] == pytest.approx(0.9)
    assert not _offending(payload)


def test_the_database_schema_has_no_confidence_column(tmp_path: Path) -> None:
    database_path = tmp_path / "app.sqlite"
    initialize_app_database(database_path)
    with sqlite3.connect(database_path) as connection:
        columns = [row[1] for row in connection.execute("pragma table_info(predictions)")]
    assert "interval_quantile_level" in columns
    assert not _offending(columns)


def test_the_fitted_calibration_payload_has_no_confidence_key() -> None:
    calibration = _fit_rul_interval_calibration(
        actual=np.array([10.0, 20.0, 30.0, 40.0]),
        predicted=np.array([12.0, 19.0, 33.0, 37.0]),
        rul_cap=125.0,
    )
    assert calibration["quantile_level"] == pytest.approx(0.9)
    assert not _offending(calibration)


def test_the_interval_payload_has_no_confidence_key() -> None:
    interval = _rul_prediction_interval(
        50.0,
        rul_cap=125.0,
        calibration={
            "method": "train_residual_absolute_quantile",
            "quantile_level": 0.9,
            "absolute_error_quantile": 8.0,
        },
    )
    assert interval["quantile_level"] == pytest.approx(0.9)
    assert not _offending(interval)


def test_an_artifact_packaged_before_the_rename_still_reports_its_level() -> None:
    # Packaged .joblib artifacts carry whichever key they were written with. Only
    # the name changed, so a pre-rename artifact must keep reporting its number
    # rather than silently serving None.
    interval = _rul_prediction_interval(
        50.0,
        rul_cap=125.0,
        calibration={
            "method": "train_residual_absolute_quantile",
            "confidence": 0.9,
            "absolute_error_quantile": 8.0,
        },
    )
    assert interval["quantile_level"] == pytest.approx(0.9)
    assert not _offending(interval)


def test_the_check_would_catch_a_reintroduced_name() -> None:
    """Guard against a matcher that silently passes because it looks at nothing."""
    assert _offending(["interval_confidence", "interval_quantile_level"]) == [
        "interval_confidence"
    ]
