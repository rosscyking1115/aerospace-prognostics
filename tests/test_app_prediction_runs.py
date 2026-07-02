from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aerospace_prognostics.app.prediction_runs import (
    PREDICTION_RUN_EVIDENCE_SCHEMA_VERSION,
    build_prediction_run_evidence_payload,
    event_from_row,
    outcome_rows,
    outcome_template_frame,
    prediction_rows,
    prediction_run_event_record,
    with_interval_availability,
)


def test_prediction_rows_accepts_prediction_document_contract() -> None:
    rows = prediction_rows(
        {
            "predictions": [
                {"unit_number": 1, "predicted_rul": 42.0, "extra": "kept"},
            ]
        }
    )

    assert rows == [{"unit_number": 1, "predicted_rul": 42.0, "extra": "kept"}]


@pytest.mark.parametrize(
    ("document", "expected_message"),
    [
        ({}, "prediction_document\\['predictions'\\] must be a list"),
        ({"predictions": ["bad-row"]}, "prediction rows must be JSON objects"),
        ({"predictions": [{"unit_number": 1}]}, "prediction rows require"),
    ],
)
def test_prediction_rows_rejects_invalid_prediction_documents(
    document: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        prediction_rows(document)


def test_prediction_run_event_record_is_stable_and_json_safe() -> None:
    event = prediction_run_event_record(
        run_id="run-1",
        event_type="operator_decision",
        status="watch",
        actor="flight-ops",
        note="Monitor on next cycle",
        payload={"ticket": "PHM-42", "priority": 2},
        timestamp="2026-01-01T00:00:00+00:00",
    )
    reordered = prediction_run_event_record(
        run_id="run-1",
        event_type="operator_decision",
        status="watch",
        actor="flight-ops",
        note="Monitor on next cycle",
        payload={"priority": 2, "ticket": "PHM-42"},
        timestamp="2026-01-01T00:00:00+00:00",
    )

    assert event["event_id"] == "event-957040e1970f07f6"
    assert reordered["event_id"] == event["event_id"]
    assert event["payload_json"] == '{"priority":2,"ticket":"PHM-42"}'
    assert event["created_at_utc"] == "2026-01-01T00:00:00+00:00"


def test_event_from_row_decodes_payload_json_tolerantly() -> None:
    event = event_from_row(
        {
            "event_id": "event-1",
            "event_type": "operator_decision",
            "payload_json": '{"ticket":"PHM-42"}',
        }
    )
    fallback = event_from_row(
        {
            "event_id": "event-2",
            "event_type": "operator_decision",
            "payload_json": "{bad-json",
        }
    )

    assert event == {
        "event_id": "event-1",
        "event_type": "operator_decision",
        "payload": {"ticket": "PHM-42"},
    }
    assert fallback["payload"] == {}


def test_outcome_rows_normalizes_numeric_outcomes() -> None:
    rows = outcome_rows(pd.DataFrame({"unit_number": ["1"], "actual_rul": ["12.5"]}))

    assert rows == [{"unit_number": 1, "actual_rul": 12.5}]


@pytest.mark.parametrize(
    ("outcomes", "expected_message"),
    [
        (pd.DataFrame({"unit_number": [1]}), "outcomes require columns: actual_rul"),
        (
            pd.DataFrame({"unit_number": [], "actual_rul": []}),
            "outcomes must contain at least one row",
        ),
        (
            pd.DataFrame({"unit_number": ["bad"], "actual_rul": [1.0]}),
            "outcome rows require numeric, non-null",
        ),
        (
            pd.DataFrame({"unit_number": [1.5], "actual_rul": [1.0]}),
            "outcome unit_number values must be whole numbers",
        ),
        (
            pd.DataFrame({"unit_number": [1], "actual_rul": [-1.0]}),
            "actual_rul values must be nonnegative",
        ),
    ],
)
def test_outcome_rows_rejects_invalid_outcomes(
    outcomes: pd.DataFrame,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        outcome_rows(outcomes)


def test_with_interval_availability_adds_prediction_and_outcome_rates() -> None:
    enriched = with_interval_availability(
        {
            "run_id": "run-1",
            "prediction_count": 4,
            "interval_count": 3,
            "outcome_count": 2,
            "interval_outcome_count": 2,
            "interval_covered_count": 1,
        }
    )

    assert enriched["interval_availability_rate"] == 0.75
    assert enriched["outcome_availability_rate"] == 0.5
    assert enriched["outcome_interval_coverage_rate"] == 0.5


def test_with_interval_availability_handles_zero_prediction_counts() -> None:
    enriched = with_interval_availability(
        {
            "prediction_count": 0,
            "interval_count": None,
            "outcome_count": None,
            "interval_outcome_count": 0,
            "interval_covered_count": 0,
        }
    )

    assert enriched["interval_count"] == 0
    assert enriched["interval_availability_rate"] is None
    assert enriched["outcome_count"] == 0
    assert enriched["outcome_availability_rate"] is None
    assert enriched["outcome_interval_coverage_rate"] is None


def test_with_interval_availability_tolerates_invalid_count_values() -> None:
    enriched = with_interval_availability(
        {
            "prediction_count": "bad",
            "interval_count": "also-bad",
            "outcome_count": "nope",
            "interval_outcome_count": "none",
            "interval_covered_count": "invalid",
        }
    )

    assert enriched["interval_count"] == 0
    assert enriched["interval_availability_rate"] is None
    assert enriched["outcome_count"] == 0
    assert enriched["outcome_availability_rate"] is None
    assert enriched["interval_outcome_count"] == 0
    assert enriched["interval_covered_count"] == 0
    assert enriched["outcome_interval_coverage_rate"] is None


def test_build_prediction_run_evidence_payload_preserves_loaded_run_contract() -> None:
    payload = build_prediction_run_evidence_payload(
        database_path="app.sqlite",
        database_schema_version="schema-v1",
        loaded_run={
            "run": {"run_id": "run-1"},
            "predictions": [{"unit_number": 1, "predicted_rul": 10.0}],
            "audit_events": [{"event_id": "event-1"}],
        },
        exported_at_utc="2026-01-01T00:00:00+00:00",
        predictions_csv_path="exports/run-1.csv",
    )

    assert payload["schema_version"] == PREDICTION_RUN_EVIDENCE_SCHEMA_VERSION
    assert payload["exported_at_utc"] == "2026-01-01T00:00:00+00:00"
    assert payload["database"] == {"path": "app.sqlite", "schema_version": "schema-v1"}
    assert payload["run"] == {"run_id": "run-1"}
    assert payload["predictions"] == [{"unit_number": 1, "predicted_rul": 10.0}]
    assert payload["audit_events"] == [{"event_id": "event-1"}]
    assert payload["files"]["predictions_csv"] == {
        "rows": 1,
        "path": str(Path("exports/run-1.csv")),
    }


def test_outcome_template_frame_keeps_prediction_units_fillable() -> None:
    frame = outcome_template_frame(
        [
            {"unit_number": 7, "predicted_rul": 10.0},
            {"unit_number": 8, "predicted_rul": 20.0},
        ]
    )

    assert list(frame.columns) == ["unit_number", "actual_rul"]
    assert list(frame["unit_number"]) == [7, 8]
    assert list(frame["actual_rul"]) == ["", ""]
