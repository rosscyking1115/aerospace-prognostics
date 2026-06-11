from __future__ import annotations

import json

from aerospace_prognostics.reports.dashboard import (
    DASHBOARD_SCHEMA_VERSION,
    build_fleet_dashboard_payload,
    render_fleet_dashboard_html,
)


def test_build_fleet_dashboard_payload_summarizes_predictions_and_evidence(tmp_path) -> None:
    prediction_json = tmp_path / "predictions.json"
    promotion_json = tmp_path / "promotion.json"
    release_bundle_json = tmp_path / "release_bundle.json"

    prediction_json.write_text(
        json.dumps(
            {
                "dataset": "C-MAPSS",
                "subset": "FD001",
                "model_name": "hist_gradient_boosting",
                "rul_cap": 125,
                "predictions": [
                    {"unit_number": 1, "predicted_rul": 12.5},
                    {"unit_number": 2, "predicted_rul": 42.0},
                    {"unit_number": 3, "predicted_rul": 95.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    promotion_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "gates": {"validation": True, "latency": True, "sbom": False},
                "artifact_identity": {"artifact_id": "fd001-demo"},
            }
        ),
        encoding="utf-8",
    )
    release_bundle_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "release_name": "fd001-demo-release",
                "artifact_identity": {"artifact_id": "fd001-demo"},
                "evidence": {"model_artifact": {}, "promotion_report": {}},
            }
        ),
        encoding="utf-8",
    )

    payload = build_fleet_dashboard_payload(
        prediction_json,
        promotion_json=promotion_json,
        release_bundle_json=release_bundle_json,
        generated_at_utc="2026-06-11T00:00:00+00:00",
    ).to_dict()

    assert payload["schema_version"] == DASHBOARD_SCHEMA_VERSION
    assert payload["summary"]["asset_count"] == 3
    assert payload["summary"]["risk_counts"] == {
        "critical": 1,
        "watch": 1,
        "nominal": 1,
        "unknown": 0,
    }
    assert payload["summary"]["attention_required_count"] == 2
    assert payload["assets"][0]["asset_id"] == "FD001-unit-1"
    assert payload["assets"][0]["priority_rank"] == 1
    assert payload["assets"][0]["status"] == "maintenance_review"
    assert payload["assets"][0]["attention_reasons"] == [
        "RUL at or below critical threshold"
    ]
    assert payload["evidence"]["promotion"]["gates_passed"] == 2
    assert payload["evidence"]["release_bundle"]["evidence_count"] == 2


def test_build_fleet_dashboard_payload_prioritizes_interval_risk(tmp_path) -> None:
    prediction_json = tmp_path / "predictions.json"
    prediction_json.write_text(
        json.dumps(
            {
                "dataset": "C-MAPSS",
                "subset": "FD001",
                "model_name": "hist_gradient_boosting",
                "rul_cap": 125,
                "predictions": [
                    {
                        "unit_number": 1,
                        "predicted_rul": 65.0,
                        "rul_interval": {"lower": 15.0, "upper": 95.0},
                    },
                    {"unit_number": 2, "predicted_rul": 18.0},
                    {"unit_number": 3, "predicted_rul": 80.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_fleet_dashboard_payload(prediction_json).to_dict()

    assert [asset["unit_number"] for asset in payload["assets"]] == [1, 2, 3]
    assert payload["assets"][0]["risk_level"] == "critical"
    assert payload["assets"][0]["priority_rank"] == 1
    assert payload["assets"][0]["rul_interval"] == {"lower": 15.0, "upper": 95.0}
    assert payload["assets"][0]["attention_reasons"] == [
        "RUL at or below critical threshold",
        "Interval lower bound crosses critical threshold",
        "Wide RUL interval",
    ]
    assert payload["summary"]["top_attention_assets"][0]["asset_id"] == "FD001-unit-1"


def test_build_fleet_dashboard_payload_rejects_missing_prediction_rows(tmp_path) -> None:
    prediction_json = tmp_path / "predictions.json"
    prediction_json.write_text(json.dumps({"predictions": {}}), encoding="utf-8")

    try:
        build_fleet_dashboard_payload(prediction_json)
    except ValueError as exc:
        assert "predictions must be a list" in str(exc)
    else:
        raise AssertionError("expected invalid prediction JSON error")


def test_render_fleet_dashboard_html_outputs_standalone_dashboard(tmp_path) -> None:
    prediction_json = tmp_path / "predictions.json"
    prediction_json.write_text(
        json.dumps(
            {
                "dataset": "C-MAPSS",
                "subset": "FD001",
                "model_name": "model <candidate>",
                "rul_cap": 100,
                "predictions": [
                    {"unit_number": 1, "predicted_rul": 18.0},
                    {"unit_number": 2, "predicted_rul": 63.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = build_fleet_dashboard_payload(
        prediction_json,
        title="Fleet <View>",
        generated_at_utc="2026-06-11T00:00:00+00:00",
    )

    html = render_fleet_dashboard_html(payload)

    assert "<!doctype html>" in html
    assert "<title>Fleet &lt;View&gt;</title>" in html
    assert "FD001-unit-1" in html
    assert "maintenance_review" in html
    assert "model &lt;candidate&gt;" in html
    assert "<th>Priority</th>" in html
    assert "<th>Interval</th>" in html
    assert "<th>Attention</th>" in html
    assert "risk-critical" in html
    assert "risk-nominal" in html
