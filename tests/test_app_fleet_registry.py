from __future__ import annotations

from aerospace_prognostics.app.fleet_registry import (
    fleet_asset_export_rows,
    fleet_asset_filters,
    fleet_asset_registry_summary,
    normalized_filter_values,
)


def test_normalized_filter_values_deduplicates_and_sorts_non_empty_values() -> None:
    assert normalized_filter_values([" watch ", "", "critical", "watch", 7]) == [
        "7",
        "critical",
        "watch",
    ]
    assert normalized_filter_values(None) == []


def test_fleet_asset_filters_normalize_review_inputs() -> None:
    assert fleet_asset_filters(
        risk_levels=["watch", "critical", "watch"],
        domains=[" spacecraft_anomaly ", ""],
        statuses=["monitor"],
        attention_only=1,
    ) == {
        "risk_levels": ["critical", "watch"],
        "domains": ["spacecraft_anomaly"],
        "statuses": ["monitor"],
        "attention_only": True,
    }


def test_fleet_asset_registry_summary_counts_unknowns_and_attention_assets() -> None:
    summary = fleet_asset_registry_summary(
        [
            {
                "latest_risk_level": "critical",
                "domain": "turbofan_rul",
                "latest_status": "maintenance_review",
                "latest_attention_reasons": [],
            },
            {
                "latest_risk_level": "nominal",
                "domain": "spacecraft_anomaly",
                "latest_status": "nominal",
                "latest_attention_reasons": ["High anomaly miss rate"],
            },
            {},
        ]
    )

    assert summary == {
        "asset_count": 3,
        "attention_required_count": 2,
        "risk_counts": {"critical": 1, "nominal": 1, "unknown": 1},
        "domain_counts": {
            "spacecraft_anomaly": 1,
            "turbofan_rul": 1,
            "unknown": 1,
        },
        "status_counts": {
            "maintenance_review": 1,
            "nominal": 1,
            "unknown": 1,
        },
    }


def test_fleet_asset_export_rows_flatten_metadata_and_reason_lists() -> None:
    rows = fleet_asset_export_rows(
        [
            {
                "asset_id": "smap-channel-p-1",
                "asset_type": "spacecraft_channel",
                "domain": "spacecraft_anomaly",
                "source_dataset": "SMAP/MSL",
                "source_subset": "SMAP",
                "external_id": "P-1",
                "latest_risk_level": "critical",
                "latest_status": "anomaly_review",
                "priority_score": 355.0,
                "priority_band": "immediate_review",
                "priority_reasons": ["Risk level is critical", "Live anomaly"],
                "latest_attention_reasons": ["Severity critical"],
                "metadata": {
                    "channel_id": "P-1",
                    "spacecraft": "SMAP",
                    "event_time_utc": "2026-01-02T00:00:00+00:00",
                    "severity": "critical",
                    "active": True,
                    "anomaly_score": 0.95,
                    "threshold": 0.8,
                    "source": "ops",
                    "f1": 0.2,
                    "predicted_positives": 2,
                },
            },
            {
                "asset_id": "FD001-unit-1",
                "latest_attention_reasons": "not-a-list",
                "metadata": "not-a-dict",
            },
        ]
    )

    assert rows[0]["asset_id"] == "smap-channel-p-1"
    assert rows[0]["priority_reasons"] == "Risk level is critical; Live anomaly"
    assert rows[0]["attention_reasons"] == "Severity critical"
    assert rows[0]["channel_id"] == "P-1"
    assert rows[0]["spacecraft"] == "SMAP"
    assert rows[0]["active"] is True
    assert rows[0]["anomaly_source"] == "ops"
    assert rows[0]["f1"] == 0.2
    assert rows[0]["predicted_positives"] == 2
    assert rows[1]["attention_reasons"] == ""
    assert rows[1]["model_name"] is None
