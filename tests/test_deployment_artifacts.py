from __future__ import annotations

import json

from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.deployment.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    benchmark_cmapss_model_artifact,
    build_cmapss_promotion_report,
    build_cmapss_release_bundle,
    load_cmapss_model_artifact,
    render_cmapss_model_card_markdown,
    render_cmapss_release_bundle_markdown,
    save_cmapss_model_artifact,
    train_cmapss_hgb_policy_artifact,
    validate_cmapss_model_artifact,
    write_cmapss_model_card_markdown,
    write_cmapss_promotion_report_markdown,
)
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_train_save_load_and_predict_cmapss_hgb_policy_artifact(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(
        tmp_path,
        "FD001",
        n_regimes=1,
    )
    artifact_path = tmp_path / "models" / "fd001.joblib"

    save_cmapss_model_artifact(packaged.artifact, artifact_path)
    loaded = load_cmapss_model_artifact(artifact_path)
    bundle = load_cmapss_subset(tmp_path, "FD001")
    predictions = loaded.predict_from_frame(bundle.test)
    monitoring = loaded.monitoring_summary(bundle.test, predictions, drift_threshold=0.1)

    assert artifact_path.exists()
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert loaded.subset == "FD001"
    assert loaded.feature_policy == "regime_engineered"
    assert "sensor_1" in loaded.reference_stats
    assert loaded.reference_stats["sensor_1"]["count"] == 6.0
    assert loaded.uncertainty_calibration["method"] == "train_residual_absolute_quantile"
    assert loaded.uncertainty_calibration["confidence"] == 0.9
    assert loaded.uncertainty_calibration["calibration_count"] == 6
    assert loaded.promotion_metadata["artifact_id"].startswith("fd001-")
    assert loaded.promotion_metadata["stage"] == "candidate"
    assert loaded.promotion_metadata["identity"]["official_test_rmse"] == round(
        packaged.result.rmse,
        12,
    )
    assert loaded.metadata()["promotion"]["rollback"]["requires_retraining"] is False
    assert monitoring["predictions"]["count"] == 2
    assert monitoring["telemetry"]["alert_column_count"] >= 1
    assert "sensor_1" in monitoring["telemetry"]["columns"]
    assert packaged.result.rmse >= 0
    assert [prediction.unit_number for prediction in predictions] == [1, 2]
    assert all(0 <= prediction.predicted_rul <= loaded.rul_cap for prediction in predictions)
    assert all(prediction.predicted_rul_lower is not None for prediction in predictions)
    assert all(prediction.predicted_rul_upper is not None for prediction in predictions)
    assert all(
        0 <= prediction.predicted_rul_lower <= prediction.predicted_rul_upper <= loaded.rul_cap
        for prediction in predictions
        if prediction.predicted_rul_lower is not None
        and prediction.predicted_rul_upper is not None
    )
    assert predictions[0].to_dict()["interval_method"] == "train_residual_absolute_quantile"


def test_cmapss_hgb_policy_artifact_rejects_missing_columns(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1).artifact
    bundle = load_cmapss_subset(tmp_path, "FD001")
    malformed = bundle.test.drop(columns=["sensor_1"])

    try:
        artifact.predict_from_frame(malformed)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("expected missing-column validation error")


def test_render_cmapss_model_card_markdown_summarizes_deployment_context(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)

    markdown = render_cmapss_model_card_markdown(packaged.artifact, packaged.result)

    assert "# C-MAPSS Deployment Model Card" in markdown
    assert "## Intended Use" in markdown
    assert "## Performance" in markdown
    assert f"| Official-test RMSE | {packaged.result.rmse:.6f} |" in markdown
    assert "## Inference Contract" in markdown
    assert "Interval method" in markdown
    assert "train_residual_absolute_quantile" in markdown
    assert "## Monitoring" in markdown
    assert "## Limitations" in markdown
    assert "Requires retraining: `False`" in markdown


def test_benchmark_cmapss_model_artifact_reports_latency_and_size(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)
    artifact_path = save_cmapss_model_artifact(
        packaged.artifact,
        tmp_path / "models" / "fd001.joblib",
    )
    input_csv = tmp_path / "fd001_input.csv"
    load_cmapss_subset(tmp_path, "FD001").test.to_csv(input_csv, index=False)

    benchmark = benchmark_cmapss_model_artifact(
        artifact_path,
        input_csv,
        runs=2,
        warmup_runs=1,
        max_p95_latency_ms=10_000.0,
    )

    assert benchmark.status == "ok"
    assert benchmark.runs == 2
    assert benchmark.warmup_runs == 1
    assert benchmark.input_rows == 4
    assert benchmark.prediction_count == 2
    assert benchmark.model_size_bytes == artifact_path.stat().st_size
    assert benchmark.latency_ms["min"] >= 0.0
    assert benchmark.latency_ms["p95"] >= benchmark.latency_ms["p50"]
    assert benchmark.artifact_identity["artifact_id"] == packaged.artifact.promotion_metadata[
        "artifact_id"
    ]
    assert benchmark.to_dict()["status"] == "ok"


def test_validate_cmapss_model_artifact_checks_metadata_and_prediction_smoke(
    tmp_path,
) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)
    artifact_path = save_cmapss_model_artifact(
        packaged.artifact,
        tmp_path / "models" / "fd001.joblib",
    )
    metadata_json = tmp_path / "models" / "fd001_metadata.json"
    input_csv = tmp_path / "fd001_input.csv"
    metadata_json.write_text(
        json.dumps(
            {
                "artifact": packaged.artifact.metadata(),
                "result": packaged.result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    load_cmapss_subset(tmp_path, "FD001").test.to_csv(input_csv, index=False)

    validation = validate_cmapss_model_artifact(
        artifact_path,
        metadata_json=metadata_json,
        input_csv=input_csv,
    )

    assert validation.status == "ok"
    assert validation.problems == []
    assert all(validation.checks.values())
    assert validation.artifact_identity["artifact_id"] == packaged.artifact.promotion_metadata[
        "artifact_id"
    ]
    assert validation.prediction_count == 2


def test_build_cmapss_promotion_report_combines_gate_evidence(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)
    artifact_path = save_cmapss_model_artifact(
        packaged.artifact,
        tmp_path / "models" / "fd001.joblib",
    )
    metadata_json = tmp_path / "models" / "fd001_metadata.json"
    validation_json = tmp_path / "models" / "fd001_validation.json"
    benchmark_json = tmp_path / "models" / "fd001_benchmark.json"
    model_card_markdown = tmp_path / "models" / "fd001_model_card.md"
    promotion_markdown = tmp_path / "models" / "fd001_promotion.md"
    sbom_json = tmp_path / "sbom" / "cyclonedx.json"
    input_csv = tmp_path / "fd001_input.csv"
    metadata_json.write_text(
        json.dumps(
            {
                "artifact": packaged.artifact.metadata(),
                "result": packaged.result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    load_cmapss_subset(tmp_path, "FD001").test.to_csv(input_csv, index=False)
    validation = validate_cmapss_model_artifact(
        artifact_path,
        metadata_json=metadata_json,
        input_csv=input_csv,
    )
    benchmark = benchmark_cmapss_model_artifact(
        artifact_path,
        input_csv,
        runs=2,
        warmup_runs=1,
        max_p95_latency_ms=10_000.0,
    )
    validation_json.write_text(json.dumps(validation.to_dict()), encoding="utf-8")
    benchmark_json.write_text(json.dumps(benchmark.to_dict()), encoding="utf-8")
    write_cmapss_model_card_markdown(packaged.artifact, packaged.result, model_card_markdown)
    sbom_json.parent.mkdir(parents=True)
    sbom_json.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "components": [{"type": "library", "name": "numpy", "version": "1.0.0"}],
            },
        ),
        encoding="utf-8",
    )

    report = build_cmapss_promotion_report(
        validation_json,
        benchmark_json,
        model_card_markdown=model_card_markdown,
        sbom_json=sbom_json,
    )
    markdown_path = write_cmapss_promotion_report_markdown(report, promotion_markdown)

    assert report.status == "ok"
    assert all(report.gates.values())
    assert report.problems == []
    assert report.artifact_identity["artifact_id"] == packaged.artifact.promotion_metadata[
        "artifact_id"
    ]
    assert report.evidence["sbom"]["component_count"] == 1
    assert "# C-MAPSS Promotion Report" in markdown_path.read_text(encoding="utf-8")


def test_build_cmapss_release_bundle_ties_model_supply_chain_and_container_evidence(
    tmp_path,
) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)
    artifact_path = save_cmapss_model_artifact(
        packaged.artifact,
        tmp_path / "models" / "fd001.joblib",
    )
    metadata_json = tmp_path / "models" / "fd001_metadata.json"
    validation_json = tmp_path / "models" / "fd001_validation.json"
    benchmark_json = tmp_path / "models" / "fd001_benchmark.json"
    model_card_markdown = tmp_path / "models" / "fd001_model_card.md"
    promotion_json = tmp_path / "models" / "fd001_promotion.json"
    sbom_json = tmp_path / "sbom" / "cyclonedx.json"
    dashboard_payload_json = tmp_path / "dashboard" / "fleet_payload.json"
    dashboard_html = tmp_path / "dashboard" / "fleet_dashboard.html"
    container_manifest_json = tmp_path / "container" / "serving_image_manifest.json"
    input_csv = tmp_path / "fd001_input.csv"
    metadata_json.write_text(
        json.dumps(
            {
                "artifact": packaged.artifact.metadata(),
                "result": packaged.result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    load_cmapss_subset(tmp_path, "FD001").test.to_csv(input_csv, index=False)
    validation = validate_cmapss_model_artifact(
        artifact_path,
        metadata_json=metadata_json,
        input_csv=input_csv,
    )
    benchmark = benchmark_cmapss_model_artifact(
        artifact_path,
        input_csv,
        runs=2,
        warmup_runs=1,
        max_p95_latency_ms=10_000.0,
    )
    validation_json.write_text(json.dumps(validation.to_dict()), encoding="utf-8")
    benchmark_json.write_text(json.dumps(benchmark.to_dict()), encoding="utf-8")
    write_cmapss_model_card_markdown(packaged.artifact, packaged.result, model_card_markdown)
    sbom_json.parent.mkdir(parents=True)
    sbom_json.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "components": [{"type": "library", "name": "numpy", "version": "1.0.0"}],
            },
        ),
        encoding="utf-8",
    )
    report = build_cmapss_promotion_report(
        validation_json,
        benchmark_json,
        model_card_markdown=model_card_markdown,
        sbom_json=sbom_json,
    )
    promotion_json.write_text(json.dumps(report.to_dict()), encoding="utf-8")
    dashboard_payload_json.parent.mkdir(parents=True)
    dashboard_payload_json.write_text(
        json.dumps(
            {
                "schema_version": "aerospace-prognostics/fleet-dashboard/v1",
                "title": "FD001 fleet",
                "generated_at_utc": "2026-06-11T16:00:00+00:00",
                "summary": {"asset_count": 1},
                "assets": [{"asset_id": "FD001-unit-1", "risk_level": "nominal"}],
            },
        ),
        encoding="utf-8",
    )
    dashboard_html.write_text("<!doctype html><title>FD001 fleet</title>", encoding="utf-8")
    container_manifest_json.parent.mkdir(parents=True)
    container_manifest_json.write_text(
        json.dumps(
            {
                "schema_version": "aerospace-prognostics/serving-image-manifest/v1",
                "image": "aerospace-prognostics:ci",
                "image_id": "sha256:image",
                "labels": {"org.opencontainers.image.revision": "abc123"},
                "validation": {
                    "has_oci_labels": True,
                    "has_healthcheck": True,
                    "revision_matches_expected": True,
                    "torch_absent": True,
                },
            },
        ),
        encoding="utf-8",
    )

    bundle = build_cmapss_release_bundle(
        release_name="fd001-ci-candidate",
        model_artifact=artifact_path,
        metadata_json=metadata_json,
        model_card_markdown=model_card_markdown,
        promotion_json=promotion_json,
        sbom_json=sbom_json,
        dashboard_payload_json=dashboard_payload_json,
        dashboard_html=dashboard_html,
        container_manifest_json=container_manifest_json,
        container_image_ref="aerospace-prognostics:ci",
    )
    markdown = render_cmapss_release_bundle_markdown(bundle)

    assert bundle.status == "ok"
    assert all(bundle.gates.values())
    assert bundle.artifact_identity["artifact_id"] == packaged.artifact.promotion_metadata[
        "artifact_id"
    ]
    assert bundle.evidence["model_artifact"]["sha256"]
    assert bundle.evidence["dashboard_payload"]["sha256"]
    assert bundle.evidence["dashboard_html"]["sha256"]
    assert bundle.gates["dashboard_payload_schema"] is True
    assert bundle.gates["dashboard_html_present"] is True
    assert bundle.evidence["container"]["image_id"] == "sha256:image"
    assert bundle.to_dict()["schema_version"] == "aerospace-prognostics/cmapss-release-bundle/v1"
    assert "# C-MAPSS Release Bundle" in markdown
    assert "Model artifact SHA-256" in markdown
    assert "Dashboard payload" in markdown
    assert "Dashboard HTML" in markdown


def test_build_cmapss_release_bundle_fails_on_identity_mismatch(tmp_path) -> None:
    artifact_path = tmp_path / "models" / "fd001.joblib"
    metadata_json = tmp_path / "models" / "fd001_metadata.json"
    model_card_markdown = tmp_path / "models" / "fd001_model_card.md"
    promotion_json = tmp_path / "models" / "fd001_promotion.json"
    sbom_json = tmp_path / "sbom" / "cyclonedx.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"model")
    metadata_json.write_text(
        json.dumps(
            {
                "artifact": {
                    "schema_version": "1.1",
                    "dataset": "C-MAPSS",
                    "subset": "FD001",
                    "model_name": "model",
                    "promotion": {"artifact_id": "metadata-id", "stage": "candidate"},
                },
            },
        ),
        encoding="utf-8",
    )
    model_card_markdown.write_text("# C-MAPSS Deployment Model Card\n", encoding="utf-8")
    promotion_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "gates": {"artifact_validation": True},
                "artifact_identity": {"artifact_id": "promotion-id"},
            },
        ),
        encoding="utf-8",
    )
    sbom_json.parent.mkdir(parents=True)
    sbom_json.write_text(
        json.dumps({"bomFormat": "CycloneDX", "components": [{"name": "numpy"}]}),
        encoding="utf-8",
    )

    bundle = build_cmapss_release_bundle(
        release_name="mismatch",
        model_artifact=artifact_path,
        metadata_json=metadata_json,
        model_card_markdown=model_card_markdown,
        promotion_json=promotion_json,
        sbom_json=sbom_json,
    )

    assert bundle.status == "failed"
    assert bundle.gates["artifact_identity_match"] is False
    assert "artifact identity mismatch" in bundle.problems[0]


def test_build_cmapss_promotion_report_keeps_failed_gate_visible(tmp_path) -> None:
    artifact_id = "fd001-example"
    validation_json = tmp_path / "validation.json"
    benchmark_json = tmp_path / "benchmark.json"
    validation_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "checks": {"artifact_loads": True},
                "problems": [],
                "artifact_identity": {"artifact_id": artifact_id},
            },
        ),
        encoding="utf-8",
    )
    benchmark_json.write_text(
        json.dumps(
            {
                "status": "failed",
                "problems": ["p95 latency exceeded budget"],
                "artifact_identity": {"artifact_id": artifact_id},
                "latency_ms": {"p95": 120.0},
            },
        ),
        encoding="utf-8",
    )

    report = build_cmapss_promotion_report(validation_json, benchmark_json)

    assert report.status == "failed"
    assert report.gates["artifact_validation"] is True
    assert report.gates["latency_benchmark"] is False
    assert "latency benchmark gate failed" in report.problems
    assert "benchmark: p95 latency exceeded budget" in report.problems


def test_validate_cmapss_model_artifact_reports_metadata_mismatch(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)
    artifact_path = save_cmapss_model_artifact(
        packaged.artifact,
        tmp_path / "models" / "fd001.joblib",
    )
    metadata = {"artifact": json.loads(json.dumps(packaged.artifact.metadata()))}
    metadata["artifact"]["promotion"]["artifact_id"] = "wrong-artifact"
    metadata_json = tmp_path / "models" / "fd001_metadata.json"
    metadata_json.write_text(json.dumps(metadata), encoding="utf-8")

    validation = validate_cmapss_model_artifact(artifact_path, metadata_json=metadata_json)

    assert validation.status == "failed"
    assert validation.checks["metadata_json_matches"] is False
    assert "metadata artifact_id mismatch" in validation.problems[0]
