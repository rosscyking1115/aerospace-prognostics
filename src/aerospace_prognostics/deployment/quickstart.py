"""No-download deployment quickstart flows."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from aerospace_prognostics.app.dashboard_state import load_quickstart_workspace
from aerospace_prognostics.app.store import (
    build_fleet_priority_policy_validation,
    export_fleet_priority_policy_validation,
    initialize_app_database,
    record_prediction_run,
    seed_quickstart_workspace,
    sync_fleet_assets_from_prediction_run,
)
from aerospace_prognostics.artifact_io import prepare_output_path


def run_cmapss_quickstart(
    *,
    root: str | Path = Path("artifacts") / "quickstart_cmapss",
    release_name: str = "quickstart-fd001-demo",
    repository: str = "local/aerospace-prognostics",
    git_sha: str = "0" * 40,
    git_ref: str = "refs/heads/local-quickstart",
    workflow: str = "local-quickstart",
    run_id: str = "local",
    lockfile: str | Path = Path("uv.lock"),
    runner: Callable[[list[str]], int] | None = None,
) -> int:
    """Generate a tiny C-MAPSS deployment evidence bundle through the public CLI."""
    if runner is None:
        from aerospace_prognostics.cli import main

        runner = main

    from aerospace_prognostics.data.cmapss import read_cmapss_frame
    from aerospace_prognostics.examples.cmapss_fixture import write_tiny_cmapss_subset

    root = Path(root)
    lockfile = Path(lockfile)
    data_dir = root / "data"
    model_dir = root / "models"
    prediction_dir = root / "predictions"
    dashboard_dir = root / "dashboard"
    sbom_dir = root / "sbom"
    artifact_path = model_dir / "fd001.joblib"
    metadata_json = model_dir / "fd001_metadata.json"
    model_card_markdown = model_dir / "fd001_model_card.md"
    inspection_json = model_dir / "fd001_inspection.json"
    validation_json = model_dir / "fd001_validation.json"
    benchmark_json = model_dir / "fd001_benchmark.json"
    promotion_json = model_dir / "fd001_promotion.json"
    promotion_markdown = model_dir / "fd001_promotion.md"
    release_bundle_json = root / "release" / "fd001_release_bundle.json"
    release_bundle_markdown = root / "release" / "fd001_release_bundle.md"
    provenance_json = root / "release" / "fd001_provenance.json"
    provenance_markdown = root / "release" / "fd001_provenance.md"
    policy_validation_json = root / "release" / "fleet_priority_policy_validation.json"
    policy_validation_markdown = root / "release" / "fleet_priority_policy_validation.md"
    app_database = root / "app" / "aerospace_prognostics.sqlite"
    sbom_json = sbom_dir / "cyclonedx.json"
    input_csv = prediction_dir / "fd001_input.csv"
    prediction_json = prediction_dir / "fd001_predictions.json"
    dashboard_payload_json = dashboard_dir / "fleet_payload.json"
    dashboard_html = dashboard_dir / "fleet_dashboard.html"

    data_dir.mkdir(parents=True, exist_ok=True)
    write_tiny_cmapss_subset(data_dir)
    input_csv = prepare_output_path(input_csv)
    read_cmapss_frame(data_dir / "test_FD001.txt").to_csv(input_csv, index=False)

    _run_cli(
        [
            "cmapss-package-hgb-policy",
            "--data-dir",
            str(data_dir),
            "--subset",
            "FD001",
            "--output-path",
            str(artifact_path),
            "--metadata-json",
            str(metadata_json),
            "--model-card-markdown",
            str(model_card_markdown),
            "--n-regimes",
            "1",
        ],
        runner,
    )
    _run_cli(
        [
            "cmapss-predict-artifact",
            "--model-artifact",
            str(artifact_path),
            "--input-csv",
            str(input_csv),
            "--output-json",
            str(prediction_json),
        ],
        runner,
    )
    _run_cli(
        [
            "cmapss-inspect-artifact",
            "--model-artifact",
            str(artifact_path),
            "--output-json",
            str(inspection_json),
        ],
        runner,
    )
    _run_cli(
        [
            "cmapss-validate-artifact",
            "--model-artifact",
            str(artifact_path),
            "--metadata-json",
            str(metadata_json),
            "--input-csv",
            str(input_csv),
            "--output-json",
            str(validation_json),
        ],
        runner,
    )
    _run_cli(
        [
            "cmapss-benchmark-artifact",
            "--model-artifact",
            str(artifact_path),
            "--input-csv",
            str(input_csv),
            "--runs",
            "2",
            "--warmup-runs",
            "1",
            "--max-p95-latency-ms",
            "10000",
            "--output-json",
            str(benchmark_json),
        ],
        runner,
    )
    _run_cli(
        ["generate-sbom", "--lockfile", str(lockfile), "--output-json", str(sbom_json)],
        runner,
    )
    _run_cli(
        [
            "cmapss-promotion-report",
            "--validation-json",
            str(validation_json),
            "--benchmark-json",
            str(benchmark_json),
            "--model-card-markdown",
            str(model_card_markdown),
            "--sbom-json",
            str(sbom_json),
            "--output-json",
            str(promotion_json),
            "--output-markdown",
            str(promotion_markdown),
        ],
        runner,
    )
    _run_cli(
        [
            "dashboard-fleet-payload",
            "--prediction-json",
            str(prediction_json),
            "--promotion-json",
            str(promotion_json),
            "--output-json",
            str(dashboard_payload_json),
        ],
        runner,
    )
    _run_cli(
        [
            "dashboard-render-html",
            "--payload-json",
            str(dashboard_payload_json),
            "--output-html",
            str(dashboard_html),
        ],
        runner,
    )
    _run_cli(
        [
            "cmapss-release-bundle",
            "--release-name",
            release_name,
            "--model-artifact",
            str(artifact_path),
            "--metadata-json",
            str(metadata_json),
            "--model-card-markdown",
            str(model_card_markdown),
            "--promotion-json",
            str(promotion_json),
            "--sbom-json",
            str(sbom_json),
            "--dashboard-payload-json",
            str(dashboard_payload_json),
            "--dashboard-html",
            str(dashboard_html),
            "--output-json",
            str(release_bundle_json),
            "--output-markdown",
            str(release_bundle_markdown),
        ],
        runner,
    )
    _run_cli(
        [
            "generate-release-provenance",
            "--release-bundle-json",
            str(release_bundle_json),
            "--repository",
            repository,
            "--git-sha",
            git_sha,
            "--git-ref",
            git_ref,
            "--workflow",
            workflow,
            "--run-id",
            run_id,
            "--output-json",
            str(provenance_json),
            "--output-markdown",
            str(provenance_markdown),
        ],
        runner,
    )
    _write_quickstart_policy_validation_evidence(
        root=root,
        app_database=app_database,
        input_csv=input_csv,
        prediction_json=prediction_json,
        metadata_json=metadata_json,
        artifact_path=artifact_path,
    )

    promotion = json.loads(promotion_json.read_text(encoding="utf-8"))
    if promotion["status"] != "ok" or not all(promotion["gates"].values()):
        raise RuntimeError(f"promotion evidence smoke failed: {promotion!r}")
    inspection = json.loads(inspection_json.read_text(encoding="utf-8"))
    if (
        inspection["schema_version"]
        != "aerospace-prognostics/cmapss-artifact-inspection/v1"
        or not all(inspection["checks"].values())
    ):
        raise RuntimeError(f"artifact inspection smoke failed: {inspection!r}")
    dashboard_payload = json.loads(dashboard_payload_json.read_text(encoding="utf-8"))
    if (
        dashboard_payload["schema_version"] != "aerospace-prognostics/fleet-dashboard/v1"
        or dashboard_payload["summary"]["asset_count"] == 0
    ):
        raise RuntimeError(f"dashboard payload smoke failed: {dashboard_payload!r}")
    dashboard_markup = dashboard_html.read_text(encoding="utf-8")
    if (
        not dashboard_markup.startswith("<!doctype html>")
        or "Aerospace PHM Fleet View" not in dashboard_markup
    ):
        raise RuntimeError(f"dashboard HTML smoke failed: {dashboard_html}")
    release_bundle = json.loads(release_bundle_json.read_text(encoding="utf-8"))
    if release_bundle["status"] != "ok" or not all(release_bundle["gates"].values()):
        raise RuntimeError(f"release bundle smoke failed: {release_bundle!r}")
    if "dashboard_payload" not in release_bundle["evidence"]:
        raise RuntimeError("release bundle is missing dashboard payload evidence")
    if "dashboard_html" not in release_bundle["evidence"]:
        raise RuntimeError("release bundle is missing dashboard HTML evidence")
    provenance = json.loads(provenance_json.read_text(encoding="utf-8"))
    if provenance["status"] != "ok":
        raise RuntimeError(f"release provenance smoke failed: {provenance!r}")
    policy_validation = json.loads(policy_validation_json.read_text(encoding="utf-8"))
    if policy_validation["overall_status"] != "pass":
        raise RuntimeError(f"priority policy smoke failed: {policy_validation!r}")
    if not policy_validation_markdown.exists():
        raise RuntimeError(f"priority policy markdown is missing: {policy_validation_markdown}")
    print(f"artifact_inspection={inspection_json}")
    print(f"promotion_report={promotion_json}")
    print(f"dashboard_payload={dashboard_payload_json}")
    print(f"dashboard_html={dashboard_html}")
    print(f"release_bundle={release_bundle_json}")
    print(f"release_provenance={provenance_json}")
    print(f"priority_policy_validation={policy_validation_json}")
    print(f"artifact_id={promotion['artifact_identity']['artifact_id']}")
    print(f"gates={len(promotion['gates'])}")
    return 0


def _write_quickstart_policy_validation_evidence(
    *,
    root: Path,
    app_database: Path,
    input_csv: Path,
    prediction_json: Path,
    metadata_json: Path,
    artifact_path: Path,
) -> None:
    workspace = load_quickstart_workspace(root)
    initialize_app_database(app_database)
    seed_quickstart_workspace(app_database, workspace)
    telemetry = pd.read_csv(input_csv)
    prediction_document = json.loads(prediction_json.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    artifact_metadata = metadata.get("artifact")
    artifact_metadata = artifact_metadata if isinstance(artifact_metadata, dict) else {}
    promotion = artifact_metadata.get("promotion")
    promotion = promotion if isinstance(promotion, dict) else {}
    prediction_document["artifact"] = {
        "schema_version": artifact_metadata.get("schema_version"),
        "artifact_id": promotion.get("artifact_id"),
        "stage": promotion.get("stage"),
    }
    run_id = record_prediction_run(
        app_database,
        telemetry=telemetry,
        prediction_document=prediction_document,
        model_artifact_path=artifact_path,
        source_name=str(input_csv),
    )
    sync_fleet_assets_from_prediction_run(app_database, run_id=run_id)
    export_fleet_priority_policy_validation(app_database, output_dir=root / "release")
    validation = build_fleet_priority_policy_validation(app_database, read_only=True)
    if validation["overall_status"] != "pass":
        raise RuntimeError(f"priority policy validation failed: {validation!r}")


def _run_cli(args: list[str], runner: Callable[[list[str]], int]) -> None:
    exit_code = runner(args)
    if exit_code != 0:
        raise RuntimeError(f"command failed with exit code {exit_code}: {args}")
