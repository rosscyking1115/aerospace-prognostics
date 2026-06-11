"""Smoke test for deployment release-evidence generation."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(
    *,
    root: str | Path = Path("artifacts") / "ci_release_evidence",
    release_name: str = "ci-fd001-candidate",
    repository: str = "rosscyking1115/aerospace-prognostics",
    git_sha: str = "0123456789abcdef0123456789abcdef01234567",
    git_ref: str = "refs/heads/main",
    workflow: str = "CI",
    run_id: str = "1",
) -> int:
    """Generate a tiny promotion-evidence bundle through the public CLI."""

    sys.path.insert(0, str(REPO_ROOT))
    from aerospace_prognostics.cli import main
    from aerospace_prognostics.data.cmapss import read_cmapss_frame
    from aerospace_prognostics.examples.cmapss_fixture import write_tiny_cmapss_subset

    root = Path(root)
    data_dir = root / "data"
    model_dir = root / "models"
    prediction_dir = root / "predictions"
    dashboard_dir = root / "dashboard"
    sbom_dir = root / "sbom"
    artifact_path = model_dir / "fd001.joblib"
    metadata_json = model_dir / "fd001_metadata.json"
    model_card_markdown = model_dir / "fd001_model_card.md"
    validation_json = model_dir / "fd001_validation.json"
    benchmark_json = model_dir / "fd001_benchmark.json"
    promotion_json = model_dir / "fd001_promotion.json"
    promotion_markdown = model_dir / "fd001_promotion.md"
    release_bundle_json = root / "release" / "fd001_release_bundle.json"
    release_bundle_markdown = root / "release" / "fd001_release_bundle.md"
    provenance_json = root / "release" / "fd001_provenance.json"
    provenance_markdown = root / "release" / "fd001_provenance.md"
    sbom_json = sbom_dir / "cyclonedx.json"
    input_csv = prediction_dir / "fd001_input.csv"
    prediction_json = prediction_dir / "fd001_predictions.json"
    dashboard_payload_json = dashboard_dir / "fleet_payload.json"
    dashboard_html = dashboard_dir / "fleet_dashboard.html"

    data_dir.mkdir(parents=True, exist_ok=True)
    write_tiny_cmapss_subset(data_dir)
    input_csv.parent.mkdir(parents=True, exist_ok=True)
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
        main,
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
        main,
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
        main,
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
        main,
    )
    _run_cli(
        ["generate-sbom", "--lockfile", "uv.lock", "--output-json", str(sbom_json)],
        main,
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
        main,
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
        main,
    )
    _run_cli(
        [
            "dashboard-render-html",
            "--payload-json",
            str(dashboard_payload_json),
            "--output-html",
            str(dashboard_html),
        ],
        main,
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
        main,
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
        main,
    )

    promotion = json.loads(promotion_json.read_text(encoding="utf-8"))
    if promotion["status"] != "ok" or not all(promotion["gates"].values()):
        raise RuntimeError(f"promotion evidence smoke failed: {promotion!r}")
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
    print(f"promotion_report={promotion_json}")
    print(f"dashboard_payload={dashboard_payload_json}")
    print(f"dashboard_html={dashboard_html}")
    print(f"release_bundle={release_bundle_json}")
    print(f"release_provenance={provenance_json}")
    print(f"artifact_id={promotion['artifact_identity']['artifact_id']}")
    print(f"gates={len(promotion['gates'])}")
    return 0


def _run_cli(args: list[str], runner: Callable[[list[str]], int]) -> None:
    exit_code = runner(args)
    if exit_code != 0:
        raise RuntimeError(f"command failed with exit code {exit_code}: {args}")


if __name__ == "__main__":
    raise SystemExit(run())
