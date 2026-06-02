"""CI smoke test for deployment release-evidence generation."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run() -> int:
    """Generate a tiny promotion-evidence bundle through the public CLI."""

    sys.path.insert(0, str(REPO_ROOT))
    from aerospace_prognostics.cli import main
    from aerospace_prognostics.data.cmapss import read_cmapss_frame
    from tests.cmapss_fixtures import write_tiny_cmapss_subset

    root = Path("artifacts") / "ci_release_evidence"
    data_dir = root / "data"
    model_dir = root / "models"
    prediction_dir = root / "predictions"
    sbom_dir = root / "sbom"
    artifact_path = model_dir / "fd001.joblib"
    metadata_json = model_dir / "fd001_metadata.json"
    model_card_markdown = model_dir / "fd001_model_card.md"
    validation_json = model_dir / "fd001_validation.json"
    benchmark_json = model_dir / "fd001_benchmark.json"
    promotion_json = model_dir / "fd001_promotion.json"
    promotion_markdown = model_dir / "fd001_promotion.md"
    sbom_json = sbom_dir / "cyclonedx.json"
    input_csv = prediction_dir / "fd001_input.csv"

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

    promotion = json.loads(promotion_json.read_text(encoding="utf-8"))
    if promotion["status"] != "ok" or not all(promotion["gates"].values()):
        raise RuntimeError(f"promotion evidence smoke failed: {promotion!r}")
    print(f"promotion_report={promotion_json}")
    print(f"artifact_id={promotion['artifact_identity']['artifact_id']}")
    print(f"gates={len(promotion['gates'])}")
    return 0


def _run_cli(args: list[str], runner: Callable[[list[str]], int]) -> None:
    exit_code = runner(args)
    if exit_code != 0:
        raise RuntimeError(f"command failed with exit code {exit_code}: {args}")


if __name__ == "__main__":
    raise SystemExit(run())
