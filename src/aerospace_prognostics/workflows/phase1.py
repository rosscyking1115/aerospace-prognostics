"""Phase 1 C-MAPSS workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aerospace_prognostics.analysis.cmapss_eda import build_cmapss_eda_report
from aerospace_prognostics.artifact_io import prepare_output_path
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS, load_cmapss_subset
from aerospace_prognostics.data.manifest import build_cmapss_manifest, verify_manifest
from aerospace_prognostics.evaluation import (
    RegressionRunResult,
    write_results_csv,
    write_results_json,
)
from aerospace_prognostics.experiments.cmapss_baseline import (
    run_all_cmapss_hist_gradient_boosting,
    run_all_cmapss_validation_selected_hgb_policy_default_windows,
    run_cmapss_validation_sensor_filter_comparison,
)


@dataclass(frozen=True)
class Phase1WorkflowResult:
    """Artifact paths and run results from the Phase 1 workflow."""

    artifact_dir: Path
    manifest_path: Path
    baseline_json_path: Path
    baseline_csv_path: Path
    hgb_policy_json_path: Path
    hgb_policy_csv_path: Path
    sensor_filter_json_path: Path
    sensor_filter_csv_path: Path
    summary_markdown_path: Path
    eda_paths: tuple[Path, ...]
    baseline_results: tuple[RegressionRunResult, ...]
    hgb_policy_results: tuple[RegressionRunResult, ...]
    sensor_filter_results: tuple[RegressionRunResult, ...]


def run_phase1_cmapss_workflow(
    data_dir: str | Path,
    artifact_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    rul_cap: int = 125,
    random_state: int = 42,
    n_regimes: int = 6,
    standardize: bool = True,
) -> Phase1WorkflowResult:
    """Run the Phase 1 C-MAPSS provenance, EDA, and baseline workflow."""
    root = Path(data_dir)
    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    manifest = build_cmapss_manifest(root, subsets=subsets)
    verification_problems = verify_manifest(manifest, root=root)
    if verification_problems:
        raise ValueError(f"manifest verification failed: {verification_problems}")

    manifest_path = artifacts / "data" / "cmapss_manifest.json"
    manifest.write_json(manifest_path)

    eda_dir = artifacts / "eda"
    eda_paths: list[Path] = []
    for subset in subsets:
        bundle = load_cmapss_subset(root, subset, rul_cap=rul_cap)
        report = build_cmapss_eda_report(bundle)
        output_path = eda_dir / f"{subset.lower()}.json"
        report.write_json(output_path)
        eda_paths.append(output_path)

    baseline_results = run_all_cmapss_hist_gradient_boosting(
        root,
        subsets=subsets,
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
    )
    baseline_json_path = artifacts / "results" / "cmapss_baseline.json"
    baseline_csv_path = artifacts / "results" / "cmapss_baseline.csv"
    write_results_json(baseline_results, baseline_json_path)
    write_results_csv(baseline_results, baseline_csv_path)

    hgb_policy_results = run_all_cmapss_validation_selected_hgb_policy_default_windows(
        root,
        subsets=subsets,
        rul_cap=rul_cap,
        random_state=random_state,
        n_regimes=n_regimes,
        standardize=standardize,
    )
    hgb_policy_json_path = artifacts / "results" / "cmapss_hgb_policy_baseline.json"
    hgb_policy_csv_path = artifacts / "results" / "cmapss_hgb_policy_baseline.csv"
    write_results_json(hgb_policy_results, hgb_policy_json_path)
    write_results_csv(hgb_policy_results, hgb_policy_csv_path)

    sensor_filter_results = run_cmapss_validation_sensor_filter_comparison(
        root,
        subsets=subsets,
        rul_cap=rul_cap,
        random_state=random_state,
        n_regimes=n_regimes,
        standardize=standardize,
    )
    sensor_filter_json_path = artifacts / "results" / "cmapss_validation_sensor_filters.json"
    sensor_filter_csv_path = artifacts / "results" / "cmapss_validation_sensor_filters.csv"
    write_results_json(sensor_filter_results, sensor_filter_json_path)
    write_results_csv(sensor_filter_results, sensor_filter_csv_path)

    summary_markdown_path = artifacts / "phase1_summary.md"
    _write_phase1_summary(
        summary_markdown_path,
        manifest_path=manifest_path,
        baseline_csv_path=baseline_csv_path,
        hgb_policy_csv_path=hgb_policy_csv_path,
        sensor_filter_csv_path=sensor_filter_csv_path,
        eda_paths=tuple(eda_paths),
        baseline_results=tuple(baseline_results),
        hgb_policy_results=tuple(hgb_policy_results),
        sensor_filter_results=tuple(sensor_filter_results),
    )

    return Phase1WorkflowResult(
        artifact_dir=artifacts,
        manifest_path=manifest_path,
        baseline_json_path=baseline_json_path,
        baseline_csv_path=baseline_csv_path,
        hgb_policy_json_path=hgb_policy_json_path,
        hgb_policy_csv_path=hgb_policy_csv_path,
        sensor_filter_json_path=sensor_filter_json_path,
        sensor_filter_csv_path=sensor_filter_csv_path,
        summary_markdown_path=summary_markdown_path,
        eda_paths=tuple(eda_paths),
        baseline_results=tuple(baseline_results),
        hgb_policy_results=tuple(hgb_policy_results),
        sensor_filter_results=tuple(sensor_filter_results),
    )


def _write_phase1_summary(
    path: Path,
    *,
    manifest_path: Path,
    baseline_csv_path: Path,
    hgb_policy_csv_path: Path,
    sensor_filter_csv_path: Path,
    eda_paths: tuple[Path, ...],
    baseline_results: tuple[RegressionRunResult, ...],
    hgb_policy_results: tuple[RegressionRunResult, ...],
    sensor_filter_results: tuple[RegressionRunResult, ...],
) -> None:
    output_path = prepare_output_path(path)
    lines = [
        "# Phase 1 C-MAPSS Summary",
        "",
        f"- Manifest: `{manifest_path.as_posix()}`",
        f"- Raw baseline table: `{baseline_csv_path.as_posix()}`",
        f"- Current HGB policy baseline table: `{hgb_policy_csv_path.as_posix()}`",
        f"- Sensor-filter validation table: `{sensor_filter_csv_path.as_posix()}`",
        f"- EDA reports: {len(eda_paths)}",
        "",
        "## Current Phase 1 Baseline",
        "",
        "| Subset | Model | Standardized | RMSE | NASA Score |",
        "|---|---|---:|---:|---:|",
    ]
    for result in hgb_policy_results:
        lines.append(
            f"| {result.subset} | {result.model_name} | {result.standardize} | "
            f"{result.rmse:.6f} | {result.nasa_score:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Raw-Cycle Sanity Baseline",
            "",
            "| Subset | Model | Standardized | RMSE | NASA Score |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for result in baseline_results:
        lines.append(
            f"| {result.subset} | {result.model_name} | {result.standardize} | "
            f"{result.rmse:.6f} | {result.nasa_score:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Sensor-Filter Validation",
            "",
            "| Subset | Model | Standardized | RMSE | NASA Score |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for result in sensor_filter_results:
        lines.append(
            f"| {result.subset} | {result.model_name} | {result.standardize} | "
            f"{result.rmse:.6f} | {result.nasa_score:.6f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
