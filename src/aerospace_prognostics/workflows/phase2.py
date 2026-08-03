"""Phase 2 C-MAPSS sequence-model workflow orchestration."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
from aerospace_prognostics.data.integrity import file_sha256
from aerospace_prognostics.evaluation import (
    RegressionRunResult,
    write_results_csv,
    write_results_json,
)
from aerospace_prognostics.experiments.cmapss_baseline import (
    run_all_cmapss_validation_selected_hgb_policy_default_windows,
)
from aerospace_prognostics.experiments.cmapss_deep_baseline import (
    CMAPSS_DEEP_COMPARISON_MODELS,
    run_cmapss_deep_baseline_comparison_runs,
    write_cmapss_deep_predictions_csv,
)
from aerospace_prognostics.reports.cmapss_model_comparison import (
    CmapssModelComparisonRow,
    build_cmapss_model_comparison,
    write_cmapss_model_comparison_csv,
    write_cmapss_model_comparison_markdown,
)
from aerospace_prognostics.reports.cmapss_prediction_diagnostics import (
    CmapssPredictionDiagnosticRow,
    CmapssPredictionMonotonicityDiagnosticRow,
    CmapssPredictionRulBinDiagnosticRow,
    CmapssPredictionUnitDiagnosticRow,
    build_cmapss_prediction_diagnostics,
    build_cmapss_prediction_monotonicity_diagnostics,
    build_cmapss_prediction_rul_bin_diagnostics,
    build_cmapss_prediction_unit_diagnostics,
    select_cmapss_high_error_predictions,
    write_cmapss_prediction_diagnostics_csv,
    write_cmapss_prediction_diagnostics_markdown,
    write_cmapss_prediction_monotonicity_diagnostics_csv,
    write_cmapss_prediction_rul_bin_diagnostics_csv,
    write_cmapss_prediction_unit_diagnostics_csv,
)
from aerospace_prognostics.sequence_exports import (
    CmapssSequenceExportResult,
    export_cmapss_sequence_splits,
)


@dataclass(frozen=True)
class Phase2WorkflowResult:
    """Artifact paths and run results from the Phase 2 workflow."""

    artifact_dir: Path
    sequence_dir: Path
    hgb_policy_json_path: Path
    hgb_policy_csv_path: Path
    deep_compare_json_path: Path
    deep_compare_csv_path: Path
    deep_predictions_csv_path: Path
    deep_validation_selection_predictions_csv_path: Path
    deep_prediction_diagnostics_csv_path: Path
    deep_validation_selection_prediction_diagnostics_csv_path: Path
    deep_prediction_rul_bin_diagnostics_csv_path: Path
    deep_validation_selection_prediction_rul_bin_diagnostics_csv_path: Path
    deep_prediction_monotonicity_diagnostics_csv_path: Path
    deep_validation_selection_prediction_monotonicity_diagnostics_csv_path: Path
    deep_prediction_unit_diagnostics_csv_path: Path
    deep_validation_selection_prediction_unit_diagnostics_csv_path: Path
    deep_prediction_diagnostics_markdown_path: Path
    deep_validation_selection_prediction_diagnostics_markdown_path: Path
    comparison_csv_path: Path
    comparison_markdown_path: Path
    summary_markdown_path: Path
    run_manifest_path: Path
    sequence_exports: tuple[CmapssSequenceExportResult, ...]
    hgb_policy_results: tuple[RegressionRunResult, ...]
    deep_compare_results: tuple[RegressionRunResult, ...]
    comparison_rows: tuple[CmapssModelComparisonRow, ...]


@dataclass(frozen=True)
class Phase2RunManifestVerification:
    """Verification result for a C-MAPSS Phase 2 run manifest."""

    manifest_path: Path
    checked_artifacts: tuple[Path, ...]
    problems: tuple[str, ...]
    artifact_root: Path = Path(".")
    manifest_payload: dict[str, object] | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


def run_phase2_cmapss_workflow(
    data_dir: str | Path,
    artifact_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    window_size: int = 30,
    stride: int = 1,
    validation_fraction: float = 0.2,
    validation_horizon: int = 30,
    rul_cap: int = 125,
    random_state: int = 42,
    n_regimes: int = 6,
    standardize: bool = True,
    models: tuple[str, ...] = ("cnn", "bilstm", "tcn", "transformer"),
    epochs: int = 5,
    batch_size: int = 256,
    learning_rates: tuple[float, ...] = (1e-3,),
    training_loss: str = "mse",
    hidden_sizes: tuple[int, ...] = (32,),
    num_layers: int = 1,
    tcn_levels: int = 3,
    tcn_normalization: str = "none",
    tcn_weight_norm: bool = False,
    tcn_pooling: str = "last",
    transformer_heads: int = 4,
    transformer_dim_feedforward: int | None = None,
    kernel_size: int = 3,
    dropout: float = 0.1,
    checkpoint_policy: str = "validation_nasa",
    device: str = "cpu",
) -> Phase2WorkflowResult:
    """Run the Phase 2 C-MAPSS sequence export, deep comparison, and report workflow."""
    unknown_models = sorted(set(models) - set(CMAPSS_DEEP_COMPARISON_MODELS))
    if unknown_models:
        raise ValueError(f"unknown Phase 2 workflow models: {', '.join(unknown_models)}")

    root = Path(data_dir)
    artifacts = Path(artifact_dir)
    results_dir = artifacts / "results"
    sequence_dir = artifacts / "sequences" / "cmapss"
    artifacts.mkdir(parents=True, exist_ok=True)

    sequence_exports = tuple(
        export_cmapss_sequence_splits(
            root,
            sequence_dir,
            subset,
            window_size=window_size,
            stride=stride,
            rul_cap=rul_cap,
            random_state=random_state,
            validation_fraction=validation_fraction,
            validation_horizon=validation_horizon,
            standardize=standardize,
        )
        for subset in subsets
    )

    hgb_policy_results = run_all_cmapss_validation_selected_hgb_policy_default_windows(
        root,
        subsets=subsets,
        rul_cap=rul_cap,
        random_state=random_state,
        n_regimes=n_regimes,
        standardize=standardize,
    )
    hgb_policy_json_path = results_dir / "cmapss_hgb_policy_baseline.json"
    hgb_policy_csv_path = results_dir / "cmapss_hgb_policy_baseline.csv"
    write_results_json(hgb_policy_results, hgb_policy_json_path)
    write_results_csv(hgb_policy_results, hgb_policy_csv_path)

    deep_compare_runs = run_cmapss_deep_baseline_comparison_runs(
        sequence_dir,
        subsets=subsets,
        models=models,
        epochs=epochs,
        batch_size=batch_size,
        learning_rates=learning_rates,
        training_loss=training_loss,
        hidden_sizes=hidden_sizes,
        num_layers=num_layers,
        tcn_levels=tcn_levels,
        tcn_normalization=tcn_normalization,
        tcn_weight_norm=tcn_weight_norm,
        tcn_pooling=tcn_pooling,
        transformer_heads=transformer_heads,
        transformer_dim_feedforward=transformer_dim_feedforward,
        kernel_size=kernel_size,
        dropout=dropout,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
    )
    deep_compare_results = [run.result for run in deep_compare_runs]
    deep_compare_json_path = results_dir / "cmapss_deep_compare.json"
    deep_compare_csv_path = results_dir / "cmapss_deep_compare.csv"
    deep_predictions_csv_path = results_dir / "cmapss_deep_predictions.csv"
    deep_validation_selection_predictions_csv_path = (
        results_dir / "cmapss_deep_validation_selection_predictions.csv"
    )
    write_results_json(deep_compare_results, deep_compare_json_path)
    write_results_csv(deep_compare_results, deep_compare_csv_path)
    write_cmapss_deep_predictions_csv(deep_compare_runs, deep_predictions_csv_path)
    write_cmapss_deep_predictions_csv(
        deep_compare_runs,
        deep_validation_selection_predictions_csv_path,
        prediction_split="validation_selection",
    )
    deep_prediction_diagnostics = build_cmapss_prediction_diagnostics(deep_predictions_csv_path)
    deep_validation_selection_prediction_diagnostics = build_cmapss_prediction_diagnostics(
        deep_validation_selection_predictions_csv_path
    )
    deep_prediction_rul_bin_diagnostics = build_cmapss_prediction_rul_bin_diagnostics(
        deep_predictions_csv_path
    )
    deep_validation_selection_prediction_rul_bin_diagnostics = (
        build_cmapss_prediction_rul_bin_diagnostics(deep_validation_selection_predictions_csv_path)
    )
    deep_prediction_monotonicity_diagnostics = (
        build_cmapss_prediction_monotonicity_diagnostics(deep_predictions_csv_path)
    )
    deep_validation_selection_prediction_monotonicity_diagnostics = (
        build_cmapss_prediction_monotonicity_diagnostics(
            deep_validation_selection_predictions_csv_path
        )
    )
    deep_prediction_unit_diagnostics = build_cmapss_prediction_unit_diagnostics(
        deep_predictions_csv_path
    )
    deep_validation_selection_prediction_unit_diagnostics = (
        build_cmapss_prediction_unit_diagnostics(deep_validation_selection_predictions_csv_path)
    )
    deep_prediction_outliers = select_cmapss_high_error_predictions(
        deep_predictions_csv_path,
        top_n=10,
    )
    deep_validation_selection_prediction_outliers = select_cmapss_high_error_predictions(
        deep_validation_selection_predictions_csv_path,
        top_n=10,
    )
    deep_prediction_diagnostics_csv_path = results_dir / "cmapss_deep_prediction_diagnostics.csv"
    deep_validation_selection_prediction_diagnostics_csv_path = (
        results_dir / "cmapss_deep_validation_selection_prediction_diagnostics.csv"
    )
    deep_prediction_rul_bin_diagnostics_csv_path = (
        results_dir / "cmapss_deep_prediction_rul_bins.csv"
    )
    deep_validation_selection_prediction_rul_bin_diagnostics_csv_path = (
        results_dir / "cmapss_deep_validation_selection_prediction_rul_bins.csv"
    )
    deep_prediction_monotonicity_diagnostics_csv_path = (
        results_dir / "cmapss_deep_prediction_monotonicity.csv"
    )
    deep_validation_selection_prediction_monotonicity_diagnostics_csv_path = (
        results_dir / "cmapss_deep_validation_selection_prediction_monotonicity.csv"
    )
    deep_prediction_unit_diagnostics_csv_path = (
        results_dir / "cmapss_deep_prediction_unit_diagnostics.csv"
    )
    deep_validation_selection_prediction_unit_diagnostics_csv_path = (
        results_dir / "cmapss_deep_validation_selection_prediction_unit_diagnostics.csv"
    )
    deep_prediction_diagnostics_markdown_path = (
        results_dir / "cmapss_deep_prediction_diagnostics.md"
    )
    deep_validation_selection_prediction_diagnostics_markdown_path = (
        results_dir / "cmapss_deep_validation_selection_prediction_diagnostics.md"
    )
    write_cmapss_prediction_diagnostics_csv(
        deep_prediction_diagnostics,
        deep_prediction_diagnostics_csv_path,
    )
    write_cmapss_prediction_diagnostics_csv(
        deep_validation_selection_prediction_diagnostics,
        deep_validation_selection_prediction_diagnostics_csv_path,
    )
    write_cmapss_prediction_rul_bin_diagnostics_csv(
        deep_prediction_rul_bin_diagnostics,
        deep_prediction_rul_bin_diagnostics_csv_path,
    )
    write_cmapss_prediction_rul_bin_diagnostics_csv(
        deep_validation_selection_prediction_rul_bin_diagnostics,
        deep_validation_selection_prediction_rul_bin_diagnostics_csv_path,
    )
    write_cmapss_prediction_monotonicity_diagnostics_csv(
        deep_prediction_monotonicity_diagnostics,
        deep_prediction_monotonicity_diagnostics_csv_path,
    )
    write_cmapss_prediction_monotonicity_diagnostics_csv(
        deep_validation_selection_prediction_monotonicity_diagnostics,
        deep_validation_selection_prediction_monotonicity_diagnostics_csv_path,
    )
    write_cmapss_prediction_unit_diagnostics_csv(
        deep_prediction_unit_diagnostics,
        deep_prediction_unit_diagnostics_csv_path,
    )
    write_cmapss_prediction_unit_diagnostics_csv(
        deep_validation_selection_prediction_unit_diagnostics,
        deep_validation_selection_prediction_unit_diagnostics_csv_path,
    )
    write_cmapss_prediction_diagnostics_markdown(
        deep_prediction_diagnostics,
        deep_prediction_outliers,
        deep_prediction_diagnostics_markdown_path,
        rul_bin_diagnostics=deep_prediction_rul_bin_diagnostics,
        monotonicity_diagnostics=deep_prediction_monotonicity_diagnostics,
        unit_diagnostics=deep_prediction_unit_diagnostics,
    )
    write_cmapss_prediction_diagnostics_markdown(
        deep_validation_selection_prediction_diagnostics,
        deep_validation_selection_prediction_outliers,
        deep_validation_selection_prediction_diagnostics_markdown_path,
        rul_bin_diagnostics=deep_validation_selection_prediction_rul_bin_diagnostics,
        monotonicity_diagnostics=deep_validation_selection_prediction_monotonicity_diagnostics,
        unit_diagnostics=deep_validation_selection_prediction_unit_diagnostics,
    )

    comparison_rows = build_cmapss_model_comparison(
        hgb_policy_csv_path,
        (deep_compare_csv_path,),
    )
    comparison_csv_path = results_dir / "cmapss_phase2_model_comparison.csv"
    comparison_markdown_path = results_dir / "cmapss_phase2_model_comparison.md"
    write_cmapss_model_comparison_csv(comparison_rows, comparison_csv_path)
    write_cmapss_model_comparison_markdown(comparison_rows, comparison_markdown_path)

    summary_markdown_path = artifacts / "phase2_summary.md"
    run_manifest_path = artifacts / "phase2_run_manifest.json"
    _write_phase2_summary(
        summary_markdown_path,
        sequence_dir=sequence_dir,
        hgb_policy_csv_path=hgb_policy_csv_path,
        deep_compare_csv_path=deep_compare_csv_path,
        deep_predictions_csv_path=deep_predictions_csv_path,
        deep_validation_selection_predictions_csv_path=(
            deep_validation_selection_predictions_csv_path
        ),
        deep_prediction_diagnostics_markdown_path=deep_prediction_diagnostics_markdown_path,
        deep_validation_selection_prediction_diagnostics_markdown_path=(
            deep_validation_selection_prediction_diagnostics_markdown_path
        ),
        comparison_markdown_path=comparison_markdown_path,
        run_manifest_path=run_manifest_path,
        sequence_exports=sequence_exports,
        deep_prediction_diagnostics=tuple(deep_prediction_diagnostics),
        deep_validation_selection_prediction_diagnostics=tuple(
            deep_validation_selection_prediction_diagnostics
        ),
        deep_prediction_rul_bin_diagnostics=tuple(deep_prediction_rul_bin_diagnostics),
        deep_validation_selection_prediction_rul_bin_diagnostics=tuple(
            deep_validation_selection_prediction_rul_bin_diagnostics
        ),
        deep_prediction_monotonicity_diagnostics=tuple(
            deep_prediction_monotonicity_diagnostics
        ),
        deep_validation_selection_prediction_monotonicity_diagnostics=tuple(
            deep_validation_selection_prediction_monotonicity_diagnostics
        ),
        deep_prediction_unit_diagnostics=tuple(deep_prediction_unit_diagnostics),
        deep_validation_selection_prediction_unit_diagnostics=tuple(
            deep_validation_selection_prediction_unit_diagnostics
        ),
        comparison_rows=tuple(comparison_rows),
        training_loss=training_loss,
    )
    artifact_paths = {
        "hgb_policy_json": _path_as_posix(hgb_policy_json_path),
        "hgb_policy_csv": _path_as_posix(hgb_policy_csv_path),
        "deep_compare_json": _path_as_posix(deep_compare_json_path),
        "deep_compare_csv": _path_as_posix(deep_compare_csv_path),
        "deep_predictions_csv": _path_as_posix(deep_predictions_csv_path),
        "deep_validation_selection_predictions_csv": _path_as_posix(
            deep_validation_selection_predictions_csv_path
        ),
        "deep_prediction_diagnostics_csv": _path_as_posix(deep_prediction_diagnostics_csv_path),
        "deep_validation_selection_prediction_diagnostics_csv": _path_as_posix(
            deep_validation_selection_prediction_diagnostics_csv_path
        ),
        "deep_prediction_rul_bin_diagnostics_csv": _path_as_posix(
            deep_prediction_rul_bin_diagnostics_csv_path
        ),
        "deep_validation_selection_prediction_rul_bin_diagnostics_csv": _path_as_posix(
            deep_validation_selection_prediction_rul_bin_diagnostics_csv_path
        ),
        "deep_prediction_monotonicity_diagnostics_csv": _path_as_posix(
            deep_prediction_monotonicity_diagnostics_csv_path
        ),
        "deep_validation_selection_prediction_monotonicity_diagnostics_csv": _path_as_posix(
            deep_validation_selection_prediction_monotonicity_diagnostics_csv_path
        ),
        "deep_prediction_unit_diagnostics_csv": _path_as_posix(
            deep_prediction_unit_diagnostics_csv_path
        ),
        "deep_validation_selection_prediction_unit_diagnostics_csv": _path_as_posix(
            deep_validation_selection_prediction_unit_diagnostics_csv_path
        ),
        "deep_prediction_diagnostics_markdown": _path_as_posix(
            deep_prediction_diagnostics_markdown_path
        ),
        "deep_validation_selection_prediction_diagnostics_markdown": _path_as_posix(
            deep_validation_selection_prediction_diagnostics_markdown_path
        ),
        "comparison_csv": _path_as_posix(comparison_csv_path),
        "comparison_markdown": _path_as_posix(comparison_markdown_path),
        "summary_markdown": _path_as_posix(summary_markdown_path),
        "run_manifest": _path_as_posix(run_manifest_path),
        **_sequence_artifact_paths(sequence_exports),
    }
    _write_phase2_run_manifest(
        run_manifest_path,
        {
            "workflow": "phase2_cmapss",
            "data_dir": _path_as_posix(root),
            "artifact_dir": _path_as_posix(artifacts),
            "runtime": _runtime_environment_payload(),
            "source_control": _source_control_payload(),
            "parameters": {
                "subsets": subsets,
                "window_size": window_size,
                "stride": stride,
                "validation_fraction": validation_fraction,
                "validation_horizon": validation_horizon,
                "rul_cap": rul_cap,
                "random_state": random_state,
                "n_regimes": n_regimes,
                "standardize": standardize,
                "models": models,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rates": learning_rates,
                "training_loss": training_loss,
                "hidden_sizes": hidden_sizes,
                "num_layers": num_layers,
                "tcn_levels": tcn_levels,
                "tcn_normalization": tcn_normalization,
                "tcn_weight_norm": tcn_weight_norm,
                "tcn_pooling": tcn_pooling,
                "transformer_heads": transformer_heads,
                "transformer_dim_feedforward": transformer_dim_feedforward,
                "kernel_size": kernel_size,
                "dropout": dropout,
                "checkpoint_policy": checkpoint_policy,
                "device": device,
            },
            "artifacts": artifact_paths,
            "artifact_integrity": _artifact_integrity_payload(artifact_paths),
            "counts": {
                "sequence_exports": len(sequence_exports),
                "hgb_policy_results": len(hgb_policy_results),
                "deep_compare_results": len(deep_compare_results),
                "deep_prediction_rows": _csv_data_row_count(deep_predictions_csv_path),
                "deep_validation_selection_prediction_rows": _csv_data_row_count(
                    deep_validation_selection_predictions_csv_path
                ),
                "deep_prediction_diagnostics": len(deep_prediction_diagnostics),
                "deep_validation_selection_prediction_diagnostics": len(
                    deep_validation_selection_prediction_diagnostics
                ),
                "deep_prediction_rul_bin_diagnostics": len(deep_prediction_rul_bin_diagnostics),
                "deep_validation_selection_prediction_rul_bin_diagnostics": len(
                    deep_validation_selection_prediction_rul_bin_diagnostics
                ),
                "deep_prediction_monotonicity_diagnostics": len(
                    deep_prediction_monotonicity_diagnostics
                ),
                "deep_validation_selection_prediction_monotonicity_diagnostics": len(
                    deep_validation_selection_prediction_monotonicity_diagnostics
                ),
                "deep_prediction_unit_diagnostics": len(deep_prediction_unit_diagnostics),
                "deep_validation_selection_prediction_unit_diagnostics": len(
                    deep_validation_selection_prediction_unit_diagnostics
                ),
                "comparison_rows": len(comparison_rows),
            },
        },
    )

    return Phase2WorkflowResult(
        artifact_dir=artifacts,
        sequence_dir=sequence_dir,
        hgb_policy_json_path=hgb_policy_json_path,
        hgb_policy_csv_path=hgb_policy_csv_path,
        deep_compare_json_path=deep_compare_json_path,
        deep_compare_csv_path=deep_compare_csv_path,
        deep_predictions_csv_path=deep_predictions_csv_path,
        deep_validation_selection_predictions_csv_path=(
            deep_validation_selection_predictions_csv_path
        ),
        deep_prediction_diagnostics_csv_path=deep_prediction_diagnostics_csv_path,
        deep_validation_selection_prediction_diagnostics_csv_path=(
            deep_validation_selection_prediction_diagnostics_csv_path
        ),
        deep_prediction_rul_bin_diagnostics_csv_path=deep_prediction_rul_bin_diagnostics_csv_path,
        deep_validation_selection_prediction_rul_bin_diagnostics_csv_path=(
            deep_validation_selection_prediction_rul_bin_diagnostics_csv_path
        ),
        deep_prediction_monotonicity_diagnostics_csv_path=(
            deep_prediction_monotonicity_diagnostics_csv_path
        ),
        deep_validation_selection_prediction_monotonicity_diagnostics_csv_path=(
            deep_validation_selection_prediction_monotonicity_diagnostics_csv_path
        ),
        deep_prediction_unit_diagnostics_csv_path=deep_prediction_unit_diagnostics_csv_path,
        deep_validation_selection_prediction_unit_diagnostics_csv_path=(
            deep_validation_selection_prediction_unit_diagnostics_csv_path
        ),
        deep_prediction_diagnostics_markdown_path=deep_prediction_diagnostics_markdown_path,
        deep_validation_selection_prediction_diagnostics_markdown_path=(
            deep_validation_selection_prediction_diagnostics_markdown_path
        ),
        comparison_csv_path=comparison_csv_path,
        comparison_markdown_path=comparison_markdown_path,
        summary_markdown_path=summary_markdown_path,
        run_manifest_path=run_manifest_path,
        sequence_exports=sequence_exports,
        hgb_policy_results=tuple(hgb_policy_results),
        deep_compare_results=tuple(deep_compare_results),
        comparison_rows=tuple(comparison_rows),
    )


def verify_phase2_cmapss_run_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path = ".",
) -> Phase2RunManifestVerification:
    """Verify that a C-MAPSS Phase 2 manifest points to a complete artifact bundle."""
    path = Path(manifest_path)
    artifact_root = Path(root)
    problems: list[str] = []
    checked_artifacts: list[Path] = []
    payload = _read_run_manifest_payload(path, problems)
    if payload is None:
        return Phase2RunManifestVerification(
            manifest_path=path,
            checked_artifacts=(),
            problems=tuple(problems),
            artifact_root=artifact_root,
        )

    if payload.get("workflow") != "phase2_cmapss":
        problems.append("workflow must be phase2_cmapss")
    for section in (
        "runtime",
        "source_control",
        "parameters",
        "artifacts",
        "artifact_integrity",
        "counts",
    ):
        if not isinstance(payload.get(section), dict):
            problems.append(f"{section} section is missing or invalid")

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        checked_artifacts.extend(_verify_manifest_artifacts(artifacts, artifact_root, problems))
    artifact_integrity = payload.get("artifact_integrity")
    if isinstance(artifacts, dict) and isinstance(artifact_integrity, dict):
        _verify_manifest_artifact_integrity(
            artifacts,
            artifact_integrity,
            artifact_root,
            problems,
        )
    counts = payload.get("counts")
    if isinstance(artifacts, dict) and isinstance(counts, dict):
        _verify_manifest_csv_counts(artifacts, counts, artifact_root, problems)

    return Phase2RunManifestVerification(
        manifest_path=path,
        checked_artifacts=tuple(checked_artifacts),
        problems=tuple(problems),
        artifact_root=artifact_root,
        manifest_payload=payload,
    )


def write_phase2_cmapss_manifest_audit_markdown(
    verification: Phase2RunManifestVerification,
    output_path: str | Path,
) -> Path:
    """Write a human-readable audit report for a C-MAPSS Phase 2 run manifest."""
    path = Path(output_path)
    payload = verification.manifest_payload or {}
    runtime = _manifest_section(payload, "runtime")
    source_control = _manifest_section(payload, "source_control")
    parameters = _manifest_section(payload, "parameters")
    counts = _manifest_section(payload, "counts")
    artifacts = _manifest_section(payload, "artifacts")
    artifact_integrity = _manifest_section(payload, "artifact_integrity")
    dependencies = _manifest_section(runtime, "dependencies")

    lines = [
        "# Phase 2 C-MAPSS Manifest Audit",
        "",
        f"- Status: {'ok' if verification.ok else 'failed'}",
        f"- Manifest: `{verification.manifest_path.as_posix()}`",
        f"- Workflow: {_markdown_inline(payload.get('workflow'))}",
        f"- Data directory: `{_markdown_inline(payload.get('data_dir'))}`",
        f"- Artifact directory: `{_markdown_inline(payload.get('artifact_dir'))}`",
        f"- Artifacts checked: {len(verification.checked_artifacts)}",
        "",
        "## Parameters",
        "",
        f"- Subsets: {_markdown_inline(parameters.get('subsets'))}",
        f"- Models: {_markdown_inline(parameters.get('models'))}",
        f"- Epochs: {_markdown_inline(parameters.get('epochs'))}",
        f"- Training loss: {_markdown_inline(parameters.get('training_loss'))}",
        f"- Checkpoint policy: {_markdown_inline(parameters.get('checkpoint_policy'))}",
        "",
        "## Runtime",
        "",
        f"- Python: {_markdown_inline(runtime.get('python_version'))}",
        f"- Platform: {_markdown_inline(runtime.get('platform'))}",
        f"- Project version: {_markdown_inline(runtime.get('project_version'))}",
        f"- Git branch: {_markdown_inline(source_control.get('git_branch'))}",
        f"- Git commit: {_markdown_inline(source_control.get('git_commit'))}",
        f"- Git dirty: {_markdown_inline(source_control.get('git_dirty'))}",
        "",
        "## Dependencies",
        "",
        "| Package | Version |",
        "| --- | --- |",
    ]
    for name, version in sorted(dependencies.items()):
        lines.append(f"| {_markdown_cell(name)} | {_markdown_cell(version)} |")

    lines.extend(
        [
            "",
            "## Counts",
            "",
            "| Metric | Count |",
            "| --- | ---: |",
        ]
    )
    for key, value in sorted(counts.items()):
        lines.append(f"| {_markdown_cell(key)} | {_markdown_cell(value)} |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Key | Exists | Size Bytes | SHA-256 | Path |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for key, value in sorted(artifacts.items()):
        if value is None:
            continue
        exists = "unknown"
        if isinstance(value, str):
            artifact_path = _resolve_manifest_path(value, verification.artifact_root)
            exists = "yes" if artifact_path.exists() else "no"
        integrity = artifact_integrity.get(key)
        size_bytes = ""
        sha256 = ""
        if isinstance(integrity, dict):
            size_bytes = _markdown_inline(integrity.get("size_bytes"), default="")
            sha256_value = integrity.get("sha256")
            sha256 = sha256_value[:12] if isinstance(sha256_value, str) else ""
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(key),
                    _markdown_cell(exists),
                    _markdown_cell(size_bytes),
                    _markdown_cell(sha256),
                    _markdown_cell(value),
                )
            )
            + " |"
        )

    lines.extend(["", "## Problems", ""])
    if verification.problems:
        lines.extend(f"- {problem}" for problem in verification.problems)
    else:
        lines.append("- None")

    output_path = prepare_output_path(path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_phase2_summary(
    path: Path,
    *,
    sequence_dir: Path,
    hgb_policy_csv_path: Path,
    deep_compare_csv_path: Path,
    deep_predictions_csv_path: Path,
    deep_validation_selection_predictions_csv_path: Path,
    deep_prediction_diagnostics_markdown_path: Path,
    deep_validation_selection_prediction_diagnostics_markdown_path: Path,
    comparison_markdown_path: Path,
    run_manifest_path: Path,
    sequence_exports: tuple[CmapssSequenceExportResult, ...],
    deep_prediction_diagnostics: tuple[CmapssPredictionDiagnosticRow, ...],
    deep_validation_selection_prediction_diagnostics: tuple[CmapssPredictionDiagnosticRow, ...],
    deep_prediction_rul_bin_diagnostics: tuple[CmapssPredictionRulBinDiagnosticRow, ...],
    deep_validation_selection_prediction_rul_bin_diagnostics: tuple[
        CmapssPredictionRulBinDiagnosticRow, ...
    ],
    deep_prediction_monotonicity_diagnostics: tuple[
        CmapssPredictionMonotonicityDiagnosticRow, ...
    ],
    deep_validation_selection_prediction_monotonicity_diagnostics: tuple[
        CmapssPredictionMonotonicityDiagnosticRow, ...
    ],
    deep_prediction_unit_diagnostics: tuple[CmapssPredictionUnitDiagnosticRow, ...],
    deep_validation_selection_prediction_unit_diagnostics: tuple[
        CmapssPredictionUnitDiagnosticRow, ...
    ],
    comparison_rows: tuple[CmapssModelComparisonRow, ...],
    training_loss: str,
) -> None:
    output_path = prepare_output_path(path)
    lines = [
        "# Phase 2 C-MAPSS Summary",
        "",
        f"- Training loss: {training_loss}",
        f"- Sequence tensors: `{sequence_dir.as_posix()}`",
        f"- Phase 1 HGB policy baseline: `{hgb_policy_csv_path.as_posix()}`",
        f"- Phase 2 deep comparison table: `{deep_compare_csv_path.as_posix()}`",
        f"- Phase 2 deep prediction diagnostics: `{deep_predictions_csv_path.as_posix()}`",
        (
            "- Phase 2 validation-selection prediction diagnostics: "
            f"`{deep_validation_selection_predictions_csv_path.as_posix()}`"
        ),
        (
            "- Phase 2 prediction diagnostics report: "
            f"`{deep_prediction_diagnostics_markdown_path.as_posix()}`"
        ),
        (
            "- Phase 2 validation-selection diagnostics report: "
            f"`{deep_validation_selection_prediction_diagnostics_markdown_path.as_posix()}`"
        ),
        f"- Ranked model comparison: `{comparison_markdown_path.as_posix()}`",
        f"- Run manifest: `{run_manifest_path.as_posix()}`",
        "",
        "## Sequence Exports",
        "",
        "| Subset | Train Windows | Validation Selection Windows | Test Windows | Features |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in sequence_exports:
        lines.append(
            f"| {result.subset} | {result.train_windows} | "
            f"{result.validation_selection_windows} | {result.test_windows} | "
            f"{len(result.feature_columns)} |"
        )
    lines.extend(
        [
            "",
            "## Deep Prediction Diagnostics",
            "",
            "| Subset | Model | Rows | Mean Error | Mean Abs Error | Max Abs Error | Late Rate |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deep_prediction_diagnostics:
        lines.append(
            f"| {row.subset} | `{row.model_name}` | {row.prediction_count} | "
            f"{row.mean_error:.6f} | {row.mean_absolute_error:.6f} | "
            f"{row.max_absolute_error:.6f} | {row.late_prediction_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Deep Prediction RUL Bins",
            "",
            "| Subset | Actual RUL Bin | Rows | Mean Error | Mean Abs Error | Late Rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in deep_prediction_rul_bin_diagnostics:
        lines.append(
            f"| {row.subset} | {row.actual_rul_bin} | {row.prediction_count} | "
            f"{row.mean_error:.6f} | {row.mean_absolute_error:.6f} | "
            f"{row.late_prediction_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Deep Prediction Monotonicity",
            "",
            (
                "| Subset | Model | Units | Transitions | Violations | Violation Rate | "
                "Max Violation |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deep_prediction_monotonicity_diagnostics:
        lines.append(
            f"| {row.subset} | `{row.model_name}` | {row.unit_count} | "
            f"{row.transition_count} | {row.violation_count} | "
            f"{row.violation_rate:.6f} | {row.max_violation_magnitude:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Deep Prediction Highest-Error Units",
            "",
            (
                "| Subset | Model | Unit | Rows | Mean Error | Mean Abs Error | "
                "Max Abs Error | Late Rate |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        deep_prediction_unit_diagnostics,
        key=lambda item: (
            -item.max_absolute_error,
            -item.mean_absolute_error,
            item.subset,
            item.model_name,
            item.unit_number,
        ),
    )[:10]:
        lines.append(
            f"| {row.subset} | `{row.model_name}` | {row.unit_number} | "
            f"{row.prediction_count} | {row.mean_error:.6f} | "
            f"{row.mean_absolute_error:.6f} | {row.max_absolute_error:.6f} | "
            f"{row.late_prediction_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation Selection Prediction Diagnostics",
            "",
            "| Subset | Model | Rows | Mean Error | Mean Abs Error | Max Abs Error | Late Rate |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deep_validation_selection_prediction_diagnostics:
        lines.append(
            f"| {row.subset} | `{row.model_name}` | {row.prediction_count} | "
            f"{row.mean_error:.6f} | {row.mean_absolute_error:.6f} | "
            f"{row.max_absolute_error:.6f} | {row.late_prediction_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation Selection RUL Bins",
            "",
            "| Subset | Actual RUL Bin | Rows | Mean Error | Mean Abs Error | Late Rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in deep_validation_selection_prediction_rul_bin_diagnostics:
        lines.append(
            f"| {row.subset} | {row.actual_rul_bin} | {row.prediction_count} | "
            f"{row.mean_error:.6f} | {row.mean_absolute_error:.6f} | "
            f"{row.late_prediction_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation Selection Prediction Monotonicity",
            "",
            (
                "| Subset | Model | Units | Transitions | Violations | Violation Rate | "
                "Max Violation |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deep_validation_selection_prediction_monotonicity_diagnostics:
        lines.append(
            f"| {row.subset} | `{row.model_name}` | {row.unit_count} | "
            f"{row.transition_count} | {row.violation_count} | "
            f"{row.violation_rate:.6f} | {row.max_violation_magnitude:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation Selection Highest-Error Units",
            "",
            (
                "| Subset | Model | Unit | Rows | Mean Error | Mean Abs Error | "
                "Max Abs Error | Late Rate | Violation Rate |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        deep_validation_selection_prediction_unit_diagnostics,
        key=lambda item: (
            -item.max_absolute_error,
            -item.mean_absolute_error,
            item.subset,
            item.model_name,
            item.unit_number,
        ),
    )[:10]:
        lines.append(
            f"| {row.subset} | `{row.model_name}` | {row.unit_number} | "
            f"{row.prediction_count} | {row.mean_error:.6f} | "
            f"{row.mean_absolute_error:.6f} | {row.max_absolute_error:.6f} | "
            f"{row.late_prediction_rate:.6f} | {row.violation_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Best Model By NASA Score",
            "",
            "| Subset | Phase | Model | RMSE | NASA Score | NASA Delta vs HGB |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for subset in sorted({row.subset for row in comparison_rows}):
        best = min(
            (row for row in comparison_rows if row.subset == subset),
            key=lambda row: (row.nasa_score, row.rmse),
        )
        lines.append(
            f"| {best.subset} | {best.phase} | {best.model_name} | "
            f"{best.rmse:.6f} | {best.nasa_score:.6f} | "
            f"{best.nasa_score_delta:.6f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sequence_artifact_paths(
    sequence_exports: tuple[CmapssSequenceExportResult, ...],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for result in sequence_exports:
        prefix = f"sequence_{result.subset.lower()}"
        artifacts.update(
            {
                f"{prefix}_metadata": _path_as_posix(result.metadata_path),
                f"{prefix}_train_npz": _path_as_posix(result.train_npz_path),
                f"{prefix}_validation_npz": _path_as_posix(result.validation_npz_path),
                f"{prefix}_validation_selection_npz": _path_as_posix(
                    result.validation_selection_npz_path
                ),
                f"{prefix}_test_npz": _path_as_posix(result.test_npz_path),
            }
        )
    return artifacts


def _write_phase2_run_manifest(path: Path, payload: dict[str, object]) -> None:
    write_json_payload(payload, path)


def _read_run_manifest_payload(path: Path, problems: list[str]) -> dict[str, object] | None:
    if not path.exists():
        problems.append(f"manifest is missing: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"manifest is not valid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        problems.append("manifest root must be an object")
        return None
    return payload


def _verify_manifest_artifacts(
    artifacts: dict[object, object],
    root: Path,
    problems: list[str],
) -> tuple[Path, ...]:
    checked_paths: list[Path] = []
    for key, value in sorted(artifacts.items()):
        if value is None:
            continue
        if not isinstance(value, str):
            problems.append(f"artifact {key} must be a string path or null")
            continue
        artifact_path = _resolve_manifest_path(value, root)
        checked_paths.append(artifact_path)
        if not artifact_path.exists():
            problems.append(f"artifact {key} is missing: {artifact_path}")
    return tuple(checked_paths)


def _verify_manifest_artifact_integrity(
    artifacts: dict[object, object],
    artifact_integrity: dict[object, object],
    root: Path,
    problems: list[str],
) -> None:
    for key, value in sorted(artifacts.items()):
        if key == "run_manifest" or value is None:
            continue
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        expected = artifact_integrity.get(key)
        if not isinstance(expected, dict):
            problems.append(f"artifact_integrity missing for {key}")
            continue
        artifact_path = _resolve_manifest_path(value, root)
        if not artifact_path.exists():
            continue
        expected_sha256 = expected.get("sha256")
        expected_size_bytes = expected.get("size_bytes")
        if not isinstance(expected_sha256, str):
            problems.append(f"artifact_integrity {key} sha256 is missing or invalid")
        elif file_sha256(artifact_path) != expected_sha256:
            problems.append(f"artifact {key} has unexpected sha256")
        if not isinstance(expected_size_bytes, int):
            problems.append(f"artifact_integrity {key} size_bytes is missing or invalid")
        elif artifact_path.stat().st_size != expected_size_bytes:
            problems.append(f"artifact {key} has unexpected size")


def _verify_manifest_csv_counts(
    artifacts: dict[object, object],
    counts: dict[object, object],
    root: Path,
    problems: list[str],
) -> None:
    count_checks = {
        "hgb_policy_csv": "hgb_policy_results",
        "deep_compare_csv": "deep_compare_results",
        "deep_predictions_csv": "deep_prediction_rows",
        "deep_validation_selection_predictions_csv": ("deep_validation_selection_prediction_rows"),
        "deep_prediction_diagnostics_csv": "deep_prediction_diagnostics",
        "deep_validation_selection_prediction_diagnostics_csv": (
            "deep_validation_selection_prediction_diagnostics"
        ),
        "deep_prediction_rul_bin_diagnostics_csv": "deep_prediction_rul_bin_diagnostics",
        "deep_validation_selection_prediction_rul_bin_diagnostics_csv": (
            "deep_validation_selection_prediction_rul_bin_diagnostics"
        ),
        "deep_prediction_monotonicity_diagnostics_csv": (
            "deep_prediction_monotonicity_diagnostics"
        ),
        "deep_validation_selection_prediction_monotonicity_diagnostics_csv": (
            "deep_validation_selection_prediction_monotonicity_diagnostics"
        ),
        "deep_prediction_unit_diagnostics_csv": "deep_prediction_unit_diagnostics",
        "deep_validation_selection_prediction_unit_diagnostics_csv": (
            "deep_validation_selection_prediction_unit_diagnostics"
        ),
        "comparison_csv": "comparison_rows",
    }
    for artifact_key, count_key in count_checks.items():
        path_value = artifacts.get(artifact_key)
        expected_count = counts.get(count_key)
        if path_value is None or expected_count is None:
            continue
        if not isinstance(path_value, str) or not isinstance(expected_count, int):
            continue
        artifact_path = _resolve_manifest_path(path_value, root)
        if not artifact_path.exists():
            continue
        row_count = _csv_data_row_count(artifact_path)
        if row_count != expected_count:
            problems.append(
                f"{artifact_key} has {row_count} rows; expected {expected_count} from {count_key}"
            )


def _csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        return sum(1 for _ in reader)


def _artifact_integrity_payload(artifacts: dict[str, str | None]) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for key, value in sorted(artifacts.items()):
        if key == "run_manifest" or value is None:
            continue
        path = Path(value)
        if not path.exists():
            continue
        payload[key] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return payload


def _runtime_environment_payload() -> dict[str, object]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "project_version": _package_version("aerospace-prognostics"),
        "dependencies": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "torch")
        },
    }


def _source_control_payload() -> dict[str, object]:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_dirty": _git_dirty(),
    }


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_dirty() -> bool | None:
    output = _git_output("status", "--short")
    if output is None:
        return None
    return bool(output)


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=_repository_root(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _manifest_section(payload: Mapping[object, object], key: str) -> dict[object, object]:
    section = payload.get(key)
    return section if isinstance(section, dict) else {}


def _markdown_inline(value: object, *, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _markdown_cell(value: object) -> str:
    return _markdown_inline(value, default="").replace("|", "\\|")


def _resolve_manifest_path(path: str, root: Path) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return artifact_path
    return root / artifact_path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_as_posix(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.as_posix()
