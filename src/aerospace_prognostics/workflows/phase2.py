"""Phase 2 C-MAPSS sequence-model workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
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
    run_cmapss_deep_baseline_comparison,
)
from aerospace_prognostics.reports.cmapss_model_comparison import (
    CmapssModelComparisonRow,
    build_cmapss_model_comparison,
    write_cmapss_model_comparison_csv,
    write_cmapss_model_comparison_markdown,
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
    comparison_csv_path: Path
    comparison_markdown_path: Path
    summary_markdown_path: Path
    sequence_exports: tuple[CmapssSequenceExportResult, ...]
    hgb_policy_results: tuple[RegressionRunResult, ...]
    deep_compare_results: tuple[RegressionRunResult, ...]
    comparison_rows: tuple[CmapssModelComparisonRow, ...]


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
    hidden_sizes: tuple[int, ...] = (32,),
    num_layers: int = 1,
    tcn_levels: int = 3,
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

    deep_compare_results = run_cmapss_deep_baseline_comparison(
        sequence_dir,
        subsets=subsets,
        models=models,
        epochs=epochs,
        batch_size=batch_size,
        learning_rates=learning_rates,
        hidden_sizes=hidden_sizes,
        num_layers=num_layers,
        tcn_levels=tcn_levels,
        transformer_heads=transformer_heads,
        transformer_dim_feedforward=transformer_dim_feedforward,
        kernel_size=kernel_size,
        dropout=dropout,
        checkpoint_policy=checkpoint_policy,
        random_state=random_state,
        device=device,
    )
    deep_compare_json_path = results_dir / "cmapss_deep_compare.json"
    deep_compare_csv_path = results_dir / "cmapss_deep_compare.csv"
    write_results_json(deep_compare_results, deep_compare_json_path)
    write_results_csv(deep_compare_results, deep_compare_csv_path)

    comparison_rows = build_cmapss_model_comparison(
        hgb_policy_csv_path,
        (deep_compare_csv_path,),
    )
    comparison_csv_path = results_dir / "cmapss_phase2_model_comparison.csv"
    comparison_markdown_path = results_dir / "cmapss_phase2_model_comparison.md"
    write_cmapss_model_comparison_csv(comparison_rows, comparison_csv_path)
    write_cmapss_model_comparison_markdown(comparison_rows, comparison_markdown_path)

    summary_markdown_path = artifacts / "phase2_summary.md"
    _write_phase2_summary(
        summary_markdown_path,
        sequence_dir=sequence_dir,
        hgb_policy_csv_path=hgb_policy_csv_path,
        deep_compare_csv_path=deep_compare_csv_path,
        comparison_markdown_path=comparison_markdown_path,
        sequence_exports=sequence_exports,
        comparison_rows=tuple(comparison_rows),
    )

    return Phase2WorkflowResult(
        artifact_dir=artifacts,
        sequence_dir=sequence_dir,
        hgb_policy_json_path=hgb_policy_json_path,
        hgb_policy_csv_path=hgb_policy_csv_path,
        deep_compare_json_path=deep_compare_json_path,
        deep_compare_csv_path=deep_compare_csv_path,
        comparison_csv_path=comparison_csv_path,
        comparison_markdown_path=comparison_markdown_path,
        summary_markdown_path=summary_markdown_path,
        sequence_exports=sequence_exports,
        hgb_policy_results=tuple(hgb_policy_results),
        deep_compare_results=tuple(deep_compare_results),
        comparison_rows=tuple(comparison_rows),
    )


def _write_phase2_summary(
    path: Path,
    *,
    sequence_dir: Path,
    hgb_policy_csv_path: Path,
    deep_compare_csv_path: Path,
    comparison_markdown_path: Path,
    sequence_exports: tuple[CmapssSequenceExportResult, ...],
    comparison_rows: tuple[CmapssModelComparisonRow, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 C-MAPSS Summary",
        "",
        f"- Sequence tensors: `{sequence_dir.as_posix()}`",
        f"- Phase 1 HGB policy baseline: `{hgb_policy_csv_path.as_posix()}`",
        f"- Phase 2 deep comparison table: `{deep_compare_csv_path.as_posix()}`",
        f"- Ranked model comparison: `{comparison_markdown_path.as_posix()}`",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
