"""Comparison reports for C-MAPSS RUL model results."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CmapssModelComparisonRow:
    """One ranked C-MAPSS model result with deltas against the baseline."""

    subset: str
    phase: str
    model_name: str
    rank_by_nasa: int
    rmse: float
    nasa_score: float
    baseline_rmse: float
    baseline_nasa_score: float
    rmse_delta: float
    nasa_score_delta: float
    nasa_score_ratio: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a flat serialisable row."""

        return asdict(self)


def build_cmapss_model_comparison(
    baseline_csv: str | Path,
    candidate_csvs: tuple[str | Path, ...],
    *,
    baseline_label: str = "phase1_hgb_policy",
    candidate_label: str = "phase2_deep",
) -> list[CmapssModelComparisonRow]:
    """Build a ranked comparison table from baseline and candidate result CSVs."""

    if not candidate_csvs:
        raise ValueError("candidate_csvs must contain at least one path")

    baseline_results = _read_result_rows(baseline_csv)
    candidate_results = [
        result
        for candidate_csv in candidate_csvs
        for result in _read_result_rows(candidate_csv)
    ]
    baseline_by_subset = _single_result_by_subset(baseline_results, baseline_csv)
    grouped: dict[str, list[dict[str, Any]]] = {
        subset: [{**result, "phase": baseline_label}]
        for subset, result in baseline_by_subset.items()
    }
    for result in candidate_results:
        subset = str(result["subset"])
        if subset not in baseline_by_subset:
            raise ValueError(f"candidate result has no baseline for subset: {subset}")
        grouped.setdefault(subset, []).append({**result, "phase": candidate_label})

    comparison_rows: list[CmapssModelComparisonRow] = []
    for subset in sorted(grouped):
        baseline = baseline_by_subset[subset]
        baseline_rmse = float(baseline["rmse"])
        baseline_nasa_score = float(baseline["nasa_score"])
        ranked_results = sorted(
            grouped[subset],
            key=lambda result: (float(result["nasa_score"]), float(result["rmse"])),
        )
        for rank, result in enumerate(ranked_results, start=1):
            nasa_score = float(result["nasa_score"])
            comparison_rows.append(
                CmapssModelComparisonRow(
                    subset=subset,
                    phase=str(result["phase"]),
                    model_name=str(result["model_name"]),
                    rank_by_nasa=rank,
                    rmse=float(result["rmse"]),
                    nasa_score=nasa_score,
                    baseline_rmse=baseline_rmse,
                    baseline_nasa_score=baseline_nasa_score,
                    rmse_delta=float(result["rmse"]) - baseline_rmse,
                    nasa_score_delta=nasa_score - baseline_nasa_score,
                    nasa_score_ratio=(
                        nasa_score / baseline_nasa_score
                        if baseline_nasa_score != 0
                        else float("inf")
                    ),
                )
            )
    return comparison_rows


def write_cmapss_model_comparison_csv(
    rows: list[CmapssModelComparisonRow],
    path: str | Path,
) -> None:
    """Write comparison rows as CSV."""

    if not rows:
        raise ValueError("rows must contain at least one item")
    output_path = _prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def write_cmapss_model_comparison_markdown(
    rows: list[CmapssModelComparisonRow],
    path: str | Path,
) -> None:
    """Write comparison rows as a compact Markdown table."""

    output_path = _prepare_output_path(path)
    output_path.write_text(render_cmapss_model_comparison_markdown(rows), encoding="utf-8")


def render_cmapss_model_comparison_markdown(
    rows: list[CmapssModelComparisonRow],
) -> str:
    """Render comparison rows as a Markdown table."""

    if not rows:
        raise ValueError("rows must contain at least one item")
    lines = [
        "# C-MAPSS Model Comparison",
        "",
        (
            "| Subset | Rank | Phase | Model | RMSE | NASA Score | "
            "RMSE Delta | NASA Delta | NASA Ratio |"
        ),
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.subset} | "
            f"{row.rank_by_nasa} | "
            f"{row.phase} | "
            f"`{row.model_name}` | "
            f"{row.rmse:.6f} | "
            f"{row.nasa_score:.6f} | "
            f"{row.rmse_delta:.6f} | "
            f"{row.nasa_score_delta:.6f} | "
            f"{row.nasa_score_ratio:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _read_result_rows(path: str | Path) -> list[dict[str, Any]]:
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"missing result CSV: {result_path}")
    with result_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"result CSV has no rows: {result_path}")
    required_columns = {"subset", "model_name", "rmse", "nasa_score"}
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"result CSV is missing required columns: {missing}")
    return rows


def _single_result_by_subset(
    rows: list[dict[str, Any]],
    source_path: str | Path,
) -> dict[str, dict[str, Any]]:
    results_by_subset: dict[str, dict[str, Any]] = {}
    for row in rows:
        subset = str(row["subset"])
        if subset in results_by_subset:
            raise ValueError(f"baseline CSV has multiple rows for subset {subset}: {source_path}")
        results_by_subset[subset] = row
    return results_by_subset


def _prepare_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
