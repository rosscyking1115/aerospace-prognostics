"""Comparison reports for C-MAPSS RUL model results."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aerospace_prognostics.metrics import nasa_rul_score, rmse


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
    candidate_csvs: tuple[str | Path, ...] = (),
    *,
    baseline_label: str = "phase1_hgb_policy",
    candidate_label: str = "phase2_deep",
    prediction_csvs: tuple[str | Path, ...] = (),
    prediction_label: str = "phase2_predictions",
    prediction_model_suffixes: tuple[str, ...] = (),
) -> list[CmapssModelComparisonRow]:
    """Build a ranked comparison table from baseline and candidate result CSVs."""

    if not candidate_csvs and not prediction_csvs:
        raise ValueError("candidate_csvs or prediction_csvs must contain at least one path")
    if prediction_model_suffixes and len(prediction_model_suffixes) != len(prediction_csvs):
        raise ValueError("prediction_model_suffixes must match prediction_csvs length")

    baseline_results = _read_result_rows(baseline_csv)
    candidate_results = [
        result
        for candidate_csv in candidate_csvs
        for result in _read_result_rows(candidate_csv)
    ]
    prediction_results = [
        result
        for index, prediction_csv in enumerate(prediction_csvs)
        for result in _read_prediction_result_rows(
            prediction_csv,
            model_name_suffix=(
                prediction_model_suffixes[index] if prediction_model_suffixes else None
            ),
        )
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
    for result in prediction_results:
        subset = str(result["subset"])
        if subset not in baseline_by_subset:
            raise ValueError(f"prediction result has no baseline for subset: {subset}")
        grouped.setdefault(subset, []).append({**result, "phase": prediction_label})

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


def _read_prediction_result_rows(
    path: str | Path,
    *,
    model_name_suffix: str | None = None,
) -> list[dict[str, Any]]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"missing prediction CSV: {prediction_path}")
    with prediction_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"prediction CSV has no rows: {prediction_path}")
    required_columns = {"subset", "model_name", "actual_rul", "predicted_rul"}
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"prediction CSV is missing required columns: {missing}")

    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        subset = str(row["subset"])
        model_name = _prediction_model_name(str(row["model_name"]), model_name_suffix)
        group = grouped.setdefault((subset, model_name), {"actual": [], "predicted": []})
        group["actual"].append(float(row["actual_rul"]))
        group["predicted"].append(float(row["predicted_rul"]))

    return [
        {
            "subset": subset,
            "model_name": model_name,
            "rmse": rmse(values["actual"], values["predicted"]),
            "nasa_score": nasa_rul_score(values["actual"], values["predicted"]),
        }
        for (subset, model_name), values in sorted(grouped.items())
    ]


def _prediction_model_name(model_name: str, suffix: str | None) -> str:
    if suffix is None or suffix == "":
        return model_name
    return f"{model_name}_{suffix}"


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
