"""Comparison reports for telemetry anomaly-detection model results."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnomalyModelComparisonRow:
    """One ranked anomaly-detection result for a telemetry channel."""

    channel_id: str
    spacecraft: str
    source: str
    model_name: str
    rank_by_f1: int
    precision: float
    recall: float
    f1: float
    point_adjusted_f1: float
    false_alarm_rate: float
    miss_rate: float
    support: int
    predicted_positives: int
    train_rows: int | None
    test_rows: int | None
    anomaly_points: int | None

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Return a flat serialisable row."""

        return asdict(self)


def build_anomaly_model_comparison(
    result_csvs: tuple[str | Path, ...],
    *,
    source_labels: tuple[str, ...] | None = None,
) -> list[AnomalyModelComparisonRow]:
    """Build a ranked comparison table from anomaly result CSVs."""

    if not result_csvs:
        raise ValueError("result_csvs must contain at least one path")
    labels = source_labels or tuple(Path(path).stem for path in result_csvs)
    if len(labels) != len(result_csvs):
        raise ValueError("source_labels length must match result_csvs length")

    rows = [
        _comparison_row_from_csv_row(row, source=label)
        for path, label in zip(result_csvs, labels, strict=True)
        for row in _read_anomaly_rows(path)
    ]
    grouped: dict[str, list[AnomalyModelComparisonRow]] = {}
    for row in rows:
        grouped.setdefault(row.channel_id, []).append(row)

    ranked_rows: list[AnomalyModelComparisonRow] = []
    for channel_id in sorted(grouped):
        channel_rows = sorted(
            grouped[channel_id],
            key=lambda row: (
                -row.f1,
                -row.point_adjusted_f1,
                row.false_alarm_rate,
                row.miss_rate,
                row.model_name,
            ),
        )
        for rank, row in enumerate(channel_rows, start=1):
            ranked_rows.append(
                AnomalyModelComparisonRow(
                    channel_id=row.channel_id,
                    spacecraft=row.spacecraft,
                    source=row.source,
                    model_name=row.model_name,
                    rank_by_f1=rank,
                    precision=row.precision,
                    recall=row.recall,
                    f1=row.f1,
                    point_adjusted_f1=row.point_adjusted_f1,
                    false_alarm_rate=row.false_alarm_rate,
                    miss_rate=row.miss_rate,
                    support=row.support,
                    predicted_positives=row.predicted_positives,
                    train_rows=row.train_rows,
                    test_rows=row.test_rows,
                    anomaly_points=row.anomaly_points,
                )
            )
    return ranked_rows


def write_anomaly_model_comparison_csv(
    rows: list[AnomalyModelComparisonRow],
    path: str | Path,
) -> None:
    """Write anomaly comparison rows as CSV."""

    if not rows:
        raise ValueError("rows must contain at least one item")
    output_path = _prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def write_anomaly_model_comparison_markdown(
    rows: list[AnomalyModelComparisonRow],
    path: str | Path,
) -> None:
    """Write anomaly comparison rows as a compact Markdown table."""

    output_path = _prepare_output_path(path)
    output_path.write_text(render_anomaly_model_comparison_markdown(rows), encoding="utf-8")


def render_anomaly_model_comparison_markdown(
    rows: list[AnomalyModelComparisonRow],
) -> str:
    """Render anomaly comparison rows as a Markdown table."""

    if not rows:
        raise ValueError("rows must contain at least one item")
    lines = [
        "# Telemetry Anomaly Model Comparison",
        "",
        (
            "| Channel | Spacecraft | Rank | Source | Model | F1 | Point-Adjusted F1 | "
            "Precision | Recall | False Alarm Rate | Miss Rate |"
        ),
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.channel_id} | "
            f"{row.spacecraft} | "
            f"{row.rank_by_f1} | "
            f"{row.source} | "
            f"`{row.model_name}` | "
            f"{row.f1:.6f} | "
            f"{row.point_adjusted_f1:.6f} | "
            f"{row.precision:.6f} | "
            f"{row.recall:.6f} | "
            f"{row.false_alarm_rate:.6f} | "
            f"{row.miss_rate:.6f} |"
        )
    _append_aggregate_table(
        lines,
        rows=_best_rows_by_channel(rows),
        heading="Winner Counts",
        count_header="Channels Won",
    )
    _append_aggregate_table(
        lines,
        rows=tuple(rows),
        heading="Average Metrics By Source And Model",
        count_header="Rows",
    )
    return "\n".join(lines) + "\n"


def _read_anomaly_rows(path: str | Path) -> list[dict[str, Any]]:
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"missing anomaly result CSV: {result_path}")
    with result_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"anomaly result CSV has no rows: {result_path}")
    required_columns = {
        "channel_id",
        "spacecraft",
        "model_name",
        "precision",
        "recall",
        "f1",
        "point_adjusted_f1",
        "false_alarm_rate",
        "miss_rate",
        "support",
        "predicted_positives",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"anomaly result CSV is missing required columns: {missing}")
    return rows


def _comparison_row_from_csv_row(
    row: dict[str, str],
    *,
    source: str,
) -> AnomalyModelComparisonRow:
    return AnomalyModelComparisonRow(
        channel_id=row["channel_id"],
        spacecraft=row["spacecraft"],
        source=source,
        model_name=row["model_name"],
        rank_by_f1=0,
        precision=float(row["precision"]),
        recall=float(row["recall"]),
        f1=float(row["f1"]),
        point_adjusted_f1=float(row["point_adjusted_f1"]),
        false_alarm_rate=float(row["false_alarm_rate"]),
        miss_rate=float(row["miss_rate"]),
        support=int(row["support"]),
        predicted_positives=int(row["predicted_positives"]),
        train_rows=_optional_int(row.get("train_rows")),
        test_rows=_optional_int(row.get("test_rows")),
        anomaly_points=_optional_int(row.get("anomaly_points")),
    )


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _best_rows_by_channel(
    rows: list[AnomalyModelComparisonRow],
) -> tuple[AnomalyModelComparisonRow, ...]:
    best_rows: list[AnomalyModelComparisonRow] = []
    for channel_id in sorted({row.channel_id for row in rows}):
        best_rows.append(
            min(
                (row for row in rows if row.channel_id == channel_id),
                key=lambda row: row.rank_by_f1,
            )
        )
    return tuple(best_rows)


def _append_aggregate_table(
    lines: list[str],
    *,
    rows: tuple[AnomalyModelComparisonRow, ...],
    heading: str,
    count_header: str,
) -> None:
    lines.extend(
        [
            "",
            f"## {heading}",
            "",
            (
                f"| Source | Model | {count_header} | Mean F1 | "
                "Mean Point-Adjusted F1 | Mean False Alarm Rate |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for source, model_name, group in _group_rows_by_model(rows):
        mean_f1 = _mean(row.f1 for row in group)
        mean_point_adjusted_f1 = _mean(row.point_adjusted_f1 for row in group)
        mean_false_alarm_rate = _mean(row.false_alarm_rate for row in group)
        lines.append(
            f"| {source} | `{model_name}` | {len(group)} | {mean_f1:.6f} | "
            f"{mean_point_adjusted_f1:.6f} | {mean_false_alarm_rate:.6f} |"
        )


def _group_rows_by_model(
    rows: tuple[AnomalyModelComparisonRow, ...],
) -> tuple[tuple[str, str, tuple[AnomalyModelComparisonRow, ...]], ...]:
    grouped: dict[tuple[str, str], list[AnomalyModelComparisonRow]] = {}
    for row in rows:
        grouped.setdefault((row.source, row.model_name), []).append(row)
    return tuple(
        (source, model_name, tuple(grouped[(source, model_name)]))
        for source, model_name in sorted(grouped)
    )


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("cannot average an empty metric sequence")
    return sum(sequence) / len(sequence)


def _prepare_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
