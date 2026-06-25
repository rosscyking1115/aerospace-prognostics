"""Generic telemetry anomaly command handlers for the project CLI."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aerospace_prognostics.anomaly.baselines import (
    CLASSICAL_ANOMALY_BASELINE_METHODS,
    ClassicalAnomalyBaselineResult,
    run_classical_anomaly_baselines,
    run_robust_zscore_baseline,
)
from aerospace_prognostics.cli_io import prepare_output_path, write_json_payload


def register_anomaly_commands(subparsers: Any) -> None:
    anomaly_baseline = subparsers.add_parser(
        "telemetry-robust-zscore-baseline",
        help="Run a robust MAD/z-score anomaly baseline on labelled telemetry CSVs",
    )
    anomaly_baseline.add_argument("--train-csv", type=Path, required=True)
    anomaly_baseline.add_argument("--test-csv", type=Path, required=True)
    anomaly_baseline.add_argument("--label-column", required=True)
    anomaly_baseline.add_argument(
        "--feature-columns",
        nargs="+",
        help="Feature columns to score; defaults to all numeric columns except the label",
    )
    anomaly_baseline.add_argument("--threshold", type=float, default=3.5)
    anomaly_baseline.add_argument("--output-json", type=Path)
    anomaly_baseline.add_argument("--predictions-csv", type=Path)

    anomaly_classical = subparsers.add_parser(
        "telemetry-classical-anomaly-baselines",
        help="Compare robust z-score, PCA reconstruction, and Isolation Forest baselines",
    )
    anomaly_classical.add_argument("--train-csv", type=Path, required=True)
    anomaly_classical.add_argument("--test-csv", type=Path, required=True)
    anomaly_classical.add_argument("--label-column", required=True)
    anomaly_classical.add_argument(
        "--feature-columns",
        nargs="+",
        help="Feature columns to score; defaults to all numeric columns except the label",
    )
    anomaly_classical.add_argument(
        "--methods",
        nargs="+",
        choices=CLASSICAL_ANOMALY_BASELINE_METHODS,
        default=list(CLASSICAL_ANOMALY_BASELINE_METHODS),
    )
    anomaly_classical.add_argument("--robust-threshold", type=float, default=3.5)
    anomaly_classical.add_argument("--pca-components", type=int)
    anomaly_classical.add_argument("--pca-threshold-quantile", type=float, default=0.99)
    anomaly_classical.add_argument("--isolation-contamination", type=float, default=0.05)
    anomaly_classical.add_argument("--random-state", type=int, default=42)
    anomaly_classical.add_argument("--output-json", type=Path)
    anomaly_classical.add_argument("--output-csv", type=Path)
    anomaly_classical.add_argument("--predictions-csv", type=Path)


def handle_anomaly_command(args: argparse.Namespace) -> int | None:
    if args.command == "telemetry-robust-zscore-baseline":
        import pandas as pd

        train_frame = pd.read_csv(args.train_csv)
        test_frame = pd.read_csv(args.test_csv)
        feature_columns = tuple(
            args.feature_columns
            if args.feature_columns is not None
            else _infer_numeric_feature_columns(train_frame, args.label_column)
        )
        _require_columns(train_frame, feature_columns, frame_name="train CSV")
        _require_columns(test_frame, (*feature_columns, args.label_column), frame_name="test CSV")
        result = run_robust_zscore_baseline(
            train_frame.loc[:, feature_columns].to_numpy(),
            test_frame.loc[:, feature_columns].to_numpy(),
            test_frame.loc[:, args.label_column].to_numpy(),
            feature_names=feature_columns,
            threshold=args.threshold,
        )
        print(f"features={len(feature_columns)}")
        print(f"test_rows={len(test_frame)}")
        print(f"threshold={args.threshold:g}")
        print(f"precision={result.metrics.precision:.6f}")
        print(f"recall={result.metrics.recall:.6f}")
        print(f"f1={result.metrics.f1:.6f}")
        print(f"point_adjusted_f1={result.point_adjusted_metrics.f1:.6f}")
        print(f"false_alarm_rate={result.metrics.false_alarm_rate:.6f}")
        if args.output_json is not None:
            write_json_payload(result.to_dict(), args.output_json)
        if args.predictions_csv is not None:
            _write_anomaly_predictions_csv(
                labels=test_frame.loc[:, args.label_column].to_numpy(),
                scores=result.scores,
                predictions=result.predictions,
                path=args.predictions_csv,
            )
        return 0

    if args.command == "telemetry-classical-anomaly-baselines":
        import pandas as pd

        train_frame = pd.read_csv(args.train_csv)
        test_frame = pd.read_csv(args.test_csv)
        feature_columns = tuple(
            args.feature_columns
            if args.feature_columns is not None
            else _infer_numeric_feature_columns(train_frame, args.label_column)
        )
        _require_columns(train_frame, feature_columns, frame_name="train CSV")
        _require_columns(test_frame, (*feature_columns, args.label_column), frame_name="test CSV")
        labels = test_frame.loc[:, args.label_column].to_numpy()
        results = run_classical_anomaly_baselines(
            train_frame.loc[:, feature_columns].to_numpy(),
            test_frame.loc[:, feature_columns].to_numpy(),
            labels,
            feature_names=feature_columns,
            methods=tuple(args.methods),
            robust_threshold=args.robust_threshold,
            pca_components=args.pca_components,
            pca_threshold_quantile=args.pca_threshold_quantile,
            isolation_contamination=args.isolation_contamination,
            random_state=args.random_state,
        )
        print(f"features={len(feature_columns)}")
        print(f"test_rows={len(test_frame)}")
        print("model,precision,recall,f1,point_adjusted_f1,false_alarm_rate")
        for result in results:
            print(
                f"{result.model_name},"
                f"{result.metrics.precision:.6f},"
                f"{result.metrics.recall:.6f},"
                f"{result.metrics.f1:.6f},"
                f"{result.point_adjusted_metrics.f1:.6f},"
                f"{result.metrics.false_alarm_rate:.6f}"
            )
        if args.output_json is not None:
            write_json_payload([result.to_dict() for result in results], args.output_json)
        if args.output_csv is not None:
            _write_classical_anomaly_summary_csv(results, args.output_csv)
        if args.predictions_csv is not None:
            _write_classical_anomaly_predictions_csv(
                labels=labels,
                results=results,
                path=args.predictions_csv,
            )
        return 0

    return None


def _infer_numeric_feature_columns(frame: object, label_column: str) -> tuple[str, ...]:
    import pandas as pd

    columns = tuple(
        column
        for column in frame.columns
        if column != label_column and pd.api.types.is_numeric_dtype(frame[column])
    )
    if not columns:
        raise ValueError("no numeric feature columns found")
    return columns


def _require_columns(frame: object, columns: Iterable[str], *, frame_name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {', '.join(missing)}")


def _write_anomaly_predictions_csv(
    *,
    labels: object,
    scores: Iterable[float],
    predictions: Iterable[int],
    path: Path,
) -> None:
    output_path = prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["row_index", "label", "anomaly_score", "prediction"],
        )
        writer.writeheader()
        for index, (label, score, prediction) in enumerate(
            zip(labels, scores, predictions, strict=True)
        ):
            writer.writerow(
                {
                    "row_index": index,
                    "label": int(label),
                    "anomaly_score": f"{score:.12g}",
                    "prediction": int(prediction),
                }
            )


def _write_classical_anomaly_summary_csv(
    results: Iterable[ClassicalAnomalyBaselineResult],
    path: Path,
) -> None:
    output_path = prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "model_name",
            "precision",
            "recall",
            "f1",
            "point_adjusted_f1",
            "false_alarm_rate",
            "miss_rate",
            "support",
            "predicted_positives",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model_name": result.model_name,
                    "precision": f"{result.metrics.precision:.12g}",
                    "recall": f"{result.metrics.recall:.12g}",
                    "f1": f"{result.metrics.f1:.12g}",
                    "point_adjusted_f1": f"{result.point_adjusted_metrics.f1:.12g}",
                    "false_alarm_rate": f"{result.metrics.false_alarm_rate:.12g}",
                    "miss_rate": f"{result.metrics.miss_rate:.12g}",
                    "support": result.metrics.support,
                    "predicted_positives": result.metrics.predicted_positives,
                }
            )


def _write_classical_anomaly_predictions_csv(
    *,
    labels: object,
    results: Iterable[ClassicalAnomalyBaselineResult],
    path: Path,
) -> None:
    output_path = prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model_name", "row_index", "label", "anomaly_score", "prediction"],
        )
        writer.writeheader()
        for result in results:
            for index, (label, score, prediction) in enumerate(
                zip(labels, result.scores, result.predictions, strict=True)
            ):
                writer.writerow(
                    {
                        "model_name": result.model_name,
                        "row_index": index,
                        "label": int(label),
                        "anomaly_score": f"{score:.12g}",
                        "prediction": int(prediction),
                    }
                )
