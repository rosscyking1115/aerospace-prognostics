"""Reporting and dashboard command handlers for the project CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aerospace_prognostics.reports.anomaly_model_comparison import (
    AnomalyModelComparisonRow,
    build_anomaly_model_comparison,
    write_anomaly_model_comparison_csv,
    write_anomaly_model_comparison_markdown,
)
from aerospace_prognostics.reports.cmapss_model_comparison import (
    CmapssModelComparisonRow,
    build_cmapss_model_comparison,
    write_cmapss_model_comparison_csv,
    write_cmapss_model_comparison_markdown,
)
from aerospace_prognostics.reports.cmapss_phase3 import run_cmapss_phase3_audit
from aerospace_prognostics.reports.cmapss_prediction_calibration import (
    calibrate_cmapss_deep_predictions,
)
from aerospace_prognostics.reports.dashboard import (
    build_fleet_dashboard_payload,
    write_fleet_dashboard_html,
    write_fleet_dashboard_payload_json,
)


def register_cmapss_report_commands(subparsers: Any) -> None:
    calibrate_deep_predictions = subparsers.add_parser(
        "cmapss-calibrate-deep-predictions",
        help=(
            "Fit validation calibration for deep C-MAPSS predictions "
            "and apply it to an official-test prediction CSV"
        ),
    )
    calibrate_deep_predictions.add_argument(
        "--method",
        choices=["affine", "predicted_bin_residual", "predicted_bin_nasa_shift"],
        default="affine",
    )
    calibrate_deep_predictions.add_argument(
        "--calibration-csv",
        type=Path,
        required=True,
        help="Validation-selection prediction CSV used to fit calibration",
    )
    calibrate_deep_predictions.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="Prediction CSV to calibrate",
    )
    calibrate_deep_predictions.add_argument("--output-csv", type=Path, required=True)
    calibrate_deep_predictions.add_argument("--output-calibration-csv", type=Path)
    calibrate_deep_predictions.add_argument("--output-diagnostics-csv", type=Path)
    calibrate_deep_predictions.add_argument("--output-rul-bins-csv", type=Path)
    calibrate_deep_predictions.add_argument("--output-unit-diagnostics-csv", type=Path)
    calibrate_deep_predictions.add_argument("--output-markdown", type=Path)
    calibrate_deep_predictions.add_argument("--top-n", type=int, default=10)
    calibrate_deep_predictions.add_argument("--clip-min", type=float, default=0.0)
    calibrate_deep_predictions.add_argument(
        "--shrinkage-strength",
        type=float,
        default=100.0,
        help="Residual-bin shrinkage strength; larger values shrink corrections more",
    )
    calibrate_deep_predictions.add_argument(
        "--nasa-shift-max",
        type=float,
        default=30.0,
        help="Maximum absolute shift searched for predicted-bin NASA-shift calibration",
    )
    calibrate_deep_predictions.add_argument(
        "--nasa-shift-step",
        type=float,
        default=1.0,
        help="Shift grid step for predicted-bin NASA-shift calibration",
    )

    phase3_audit = subparsers.add_parser(
        "cmapss-phase3-audit",
        help=(
            "Build Phase 3 C-MAPSS uncertainty and monotonicity evidence "
            "from validation and official-test prediction CSVs"
        ),
    )
    phase3_audit.add_argument(
        "--calibration-csv",
        type=Path,
        required=True,
        help="Validation-selection prediction CSV used to fit interval calibration",
    )
    phase3_audit.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="Official-test or holdout prediction CSV to audit",
    )
    phase3_audit.add_argument(
        "--calibrated-predictions-csv",
        type=Path,
        help="Optional calibrated prediction CSV for raw-vs-calibrated monotonicity",
    )
    phase3_audit.add_argument("--output-json", type=Path, required=True)
    phase3_audit.add_argument("--output-markdown", type=Path)
    phase3_audit.add_argument("--confidence", type=float, default=0.9)
    phase3_audit.add_argument("--top-n", type=int, default=10)
    phase3_audit.add_argument("--clip-min", type=float, default=0.0)

    compare_rul_results = subparsers.add_parser(
        "cmapss-compare-rul-results",
        help="Compare Phase 2 RUL result tables against the Phase 1 HGB policy baseline",
    )
    compare_rul_results.add_argument("--baseline-csv", type=Path, required=True)
    compare_rul_results.add_argument(
        "--candidate-csv",
        nargs="*",
        type=Path,
        default=[],
    )
    compare_rul_results.add_argument("--prediction-csv", nargs="*", type=Path, default=[])
    compare_rul_results.add_argument("--prediction-label", default="phase2_predictions")
    compare_rul_results.add_argument("--prediction-model-suffixes", nargs="+")
    compare_rul_results.add_argument("--baseline-label", default="phase1_hgb_policy")
    compare_rul_results.add_argument("--candidate-label", default="phase2_deep")
    compare_rul_results.add_argument("--output-csv", type=Path)
    compare_rul_results.add_argument("--output-markdown", type=Path)


def register_smap_msl_report_commands(subparsers: Any) -> None:
    smap_msl_compare = subparsers.add_parser(
        "smap-msl-compare-anomaly-results",
        help="Rank SMAP/MSL anomaly result CSVs across classical and forecast baselines",
    )
    smap_msl_compare.add_argument("--result-csv", nargs="+", type=Path, required=True)
    smap_msl_compare.add_argument("--source-labels", nargs="+")
    smap_msl_compare.add_argument("--output-csv", type=Path)
    smap_msl_compare.add_argument("--output-markdown", type=Path)


def register_dashboard_commands(subparsers: Any) -> None:
    dashboard_payload = subparsers.add_parser(
        "dashboard-fleet-payload",
        help="Build dashboard-ready fleet JSON from prediction and release evidence",
    )
    dashboard_payload.add_argument("--prediction-json", type=Path, required=True)
    dashboard_payload.add_argument("--promotion-json", type=Path)
    dashboard_payload.add_argument("--release-bundle-json", type=Path)
    dashboard_payload.add_argument("--title", default="Aerospace PHM Fleet View")
    dashboard_payload.add_argument("--output-json", type=Path, required=True)

    dashboard_html = subparsers.add_parser(
        "dashboard-render-html",
        help="Render standalone HTML from a dashboard fleet payload JSON",
    )
    dashboard_html.add_argument("--payload-json", type=Path, required=True)
    dashboard_html.add_argument("--output-html", type=Path, required=True)


def handle_report_command(args: argparse.Namespace) -> int | None:
    if args.command == "cmapss-calibrate-deep-predictions":
        result = calibrate_cmapss_deep_predictions(
            calibration_csv=args.calibration_csv,
            predictions_csv=args.predictions_csv,
            output_csv=args.output_csv,
            method=args.method,
            output_calibration_csv=args.output_calibration_csv,
            output_diagnostics_csv=args.output_diagnostics_csv,
            output_rul_bins_csv=args.output_rul_bins_csv,
            output_unit_diagnostics_csv=args.output_unit_diagnostics_csv,
            output_markdown=args.output_markdown,
            top_n=args.top_n,
            clip_min=args.clip_min,
            shrinkage_strength=args.shrinkage_strength,
            nasa_shift_max=args.nasa_shift_max,
            nasa_shift_step=args.nasa_shift_step,
        )
        print(f"calibration_method={args.method}")
        print(f"calibration_groups={len(result.calibrations)}")
        print(f"calibrated_prediction_rows={result.calibrated_prediction_count}")
        print(f"calibrated_predictions_csv={result.calibrated_predictions_csv_path}")
        for calibration in result.calibrations:
            if hasattr(calibration, "predicted_rul_bin"):
                print(
                    "calibration="
                    f"{calibration.subset}:{calibration.model_name}:"
                    f"bin={calibration.predicted_rul_bin}:"
                    f"rows={calibration.calibration_count}:"
                    f"correction={_format_cli_float(calibration.correction)}"
                )
            else:
                print(
                    "calibration="
                    f"{calibration.subset}:{calibration.model_name}:"
                    f"rows={calibration.calibration_count}:"
                    f"intercept={_format_cli_float(calibration.intercept)}:"
                    f"slope={_format_cli_float(calibration.slope)}"
                )
        if result.calibration_csv_path is not None:
            print(f"calibration_csv={result.calibration_csv_path}")
        if result.diagnostics_csv_path is not None:
            print(f"diagnostics_csv={result.diagnostics_csv_path}")
        if result.rul_bins_csv_path is not None:
            print(f"rul_bins_csv={result.rul_bins_csv_path}")
        if result.unit_diagnostics_csv_path is not None:
            print(f"unit_diagnostics_csv={result.unit_diagnostics_csv_path}")
        if result.diagnostics_markdown_path is not None:
            print(f"diagnostics_markdown={result.diagnostics_markdown_path}")
        return 0

    if args.command == "cmapss-phase3-audit":
        result = run_cmapss_phase3_audit(
            calibration_csv=args.calibration_csv,
            predictions_csv=args.predictions_csv,
            calibrated_predictions_csv=args.calibrated_predictions_csv,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            confidence=args.confidence,
            top_n=args.top_n,
            clip_min=args.clip_min,
        )
        print("schema_version=aerospace-prognostics/cmapss-phase3-audit/v1")
        print(f"interval_calibrations={len(result.interval_calibrations)}")
        print(f"uncertainty_summaries={len(result.uncertainty_summaries)}")
        print(f"uncertainty_bins={len(result.uncertainty_bins)}")
        print(f"failure_cases={len(result.failure_cases)}")
        print(
            "predicted_bin_interval_calibrations="
            f"{len(result.predicted_bin_interval_calibrations)}"
        )
        print(
            "predicted_bin_floor_summaries="
            f"{len(result.predicted_bin_floor_uncertainty_summaries)}"
        )
        print(f"interval_comparisons={len(result.interval_comparisons)}")
        print(f"monotonicity_diagnostics={len(result.monotonicity_diagnostics)}")
        print(f"monotonicity_comparisons={len(result.monotonicity_comparisons)}")
        print(f"training_recommendation={result.training_recommendation}")
        if result.output_json_path is not None:
            print(f"output_json={result.output_json_path}")
        if result.output_markdown_path is not None:
            print(f"output_markdown={result.output_markdown_path}")
        return 0

    if args.command == "cmapss-compare-rul-results":
        rows = build_cmapss_model_comparison(
            args.baseline_csv,
            tuple(args.candidate_csv),
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            prediction_csvs=tuple(args.prediction_csv),
            prediction_label=args.prediction_label,
            prediction_model_suffixes=tuple(args.prediction_model_suffixes or ()),
        )
        print(f"rows={len(rows)}")
        print(f"subsets={','.join(_comparison_subsets(rows))}")
        print(
            "best_by_nasa="
            + ",".join(
                f"{subset}:{_best_comparison_row_for_subset(rows, subset).phase}:"
                f"{_best_comparison_row_for_subset(rows, subset).model_name}"
                for subset in _comparison_subsets(rows)
            )
        )
        _print_comparison_table(rows)
        if args.output_csv is not None:
            write_cmapss_model_comparison_csv(rows, args.output_csv)
        if args.output_markdown is not None:
            write_cmapss_model_comparison_markdown(rows, args.output_markdown)
        return 0

    if args.command == "smap-msl-compare-anomaly-results":
        rows = build_anomaly_model_comparison(
            tuple(args.result_csv),
            source_labels=tuple(args.source_labels) if args.source_labels is not None else None,
        )
        print(f"channels={len(_anomaly_comparison_channels(rows))}")
        print(f"rows={len(rows)}")
        _print_anomaly_comparison_table(rows)
        if args.output_csv is not None:
            write_anomaly_model_comparison_csv(rows, args.output_csv)
        if args.output_markdown is not None:
            write_anomaly_model_comparison_markdown(rows, args.output_markdown)
        return 0

    if args.command == "dashboard-fleet-payload":
        payload = build_fleet_dashboard_payload(
            args.prediction_json,
            title=args.title,
            promotion_json=args.promotion_json,
            release_bundle_json=args.release_bundle_json,
        )
        output_path = write_fleet_dashboard_payload_json(payload, args.output_json)
        payload_dict = payload.to_dict()
        summary = payload_dict["summary"]
        risk_counts = summary["risk_counts"]
        print(f"schema_version={payload.schema_version}")
        print(f"assets={summary['asset_count']}")
        print(f"attention_required={summary['attention_required_count']}")
        print(
            "risk_counts="
            f"critical:{risk_counts['critical']},"
            f"watch:{risk_counts['watch']},"
            f"nominal:{risk_counts['nominal']},"
            f"unknown:{risk_counts['unknown']}"
        )
        print(f"output_json={output_path}")
        return 0

    if args.command == "dashboard-render-html":
        payload = json.loads(args.payload_json.read_text(encoding="utf-8"))
        output_path = write_fleet_dashboard_html(payload, args.output_html)
        assets = len(payload.get("assets", [])) if isinstance(payload, dict) else 0
        print("schema_version=aerospace-prognostics/fleet-dashboard/v1")
        print(f"assets={assets}")
        print(f"output_html={output_path}")
        return 0

    return None


def _print_comparison_table(rows: Iterable[CmapssModelComparisonRow]) -> None:
    print(
        "subset,rank_by_nasa,phase,model,rmse,nasa_score,"
        "rmse_delta,nasa_score_delta,nasa_score_ratio"
    )
    for row in rows:
        print(
            f"{row.subset},{row.rank_by_nasa},{row.phase},{row.model_name},"
            f"{row.rmse:.6f},{row.nasa_score:.6f},"
            f"{row.rmse_delta:.6f},{row.nasa_score_delta:.6f},"
            f"{row.nasa_score_ratio:.6f}"
        )


def _print_anomaly_comparison_table(rows: Iterable[AnomalyModelComparisonRow]) -> None:
    print(
        "channel_id,spacecraft,rank_by_f1,source,model,"
        "f1,point_adjusted_f1,precision,recall,false_alarm_rate"
    )
    for row in rows:
        print(
            f"{row.channel_id},"
            f"{row.spacecraft},"
            f"{row.rank_by_f1},"
            f"{row.source},"
            f"{row.model_name},"
            f"{row.f1:.6f},"
            f"{row.point_adjusted_f1:.6f},"
            f"{row.precision:.6f},"
            f"{row.recall:.6f},"
            f"{row.false_alarm_rate:.6f}"
        )


def _anomaly_comparison_channels(rows: Iterable[AnomalyModelComparisonRow]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(row.channel_id for row in rows))


def _comparison_subsets(rows: Iterable[CmapssModelComparisonRow]) -> list[str]:
    return sorted({row.subset for row in rows})


def _best_comparison_row_for_subset(
    rows: Iterable[CmapssModelComparisonRow],
    subset: str,
) -> CmapssModelComparisonRow:
    subset_rows = [row for row in rows if row.subset == subset]
    if not subset_rows:
        raise ValueError(f"no comparison rows for subset {subset}")
    return min(subset_rows, key=lambda row: (row.nasa_score, row.rmse))


def _format_cli_float(value: float) -> str:
    return f"{value:g}"
