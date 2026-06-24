"""Command-line utilities for local dataset checks."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

from aerospace_prognostics.analysis.cmapss_eda import build_cmapss_eda_report
from aerospace_prognostics.anomaly.baselines import (
    CLASSICAL_ANOMALY_BASELINE_METHODS,
    ClassicalAnomalyBaselineResult,
    run_classical_anomaly_baselines,
    run_robust_zscore_baseline,
)
from aerospace_prognostics.cli_app import handle_app_command, register_app_commands
from aerospace_prognostics.cli_deployment import (
    handle_deployment_command,
    register_deployment_artifact_commands,
    register_deployment_release_commands,
)
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS, load_cmapss_subset
from aerospace_prognostics.data.downloads import (
    NASA_CMAPSS_URL,
    TELEMANOM_SMAP_MSL_DATA_URL,
    TELEMANOM_SMAP_MSL_LABELS_URL,
    download_cmapss_dataset,
    download_smap_msl_dataset,
)
from aerospace_prognostics.data.manifest import (
    build_cmapss_manifest,
    read_manifest,
    verify_manifest,
)
from aerospace_prognostics.data.smap_msl import (
    SmapMslChannelSelection,
    export_smap_msl_channel_csv,
    load_smap_msl_channel,
    read_smap_msl_labels,
    select_smap_msl_channels,
    write_smap_msl_channel_selection_csv,
    write_smap_msl_channel_selection_json,
)
from aerospace_prognostics.data.summary import summarise_cmapss_frame
from aerospace_prognostics.deployment.quickstart import run_cmapss_quickstart
from aerospace_prognostics.evaluation import (
    RegressionRunResult,
    write_results_csv,
    write_results_json,
)
from aerospace_prognostics.experiments.cmapss_baseline import (
    CMAPSS_ENGINEERED_DEFAULT_WINDOWS,
    CMAPSS_HGB_PARAM_GRID,
    CMAPSS_SENSOR_FILTER_CANDIDATES,
    CMAPSS_VALIDATION_SELECTED_FEATURES,
    CMAPSS_VALIDATION_SELECTED_HGB_PARAMS,
    CmapssValidationAggregateResult,
    run_all_cmapss_engineered_default_windows,
    run_all_cmapss_engineered_hist_gradient_boosting,
    run_all_cmapss_hist_gradient_boosting,
    run_all_cmapss_regime_aware_engineered_default_windows,
    run_all_cmapss_validation_selected_default_windows,
    run_all_cmapss_validation_selected_hgb_policy_default_windows,
    run_cmapss_engineered_hist_gradient_boosting,
    run_cmapss_engineered_window_sweep,
    run_cmapss_hist_gradient_boosting,
    run_cmapss_repeated_validation_feature_comparison,
    run_cmapss_validation_feature_comparison,
    run_cmapss_validation_selected_hgb_grid,
    run_cmapss_validation_sensor_filter_comparison,
)
from aerospace_prognostics.experiments.smap_msl_anomaly import (
    LSTM_FORECAST_THRESHOLD_METHODS,
    SmapMslClassicalBaselineRun,
    SmapMslLstmForecastBaselineRun,
    SmapMslRobustThresholdOperatingPoint,
    SmapMslRobustThresholdSweepAggregate,
    SmapMslRobustThresholdSweepRun,
    aggregate_smap_msl_robust_threshold_sweep,
    run_smap_msl_classical_baselines,
    run_smap_msl_lstm_forecast_baseline,
    run_smap_msl_robust_threshold_sweep,
    select_smap_msl_robust_threshold_operating_points,
    select_smap_msl_robust_threshold_policy_runs,
    write_smap_msl_classical_baselines_csv,
    write_smap_msl_classical_baselines_json,
    write_smap_msl_lstm_forecast_baseline_csv,
    write_smap_msl_lstm_forecast_baseline_json,
    write_smap_msl_robust_threshold_operating_points_csv,
    write_smap_msl_robust_threshold_operating_points_json,
    write_smap_msl_robust_threshold_policy_csv,
    write_smap_msl_robust_threshold_policy_json,
    write_smap_msl_robust_threshold_sweep_aggregate_csv,
    write_smap_msl_robust_threshold_sweep_aggregate_json,
    write_smap_msl_robust_threshold_sweep_csv,
    write_smap_msl_robust_threshold_sweep_json,
)
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
from aerospace_prognostics.reports.cmapss_prediction_calibration import (
    calibrate_cmapss_deep_predictions,
)
from aerospace_prognostics.reports.dashboard import (
    build_fleet_dashboard_payload,
    write_fleet_dashboard_html,
    write_fleet_dashboard_payload_json,
)
from aerospace_prognostics.sequence_exports import export_cmapss_sequence_splits
from aerospace_prognostics.workflows.phase1 import run_phase1_cmapss_workflow

CMAPSS_DEEP_COMPARISON_MODELS = ("cnn", "rescnn", "lstm", "bilstm", "tcn", "transformer")
CMAPSS_DEEP_TRAINING_LOSSES = (
    "mse",
    "nasa_surrogate",
    "mse_nasa_blend_w0p001",
    "mse_nasa_blend_w0p0001",
    "asymmetric_mse_late_w1p5",
    "asymmetric_mse_late_w2",
    "asymmetric_mse_late_w3",
    "target_weighted_mse_high_w2",
    "target_weighted_mse_mid_high_w1p5",
    "mse_monotonic_w0p1",
    "asymmetric_mse_late_w1p5_monotonic_w0p1",
    "mse_unit_monotonic_w0p1",
    "asymmetric_mse_late_w1p5_unit_monotonic_w0p1",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aerospace-prognostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quickstart = subparsers.add_parser(
        "quickstart-cmapss-demo",
        help="Run the no-download C-MAPSS deployment quickstart",
    )
    quickstart.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "quickstart_cmapss",
    )
    quickstart.add_argument("--release-name", default="quickstart-fd001-demo")
    quickstart.add_argument("--repository", default="local/aerospace-prognostics")
    quickstart.add_argument("--git-sha", default="0" * 40)
    quickstart.add_argument("--git-ref", default="refs/heads/local-quickstart")
    quickstart.add_argument("--workflow", default="local-quickstart")
    quickstart.add_argument("--run-id", default="local")
    quickstart.add_argument("--lockfile", type=Path, default=Path("uv.lock"))

    register_app_commands(subparsers)

    summary = subparsers.add_parser("cmapss-summary", help="Summarise a local C-MAPSS subset")
    summary.add_argument("--data-dir", type=Path, required=True)
    summary.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], required=True)
    summary.add_argument("--rul-cap", type=int, default=125)

    baseline = subparsers.add_parser(
        "cmapss-baseline",
        help="Train a first-pass C-MAPSS gradient-boosting baseline",
    )
    baseline.add_argument("--data-dir", type=Path, required=True)
    baseline.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], required=True)
    baseline.add_argument("--rul-cap", type=int, default=125)
    baseline.add_argument("--random-state", type=int, default=42)
    baseline.add_argument("--output-json", type=Path)
    baseline.add_argument("--standardize", action="store_true")

    baseline_all = subparsers.add_parser(
        "cmapss-baseline-all",
        help="Train the first-pass C-MAPSS baseline on multiple subsets",
    )
    baseline_all.add_argument("--data-dir", type=Path, required=True)
    baseline_all.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    baseline_all.add_argument("--rul-cap", type=int, default=125)
    baseline_all.add_argument("--random-state", type=int, default=42)
    baseline_all.add_argument("--output-json", type=Path)
    baseline_all.add_argument("--output-csv", type=Path)
    baseline_all.add_argument("--standardize", action="store_true")

    engineered = subparsers.add_parser(
        "cmapss-engineered-baseline",
        help="Train a feature-engineered C-MAPSS gradient-boosting baseline",
    )
    engineered.add_argument("--data-dir", type=Path, required=True)
    engineered.add_argument("--subset", choices=CMAPSS_SUBSETS, required=True)
    engineered.add_argument("--rul-cap", type=int, default=125)
    engineered.add_argument("--random-state", type=int, default=42)
    engineered.add_argument("--rolling-window", type=int, default=5)
    engineered.add_argument("--output-json", type=Path)
    engineered.add_argument("--no-standardize", action="store_true")

    engineered_all = subparsers.add_parser(
        "cmapss-engineered-baseline-all",
        help="Train the feature-engineered C-MAPSS baseline on multiple subsets",
    )
    engineered_all.add_argument("--data-dir", type=Path, required=True)
    engineered_all.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    engineered_all.add_argument("--rul-cap", type=int, default=125)
    engineered_all.add_argument("--random-state", type=int, default=42)
    engineered_all.add_argument("--rolling-window", type=int, default=5)
    engineered_all.add_argument("--output-json", type=Path)
    engineered_all.add_argument("--output-csv", type=Path)
    engineered_all.add_argument("--no-standardize", action="store_true")

    engineered_sweep = subparsers.add_parser(
        "cmapss-engineered-window-sweep",
        help="Compare engineered C-MAPSS baselines across rolling-window sizes",
    )
    engineered_sweep.add_argument("--data-dir", type=Path, required=True)
    engineered_sweep.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    engineered_sweep.add_argument("--rolling-windows", nargs="+", type=int, default=[3, 5, 10])
    engineered_sweep.add_argument("--rul-cap", type=int, default=125)
    engineered_sweep.add_argument("--random-state", type=int, default=42)
    engineered_sweep.add_argument("--output-json", type=Path)
    engineered_sweep.add_argument("--output-csv", type=Path)
    engineered_sweep.add_argument("--no-standardize", action="store_true")

    engineered_best = subparsers.add_parser(
        "cmapss-engineered-best-baseline-all",
        help="Train engineered baselines using current per-subset rolling-window defaults",
    )
    engineered_best.add_argument("--data-dir", type=Path, required=True)
    engineered_best.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    engineered_best.add_argument("--rul-cap", type=int, default=125)
    engineered_best.add_argument("--random-state", type=int, default=42)
    engineered_best.add_argument("--output-json", type=Path)
    engineered_best.add_argument("--output-csv", type=Path)
    engineered_best.add_argument("--no-standardize", action="store_true")

    regime_engineered = subparsers.add_parser(
        "cmapss-regime-engineered-best-baseline-all",
        help="Train regime-aware engineered baselines with per-subset windows",
    )
    regime_engineered.add_argument("--data-dir", type=Path, required=True)
    regime_engineered.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    regime_engineered.add_argument("--rul-cap", type=int, default=125)
    regime_engineered.add_argument("--random-state", type=int, default=42)
    regime_engineered.add_argument("--n-regimes", type=int, default=6)
    regime_engineered.add_argument("--output-json", type=Path)
    regime_engineered.add_argument("--output-csv", type=Path)
    regime_engineered.add_argument("--no-standardize", action="store_true")

    validation_selected = subparsers.add_parser(
        "cmapss-validation-selected-baseline-all",
        help="Train official-test baselines using the repeated-validation feature policy",
    )
    validation_selected.add_argument("--data-dir", type=Path, required=True)
    validation_selected.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    validation_selected.add_argument("--rul-cap", type=int, default=125)
    validation_selected.add_argument("--random-state", type=int, default=42)
    validation_selected.add_argument("--n-regimes", type=int, default=6)
    validation_selected.add_argument("--output-json", type=Path)
    validation_selected.add_argument("--output-csv", type=Path)
    validation_selected.add_argument("--no-standardize", action="store_true")

    hgb_policy = subparsers.add_parser(
        "cmapss-hgb-policy-baseline-all",
        help="Train official-test baselines using validation-selected features and HGB params",
    )
    hgb_policy.add_argument("--data-dir", type=Path, required=True)
    hgb_policy.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    hgb_policy.add_argument("--rul-cap", type=int, default=125)
    hgb_policy.add_argument("--random-state", type=int, default=42)
    hgb_policy.add_argument("--n-regimes", type=int, default=6)
    hgb_policy.add_argument("--output-json", type=Path)
    hgb_policy.add_argument("--output-csv", type=Path)
    hgb_policy.add_argument("--no-standardize", action="store_true")

    validation_candidates = subparsers.add_parser(
        "cmapss-validate-feature-candidates",
        help="Compare engineered and regime-aware candidates on temporal validation splits",
    )
    validation_candidates.add_argument("--data-dir", type=Path, required=True)
    validation_candidates.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    validation_candidates.add_argument("--rul-cap", type=int, default=125)
    validation_candidates.add_argument("--random-state", type=int, default=42)
    validation_candidates.add_argument("--n-regimes", type=int, default=6)
    validation_candidates.add_argument("--validation-fraction", type=float, default=0.2)
    validation_candidates.add_argument("--validation-horizon", type=int, default=30)
    validation_candidates.add_argument("--output-json", type=Path)
    validation_candidates.add_argument("--output-csv", type=Path)
    validation_candidates.add_argument("--no-standardize", action="store_true")

    repeated_validation = subparsers.add_parser(
        "cmapss-validate-feature-candidates-repeated",
        help="Aggregate candidate validation across multiple seeds and horizons",
    )
    repeated_validation.add_argument("--data-dir", type=Path, required=True)
    repeated_validation.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    repeated_validation.add_argument("--rul-cap", type=int, default=125)
    repeated_validation.add_argument("--random-states", nargs="+", type=int, default=[11, 42])
    repeated_validation.add_argument("--n-regimes", type=int, default=6)
    repeated_validation.add_argument("--validation-fraction", type=float, default=0.2)
    repeated_validation.add_argument("--validation-horizons", nargs="+", type=int, default=[20, 30])
    repeated_validation.add_argument("--output-json", type=Path)
    repeated_validation.add_argument("--output-csv", type=Path)
    repeated_validation.add_argument("--no-standardize", action="store_true")

    hgb_grid = subparsers.add_parser(
        "cmapss-validate-hgb-grid",
        help="Validate compact HGB parameter candidates for the current feature policy",
    )
    hgb_grid.add_argument("--data-dir", type=Path, required=True)
    hgb_grid.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    hgb_grid.add_argument("--rul-cap", type=int, default=125)
    hgb_grid.add_argument("--random-state", type=int, default=42)
    hgb_grid.add_argument("--n-regimes", type=int, default=6)
    hgb_grid.add_argument("--validation-fraction", type=float, default=0.2)
    hgb_grid.add_argument("--validation-horizon", type=int, default=30)
    hgb_grid.add_argument("--output-json", type=Path)
    hgb_grid.add_argument("--output-csv", type=Path)
    hgb_grid.add_argument("--no-standardize", action="store_true")

    sensor_filters = subparsers.add_parser(
        "cmapss-validate-sensor-filters",
        help="Validate full versus EDA-filtered sensor sets for the current policy",
    )
    sensor_filters.add_argument("--data-dir", type=Path, required=True)
    sensor_filters.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    sensor_filters.add_argument("--rul-cap", type=int, default=125)
    sensor_filters.add_argument("--random-state", type=int, default=42)
    sensor_filters.add_argument("--n-regimes", type=int, default=6)
    sensor_filters.add_argument("--validation-fraction", type=float, default=0.2)
    sensor_filters.add_argument("--validation-horizon", type=int, default=30)
    sensor_filters.add_argument("--min-abs-rul-correlation", type=float, default=0.05)
    sensor_filters.add_argument("--min-abs-standardized-drift", type=float, default=0.2)
    sensor_filters.add_argument("--output-json", type=Path)
    sensor_filters.add_argument("--output-csv", type=Path)
    sensor_filters.add_argument("--no-standardize", action="store_true")

    eda = subparsers.add_parser("cmapss-eda", help="Build a C-MAPSS EDA summary report")
    eda.add_argument("--data-dir", type=Path, required=True)
    eda.add_argument("--subset", choices=CMAPSS_SUBSETS, required=True)
    eda.add_argument("--rul-cap", type=int, default=125)
    eda.add_argument("--output-json", type=Path)

    manifest = subparsers.add_parser("cmapss-manifest", help="Write a C-MAPSS file manifest")
    manifest.add_argument("--data-dir", type=Path, required=True)
    manifest.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    manifest.add_argument("--output-json", type=Path, required=True)

    verify = subparsers.add_parser("cmapss-verify", help="Verify C-MAPSS files against a manifest")
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    download = subparsers.add_parser(
        "cmapss-download",
        help="Download and extract the official NASA C-MAPSS raw text files",
    )
    download.add_argument("--output-dir", type=Path, default=Path("data/raw/cmapss"))
    download.add_argument(
        "--archive-path",
        type=Path,
        default=Path("data/raw/downloads/cmapss_nasa.zip"),
    )
    download.add_argument("--source-url", default=NASA_CMAPSS_URL)
    download.add_argument("--force", action="store_true")

    smap_msl_download = subparsers.add_parser(
        "smap-msl-download",
        help="Download and extract the Telemanom SMAP/MSL raw arrays and labels",
    )
    smap_msl_download.add_argument("--output-dir", type=Path, default=Path("data/raw/smap_msl"))
    smap_msl_download.add_argument(
        "--archive-path",
        type=Path,
        default=Path("data/raw/downloads/smap_msl_telemanom.zip"),
        help="Destination for the downloaded archive, or an existing Kaggle archive to import",
    )
    smap_msl_download.add_argument("--source-url", default=TELEMANOM_SMAP_MSL_DATA_URL)
    smap_msl_download.add_argument("--labels-url", default=TELEMANOM_SMAP_MSL_LABELS_URL)
    smap_msl_download.add_argument("--force", action="store_true")

    phase1 = subparsers.add_parser(
        "phase1-cmapss",
        help="Run the Phase 1 C-MAPSS provenance, EDA, and baseline workflow",
    )
    phase1.add_argument("--data-dir", type=Path, required=True)
    phase1.add_argument("--artifact-dir", type=Path, default=Path("artifacts/phase1"))
    phase1.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    phase1.add_argument("--rul-cap", type=int, default=125)
    phase1.add_argument("--random-state", type=int, default=42)
    phase1.add_argument("--n-regimes", type=int, default=6)
    phase1.add_argument("--no-standardize", action="store_true")

    phase2 = subparsers.add_parser(
        "phase2-cmapss",
        help="Run the Phase 2 C-MAPSS sequence-model comparison workflow",
    )
    phase2.add_argument("--data-dir", type=Path, required=True)
    phase2.add_argument("--artifact-dir", type=Path, default=Path("artifacts/phase2"))
    phase2.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    phase2.add_argument("--window-size", type=int, default=30)
    phase2.add_argument("--stride", type=int, default=1)
    phase2.add_argument("--validation-fraction", type=float, default=0.2)
    phase2.add_argument("--validation-horizon", type=int, default=30)
    phase2.add_argument("--rul-cap", type=int, default=125)
    phase2.add_argument("--random-state", type=int, default=42)
    phase2.add_argument("--n-regimes", type=int, default=6)
    phase2.add_argument("--no-standardize", action="store_true")
    phase2.add_argument(
        "--models",
        nargs="+",
        choices=CMAPSS_DEEP_COMPARISON_MODELS,
        default=["cnn", "bilstm", "tcn", "transformer"],
    )
    phase2.add_argument("--epochs", type=int, default=5)
    phase2.add_argument("--batch-size", type=int, default=256)
    phase2.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3])
    phase2.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    phase2.add_argument("--hidden-sizes", nargs="+", type=int, default=[32])
    phase2.add_argument("--num-layers", type=int, default=1)
    phase2.add_argument("--tcn-levels", type=int, default=3)
    phase2.add_argument(
        "--tcn-normalization",
        choices=["none", "layer_norm"],
        default="none",
    )
    phase2.add_argument("--tcn-weight-norm", action="store_true")
    phase2.add_argument("--tcn-pooling", choices=["last", "mean"], default="last")
    phase2.add_argument("--transformer-heads", type=int, default=4)
    phase2.add_argument("--transformer-dim-feedforward", type=int)
    phase2.add_argument("--kernel-size", type=int, default=3)
    phase2.add_argument("--dropout", type=float, default=0.1)
    phase2.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    phase2.add_argument("--device", default="cpu")

    phase2_verify = subparsers.add_parser(
        "phase2-cmapss-verify-manifest",
        help="Verify a Phase 2 C-MAPSS run manifest and referenced artifacts",
    )
    phase2_verify.add_argument("--manifest", type=Path, required=True)
    phase2_verify.add_argument("--root", type=Path, default=Path("."))
    phase2_verify.add_argument("--output-markdown", type=Path)

    phase2_smap_msl = subparsers.add_parser(
        "phase2-smap-msl",
        help="Run the Phase 2 SMAP/MSL classical, forecast, and comparison workflow",
    )
    phase2_smap_msl.add_argument("--data-dir", type=Path, required=True)
    phase2_smap_msl.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/phase2_smap_msl"),
    )
    phase2_smap_msl.add_argument("--channels", nargs="+")
    phase2_smap_msl.add_argument("--max-channels", type=int)
    phase2_smap_msl.add_argument(
        "--classical-methods",
        nargs="+",
        choices=CLASSICAL_ANOMALY_BASELINE_METHODS,
        default=list(CLASSICAL_ANOMALY_BASELINE_METHODS),
    )
    phase2_smap_msl.add_argument("--robust-threshold", type=float, default=3.5)
    phase2_smap_msl.add_argument("--pca-components", type=int)
    phase2_smap_msl.add_argument("--pca-threshold-quantile", type=float, default=0.99)
    phase2_smap_msl.add_argument("--isolation-contamination", type=float, default=0.05)
    phase2_smap_msl.add_argument("--window-size", type=int, default=30)
    phase2_smap_msl.add_argument("--hidden-size", type=int, default=32)
    phase2_smap_msl.add_argument("--num-layers", type=int, default=1)
    phase2_smap_msl.add_argument("--dropout", type=float, default=0.0)
    phase2_smap_msl.add_argument("--epochs", type=int, default=10)
    phase2_smap_msl.add_argument("--batch-size", type=int, default=64)
    phase2_smap_msl.add_argument("--learning-rate", type=float, default=1e-3)
    phase2_smap_msl.add_argument("--threshold-sigma", type=float, default=3.0)
    phase2_smap_msl.add_argument("--robust-policy-false-alarm-budget", type=float)
    phase2_smap_msl.add_argument(
        "--robust-policy-thresholds",
        nargs="+",
        type=float,
        default=[3.5, 5.0, 7.0, 10.0, 15.0],
    )
    phase2_smap_msl.add_argument(
        "--robust-policy-group-by",
        choices=["spacecraft", "global"],
        default="spacecraft",
    )
    phase2_smap_msl.add_argument("--dynamic-batch-size", type=int, default=70)
    phase2_smap_msl.add_argument("--dynamic-window-batches", type=int, default=30)
    phase2_smap_msl.add_argument("--dynamic-smoothing-fraction", type=float, default=0.05)
    phase2_smap_msl.add_argument("--dynamic-z-start", type=float, default=2.5)
    phase2_smap_msl.add_argument("--dynamic-z-stop", type=float, default=12.0)
    phase2_smap_msl.add_argument("--dynamic-z-step", type=float, default=0.5)
    phase2_smap_msl.add_argument("--dynamic-error-buffer", type=int, default=100)
    phase2_smap_msl.add_argument("--dynamic-prune-p", type=float, default=0.13)
    phase2_smap_msl.add_argument("--random-state", type=int, default=42)
    phase2_smap_msl.add_argument("--device", default="cpu")

    phase2_smap_msl_verify = subparsers.add_parser(
        "phase2-smap-msl-verify-manifest",
        help="Verify a Phase 2 SMAP/MSL run manifest and referenced artifacts",
    )
    phase2_smap_msl_verify.add_argument("--manifest", type=Path, required=True)
    phase2_smap_msl_verify.add_argument("--root", type=Path, default=Path("."))
    phase2_smap_msl_verify.add_argument("--output-markdown", type=Path)

    sequence_export = subparsers.add_parser(
        "cmapss-export-sequences",
        help="Export C-MAPSS train/validation/test sequence tensors for Phase 2 models",
    )
    sequence_export.add_argument("--data-dir", type=Path, required=True)
    sequence_export.add_argument("--output-dir", type=Path, required=True)
    sequence_export.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    sequence_export.add_argument("--window-size", type=int, default=30)
    sequence_export.add_argument("--stride", type=int, default=1)
    sequence_export.add_argument("--rul-cap", type=int, default=125)
    sequence_export.add_argument("--random-state", type=int, default=42)
    sequence_export.add_argument("--validation-fraction", type=float, default=0.2)
    sequence_export.add_argument("--validation-horizon", type=int, default=30)
    sequence_export.add_argument("--no-standardize", action="store_true")

    cnn_baseline = subparsers.add_parser(
        "cmapss-cnn-baseline",
        help="Train a 1D-CNN RUL baseline from exported C-MAPSS sequence tensors",
    )
    cnn_baseline.add_argument("--sequence-dir", type=Path, required=True)
    cnn_baseline.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    cnn_baseline.add_argument("--epochs", type=int, default=5)
    cnn_baseline.add_argument("--batch-size", type=int, default=256)
    cnn_baseline.add_argument("--learning-rate", type=float, default=1e-3)
    cnn_baseline.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    cnn_baseline.add_argument("--hidden-channels", type=int, default=32)
    cnn_baseline.add_argument("--kernel-size", type=int, default=3)
    cnn_baseline.add_argument("--dropout", type=float, default=0.1)
    cnn_baseline.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    cnn_baseline.add_argument("--random-state", type=int, default=42)
    cnn_baseline.add_argument("--device", default="cpu")
    cnn_baseline.add_argument("--output-json", type=Path)
    cnn_baseline.add_argument("--output-csv", type=Path)
    cnn_baseline.add_argument("--history-json", type=Path)

    lstm_baseline = subparsers.add_parser(
        "cmapss-lstm-baseline",
        help="Train an LSTM/BiLSTM RUL baseline from exported C-MAPSS sequence tensors",
    )
    lstm_baseline.add_argument("--sequence-dir", type=Path, required=True)
    lstm_baseline.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    lstm_baseline.add_argument("--epochs", type=int, default=5)
    lstm_baseline.add_argument("--batch-size", type=int, default=256)
    lstm_baseline.add_argument("--learning-rate", type=float, default=1e-3)
    lstm_baseline.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    lstm_baseline.add_argument("--hidden-size", type=int, default=32)
    lstm_baseline.add_argument("--num-layers", type=int, default=1)
    lstm_baseline.add_argument("--dropout", type=float, default=0.1)
    lstm_baseline.add_argument("--bidirectional", action="store_true")
    lstm_baseline.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    lstm_baseline.add_argument("--random-state", type=int, default=42)
    lstm_baseline.add_argument("--device", default="cpu")
    lstm_baseline.add_argument("--output-json", type=Path)
    lstm_baseline.add_argument("--output-csv", type=Path)
    lstm_baseline.add_argument("--history-json", type=Path)

    tcn_baseline = subparsers.add_parser(
        "cmapss-tcn-baseline",
        help="Train a TCN RUL baseline from exported C-MAPSS sequence tensors",
    )
    tcn_baseline.add_argument("--sequence-dir", type=Path, required=True)
    tcn_baseline.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    tcn_baseline.add_argument("--epochs", type=int, default=5)
    tcn_baseline.add_argument("--batch-size", type=int, default=256)
    tcn_baseline.add_argument("--learning-rate", type=float, default=1e-3)
    tcn_baseline.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    tcn_baseline.add_argument("--hidden-channels", type=int, default=32)
    tcn_baseline.add_argument("--num-levels", type=int, default=3)
    tcn_baseline.add_argument("--kernel-size", type=int, default=3)
    tcn_baseline.add_argument("--dropout", type=float, default=0.1)
    tcn_baseline.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    tcn_baseline.add_argument("--random-state", type=int, default=42)
    tcn_baseline.add_argument("--device", default="cpu")
    tcn_baseline.add_argument("--output-json", type=Path)
    tcn_baseline.add_argument("--output-csv", type=Path)
    tcn_baseline.add_argument("--history-json", type=Path)

    transformer_baseline = subparsers.add_parser(
        "cmapss-transformer-baseline",
        help="Train a Transformer encoder RUL baseline from exported C-MAPSS sequence tensors",
    )
    transformer_baseline.add_argument("--sequence-dir", type=Path, required=True)
    transformer_baseline.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    transformer_baseline.add_argument("--epochs", type=int, default=5)
    transformer_baseline.add_argument("--batch-size", type=int, default=256)
    transformer_baseline.add_argument("--learning-rate", type=float, default=1e-3)
    transformer_baseline.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    transformer_baseline.add_argument("--d-model", type=int, default=32)
    transformer_baseline.add_argument("--num-heads", type=int, default=4)
    transformer_baseline.add_argument("--num-layers", type=int, default=2)
    transformer_baseline.add_argument("--dim-feedforward", type=int, default=64)
    transformer_baseline.add_argument("--dropout", type=float, default=0.1)
    transformer_baseline.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    transformer_baseline.add_argument("--random-state", type=int, default=42)
    transformer_baseline.add_argument("--device", default="cpu")
    transformer_baseline.add_argument("--output-json", type=Path)
    transformer_baseline.add_argument("--output-csv", type=Path)
    transformer_baseline.add_argument("--history-json", type=Path)

    deep_compare = subparsers.add_parser(
        "cmapss-deep-baseline-compare",
        help="Compare Phase 2 deep RUL baselines across compact hyperparameter grids",
    )
    deep_compare.add_argument("--sequence-dir", type=Path, required=True)
    deep_compare.add_argument(
        "--subsets",
        nargs="+",
        choices=CMAPSS_SUBSETS,
        default=list(CMAPSS_SUBSETS),
    )
    deep_compare.add_argument(
        "--models",
        nargs="+",
        choices=CMAPSS_DEEP_COMPARISON_MODELS,
        default=["cnn", "bilstm", "tcn"],
    )
    deep_compare.add_argument("--epochs", type=int, default=5)
    deep_compare.add_argument("--batch-size", type=int, default=256)
    deep_compare.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3])
    deep_compare.add_argument(
        "--training-loss",
        choices=CMAPSS_DEEP_TRAINING_LOSSES,
        default="mse",
    )
    deep_compare.add_argument("--hidden-sizes", nargs="+", type=int, default=[32])
    deep_compare.add_argument("--num-layers", type=int, default=1)
    deep_compare.add_argument("--tcn-levels", type=int, default=3)
    deep_compare.add_argument(
        "--tcn-normalization",
        choices=["none", "layer_norm"],
        default="none",
    )
    deep_compare.add_argument("--tcn-weight-norm", action="store_true")
    deep_compare.add_argument("--tcn-pooling", choices=["last", "mean"], default="last")
    deep_compare.add_argument("--transformer-heads", type=int, default=4)
    deep_compare.add_argument("--transformer-dim-feedforward", type=int)
    deep_compare.add_argument("--kernel-size", type=int, default=3)
    deep_compare.add_argument("--dropout", type=float, default=0.1)
    deep_compare.add_argument(
        "--checkpoint-policy",
        choices=["validation_nasa", "final"],
        default="validation_nasa",
    )
    deep_compare.add_argument("--random-state", type=int, default=42)
    deep_compare.add_argument("--device", default="cpu")
    deep_compare.add_argument("--output-json", type=Path)
    deep_compare.add_argument("--output-csv", type=Path)

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

    smap_msl_summary = subparsers.add_parser(
        "smap-msl-summary",
        help="Summarise local Telemanom SMAP/MSL labels and optional channel arrays",
    )
    smap_msl_summary.add_argument("--data-dir", type=Path, required=True)
    smap_msl_summary.add_argument("--channel-id")

    smap_msl_select = subparsers.add_parser(
        "smap-msl-select-channels",
        help="Select deterministic SMAP/MSL channels for bounded benchmark sweeps",
    )
    smap_msl_select.add_argument("--data-dir", type=Path, required=True)
    smap_msl_select.add_argument("--count", type=int, default=20)
    smap_msl_select.add_argument(
        "--strategy",
        choices=["balanced", "label_order"],
        default="balanced",
    )
    smap_msl_select.add_argument("--spacecraft", nargs="+")
    smap_msl_select.add_argument("--min-anomaly-sequences", type=int, default=1)
    smap_msl_select.add_argument("--output-json", type=Path)
    smap_msl_select.add_argument("--output-csv", type=Path)

    smap_msl_export = subparsers.add_parser(
        "smap-msl-export-channel-csv",
        help="Export one Telemanom SMAP/MSL channel to train/test CSVs",
    )
    smap_msl_export.add_argument("--data-dir", type=Path, required=True)
    smap_msl_export.add_argument("--channel-id", required=True)
    smap_msl_export.add_argument("--output-dir", type=Path, required=True)
    smap_msl_export.add_argument("--metadata-json", type=Path)

    smap_msl_classical = subparsers.add_parser(
        "smap-msl-classical-baselines",
        help="Run classical anomaly baselines across raw Telemanom SMAP/MSL channels",
    )
    smap_msl_classical.add_argument("--data-dir", type=Path, required=True)
    smap_msl_classical.add_argument("--channels", nargs="+")
    smap_msl_classical.add_argument("--max-channels", type=int)
    smap_msl_classical.add_argument(
        "--methods",
        nargs="+",
        choices=CLASSICAL_ANOMALY_BASELINE_METHODS,
        default=list(CLASSICAL_ANOMALY_BASELINE_METHODS),
    )
    smap_msl_classical.add_argument("--robust-threshold", type=float, default=3.5)
    smap_msl_classical.add_argument("--pca-components", type=int)
    smap_msl_classical.add_argument("--pca-threshold-quantile", type=float, default=0.99)
    smap_msl_classical.add_argument("--isolation-contamination", type=float, default=0.05)
    smap_msl_classical.add_argument("--random-state", type=int, default=42)
    smap_msl_classical.add_argument("--output-json", type=Path)
    smap_msl_classical.add_argument("--output-csv", type=Path)

    smap_msl_robust_sweep = subparsers.add_parser(
        "smap-msl-robust-threshold-sweep",
        help="Sweep robust z-score thresholds across raw Telemanom SMAP/MSL channels",
    )
    smap_msl_robust_sweep.add_argument("--data-dir", type=Path, required=True)
    smap_msl_robust_sweep.add_argument("--channels", nargs="+")
    smap_msl_robust_sweep.add_argument("--max-channels", type=int)
    smap_msl_robust_sweep.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[3.5, 5.0, 7.0, 10.0],
    )
    smap_msl_robust_sweep.add_argument("--output-json", type=Path)
    smap_msl_robust_sweep.add_argument("--output-csv", type=Path)
    smap_msl_robust_sweep.add_argument("--aggregate-json", type=Path)
    smap_msl_robust_sweep.add_argument("--aggregate-csv", type=Path)
    smap_msl_robust_sweep.add_argument("--false-alarm-budget", type=float)
    smap_msl_robust_sweep.add_argument(
        "--selection-group",
        choices=["spacecraft", "global"],
        default="spacecraft",
    )
    smap_msl_robust_sweep.add_argument("--operating-point-json", type=Path)
    smap_msl_robust_sweep.add_argument("--operating-point-csv", type=Path)
    smap_msl_robust_sweep.add_argument("--policy-json", type=Path)
    smap_msl_robust_sweep.add_argument("--policy-csv", type=Path)

    smap_msl_lstm = subparsers.add_parser(
        "smap-msl-lstm-forecast-baseline",
        help="Run an LSTM one-step forecast anomaly baseline across SMAP/MSL channels",
    )
    smap_msl_lstm.add_argument("--data-dir", type=Path, required=True)
    smap_msl_lstm.add_argument("--channels", nargs="+")
    smap_msl_lstm.add_argument("--max-channels", type=int)
    smap_msl_lstm.add_argument("--window-size", type=int, default=30)
    smap_msl_lstm.add_argument("--hidden-size", type=int, default=32)
    smap_msl_lstm.add_argument("--num-layers", type=int, default=1)
    smap_msl_lstm.add_argument("--dropout", type=float, default=0.0)
    smap_msl_lstm.add_argument("--epochs", type=int, default=10)
    smap_msl_lstm.add_argument("--batch-size", type=int, default=64)
    smap_msl_lstm.add_argument("--learning-rate", type=float, default=1e-3)
    smap_msl_lstm.add_argument("--threshold-sigma", type=float, default=3.0)
    smap_msl_lstm.add_argument(
        "--threshold-method",
        choices=LSTM_FORECAST_THRESHOLD_METHODS,
        default="robust",
    )
    smap_msl_lstm.add_argument("--dynamic-batch-size", type=int, default=70)
    smap_msl_lstm.add_argument("--dynamic-window-batches", type=int, default=30)
    smap_msl_lstm.add_argument("--dynamic-smoothing-fraction", type=float, default=0.05)
    smap_msl_lstm.add_argument("--dynamic-z-start", type=float, default=2.5)
    smap_msl_lstm.add_argument("--dynamic-z-stop", type=float, default=12.0)
    smap_msl_lstm.add_argument("--dynamic-z-step", type=float, default=0.5)
    smap_msl_lstm.add_argument("--dynamic-error-buffer", type=int, default=100)
    smap_msl_lstm.add_argument("--dynamic-prune-p", type=float, default=0.13)
    smap_msl_lstm.add_argument("--random-state", type=int, default=42)
    smap_msl_lstm.add_argument("--device", default="cpu")
    smap_msl_lstm.add_argument("--output-json", type=Path)
    smap_msl_lstm.add_argument("--output-csv", type=Path)

    smap_msl_compare = subparsers.add_parser(
        "smap-msl-compare-anomaly-results",
        help="Rank SMAP/MSL anomaly result CSVs across classical and forecast baselines",
    )
    smap_msl_compare.add_argument("--result-csv", nargs="+", type=Path, required=True)
    smap_msl_compare.add_argument("--source-labels", nargs="+")
    smap_msl_compare.add_argument("--output-csv", type=Path)
    smap_msl_compare.add_argument("--output-markdown", type=Path)

    register_deployment_artifact_commands(subparsers)

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

    register_deployment_release_commands(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "quickstart-cmapss-demo":
        return run_cmapss_quickstart(
            root=args.output_dir,
            release_name=args.release_name,
            repository=args.repository,
            git_sha=args.git_sha,
            git_ref=args.git_ref,
            workflow=args.workflow,
            run_id=args.run_id,
            lockfile=args.lockfile,
            runner=main,
        )

    app_result = handle_app_command(args)
    if app_result is not None:
        return app_result

    deployment_result = handle_deployment_command(args)
    if deployment_result is not None:
        return deployment_result

    if args.command == "cmapss-summary":
        bundle = load_cmapss_subset(args.data_dir, args.subset, rul_cap=args.rul_cap)
        train_summary = summarise_cmapss_frame(bundle.train)
        test_summary = summarise_cmapss_frame(bundle.test)
        print(f"subset={bundle.subset}")
        print(
            "train_rows="
            f"{train_summary.rows} train_units={train_summary.units} "
            f"train_cycle_range={train_summary.min_cycle}-{train_summary.max_cycle} "
            f"train_unit_cycle_range={train_summary.min_unit_cycles}-{train_summary.max_unit_cycles}"
        )
        print(
            "test_rows="
            f"{test_summary.rows} test_units={test_summary.units} "
            f"test_cycle_range={test_summary.min_cycle}-{test_summary.max_cycle} "
            f"test_unit_cycle_range={test_summary.min_unit_cycles}-{test_summary.max_unit_cycles}"
        )
        print(f"test_rul_values={len(bundle.test_rul)}")
        return 0

    if args.command == "cmapss-baseline":
        result = run_cmapss_hist_gradient_boosting(
            args.data_dir,
            args.subset,
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=args.standardize,
        )
        print(f"dataset={result.dataset}")
        print(f"subset={result.subset}")
        print(f"model={result.model_name}")
        print(f"standardize={result.standardize}")
        print(f"rmse={result.rmse:.6f}")
        print(f"nasa_score={result.nasa_score:.6f}")
        if args.output_json is not None:
            result.write_json(args.output_json)
        return 0

    if args.command == "cmapss-baseline-all":
        results = run_all_cmapss_hist_gradient_boosting(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=args.standardize,
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-engineered-baseline":
        result = run_cmapss_engineered_hist_gradient_boosting(
            args.data_dir,
            args.subset,
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            rolling_window=args.rolling_window,
            standardize=not args.no_standardize,
        )
        print(f"dataset={result.dataset}")
        print(f"subset={result.subset}")
        print(f"model={result.model_name}")
        print(f"standardize={result.standardize}")
        print(f"rmse={result.rmse:.6f}")
        print(f"nasa_score={result.nasa_score:.6f}")
        if args.output_json is not None:
            result.write_json(args.output_json)
        return 0

    if args.command == "cmapss-engineered-baseline-all":
        results = run_all_cmapss_engineered_hist_gradient_boosting(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            rolling_window=args.rolling_window,
            standardize=not args.no_standardize,
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-engineered-window-sweep":
        results = run_cmapss_engineered_window_sweep(
            args.data_dir,
            subsets=tuple(args.subsets),
            rolling_windows=tuple(args.rolling_windows),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=not args.no_standardize,
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-engineered-best-baseline-all":
        results = run_all_cmapss_engineered_default_windows(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            standardize=not args.no_standardize,
        )
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-regime-engineered-best-baseline-all":
        results = run_all_cmapss_regime_aware_engineered_default_windows(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
        )
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-validation-selected-baseline-all":
        results = run_all_cmapss_validation_selected_default_windows(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
        )
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(
            "feature_policy="
            + ",".join(
                f"{subset}:{CMAPSS_VALIDATION_SELECTED_FEATURES[subset]}"
                for subset in args.subsets
            )
        )
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-hgb-policy-baseline-all":
        results = run_all_cmapss_validation_selected_hgb_policy_default_windows(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
        )
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(
            "feature_policy="
            + ",".join(
                f"{subset}:{CMAPSS_VALIDATION_SELECTED_FEATURES[subset]}"
                for subset in args.subsets
            )
        )
        print(
            "hgb_policy="
            + ",".join(
                f"{subset}:{CMAPSS_VALIDATION_SELECTED_HGB_PARAMS[subset]}"
                for subset in args.subsets
            )
        )
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-validate-feature-candidates":
        results = run_cmapss_validation_feature_comparison(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            validation_fraction=args.validation_fraction,
            validation_horizon=args.validation_horizon,
            standardize=not args.no_standardize,
        )
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizon={args.validation_horizon}")
        print(f"max_regimes={args.n_regimes}")
        _print_results_table(results)
        print(
            "selected_by_nasa="
            + ",".join(
                f"{subset}:{_best_result_for_subset(results, subset).model_name}"
                for subset in args.subsets
            )
        )
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-validate-feature-candidates-repeated":
        results = run_cmapss_repeated_validation_feature_comparison(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_states=tuple(args.random_states),
            n_regimes=args.n_regimes,
            validation_fraction=args.validation_fraction,
            validation_horizons=tuple(args.validation_horizons),
            standardize=not args.no_standardize,
        )
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizons={','.join(str(value) for value in args.validation_horizons)}")
        print(f"random_states={','.join(str(value) for value in args.random_states)}")
        print(f"max_regimes={args.n_regimes}")
        _print_validation_aggregate_table(results)
        print(
            "selected_by_mean_nasa="
            + ",".join(
                f"{subset}:{_best_aggregate_for_subset(results, subset).model_name}"
                for subset in args.subsets
            )
        )
        if args.output_json is not None:
            _write_validation_aggregate_json(results, args.output_json)
        if args.output_csv is not None:
            _write_validation_aggregate_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-validate-hgb-grid":
        results = run_cmapss_validation_selected_hgb_grid(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            validation_fraction=args.validation_fraction,
            validation_horizon=args.validation_horizon,
            standardize=not args.no_standardize,
        )
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(
            "feature_policy="
            + ",".join(
                f"{subset}:{CMAPSS_VALIDATION_SELECTED_FEATURES[subset]}"
                for subset in args.subsets
            )
        )
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizon={args.validation_horizon}")
        print(f"param_grid={','.join(str(params['label']) for params in CMAPSS_HGB_PARAM_GRID)}")
        _print_results_table(results)
        print(
            "selected_by_nasa="
            + ",".join(
                f"{subset}:{_best_result_for_subset(results, subset).model_name}"
                for subset in args.subsets
            )
        )
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-validate-sensor-filters":
        results = run_cmapss_validation_sensor_filter_comparison(
            args.data_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            validation_fraction=args.validation_fraction,
            validation_horizon=args.validation_horizon,
            min_abs_rul_correlation=args.min_abs_rul_correlation,
            min_abs_standardized_drift=args.min_abs_standardized_drift,
            standardize=not args.no_standardize,
        )
        print(
            "rolling_windows="
            + ",".join(
                f"{subset}:{CMAPSS_ENGINEERED_DEFAULT_WINDOWS[subset]}"
                for subset in args.subsets
            )
        )
        print(
            "feature_policy="
            + ",".join(
                f"{subset}:{CMAPSS_VALIDATION_SELECTED_FEATURES[subset]}"
                for subset in args.subsets
            )
        )
        print(
            "hgb_policy="
            + ",".join(
                f"{subset}:{CMAPSS_VALIDATION_SELECTED_HGB_PARAMS[subset]}"
                for subset in args.subsets
            )
        )
        print(f"sensor_filter_candidates={','.join(CMAPSS_SENSOR_FILTER_CANDIDATES)}")
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizon={args.validation_horizon}")
        print(f"min_abs_rul_correlation={args.min_abs_rul_correlation}")
        print(f"min_abs_standardized_drift={args.min_abs_standardized_drift}")
        _print_results_table(results)
        print(
            "selected_by_nasa="
            + ",".join(
                f"{subset}:{_best_result_for_subset(results, subset).model_name}"
                for subset in args.subsets
            )
        )
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

    if args.command == "cmapss-eda":
        bundle = load_cmapss_subset(args.data_dir, args.subset, rul_cap=args.rul_cap)
        report = build_cmapss_eda_report(bundle)
        flat_sensors = [
            summary.sensor for summary in report.sensor_summaries if summary.is_near_constant
        ]
        largest_drift = max(report.sensor_summaries, key=lambda summary: abs(summary.drift))
        print(f"subset={report.subset}")
        print(f"train_rows={report.train_rows} train_units={report.train_units}")
        print(f"test_rows={report.test_rows} test_units={report.test_units}")
        print(f"near_constant_sensors={','.join(flat_sensors) if flat_sensors else 'none'}")
        print(f"largest_abs_drift_sensor={largest_drift.sensor} drift={largest_drift.drift:.6f}")
        if args.output_json is not None:
            report.write_json(args.output_json)
        return 0

    if args.command == "cmapss-manifest":
        manifest = build_cmapss_manifest(args.data_dir, subsets=tuple(args.subsets))
        manifest.write_json(args.output_json)
        print(f"dataset={manifest.dataset}")
        print(f"files={len(manifest.entries)}")
        print(f"manifest={args.output_json}")
        return 0

    if args.command == "cmapss-verify":
        manifest = read_manifest(args.manifest)
        problems = verify_manifest(manifest, root=args.data_dir)
        if problems:
            print("status=failed")
            for problem in problems:
                print(f"problem={problem}")
            return 1
        print("status=ok")
        print(f"files={len(manifest.entries)}")
        return 0

    if args.command == "cmapss-download":
        result = download_cmapss_dataset(
            args.output_dir,
            source_url=args.source_url,
            archive_path=args.archive_path,
            force=args.force,
        )
        print(f"source_url={result.source_url}")
        print(f"archive={result.archive_path}")
        print(f"output_dir={result.output_dir}")
        print(f"metadata={result.metadata_path}")
        print(f"files={len(result.extracted_files)}")
        return 0

    if args.command == "smap-msl-download":
        try:
            result = download_smap_msl_dataset(
                args.output_dir,
                source_url=args.source_url,
                labels_url=args.labels_url,
                archive_path=args.archive_path,
                force=args.force,
            )
        except RuntimeError as exc:
            print("status=failed")
            for line in str(exc).splitlines():
                print(f"problem={line}")
            return 1
        print(f"source_url={result.source_url}")
        print(f"labels_url={result.labels_url}")
        print(f"archive={result.archive_path}")
        print(f"output_dir={result.output_dir}")
        print(f"labels={result.labels_path}")
        print(f"metadata={result.metadata_path}")
        print(f"arrays={len(result.extracted_arrays)}")
        return 0

    if args.command == "phase1-cmapss":
        result = run_phase1_cmapss_workflow(
            args.data_dir,
            args.artifact_dir,
            subsets=tuple(args.subsets),
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
        )
        print(f"artifact_dir={result.artifact_dir}")
        print(f"manifest={result.manifest_path}")
        print(f"baseline_json={result.baseline_json_path}")
        print(f"baseline_csv={result.baseline_csv_path}")
        print(f"hgb_policy_json={result.hgb_policy_json_path}")
        print(f"hgb_policy_csv={result.hgb_policy_csv_path}")
        print(f"sensor_filter_json={result.sensor_filter_json_path}")
        print(f"sensor_filter_csv={result.sensor_filter_csv_path}")
        print(f"summary={result.summary_markdown_path}")
        print(f"eda_reports={len(result.eda_paths)}")
        return 0

    if args.command == "phase2-cmapss":
        from aerospace_prognostics.workflows.phase2 import run_phase2_cmapss_workflow

        result = run_phase2_cmapss_workflow(
            args.data_dir,
            args.artifact_dir,
            subsets=tuple(args.subsets),
            window_size=args.window_size,
            stride=args.stride,
            validation_fraction=args.validation_fraction,
            validation_horizon=args.validation_horizon,
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
            models=tuple(args.models),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rates=tuple(args.learning_rates),
            training_loss=args.training_loss,
            hidden_sizes=tuple(args.hidden_sizes),
            num_layers=args.num_layers,
            tcn_levels=args.tcn_levels,
            tcn_normalization=args.tcn_normalization,
            tcn_weight_norm=args.tcn_weight_norm,
            tcn_pooling=args.tcn_pooling,
            transformer_heads=args.transformer_heads,
            transformer_dim_feedforward=args.transformer_dim_feedforward,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            checkpoint_policy=args.checkpoint_policy,
            device=args.device,
        )
        print(f"artifact_dir={result.artifact_dir}")
        print(f"sequence_dir={result.sequence_dir}")
        print(f"hgb_policy_json={result.hgb_policy_json_path}")
        print(f"hgb_policy_csv={result.hgb_policy_csv_path}")
        print(f"deep_compare_json={result.deep_compare_json_path}")
        print(f"deep_compare_csv={result.deep_compare_csv_path}")
        print(f"deep_predictions_csv={result.deep_predictions_csv_path}")
        print(
            "deep_validation_selection_predictions_csv="
            f"{result.deep_validation_selection_predictions_csv_path}"
        )
        print(f"deep_prediction_diagnostics_csv={result.deep_prediction_diagnostics_csv_path}")
        print(
            "deep_validation_selection_prediction_diagnostics_csv="
            f"{result.deep_validation_selection_prediction_diagnostics_csv_path}"
        )
        print(
            "deep_prediction_rul_bin_diagnostics_csv="
            f"{result.deep_prediction_rul_bin_diagnostics_csv_path}"
        )
        print(
            "deep_validation_selection_prediction_rul_bin_diagnostics_csv="
            f"{result.deep_validation_selection_prediction_rul_bin_diagnostics_csv_path}"
        )
        print(
            "deep_prediction_monotonicity_diagnostics_csv="
            f"{result.deep_prediction_monotonicity_diagnostics_csv_path}"
        )
        print(
            "deep_validation_selection_prediction_monotonicity_diagnostics_csv="
            f"{result.deep_validation_selection_prediction_monotonicity_diagnostics_csv_path}"
        )
        print(
            "deep_prediction_unit_diagnostics_csv="
            f"{result.deep_prediction_unit_diagnostics_csv_path}"
        )
        print(
            "deep_validation_selection_prediction_unit_diagnostics_csv="
            f"{result.deep_validation_selection_prediction_unit_diagnostics_csv_path}"
        )
        print(
            "deep_prediction_diagnostics_markdown="
            f"{result.deep_prediction_diagnostics_markdown_path}"
        )
        print(
            "deep_validation_selection_prediction_diagnostics_markdown="
            f"{result.deep_validation_selection_prediction_diagnostics_markdown_path}"
        )
        print(f"comparison_csv={result.comparison_csv_path}")
        print(f"comparison_markdown={result.comparison_markdown_path}")
        print(f"summary={result.summary_markdown_path}")
        print(f"run_manifest={result.run_manifest_path}")
        print(f"sequence_exports={len(result.sequence_exports)}")
        print(f"deep_results={len(result.deep_compare_results)}")
        print(f"comparison_rows={len(result.comparison_rows)}")
        return 0

    if args.command == "phase2-cmapss-verify-manifest":
        from aerospace_prognostics.workflows.phase2 import (
            verify_phase2_cmapss_run_manifest,
            write_phase2_cmapss_manifest_audit_markdown,
        )

        result = verify_phase2_cmapss_run_manifest(args.manifest, root=args.root)
        if args.output_markdown is not None:
            audit_path = write_phase2_cmapss_manifest_audit_markdown(
                result,
                args.output_markdown,
            )
            print(f"audit_markdown={audit_path}")
        print(f"status={'ok' if result.ok else 'failed'}")
        print(f"manifest={result.manifest_path}")
        print(f"artifacts_checked={len(result.checked_artifacts)}")
        for problem in result.problems:
            print(f"problem={problem}")
        return 0 if result.ok else 1

    if args.command == "phase2-smap-msl":
        from aerospace_prognostics.anomaly.forecasting import DynamicThresholdConfig
        from aerospace_prognostics.workflows.phase2_smap_msl import run_phase2_smap_msl_workflow

        result = run_phase2_smap_msl_workflow(
            args.data_dir,
            args.artifact_dir,
            channels=tuple(args.channels) if args.channels is not None else None,
            max_channels=(
                args.max_channels
                if args.max_channels is not None
                else _phase2_smap_msl_default_max_channels(args.channels)
            ),
            classical_methods=tuple(args.classical_methods),
            robust_threshold=args.robust_threshold,
            pca_components=args.pca_components,
            pca_threshold_quantile=args.pca_threshold_quantile,
            isolation_contamination=args.isolation_contamination,
            window_size=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold_sigma=args.threshold_sigma,
            robust_policy_false_alarm_budget=args.robust_policy_false_alarm_budget,
            robust_policy_thresholds=tuple(args.robust_policy_thresholds),
            robust_policy_group_by=args.robust_policy_group_by,
            dynamic_threshold_config=DynamicThresholdConfig(
                batch_size=args.dynamic_batch_size,
                window_batches=args.dynamic_window_batches,
                smoothing_fraction=args.dynamic_smoothing_fraction,
                z_start=args.dynamic_z_start,
                z_stop=args.dynamic_z_stop,
                z_step=args.dynamic_z_step,
                error_buffer=args.dynamic_error_buffer,
                p=args.dynamic_prune_p,
            ),
            random_state=args.random_state,
            device=args.device,
        )
        print(f"artifact_dir={result.artifact_dir}")
        print(f"classical_csv={result.classical_csv_path}")
        print(f"lstm_robust_csv={result.lstm_robust_csv_path}")
        print(f"lstm_dynamic_csv={result.lstm_dynamic_csv_path}")
        if result.robust_threshold_policy_csv_path is not None:
            print(f"robust_threshold_policy_csv={result.robust_threshold_policy_csv_path}")
            print(
                "robust_threshold_operating_point_csv="
                f"{result.robust_threshold_operating_point_csv_path}"
            )
        print(f"comparison_csv={result.comparison_csv_path}")
        print(f"comparison_markdown={result.comparison_markdown_path}")
        print(f"summary={result.summary_markdown_path}")
        print(f"run_manifest={result.run_manifest_path}")
        print(f"classical_runs={len(result.classical_runs)}")
        print(f"lstm_robust_runs={len(result.lstm_robust_runs)}")
        print(f"lstm_dynamic_runs={len(result.lstm_dynamic_runs)}")
        print(f"robust_threshold_policy_runs={len(result.robust_threshold_policy_runs)}")
        print(f"comparison_rows={len(result.comparison_rows)}")
        return 0

    if args.command == "phase2-smap-msl-verify-manifest":
        from aerospace_prognostics.workflows.phase2_smap_msl import (
            verify_phase2_smap_msl_run_manifest,
            write_phase2_smap_msl_manifest_audit_markdown,
        )

        result = verify_phase2_smap_msl_run_manifest(args.manifest, root=args.root)
        if args.output_markdown is not None:
            audit_path = write_phase2_smap_msl_manifest_audit_markdown(
                result,
                args.output_markdown,
            )
            print(f"audit_markdown={audit_path}")
        print(f"status={'ok' if result.ok else 'failed'}")
        print(f"manifest={result.manifest_path}")
        print(f"artifacts_checked={len(result.checked_artifacts)}")
        for problem in result.problems:
            print(f"problem={problem}")
        return 0 if result.ok else 1

    if args.command == "cmapss-export-sequences":
        results = [
            export_cmapss_sequence_splits(
                args.data_dir,
                args.output_dir,
                subset,
                window_size=args.window_size,
                stride=args.stride,
                rul_cap=args.rul_cap,
                random_state=args.random_state,
                validation_fraction=args.validation_fraction,
                validation_horizon=args.validation_horizon,
                standardize=not args.no_standardize,
            )
            for subset in args.subsets
        ]
        print(f"window_size={args.window_size}")
        print(f"stride={args.stride}")
        print(f"validation_fraction={args.validation_fraction}")
        print(f"validation_horizon={args.validation_horizon}")
        print(f"standardize={not args.no_standardize}")
        for result in results:
            print(
                f"{result.subset},"
                f"train_windows={result.train_windows},"
                f"validation_windows={result.validation_windows},"
                f"validation_selection_windows={result.validation_selection_windows},"
                f"test_windows={result.test_windows},"
                f"features={len(result.feature_columns)},"
                f"metadata={result.metadata_path}"
            )
        return 0

    if args.command == "cmapss-cnn-baseline":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_all_cmapss_cnn_baseline_runs,
        )

        runs = run_all_cmapss_cnn_baseline_runs(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            training_loss=args.training_loss,
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        results = [run.result for run in runs]
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(f"learning_rate={args.learning_rate}")
        print(f"training_loss={args.training_loss}")
        print(f"hidden_channels={args.hidden_channels}")
        print(f"kernel_size={args.kernel_size}")
        print(f"dropout={args.dropout}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        print(
            "selected_epochs="
            + ",".join(f"{run.result.subset}:{run.selected_epoch}" for run in runs)
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        if args.history_json is not None:
            _write_deep_history_json(runs, args.history_json)
        return 0

    if args.command == "cmapss-lstm-baseline":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_all_cmapss_lstm_baseline_runs,
        )

        runs = run_all_cmapss_lstm_baseline_runs(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            training_loss=args.training_loss,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        results = [run.result for run in runs]
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(f"learning_rate={args.learning_rate}")
        print(f"training_loss={args.training_loss}")
        print(f"hidden_size={args.hidden_size}")
        print(f"num_layers={args.num_layers}")
        print(f"dropout={args.dropout}")
        print(f"bidirectional={args.bidirectional}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        print(
            "selected_epochs="
            + ",".join(f"{run.result.subset}:{run.selected_epoch}" for run in runs)
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        if args.history_json is not None:
            _write_deep_history_json(runs, args.history_json)
        return 0

    if args.command == "cmapss-tcn-baseline":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_all_cmapss_tcn_baseline_runs,
        )

        runs = run_all_cmapss_tcn_baseline_runs(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            training_loss=args.training_loss,
            hidden_channels=args.hidden_channels,
            num_levels=args.num_levels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        results = [run.result for run in runs]
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(f"learning_rate={args.learning_rate}")
        print(f"training_loss={args.training_loss}")
        print(f"hidden_channels={args.hidden_channels}")
        print(f"num_levels={args.num_levels}")
        print(f"kernel_size={args.kernel_size}")
        print(f"dropout={args.dropout}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        print(
            "selected_epochs="
            + ",".join(f"{run.result.subset}:{run.selected_epoch}" for run in runs)
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        if args.history_json is not None:
            _write_deep_history_json(runs, args.history_json)
        return 0

    if args.command == "cmapss-transformer-baseline":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_all_cmapss_transformer_baseline_runs,
        )

        runs = run_all_cmapss_transformer_baseline_runs(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            training_loss=args.training_loss,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        results = [run.result for run in runs]
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(f"learning_rate={args.learning_rate}")
        print(f"training_loss={args.training_loss}")
        print(f"d_model={args.d_model}")
        print(f"num_heads={args.num_heads}")
        print(f"num_layers={args.num_layers}")
        print(f"dim_feedforward={args.dim_feedforward}")
        print(f"dropout={args.dropout}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        print(
            "selected_epochs="
            + ",".join(f"{run.result.subset}:{run.selected_epoch}" for run in runs)
        )
        _print_results_table(results)
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        if args.history_json is not None:
            _write_deep_history_json(runs, args.history_json)
        return 0

    if args.command == "cmapss-deep-baseline-compare":
        from aerospace_prognostics.experiments.cmapss_deep_baseline import (
            run_cmapss_deep_baseline_comparison,
        )

        results = run_cmapss_deep_baseline_comparison(
            args.sequence_dir,
            subsets=tuple(args.subsets),
            models=tuple(args.models),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rates=tuple(args.learning_rates),
            training_loss=args.training_loss,
            hidden_sizes=tuple(args.hidden_sizes),
            num_layers=args.num_layers,
            tcn_levels=args.tcn_levels,
            tcn_normalization=args.tcn_normalization,
            tcn_weight_norm=args.tcn_weight_norm,
            tcn_pooling=args.tcn_pooling,
            transformer_heads=args.transformer_heads,
            transformer_dim_feedforward=args.transformer_dim_feedforward,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            checkpoint_policy=args.checkpoint_policy,
            random_state=args.random_state,
            device=args.device,
        )
        print(f"models={','.join(args.models)}")
        print(f"epochs={args.epochs}")
        print(f"batch_size={args.batch_size}")
        print(
            "learning_rates="
            + ",".join(_format_cli_float(value) for value in args.learning_rates)
        )
        print(f"training_loss={args.training_loss}")
        print(f"hidden_sizes={','.join(str(value) for value in args.hidden_sizes)}")
        print(f"num_layers={args.num_layers}")
        print(f"tcn_levels={args.tcn_levels}")
        print(f"tcn_normalization={args.tcn_normalization}")
        print(f"tcn_weight_norm={args.tcn_weight_norm}")
        print(f"tcn_pooling={args.tcn_pooling}")
        print(f"transformer_heads={args.transformer_heads}")
        print(f"transformer_dim_feedforward={args.transformer_dim_feedforward}")
        print(f"kernel_size={args.kernel_size}")
        print(f"dropout={args.dropout}")
        print(f"checkpoint_policy={args.checkpoint_policy}")
        print(f"device={args.device}")
        _print_results_table(results)
        print(
            "selected_by_nasa="
            + ",".join(
                f"{subset}:{_best_result_for_subset(results, subset).model_name}"
                for subset in args.subsets
            )
        )
        if args.output_json is not None:
            write_results_json(results, args.output_json)
        if args.output_csv is not None:
            write_results_csv(results, args.output_csv)
        return 0

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
            _write_json_payload(result.to_dict(), args.output_json)
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
            _write_json_payload([result.to_dict() for result in results], args.output_json)
        if args.output_csv is not None:
            _write_classical_anomaly_summary_csv(results, args.output_csv)
        if args.predictions_csv is not None:
            _write_classical_anomaly_predictions_csv(
                labels=labels,
                results=results,
                path=args.predictions_csv,
            )
        return 0

    if args.command == "smap-msl-summary":
        if args.channel_id is None:
            labels = read_smap_msl_labels(args.data_dir)
            spacecraft_counts: dict[str, int] = {}
            for metadata in labels:
                spacecraft_counts[metadata.spacecraft] = (
                    spacecraft_counts.get(metadata.spacecraft, 0) + 1
                )
            print(f"channels={len(labels)}")
            print(f"anomaly_sequences={sum(len(item.anomaly_sequences) for item in labels)}")
            print(
                "spacecraft="
                + ",".join(
                    f"{spacecraft}:{count}"
                    for spacecraft, count in sorted(spacecraft_counts.items())
                )
            )
            return 0

        channel = load_smap_msl_channel(args.data_dir, args.channel_id)
        print(f"channel_id={channel.metadata.channel_id}")
        print(f"spacecraft={channel.metadata.spacecraft}")
        print(f"train_shape={channel.train_values.shape[0]}x{channel.train_values.shape[1]}")
        print(f"test_shape={channel.test_values.shape[0]}x{channel.test_values.shape[1]}")
        print(f"anomaly_sequences={len(channel.metadata.anomaly_sequences)}")
        print(f"labelled_anomaly_points={int(channel.test_labels.sum())}")
        return 0

    if args.command == "smap-msl-select-channels":
        selections = select_smap_msl_channels(
            args.data_dir,
            count=args.count,
            strategy=args.strategy,
            spacecraft=tuple(args.spacecraft) if args.spacecraft is not None else None,
            min_anomaly_sequences=args.min_anomaly_sequences,
        )
        print(f"selected_channels={len(selections)}")
        print("channels=" + " ".join(selection.channel_id for selection in selections))
        _print_smap_msl_channel_selection_table(selections)
        if args.output_json is not None:
            write_smap_msl_channel_selection_json(selections, args.output_json)
        if args.output_csv is not None:
            write_smap_msl_channel_selection_csv(selections, args.output_csv)
        return 0

    if args.command == "smap-msl-export-channel-csv":
        export = export_smap_msl_channel_csv(args.data_dir, args.channel_id, args.output_dir)
        print(f"channel_id={export.channel_id}")
        print(f"train_csv={export.train_csv}")
        print(f"test_csv={export.test_csv}")
        print(f"train_rows={export.train_rows}")
        print(f"test_rows={export.test_rows}")
        print(f"features={len(export.feature_names)}")
        if args.metadata_json is not None:
            _write_json_payload(export.to_dict(), args.metadata_json)
        return 0

    if args.command == "smap-msl-classical-baselines":
        runs = run_smap_msl_classical_baselines(
            args.data_dir,
            channels=tuple(args.channels) if args.channels is not None else None,
            max_channels=args.max_channels,
            methods=tuple(args.methods),
            robust_threshold=args.robust_threshold,
            pca_components=args.pca_components,
            pca_threshold_quantile=args.pca_threshold_quantile,
            isolation_contamination=args.isolation_contamination,
            random_state=args.random_state,
        )
        print(f"channels={len(_smap_msl_result_channels(runs))}")
        print(f"runs={len(runs)}")
        _print_smap_msl_classical_table(runs)
        if args.output_json is not None:
            write_smap_msl_classical_baselines_json(runs, args.output_json)
        if args.output_csv is not None:
            write_smap_msl_classical_baselines_csv(runs, args.output_csv)
        return 0

    if args.command == "smap-msl-robust-threshold-sweep":
        runs = run_smap_msl_robust_threshold_sweep(
            args.data_dir,
            thresholds=tuple(args.thresholds),
            channels=tuple(args.channels) if args.channels is not None else None,
            max_channels=args.max_channels,
        )
        aggregates = aggregate_smap_msl_robust_threshold_sweep(runs)
        print(f"channels={len(_smap_msl_robust_sweep_channels(runs))}")
        print(f"thresholds={','.join(_format_cli_float(value) for value in args.thresholds)}")
        print(f"runs={len(runs)}")
        _print_smap_msl_robust_threshold_aggregate_table(aggregates)
        operating_point_outputs_requested = (
            args.operating_point_json is not None
            or args.operating_point_csv is not None
            or args.policy_json is not None
            or args.policy_csv is not None
        )
        if args.false_alarm_budget is not None or operating_point_outputs_requested:
            if args.false_alarm_budget is None:
                raise ValueError(
                    "--false-alarm-budget is required when writing threshold-policy outputs"
                )
            operating_points = select_smap_msl_robust_threshold_operating_points(
                runs,
                false_alarm_budget=args.false_alarm_budget,
                group_by=args.selection_group,
            )
            print(f"false_alarm_budget={_format_cli_float(args.false_alarm_budget)}")
            print(f"selection_group={args.selection_group}")
            _print_smap_msl_robust_threshold_operating_point_table(operating_points)
            policy_runs = select_smap_msl_robust_threshold_policy_runs(
                runs,
                operating_points,
            )
            print(f"policy_runs={len(policy_runs)}")
            if args.operating_point_json is not None:
                write_smap_msl_robust_threshold_operating_points_json(
                    operating_points,
                    args.operating_point_json,
                )
            if args.operating_point_csv is not None:
                write_smap_msl_robust_threshold_operating_points_csv(
                    operating_points,
                    args.operating_point_csv,
                )
            if args.policy_json is not None:
                write_smap_msl_robust_threshold_policy_json(
                    policy_runs,
                    operating_points,
                    args.policy_json,
                )
            if args.policy_csv is not None:
                write_smap_msl_robust_threshold_policy_csv(
                    policy_runs,
                    operating_points,
                    args.policy_csv,
                )
        if args.output_json is not None:
            write_smap_msl_robust_threshold_sweep_json(runs, args.output_json)
        if args.output_csv is not None:
            write_smap_msl_robust_threshold_sweep_csv(runs, args.output_csv)
        if args.aggregate_json is not None:
            write_smap_msl_robust_threshold_sweep_aggregate_json(
                aggregates,
                args.aggregate_json,
            )
        if args.aggregate_csv is not None:
            write_smap_msl_robust_threshold_sweep_aggregate_csv(
                aggregates,
                args.aggregate_csv,
            )
        return 0

    if args.command == "smap-msl-lstm-forecast-baseline":
        from aerospace_prognostics.anomaly.forecasting import DynamicThresholdConfig

        runs = run_smap_msl_lstm_forecast_baseline(
            args.data_dir,
            channels=tuple(args.channels) if args.channels is not None else None,
            max_channels=args.max_channels,
            window_size=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold_sigma=args.threshold_sigma,
            threshold_method=args.threshold_method,
            dynamic_threshold_config=DynamicThresholdConfig(
                batch_size=args.dynamic_batch_size,
                window_batches=args.dynamic_window_batches,
                smoothing_fraction=args.dynamic_smoothing_fraction,
                z_start=args.dynamic_z_start,
                z_stop=args.dynamic_z_stop,
                z_step=args.dynamic_z_step,
                error_buffer=args.dynamic_error_buffer,
                p=args.dynamic_prune_p,
            ),
            random_state=args.random_state,
            device=args.device,
        )
        print(f"channels={len(_smap_msl_result_channels(runs))}")
        print(f"runs={len(runs)}")
        _print_smap_msl_forecast_table(runs)
        if args.output_json is not None:
            write_smap_msl_lstm_forecast_baseline_json(runs, args.output_json)
        if args.output_csv is not None:
            write_smap_msl_lstm_forecast_baseline_csv(runs, args.output_csv)
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

    parser.error(f"unknown command: {args.command}")
    return 2


def _print_results_table(results: Iterable[RegressionRunResult]) -> None:
    print("subset,model,standardize,rmse,nasa_score")
    for result in results:
        print(
            f"{result.subset},"
            f"{result.model_name},"
            f"{result.standardize},"
            f"{result.rmse:.6f},"
            f"{result.nasa_score:.6f}"
        )


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


def _print_smap_msl_classical_table(runs: Iterable[SmapMslClassicalBaselineRun]) -> None:
    print("channel_id,spacecraft,model,precision,recall,f1,point_adjusted_f1,false_alarm_rate")
    for run in runs:
        print(
            f"{run.channel_id},"
            f"{run.spacecraft},"
            f"{run.model_name},"
            f"{run.metrics.precision:.6f},"
            f"{run.metrics.recall:.6f},"
            f"{run.metrics.f1:.6f},"
            f"{run.point_adjusted_metrics.f1:.6f},"
            f"{run.metrics.false_alarm_rate:.6f}"
        )


def _print_smap_msl_forecast_table(runs: Iterable[SmapMslLstmForecastBaselineRun]) -> None:
    print(
        "channel_id,spacecraft,model,epochs,final_train_loss,"
        "precision,recall,f1,point_adjusted_f1,false_alarm_rate"
    )
    for run in runs:
        final_train_loss = run.history[-1].train_loss if run.history else 0.0
        print(
            f"{run.channel_id},"
            f"{run.spacecraft},"
            f"{run.model_name},"
            f"{len(run.history)},"
            f"{final_train_loss:.6f},"
            f"{run.metrics.precision:.6f},"
            f"{run.metrics.recall:.6f},"
            f"{run.metrics.f1:.6f},"
            f"{run.point_adjusted_metrics.f1:.6f},"
            f"{run.metrics.false_alarm_rate:.6f}"
        )


def _print_smap_msl_robust_threshold_aggregate_table(
    aggregates: Iterable[SmapMslRobustThresholdSweepAggregate],
) -> None:
    print(
        "threshold,channels,wins_by_f1,mean_precision,mean_recall,mean_f1,"
        "mean_point_adjusted_f1,mean_false_alarm_rate,mean_miss_rate"
    )
    for aggregate in aggregates:
        print(
            f"{aggregate.threshold:g},"
            f"{aggregate.channels},"
            f"{aggregate.wins_by_f1},"
            f"{aggregate.mean_precision:.6f},"
            f"{aggregate.mean_recall:.6f},"
            f"{aggregate.mean_f1:.6f},"
            f"{aggregate.mean_point_adjusted_f1:.6f},"
            f"{aggregate.mean_false_alarm_rate:.6f},"
            f"{aggregate.mean_miss_rate:.6f}"
        )


def _print_smap_msl_robust_threshold_operating_point_table(
    operating_points: Iterable[SmapMslRobustThresholdOperatingPoint],
) -> None:
    print(
        "scope,group,false_alarm_budget,selected_threshold,feasible,channels,"
        "mean_precision,mean_recall,mean_f1,mean_point_adjusted_f1,"
        "mean_false_alarm_rate,mean_miss_rate"
    )
    for operating_point in operating_points:
        print(
            f"{operating_point.scope},"
            f"{operating_point.group},"
            f"{operating_point.false_alarm_budget:g},"
            f"{operating_point.selected_threshold:g},"
            f"{operating_point.feasible},"
            f"{operating_point.channels},"
            f"{operating_point.mean_precision:.6f},"
            f"{operating_point.mean_recall:.6f},"
            f"{operating_point.mean_f1:.6f},"
            f"{operating_point.mean_point_adjusted_f1:.6f},"
            f"{operating_point.mean_false_alarm_rate:.6f},"
            f"{operating_point.mean_miss_rate:.6f}"
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


def _print_smap_msl_channel_selection_table(
    selections: Iterable[SmapMslChannelSelection],
) -> None:
    print("rank,channel_id,spacecraft,anomaly_sequences,anomaly_points,num_values")
    for selection in selections:
        print(
            f"{selection.rank},"
            f"{selection.channel_id},"
            f"{selection.spacecraft},"
            f"{selection.anomaly_sequences},"
            f"{selection.anomaly_points},"
            f"{selection.num_values if selection.num_values is not None else ''}"
        )


def _anomaly_comparison_channels(rows: Iterable[AnomalyModelComparisonRow]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(row.channel_id for row in rows))


def _smap_msl_result_channels(
    runs: Iterable[SmapMslClassicalBaselineRun | SmapMslLstmForecastBaselineRun],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(run.channel_id for run in runs))


def _smap_msl_robust_sweep_channels(
    runs: Iterable[SmapMslRobustThresholdSweepRun],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(run.channel_id for run in runs))


def _phase2_smap_msl_default_max_channels(channels: list[str] | None) -> int | None:
    if channels is not None:
        return None
    return 5


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


def _best_result_for_subset(
    results: Iterable[RegressionRunResult],
    subset: str,
) -> RegressionRunResult:
    subset_results = [result for result in results if result.subset == subset]
    if not subset_results:
        raise ValueError(f"no results for subset: {subset}")
    return min(subset_results, key=lambda result: result.nasa_score)


def _print_validation_aggregate_table(
    results: Iterable[CmapssValidationAggregateResult],
) -> None:
    print("subset,model,standardize,runs,wins_by_nasa,mean_rmse,mean_nasa_score")
    for result in results:
        print(
            f"{result.subset},"
            f"{result.model_name},"
            f"{result.standardize},"
            f"{result.runs},"
            f"{result.wins_by_nasa},"
            f"{result.mean_rmse:.6f},"
            f"{result.mean_nasa_score:.6f}"
        )


def _best_aggregate_for_subset(
    results: Iterable[CmapssValidationAggregateResult],
    subset: str,
) -> CmapssValidationAggregateResult:
    subset_results = [result for result in results if result.subset == subset]
    if not subset_results:
        raise ValueError(f"no aggregate results for subset: {subset}")
    return min(subset_results, key=lambda result: result.mean_nasa_score)


def _format_cli_float(value: float) -> str:
    return f"{value:g}"


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
    output_path = _prepare_output_path(path)
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
    output_path = _prepare_output_path(path)
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
    output_path = _prepare_output_path(path)
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


def _write_validation_aggregate_json(
    results: list[CmapssValidationAggregateResult],
    path: Path,
) -> None:
    output_path = _prepare_output_path(path)
    payload = [result.to_dict() for result in results]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_validation_aggregate_csv(
    results: list[CmapssValidationAggregateResult],
    path: Path,
) -> None:
    if not results:
        raise ValueError("results must contain at least one item")
    output_path = _prepare_output_path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].to_dict()))
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)


def _write_deep_history_json(
    results: list[object],
    path: Path,
) -> None:
    output_path = _prepare_output_path(path)
    payload = [result.to_dict() for result in results]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_json_payload(payload: object, path: Path) -> None:
    output_path = _prepare_output_path(path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
