"""Command-line utilities for local dataset checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from aerospace_prognostics.anomaly.baselines import CLASSICAL_ANOMALY_BASELINE_METHODS
from aerospace_prognostics.cli_anomaly import handle_anomaly_command, register_anomaly_commands
from aerospace_prognostics.cli_app import handle_app_command, register_app_commands
from aerospace_prognostics.cli_cmapss_baselines import (
    handle_cmapss_baseline_command,
    register_cmapss_baseline_commands,
)
from aerospace_prognostics.cli_cmapss_data import (
    handle_cmapss_data_command,
    register_cmapss_data_commands,
)
from aerospace_prognostics.cli_cmapss_validation import (
    handle_cmapss_validation_command,
    register_cmapss_validation_commands,
)
from aerospace_prognostics.cli_deployment import (
    handle_deployment_command,
    register_deployment_artifact_commands,
    register_deployment_release_commands,
)
from aerospace_prognostics.cli_reports import (
    handle_report_command,
    register_cmapss_report_commands,
    register_dashboard_commands,
    register_smap_msl_report_commands,
)
from aerospace_prognostics.cli_smap_msl import (
    handle_smap_msl_data_command,
    register_smap_msl_data_commands,
    register_smap_msl_download_command,
)
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
from aerospace_prognostics.deployment.quickstart import run_cmapss_quickstart
from aerospace_prognostics.evaluation import (
    RegressionRunResult,
    write_results_csv,
    write_results_json,
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
    register_cmapss_data_commands(subparsers)
    register_cmapss_baseline_commands(subparsers)
    register_cmapss_validation_commands(subparsers)

    register_smap_msl_download_command(subparsers)

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

    register_cmapss_report_commands(subparsers)
    register_anomaly_commands(subparsers)

    register_smap_msl_data_commands(subparsers)

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

    register_smap_msl_report_commands(subparsers)

    register_deployment_artifact_commands(subparsers)

    register_dashboard_commands(subparsers)

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

    report_result = handle_report_command(args)
    if report_result is not None:
        return report_result

    smap_msl_data_result = handle_smap_msl_data_command(args)
    if smap_msl_data_result is not None:
        return smap_msl_data_result

    cmapss_data_result = handle_cmapss_data_command(args)
    if cmapss_data_result is not None:
        return cmapss_data_result

    cmapss_baseline_result = handle_cmapss_baseline_command(args)
    if cmapss_baseline_result is not None:
        return cmapss_baseline_result

    cmapss_validation_result = handle_cmapss_validation_command(args)
    if cmapss_validation_result is not None:
        return cmapss_validation_result

    anomaly_result = handle_anomaly_command(args)
    if anomaly_result is not None:
        return anomaly_result

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


def _best_result_for_subset(
    results: Iterable[RegressionRunResult],
    subset: str,
) -> RegressionRunResult:
    subset_results = [result for result in results if result.subset == subset]
    if not subset_results:
        raise ValueError(f"no results for subset: {subset}")
    return min(subset_results, key=lambda result: result.nasa_score)


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


def _format_cli_float(value: float) -> str:
    return f"{value:g}"


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
