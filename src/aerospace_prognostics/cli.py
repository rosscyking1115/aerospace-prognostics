"""Command-line utilities for local dataset checks."""

from __future__ import annotations

import argparse

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
from aerospace_prognostics.cli_cmapss_deep import (
    CMAPSS_DEEP_COMPARISON_MODELS,
    CMAPSS_DEEP_TRAINING_LOSSES,
    handle_cmapss_deep_command,
    register_cmapss_deep_commands,
)
from aerospace_prognostics.cli_cmapss_validation import (
    handle_cmapss_validation_command,
    register_cmapss_validation_commands,
)
from aerospace_prognostics.cli_conformal import (
    handle_conformal_command,
    register_conformal_commands,
)
from aerospace_prognostics.cli_deployment import (
    handle_deployment_command,
    register_deployment_artifact_commands,
    register_deployment_release_commands,
)
from aerospace_prognostics.cli_esa_adb import handle_esa_adb_command, register_esa_adb_commands
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
from aerospace_prognostics.cli_smap_msl_experiments import (
    handle_smap_msl_experiment_command,
    register_smap_msl_experiment_commands,
)
from aerospace_prognostics.cli_workflows import handle_workflow_command, register_workflow_commands
from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aerospace-prognostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_app_commands(subparsers)
    register_cmapss_data_commands(subparsers)
    register_cmapss_baseline_commands(subparsers)
    register_cmapss_validation_commands(subparsers)

    register_smap_msl_download_command(subparsers)
    register_workflow_commands(
        subparsers,
        cmapss_subsets=CMAPSS_SUBSETS,
        cmapss_deep_models=CMAPSS_DEEP_COMPARISON_MODELS,
        cmapss_training_losses=CMAPSS_DEEP_TRAINING_LOSSES,
        classical_anomaly_methods=CLASSICAL_ANOMALY_BASELINE_METHODS,
    )
    register_cmapss_deep_commands(subparsers)

    register_cmapss_report_commands(subparsers)
    register_conformal_commands(subparsers)
    register_anomaly_commands(subparsers)

    register_smap_msl_data_commands(subparsers)
    register_smap_msl_experiment_commands(subparsers)

    register_smap_msl_report_commands(subparsers)

    register_esa_adb_commands(subparsers)

    register_deployment_artifact_commands(subparsers)

    register_dashboard_commands(subparsers)

    register_deployment_release_commands(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    workflow_result = handle_workflow_command(args, runner=main)
    if workflow_result is not None:
        return workflow_result

    app_result = handle_app_command(args)
    if app_result is not None:
        return app_result

    deployment_result = handle_deployment_command(args)
    if deployment_result is not None:
        return deployment_result

    report_result = handle_report_command(args)
    if report_result is not None:
        return report_result

    conformal_result = handle_conformal_command(args)
    if conformal_result is not None:
        return conformal_result

    smap_msl_data_result = handle_smap_msl_data_command(args)
    if smap_msl_data_result is not None:
        return smap_msl_data_result

    esa_adb_result = handle_esa_adb_command(args)
    if esa_adb_result is not None:
        return esa_adb_result

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

    cmapss_deep_result = handle_cmapss_deep_command(args)
    if cmapss_deep_result is not None:
        return cmapss_deep_result

    smap_msl_experiment_result = handle_smap_msl_experiment_command(args)
    if smap_msl_experiment_result is not None:
        return smap_msl_experiment_result

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
