"""Workflow command handlers for the project CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from aerospace_prognostics.deployment.quickstart import run_cmapss_quickstart
from aerospace_prognostics.workflows.phase1 import run_phase1_cmapss_workflow


def register_workflow_commands(
    subparsers: Any,
    *,
    cmapss_subsets: Sequence[str],
    cmapss_deep_models: Sequence[str],
    cmapss_training_losses: Sequence[str],
    classical_anomaly_methods: Sequence[str],
) -> None:
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

    phase1 = subparsers.add_parser(
        "phase1-cmapss",
        help="Run the Phase 1 C-MAPSS provenance, EDA, and baseline workflow",
    )
    phase1.add_argument("--data-dir", type=Path, required=True)
    phase1.add_argument("--artifact-dir", type=Path, default=Path("artifacts/phase1"))
    phase1.add_argument(
        "--subsets",
        nargs="+",
        choices=cmapss_subsets,
        default=list(cmapss_subsets),
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
        choices=cmapss_subsets,
        default=list(cmapss_subsets),
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
        choices=cmapss_deep_models,
        default=["cnn", "bilstm", "tcn", "transformer"],
    )
    phase2.add_argument("--epochs", type=int, default=5)
    phase2.add_argument("--batch-size", type=int, default=256)
    phase2.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3])
    phase2.add_argument(
        "--training-loss",
        choices=cmapss_training_losses,
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
        choices=classical_anomaly_methods,
        default=list(classical_anomaly_methods),
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

    phase2_completion_audit = subparsers.add_parser(
        "phase2-completion-audit",
        help="Verify both Phase 2 track manifests and write a combined audit",
    )
    phase2_completion_audit.add_argument("--cmapss-manifest", type=Path, required=True)
    phase2_completion_audit.add_argument("--smap-msl-manifest", type=Path, required=True)
    phase2_completion_audit.add_argument("--root", type=Path, default=Path("."))
    phase2_completion_audit.add_argument("--output-json", type=Path)
    phase2_completion_audit.add_argument("--output-markdown", type=Path)

    pre_phase3_readiness = subparsers.add_parser(
        "pre-phase3-readiness-audit",
        help="Audit historical launch-readiness gates; quarantined from active Phase 3 planning",
    )
    pre_phase3_readiness.add_argument("--root", type=Path, default=Path("."))
    pre_phase3_readiness.add_argument("--hosted-demo-url")
    pre_phase3_readiness.add_argument("--hosted-demo-proof", type=Path)
    pre_phase3_readiness.add_argument("--license-decision")
    pre_phase3_readiness.add_argument("--output-json", type=Path)
    pre_phase3_readiness.add_argument("--output-markdown", type=Path)


def handle_workflow_command(
    args: argparse.Namespace,
    *,
    runner: Callable[[list[str] | None], int],
) -> int | None:
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
            runner=runner,
        )

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

    if args.command == "phase2-completion-audit":
        from aerospace_prognostics.workflows.phase2_completion import (
            run_phase2_completion_audit,
            write_phase2_completion_audit_json,
            write_phase2_completion_audit_markdown,
        )

        audit = run_phase2_completion_audit(
            cmapss_manifest=args.cmapss_manifest,
            smap_msl_manifest=args.smap_msl_manifest,
            root=args.root,
        )
        if args.output_json is not None:
            json_path = write_phase2_completion_audit_json(audit, args.output_json)
            print(f"audit_json={json_path}")
        if args.output_markdown is not None:
            markdown_path = write_phase2_completion_audit_markdown(
                audit,
                args.output_markdown,
            )
            print(f"audit_markdown={markdown_path}")
        print(f"status={audit.status}")
        print(f"cmapss_artifacts_checked={audit.cmapss.artifacts_checked}")
        print(f"smap_msl_artifacts_checked={audit.smap_msl.artifacts_checked}")
        for track in (audit.cmapss, audit.smap_msl):
            for problem in track.problems:
                print(f"problem={track.track}: {problem}")
        return 0 if audit.ok else 1

    if args.command == "pre-phase3-readiness-audit":
        from aerospace_prognostics.workflows.pre_phase3_readiness import (
            run_pre_phase3_readiness_audit,
            write_pre_phase3_readiness_json,
            write_pre_phase3_readiness_markdown,
        )

        audit = run_pre_phase3_readiness_audit(
            args.root,
            hosted_demo_url=args.hosted_demo_url,
            hosted_demo_proof=args.hosted_demo_proof,
            license_decision=args.license_decision,
        )
        if args.output_json is not None:
            json_path = write_pre_phase3_readiness_json(audit, args.output_json)
            print(f"audit_json={json_path}")
        if args.output_markdown is not None:
            markdown_path = write_pre_phase3_readiness_markdown(
                audit,
                args.output_markdown,
            )
            print(f"audit_markdown={markdown_path}")
        print(f"status={audit.status}")
        print(f"gates={len(audit.gates)}")
        print(f"blockers={len(audit.blockers)}")
        for gate in audit.blockers:
            print(f"blocker={gate.gate_id}: {gate.next_action}")
        return 0 if audit.ok else 1

    return None


def _phase2_smap_msl_default_max_channels(channels: list[str] | None) -> int | None:
    if channels is not None:
        return None
    return 5
