"""Phase 2 SMAP/MSL anomaly-baseline workflow orchestration."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

from aerospace_prognostics.anomaly.baselines import CLASSICAL_ANOMALY_BASELINE_METHODS
from aerospace_prognostics.anomaly.forecasting import DynamicThresholdConfig
from aerospace_prognostics.data.integrity import file_sha256
from aerospace_prognostics.experiments.smap_msl_anomaly import (
    SmapMslClassicalBaselineRun,
    SmapMslLstmForecastBaselineRun,
    SmapMslRobustThresholdOperatingPoint,
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


@dataclass(frozen=True)
class Phase2SmapMslWorkflowResult:
    """Artifact paths and run results from the SMAP/MSL Phase 2 workflow."""

    artifact_dir: Path
    classical_json_path: Path
    classical_csv_path: Path
    lstm_robust_json_path: Path
    lstm_robust_csv_path: Path
    lstm_dynamic_json_path: Path
    lstm_dynamic_csv_path: Path
    robust_threshold_sweep_json_path: Path | None
    robust_threshold_sweep_csv_path: Path | None
    robust_threshold_sweep_aggregate_json_path: Path | None
    robust_threshold_sweep_aggregate_csv_path: Path | None
    robust_threshold_operating_point_json_path: Path | None
    robust_threshold_operating_point_csv_path: Path | None
    robust_threshold_policy_json_path: Path | None
    robust_threshold_policy_csv_path: Path | None
    comparison_csv_path: Path
    comparison_markdown_path: Path
    summary_markdown_path: Path
    run_manifest_path: Path
    classical_runs: tuple[SmapMslClassicalBaselineRun, ...]
    lstm_robust_runs: tuple[SmapMslLstmForecastBaselineRun, ...]
    lstm_dynamic_runs: tuple[SmapMslLstmForecastBaselineRun, ...]
    robust_threshold_sweep_runs: tuple[SmapMslRobustThresholdSweepRun, ...]
    robust_threshold_operating_points: tuple[SmapMslRobustThresholdOperatingPoint, ...]
    robust_threshold_policy_runs: tuple[SmapMslRobustThresholdSweepRun, ...]
    comparison_rows: tuple[AnomalyModelComparisonRow, ...]


@dataclass(frozen=True)
class Phase2SmapMslRunManifestVerification:
    """Verification result for a SMAP/MSL Phase 2 run manifest."""

    manifest_path: Path
    checked_artifacts: tuple[Path, ...]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def run_phase2_smap_msl_workflow(
    data_dir: str | Path,
    artifact_dir: str | Path,
    *,
    channels: tuple[str, ...] | None = None,
    max_channels: int | None = None,
    classical_methods: tuple[str, ...] = CLASSICAL_ANOMALY_BASELINE_METHODS,
    robust_threshold: float = 3.5,
    pca_components: int | None = None,
    pca_threshold_quantile: float = 0.99,
    isolation_contamination: float = 0.05,
    window_size: int = 30,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    threshold_sigma: float = 3.0,
    robust_policy_false_alarm_budget: float | None = None,
    robust_policy_thresholds: tuple[float, ...] = (3.5, 5.0, 7.0, 10.0, 15.0),
    robust_policy_group_by: str = "spacecraft",
    dynamic_threshold_config: DynamicThresholdConfig | None = None,
    random_state: int = 42,
    device: str = "cpu",
) -> Phase2SmapMslWorkflowResult:
    """Run classical, LSTM forecast, dynamic-threshold, and comparison SMAP/MSL outputs."""

    root = Path(data_dir)
    artifacts = Path(artifact_dir)
    results_dir = artifacts / "results"
    artifacts.mkdir(parents=True, exist_ok=True)

    classical_runs = run_smap_msl_classical_baselines(
        root,
        channels=channels,
        max_channels=max_channels,
        methods=classical_methods,
        robust_threshold=robust_threshold,
        pca_components=pca_components,
        pca_threshold_quantile=pca_threshold_quantile,
        isolation_contamination=isolation_contamination,
        random_state=random_state,
    )
    classical_json_path = results_dir / "smap_msl_classical_baselines.json"
    classical_csv_path = results_dir / "smap_msl_classical_baselines.csv"
    write_smap_msl_classical_baselines_json(classical_runs, classical_json_path)
    write_smap_msl_classical_baselines_csv(classical_runs, classical_csv_path)

    lstm_robust_runs = run_smap_msl_lstm_forecast_baseline(
        root,
        channels=channels,
        max_channels=max_channels,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        threshold_sigma=threshold_sigma,
        threshold_method="robust",
        random_state=random_state,
        device=device,
    )
    lstm_robust_json_path = results_dir / "smap_msl_lstm_forecast_robust.json"
    lstm_robust_csv_path = results_dir / "smap_msl_lstm_forecast_robust.csv"
    write_smap_msl_lstm_forecast_baseline_json(lstm_robust_runs, lstm_robust_json_path)
    write_smap_msl_lstm_forecast_baseline_csv(lstm_robust_runs, lstm_robust_csv_path)

    lstm_dynamic_runs = run_smap_msl_lstm_forecast_baseline(
        root,
        channels=channels,
        max_channels=max_channels,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        threshold_sigma=threshold_sigma,
        threshold_method="dynamic",
        dynamic_threshold_config=dynamic_threshold_config,
        random_state=random_state,
        device=device,
    )
    lstm_dynamic_json_path = results_dir / "smap_msl_lstm_forecast_dynamic.json"
    lstm_dynamic_csv_path = results_dir / "smap_msl_lstm_forecast_dynamic.csv"
    write_smap_msl_lstm_forecast_baseline_json(lstm_dynamic_runs, lstm_dynamic_json_path)
    write_smap_msl_lstm_forecast_baseline_csv(lstm_dynamic_runs, lstm_dynamic_csv_path)

    robust_threshold_sweep_runs: tuple[SmapMslRobustThresholdSweepRun, ...] = ()
    robust_threshold_operating_points: tuple[SmapMslRobustThresholdOperatingPoint, ...] = ()
    robust_threshold_policy_runs: tuple[SmapMslRobustThresholdSweepRun, ...] = ()
    robust_threshold_sweep_json_path: Path | None = None
    robust_threshold_sweep_csv_path: Path | None = None
    robust_threshold_sweep_aggregate_json_path: Path | None = None
    robust_threshold_sweep_aggregate_csv_path: Path | None = None
    robust_threshold_operating_point_json_path: Path | None = None
    robust_threshold_operating_point_csv_path: Path | None = None
    robust_threshold_policy_json_path: Path | None = None
    robust_threshold_policy_csv_path: Path | None = None
    comparison_paths = [classical_csv_path, lstm_robust_csv_path, lstm_dynamic_csv_path]
    source_labels = ["classical", "lstm_robust", "lstm_dynamic"]
    if robust_policy_false_alarm_budget is not None:
        robust_threshold_sweep_runs = run_smap_msl_robust_threshold_sweep(
            root,
            thresholds=robust_policy_thresholds,
            channels=channels,
            max_channels=max_channels,
        )
        robust_threshold_aggregates = aggregate_smap_msl_robust_threshold_sweep(
            robust_threshold_sweep_runs
        )
        robust_threshold_operating_points = select_smap_msl_robust_threshold_operating_points(
            robust_threshold_sweep_runs,
            false_alarm_budget=robust_policy_false_alarm_budget,
            group_by=robust_policy_group_by,
        )
        robust_threshold_policy_runs = select_smap_msl_robust_threshold_policy_runs(
            robust_threshold_sweep_runs,
            robust_threshold_operating_points,
        )
        robust_threshold_sweep_json_path = results_dir / "smap_msl_robust_threshold_sweep.json"
        robust_threshold_sweep_csv_path = results_dir / "smap_msl_robust_threshold_sweep.csv"
        robust_threshold_sweep_aggregate_json_path = (
            results_dir / "smap_msl_robust_threshold_sweep_aggregate.json"
        )
        robust_threshold_sweep_aggregate_csv_path = (
            results_dir / "smap_msl_robust_threshold_sweep_aggregate.csv"
        )
        robust_threshold_operating_point_json_path = (
            results_dir / "smap_msl_robust_threshold_operating_points.json"
        )
        robust_threshold_operating_point_csv_path = (
            results_dir / "smap_msl_robust_threshold_operating_points.csv"
        )
        robust_threshold_policy_json_path = (
            results_dir / "smap_msl_robust_threshold_policy.json"
        )
        robust_threshold_policy_csv_path = results_dir / "smap_msl_robust_threshold_policy.csv"
        write_smap_msl_robust_threshold_sweep_json(
            robust_threshold_sweep_runs,
            robust_threshold_sweep_json_path,
        )
        write_smap_msl_robust_threshold_sweep_csv(
            robust_threshold_sweep_runs,
            robust_threshold_sweep_csv_path,
        )
        write_smap_msl_robust_threshold_sweep_aggregate_json(
            robust_threshold_aggregates,
            robust_threshold_sweep_aggregate_json_path,
        )
        write_smap_msl_robust_threshold_sweep_aggregate_csv(
            robust_threshold_aggregates,
            robust_threshold_sweep_aggregate_csv_path,
        )
        write_smap_msl_robust_threshold_operating_points_json(
            robust_threshold_operating_points,
            robust_threshold_operating_point_json_path,
        )
        write_smap_msl_robust_threshold_operating_points_csv(
            robust_threshold_operating_points,
            robust_threshold_operating_point_csv_path,
        )
        write_smap_msl_robust_threshold_policy_json(
            robust_threshold_policy_runs,
            robust_threshold_operating_points,
            robust_threshold_policy_json_path,
        )
        write_smap_msl_robust_threshold_policy_csv(
            robust_threshold_policy_runs,
            robust_threshold_operating_points,
            robust_threshold_policy_csv_path,
        )
        comparison_paths.append(robust_threshold_policy_csv_path)
        source_labels.append(f"robust_policy_far_{robust_policy_false_alarm_budget:g}")

    comparison_rows = build_anomaly_model_comparison(
        tuple(comparison_paths),
        source_labels=tuple(source_labels),
    )
    comparison_csv_path = results_dir / "smap_msl_anomaly_model_comparison.csv"
    comparison_markdown_path = results_dir / "smap_msl_anomaly_model_comparison.md"
    write_anomaly_model_comparison_csv(comparison_rows, comparison_csv_path)
    write_anomaly_model_comparison_markdown(comparison_rows, comparison_markdown_path)

    summary_markdown_path = artifacts / "phase2_smap_msl_summary.md"
    run_manifest_path = artifacts / "phase2_smap_msl_run_manifest.json"
    artifact_paths = {
        "classical_json": _path_as_posix(classical_json_path),
        "classical_csv": _path_as_posix(classical_csv_path),
        "lstm_robust_json": _path_as_posix(lstm_robust_json_path),
        "lstm_robust_csv": _path_as_posix(lstm_robust_csv_path),
        "lstm_dynamic_json": _path_as_posix(lstm_dynamic_json_path),
        "lstm_dynamic_csv": _path_as_posix(lstm_dynamic_csv_path),
        "robust_threshold_sweep_json": _path_as_posix(robust_threshold_sweep_json_path),
        "robust_threshold_sweep_csv": _path_as_posix(robust_threshold_sweep_csv_path),
        "robust_threshold_sweep_aggregate_json": _path_as_posix(
            robust_threshold_sweep_aggregate_json_path
        ),
        "robust_threshold_sweep_aggregate_csv": _path_as_posix(
            robust_threshold_sweep_aggregate_csv_path
        ),
        "robust_threshold_operating_point_json": _path_as_posix(
            robust_threshold_operating_point_json_path
        ),
        "robust_threshold_operating_point_csv": _path_as_posix(
            robust_threshold_operating_point_csv_path
        ),
        "robust_threshold_policy_json": _path_as_posix(robust_threshold_policy_json_path),
        "robust_threshold_policy_csv": _path_as_posix(robust_threshold_policy_csv_path),
        "comparison_csv": _path_as_posix(comparison_csv_path),
        "comparison_markdown": _path_as_posix(comparison_markdown_path),
        "summary_markdown": _path_as_posix(summary_markdown_path),
        "run_manifest": _path_as_posix(run_manifest_path),
    }
    _write_phase2_smap_msl_summary(
        summary_markdown_path,
        classical_csv_path=classical_csv_path,
        lstm_robust_csv_path=lstm_robust_csv_path,
        lstm_dynamic_csv_path=lstm_dynamic_csv_path,
        robust_threshold_policy_csv_path=robust_threshold_policy_csv_path,
        robust_threshold_operating_point_csv_path=robust_threshold_operating_point_csv_path,
        comparison_markdown_path=comparison_markdown_path,
        run_manifest_path=run_manifest_path,
        comparison_rows=tuple(comparison_rows),
    )
    _write_phase2_smap_msl_run_manifest(
        run_manifest_path,
        {
            "workflow": "phase2_smap_msl",
            "data_dir": root.as_posix(),
            "artifact_dir": artifacts.as_posix(),
            "selection": {
                "channels": list(channels) if channels is not None else None,
                "max_channels": max_channels,
            },
            "runtime": _runtime_environment_payload(),
            "source_control": _source_control_payload(),
            "parameters": {
                "classical_methods": list(classical_methods),
                "robust_threshold": robust_threshold,
                "pca_components": pca_components,
                "pca_threshold_quantile": pca_threshold_quantile,
                "isolation_contamination": isolation_contamination,
                "window_size": window_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "threshold_sigma": threshold_sigma,
                "robust_policy_false_alarm_budget": robust_policy_false_alarm_budget,
                "robust_policy_thresholds": list(robust_policy_thresholds),
                "robust_policy_group_by": robust_policy_group_by,
                "dynamic_threshold_config": (
                    dynamic_threshold_config or DynamicThresholdConfig()
                ).to_dict(),
                "random_state": random_state,
                "device": device,
            },
            "artifacts": artifact_paths,
            "artifact_integrity": _artifact_integrity_payload(artifact_paths),
            "counts": {
                "channels": len({row.channel_id for row in comparison_rows}),
                "classical_runs": len(classical_runs),
                "lstm_robust_runs": len(lstm_robust_runs),
                "lstm_dynamic_runs": len(lstm_dynamic_runs),
                "robust_threshold_sweep_runs": len(robust_threshold_sweep_runs),
                "robust_threshold_operating_points": len(robust_threshold_operating_points),
                "robust_threshold_policy_runs": len(robust_threshold_policy_runs),
                "comparison_rows": len(comparison_rows),
            },
        },
    )

    return Phase2SmapMslWorkflowResult(
        artifact_dir=artifacts,
        classical_json_path=classical_json_path,
        classical_csv_path=classical_csv_path,
        lstm_robust_json_path=lstm_robust_json_path,
        lstm_robust_csv_path=lstm_robust_csv_path,
        lstm_dynamic_json_path=lstm_dynamic_json_path,
        lstm_dynamic_csv_path=lstm_dynamic_csv_path,
        robust_threshold_sweep_json_path=robust_threshold_sweep_json_path,
        robust_threshold_sweep_csv_path=robust_threshold_sweep_csv_path,
        robust_threshold_sweep_aggregate_json_path=robust_threshold_sweep_aggregate_json_path,
        robust_threshold_sweep_aggregate_csv_path=robust_threshold_sweep_aggregate_csv_path,
        robust_threshold_operating_point_json_path=robust_threshold_operating_point_json_path,
        robust_threshold_operating_point_csv_path=robust_threshold_operating_point_csv_path,
        robust_threshold_policy_json_path=robust_threshold_policy_json_path,
        robust_threshold_policy_csv_path=robust_threshold_policy_csv_path,
        comparison_csv_path=comparison_csv_path,
        comparison_markdown_path=comparison_markdown_path,
        summary_markdown_path=summary_markdown_path,
        run_manifest_path=run_manifest_path,
        classical_runs=tuple(classical_runs),
        lstm_robust_runs=tuple(lstm_robust_runs),
        lstm_dynamic_runs=tuple(lstm_dynamic_runs),
        robust_threshold_sweep_runs=robust_threshold_sweep_runs,
        robust_threshold_operating_points=robust_threshold_operating_points,
        robust_threshold_policy_runs=robust_threshold_policy_runs,
        comparison_rows=tuple(comparison_rows),
    )


def verify_phase2_smap_msl_run_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path = ".",
) -> Phase2SmapMslRunManifestVerification:
    """Verify that a SMAP/MSL Phase 2 manifest points to a complete artifact bundle."""

    path = Path(manifest_path)
    artifact_root = Path(root)
    problems: list[str] = []
    checked_artifacts: list[Path] = []
    payload = _read_run_manifest_payload(path, problems)
    if payload is None:
        return Phase2SmapMslRunManifestVerification(path, (), tuple(problems))

    if payload.get("workflow") != "phase2_smap_msl":
        problems.append("workflow must be phase2_smap_msl")
    for section in (
        "selection",
        "runtime",
        "source_control",
        "parameters",
        "artifacts",
        "artifact_integrity",
        "counts",
    ):
        if not isinstance(payload.get(section), dict):
            problems.append(f"{section} section is missing or invalid")

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        checked_artifacts.extend(_verify_manifest_artifacts(artifacts, artifact_root, problems))
    artifact_integrity = payload.get("artifact_integrity")
    if isinstance(artifacts, dict) and isinstance(artifact_integrity, dict):
        _verify_manifest_artifact_integrity(
            artifacts,
            artifact_integrity,
            artifact_root,
            problems,
        )
    counts = payload.get("counts")
    if isinstance(artifacts, dict) and isinstance(counts, dict):
        _verify_manifest_csv_counts(artifacts, counts, artifact_root, problems)

    return Phase2SmapMslRunManifestVerification(
        manifest_path=path,
        checked_artifacts=tuple(checked_artifacts),
        problems=tuple(problems),
    )


def _write_phase2_smap_msl_summary(
    path: Path,
    *,
    classical_csv_path: Path,
    lstm_robust_csv_path: Path,
    lstm_dynamic_csv_path: Path,
    robust_threshold_policy_csv_path: Path | None,
    robust_threshold_operating_point_csv_path: Path | None,
    comparison_markdown_path: Path,
    run_manifest_path: Path,
    comparison_rows: tuple[AnomalyModelComparisonRow, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 SMAP/MSL Summary",
        "",
        f"- Classical baseline table: `{classical_csv_path.as_posix()}`",
        f"- LSTM robust-threshold table: `{lstm_robust_csv_path.as_posix()}`",
        f"- LSTM dynamic-threshold table: `{lstm_dynamic_csv_path.as_posix()}`",
    ]
    if robust_threshold_policy_csv_path is not None:
        lines.extend(
            [
                (
                    "- Robust threshold policy table: "
                    f"`{robust_threshold_policy_csv_path.as_posix()}`"
                ),
                (
                    "- Robust threshold operating points: "
                    f"`{robust_threshold_operating_point_csv_path.as_posix()}`"
                ),
            ]
        )
    lines.extend(
        [
            f"- Ranked anomaly comparison: `{comparison_markdown_path.as_posix()}`",
            f"- Run manifest: `{run_manifest_path.as_posix()}`",
            "",
            "## Best Model By Point-Wise F1",
            "",
            (
                "| Channel | Spacecraft | Source | Model | F1 | "
                "Point-Adjusted F1 | False Alarm Rate |"
            ),
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    best_rows = _best_smap_msl_rows_by_channel(comparison_rows)
    for best in best_rows:
        lines.append(
            f"| {best.channel_id} | {best.spacecraft} | {best.source} | "
            f"{best.model_name} | {best.f1:.6f} | {best.point_adjusted_f1:.6f} | "
            f"{best.false_alarm_rate:.6f} |"
        )
    _append_smap_msl_aggregate_table(
        lines,
        rows=best_rows,
        heading="Winner Counts",
        count_header="Channels Won",
    )
    _append_smap_msl_aggregate_table(
        lines,
        rows=comparison_rows,
        heading="Average Metrics By Source And Model",
        count_header="Rows",
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_run_manifest_payload(path: Path, problems: list[str]) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append(f"{path} is missing")
        return None
    except json.JSONDecodeError as error:
        problems.append(f"{path} is not valid JSON: {error.msg}")
        return None
    if not isinstance(payload, dict):
        problems.append(f"{path} must contain a JSON object")
        return None
    return payload


def _verify_manifest_artifacts(
    artifacts: dict[object, object],
    root: Path,
    problems: list[str],
) -> tuple[Path, ...]:
    checked_paths: list[Path] = []
    for key, value in sorted(artifacts.items()):
        if value is None:
            continue
        if not isinstance(value, str):
            problems.append(f"artifact {key} must be a string path or null")
            continue
        artifact_path = _resolve_manifest_path(value, root)
        checked_paths.append(artifact_path)
        if not artifact_path.exists():
            problems.append(f"artifact {key} is missing: {artifact_path}")
    return tuple(checked_paths)


def _verify_manifest_csv_counts(
    artifacts: dict[object, object],
    counts: dict[object, object],
    root: Path,
    problems: list[str],
) -> None:
    count_checks = {
        "classical_csv": "classical_runs",
        "lstm_robust_csv": "lstm_robust_runs",
        "lstm_dynamic_csv": "lstm_dynamic_runs",
        "robust_threshold_sweep_csv": "robust_threshold_sweep_runs",
        "robust_threshold_operating_point_csv": "robust_threshold_operating_points",
        "robust_threshold_policy_csv": "robust_threshold_policy_runs",
        "comparison_csv": "comparison_rows",
    }
    for artifact_key, count_key in count_checks.items():
        path_value = artifacts.get(artifact_key)
        expected_count = counts.get(count_key)
        if path_value is None or expected_count is None:
            continue
        if not isinstance(path_value, str) or not isinstance(expected_count, int):
            continue
        artifact_path = _resolve_manifest_path(path_value, root)
        if not artifact_path.exists():
            continue
        row_count = _csv_data_row_count(artifact_path)
        if row_count != expected_count:
            problems.append(
                f"{artifact_key} has {row_count} rows; expected {expected_count} from {count_key}"
            )


def _verify_manifest_artifact_integrity(
    artifacts: dict[object, object],
    artifact_integrity: dict[object, object],
    root: Path,
    problems: list[str],
) -> None:
    for key, value in sorted(artifacts.items()):
        if key == "run_manifest" or value is None:
            continue
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        expected = artifact_integrity.get(key)
        if not isinstance(expected, dict):
            problems.append(f"artifact_integrity missing for {key}")
            continue
        artifact_path = _resolve_manifest_path(value, root)
        if not artifact_path.exists():
            continue
        expected_sha256 = expected.get("sha256")
        expected_size_bytes = expected.get("size_bytes")
        if not isinstance(expected_sha256, str):
            problems.append(f"artifact_integrity {key} sha256 is missing or invalid")
        elif file_sha256(artifact_path) != expected_sha256:
            problems.append(f"artifact {key} has unexpected sha256")
        if not isinstance(expected_size_bytes, int):
            problems.append(f"artifact_integrity {key} size_bytes is missing or invalid")
        elif artifact_path.stat().st_size != expected_size_bytes:
            problems.append(f"artifact {key} has unexpected size")


def _csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        return sum(1 for _ in reader)


def _resolve_manifest_path(path: str, root: Path) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return artifact_path
    return root / artifact_path


def _write_phase2_smap_msl_run_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_integrity_payload(artifacts: dict[str, str | None]) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for key, value in sorted(artifacts.items()):
        if key == "run_manifest" or value is None:
            continue
        path = Path(value)
        if not path.exists():
            continue
        payload[key] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return payload


def _runtime_environment_payload() -> dict[str, object]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "project_version": _package_version("aerospace-prognostics"),
        "dependencies": {
            name: _package_version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "torch",
            )
        },
    }


def _source_control_payload() -> dict[str, object]:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": _git_dirty(),
    }


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_dirty() -> bool | None:
    status = _git_output("status", "--porcelain")
    if status is None:
        return None
    return bool(status)


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=_repository_root(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_as_posix(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.as_posix()


def _best_smap_msl_rows_by_channel(
    comparison_rows: tuple[AnomalyModelComparisonRow, ...],
) -> tuple[AnomalyModelComparisonRow, ...]:
    best_rows: list[AnomalyModelComparisonRow] = []
    for channel_id in sorted({row.channel_id for row in comparison_rows}):
        best_rows.append(
            min(
                (row for row in comparison_rows if row.channel_id == channel_id),
                key=lambda row: row.rank_by_f1,
            )
        )
    return tuple(best_rows)


def _append_smap_msl_aggregate_table(
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
    for source, model_name, group in _group_smap_msl_rows_by_model(rows):
        mean_f1 = _mean_smap_msl_metric(row.f1 for row in group)
        mean_point_adjusted_f1 = _mean_smap_msl_metric(row.point_adjusted_f1 for row in group)
        mean_false_alarm_rate = _mean_smap_msl_metric(row.false_alarm_rate for row in group)
        lines.append(
            f"| {source} | {model_name} | {len(group)} | {mean_f1:.6f} | "
            f"{mean_point_adjusted_f1:.6f} | {mean_false_alarm_rate:.6f} |"
        )


def _group_smap_msl_rows_by_model(
    rows: tuple[AnomalyModelComparisonRow, ...],
) -> tuple[tuple[str, str, tuple[AnomalyModelComparisonRow, ...]], ...]:
    grouped: dict[tuple[str, str], list[AnomalyModelComparisonRow]] = {}
    for row in rows:
        grouped.setdefault((row.source, row.model_name), []).append(row)
    return tuple(
        (source, model_name, tuple(grouped[(source, model_name)]))
        for source, model_name in sorted(grouped)
    )


def _mean_smap_msl_metric(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("cannot average an empty metric sequence")
    return sum(sequence) / len(sequence)
