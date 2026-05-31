from __future__ import annotations

import json

import numpy as np

from aerospace_prognostics.workflows.phase2_smap_msl import (
    run_phase2_smap_msl_workflow,
    verify_phase2_smap_msl_run_manifest,
)


def test_run_phase2_smap_msl_workflow_writes_expected_artifacts(tmp_path) -> None:
    _write_phase2_smap_msl_fixture(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    result = run_phase2_smap_msl_workflow(
        tmp_path,
        artifact_dir,
        channels=("P-1", "M-1"),
        window_size=2,
        hidden_size=4,
        epochs=1,
        batch_size=2,
        classical_methods=("robust_zscore", "pca_reconstruction"),
        pca_components=1,
        pca_threshold_quantile=0.95,
        robust_policy_false_alarm_budget=1.0,
        robust_policy_thresholds=(2.0, 4.0),
        robust_policy_group_by="spacecraft",
    )

    assert result.classical_json_path.exists()
    assert result.classical_csv_path.exists()
    assert result.lstm_robust_json_path.exists()
    assert result.lstm_robust_csv_path.exists()
    assert result.lstm_dynamic_json_path.exists()
    assert result.lstm_dynamic_csv_path.exists()
    assert result.robust_threshold_sweep_csv_path is not None
    assert result.robust_threshold_sweep_csv_path.exists()
    assert result.robust_threshold_operating_point_csv_path is not None
    assert result.robust_threshold_operating_point_csv_path.exists()
    assert result.robust_threshold_policy_csv_path is not None
    assert result.robust_threshold_policy_csv_path.exists()
    assert result.comparison_csv_path.exists()
    assert result.comparison_markdown_path.exists()
    assert result.summary_markdown_path.exists()
    assert result.run_manifest_path.exists()
    assert len(result.classical_runs) == 4
    assert len(result.lstm_robust_runs) == 2
    assert len(result.lstm_dynamic_runs) == 2
    assert len(result.robust_threshold_sweep_runs) == 4
    assert len(result.robust_threshold_operating_points) == 2
    assert len(result.robust_threshold_policy_runs) == 2
    assert len(result.comparison_rows) == 10

    classical_payload = json.loads(result.classical_json_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    summary = result.summary_markdown_path.read_text(encoding="utf-8")

    assert {row["channel_id"] for row in classical_payload} == {"P-1", "M-1"}
    assert run_manifest["workflow"] == "phase2_smap_msl"
    assert run_manifest["selection"]["channels"] == ["P-1", "M-1"]
    assert run_manifest["parameters"]["robust_policy_false_alarm_budget"] == 1.0
    assert run_manifest["counts"]["robust_threshold_policy_runs"] == 2
    assert run_manifest["counts"]["comparison_rows"] == 10
    assert run_manifest["runtime"]["python_version"]
    assert "numpy" in run_manifest["runtime"]["dependencies"]
    assert "git_commit" in run_manifest["source_control"]
    assert "git_dirty" in run_manifest["source_control"]
    assert len(run_manifest["artifact_integrity"]) == 17
    assert "sha256" in run_manifest["artifact_integrity"]["classical_csv"]
    assert run_manifest["artifacts"]["robust_threshold_policy_csv"].endswith(
        "smap_msl_robust_threshold_policy.csv"
    )
    assert "# Phase 2 SMAP/MSL Summary" in summary
    assert "## Best Model By Point-Wise F1" in summary
    assert "## Winner Counts" in summary
    assert "## Average Metrics By Source And Model" in summary
    assert "| classical | robust_zscore |" in summary
    assert "Robust threshold policy table" in summary
    assert "Run manifest" in summary

    verification = verify_phase2_smap_msl_run_manifest(result.run_manifest_path)
    assert verification.ok
    assert len(verification.checked_artifacts) == 18

    with result.classical_csv_path.open("a", encoding="utf-8") as file:
        file.write("tampered\n")
    tampered_verification = verify_phase2_smap_msl_run_manifest(result.run_manifest_path)
    assert not tampered_verification.ok
    assert any(
        "artifact classical_csv has unexpected sha256" in problem
        for problem in tampered_verification.problems
    )

    result.comparison_csv_path.unlink()
    failed_verification = verify_phase2_smap_msl_run_manifest(result.run_manifest_path)
    assert not failed_verification.ok
    assert any("comparison_csv is missing" in problem for problem in failed_verification.problems)


def _write_phase2_smap_msl_fixture(root) -> None:
    (root / "data" / "train").mkdir(parents=True)
    (root / "data" / "test").mkdir(parents=True)
    (root / "labeled_anomalies.csv").write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                '"P-1",SMAP,"[[2, 3]]","[contextual]",6',
                '"M-1",MSL,"[[2, 3]]","[contextual]",6',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(
        root / "data" / "train" / "P-1.npy",
        np.array(
            [
                [-0.2, -0.2],
                [-0.1, -0.1],
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
            ]
        ),
    )
    np.save(
        root / "data" / "test" / "P-1.npy",
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [8.0, -8.0],
                [7.0, -7.0],
                [0.2, 0.2],
                [0.3, 0.3],
            ]
        ),
    )
    np.save(
        root / "data" / "train" / "M-1.npy",
        np.array(
            [
                [0.3, 0.3],
                [0.2, 0.2],
                [0.1, 0.1],
                [0.0, 0.0],
                [-0.1, -0.1],
                [-0.2, -0.2],
            ]
        ),
    )
    np.save(
        root / "data" / "test" / "M-1.npy",
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [-8.0, 8.0],
                [-7.0, 7.0],
                [0.2, 0.2],
                [0.3, 0.3],
            ]
        ),
    )
