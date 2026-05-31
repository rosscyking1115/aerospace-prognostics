from __future__ import annotations

import json

import numpy as np

from aerospace_prognostics.workflows.phase2_smap_msl import run_phase2_smap_msl_workflow


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
    )

    assert result.classical_json_path.exists()
    assert result.classical_csv_path.exists()
    assert result.lstm_robust_json_path.exists()
    assert result.lstm_robust_csv_path.exists()
    assert result.lstm_dynamic_json_path.exists()
    assert result.lstm_dynamic_csv_path.exists()
    assert result.comparison_csv_path.exists()
    assert result.comparison_markdown_path.exists()
    assert result.summary_markdown_path.exists()
    assert len(result.classical_runs) == 4
    assert len(result.lstm_robust_runs) == 2
    assert len(result.lstm_dynamic_runs) == 2
    assert len(result.comparison_rows) == 8

    classical_payload = json.loads(result.classical_json_path.read_text(encoding="utf-8"))
    summary = result.summary_markdown_path.read_text(encoding="utf-8")

    assert {row["channel_id"] for row in classical_payload} == {"P-1", "M-1"}
    assert "# Phase 2 SMAP/MSL Summary" in summary
    assert "## Best Model By Point-Wise F1" in summary


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
