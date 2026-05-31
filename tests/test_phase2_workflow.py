from __future__ import annotations

import json

from aerospace_prognostics.workflows.phase2 import run_phase2_cmapss_workflow
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_run_phase2_cmapss_workflow_writes_expected_artifacts(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    result = run_phase2_cmapss_workflow(
        tmp_path,
        artifact_dir,
        subsets=("FD001",),
        window_size=2,
        validation_horizon=1,
        n_regimes=1,
        models=("cnn", "tcn"),
        epochs=1,
        batch_size=2,
        hidden_sizes=(4,),
        tcn_levels=1,
    )

    assert result.sequence_dir.exists()
    assert result.hgb_policy_json_path.exists()
    assert result.hgb_policy_csv_path.exists()
    assert result.deep_compare_json_path.exists()
    assert result.deep_compare_csv_path.exists()
    assert result.comparison_csv_path.exists()
    assert result.comparison_markdown_path.exists()
    assert result.summary_markdown_path.exists()
    assert result.run_manifest_path.exists()
    assert len(result.sequence_exports) == 1
    assert len(result.hgb_policy_results) == 1
    assert len(result.deep_compare_results) == 2
    assert len(result.comparison_rows) == 3

    deep_payload = json.loads(result.deep_compare_json_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    summary = result.summary_markdown_path.read_text(encoding="utf-8")

    assert deep_payload[0]["subset"] == "FD001"
    assert run_manifest["workflow"] == "phase2_cmapss"
    assert run_manifest["parameters"]["subsets"] == ["FD001"]
    assert run_manifest["counts"]["sequence_exports"] == 1
    assert run_manifest["counts"]["deep_compare_results"] == 2
    assert run_manifest["counts"]["comparison_rows"] == 3
    assert "numpy" in run_manifest["runtime"]["dependencies"]
    assert "git_commit" in run_manifest["source_control"]
    assert len(run_manifest["artifact_integrity"]) == 12
    assert "sha256" in run_manifest["artifact_integrity"]["deep_compare_csv"]
    assert run_manifest["artifacts"]["sequence_fd001_train_npz"].endswith(
        "train_sequences.npz"
    )
    assert "# Phase 2 C-MAPSS Summary" in summary
    assert "## Best Model By NASA Score" in summary
    assert "Run manifest" in summary
