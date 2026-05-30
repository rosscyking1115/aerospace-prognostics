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
    assert len(result.sequence_exports) == 1
    assert len(result.hgb_policy_results) == 1
    assert len(result.deep_compare_results) == 2
    assert len(result.comparison_rows) == 3

    deep_payload = json.loads(result.deep_compare_json_path.read_text(encoding="utf-8"))
    summary = result.summary_markdown_path.read_text(encoding="utf-8")

    assert deep_payload[0]["subset"] == "FD001"
    assert "# Phase 2 C-MAPSS Summary" in summary
    assert "## Best Model By NASA Score" in summary
