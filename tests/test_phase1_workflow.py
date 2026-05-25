from __future__ import annotations

import json

from aerospace_prognostics.workflows.phase1 import run_phase1_cmapss_workflow
from tests.cmapss_fixtures import write_all_tiny_cmapss_subsets


def test_run_phase1_cmapss_workflow_writes_expected_artifacts(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    result = run_phase1_cmapss_workflow(tmp_path, artifact_dir)

    assert result.manifest_path.exists()
    assert result.baseline_json_path.exists()
    assert result.baseline_csv_path.exists()
    assert result.summary_markdown_path.exists()
    assert len(result.eda_paths) == 4
    assert all(path.exists() for path in result.eda_paths)
    assert len(result.baseline_results) == 4

    baseline_payload = json.loads(result.baseline_json_path.read_text(encoding="utf-8"))
    summary_text = result.summary_markdown_path.read_text(encoding="utf-8")

    assert baseline_payload[0]["subset"] == "FD001"
    assert "| FD004 | hist_gradient_boosting | True |" in summary_text
