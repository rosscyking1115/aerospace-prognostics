from __future__ import annotations

import csv
import json

from aerospace_prognostics.workflows.phase2 import (
    run_phase2_cmapss_workflow,
    verify_phase2_cmapss_run_manifest,
    write_phase2_cmapss_manifest_audit_markdown,
)
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
    assert result.deep_predictions_csv_path.exists()
    assert result.deep_validation_selection_predictions_csv_path.exists()
    assert result.deep_prediction_diagnostics_csv_path.exists()
    assert result.deep_validation_selection_prediction_diagnostics_csv_path.exists()
    assert result.deep_prediction_rul_bin_diagnostics_csv_path.exists()
    assert result.deep_validation_selection_prediction_rul_bin_diagnostics_csv_path.exists()
    assert result.deep_prediction_diagnostics_markdown_path.exists()
    assert result.deep_validation_selection_prediction_diagnostics_markdown_path.exists()
    assert result.comparison_csv_path.exists()
    assert result.comparison_markdown_path.exists()
    assert result.summary_markdown_path.exists()
    assert result.run_manifest_path.exists()
    assert len(result.sequence_exports) == 1
    assert len(result.hgb_policy_results) == 1
    assert len(result.deep_compare_results) == 2
    assert len(result.comparison_rows) == 3

    deep_payload = json.loads(result.deep_compare_json_path.read_text(encoding="utf-8"))
    with result.deep_predictions_csv_path.open("r", encoding="utf-8", newline="") as file:
        prediction_rows = list(csv.DictReader(file))
    with result.deep_validation_selection_predictions_csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        validation_selection_prediction_rows = list(csv.DictReader(file))
    with result.deep_prediction_diagnostics_csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        diagnostic_rows = list(csv.DictReader(file))
    with result.deep_prediction_rul_bin_diagnostics_csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rul_bin_rows = list(csv.DictReader(file))
    run_manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    summary = result.summary_markdown_path.read_text(encoding="utf-8")

    assert deep_payload[0]["subset"] == "FD001"
    assert len(prediction_rows) == 4
    assert prediction_rows[0]["prediction_split"] == "official_test"
    assert validation_selection_prediction_rows
    assert validation_selection_prediction_rows[0]["prediction_split"] == "validation_selection"
    assert prediction_rows[0]["model_name"].startswith("compare_")
    assert len(diagnostic_rows) == 2
    assert diagnostic_rows[0]["mean_absolute_error"]
    assert rul_bin_rows
    assert "actual_rul_bin" in rul_bin_rows[0]
    assert run_manifest["workflow"] == "phase2_cmapss"
    assert run_manifest["parameters"]["subsets"] == ["FD001"]
    assert run_manifest["counts"]["sequence_exports"] == 1
    assert run_manifest["counts"]["deep_compare_results"] == 2
    assert run_manifest["counts"]["deep_prediction_rows"] == 4
    assert run_manifest["counts"]["deep_validation_selection_prediction_rows"] == len(
        validation_selection_prediction_rows
    )
    assert run_manifest["counts"]["deep_prediction_diagnostics"] == 2
    assert run_manifest["counts"]["deep_validation_selection_prediction_diagnostics"] == 2
    assert run_manifest["counts"]["deep_prediction_rul_bin_diagnostics"] >= 2
    assert run_manifest["counts"]["deep_validation_selection_prediction_rul_bin_diagnostics"] >= 1
    assert run_manifest["counts"]["comparison_rows"] == 3
    assert "numpy" in run_manifest["runtime"]["dependencies"]
    assert "git_commit" in run_manifest["source_control"]
    assert len(run_manifest["artifact_integrity"]) == 20
    assert "sha256" in run_manifest["artifact_integrity"]["deep_compare_csv"]
    assert "sha256" in run_manifest["artifact_integrity"]["deep_predictions_csv"]
    assert (
        "sha256" in run_manifest["artifact_integrity"]["deep_validation_selection_predictions_csv"]
    )
    assert "sha256" in run_manifest["artifact_integrity"]["deep_prediction_diagnostics_csv"]
    assert "sha256" in run_manifest["artifact_integrity"]["deep_prediction_rul_bin_diagnostics_csv"]
    assert (
        "sha256"
        in run_manifest["artifact_integrity"][
            "deep_validation_selection_prediction_rul_bin_diagnostics_csv"
        ]
    )
    assert run_manifest["artifacts"]["sequence_fd001_train_npz"].endswith("train_sequences.npz")
    assert "# Phase 2 C-MAPSS Summary" in summary
    assert "## Best Model By NASA Score" in summary
    assert "## Deep Prediction Diagnostics" in summary
    assert "## Deep Prediction RUL Bins" in summary
    assert "## Validation Selection Prediction Diagnostics" in summary
    assert "## Validation Selection RUL Bins" in summary
    assert "deep prediction diagnostics" in summary
    assert "Run manifest" in summary

    verification = verify_phase2_cmapss_run_manifest(result.run_manifest_path)
    assert verification.ok
    assert len(verification.checked_artifacts) == 21
    assert verification.manifest_payload is not None

    audit_path = write_phase2_cmapss_manifest_audit_markdown(
        verification,
        artifact_dir / "phase2_manifest_audit.md",
    )
    audit_markdown = audit_path.read_text(encoding="utf-8")
    assert "# Phase 2 C-MAPSS Manifest Audit" in audit_markdown
    assert "- Status: ok" in audit_markdown
    assert "- Artifacts checked: 21" in audit_markdown
    assert "| deep_compare_csv | yes |" in audit_markdown
    assert "| deep_predictions_csv | yes |" in audit_markdown
    assert "| deep_validation_selection_predictions_csv | yes |" in audit_markdown
    assert "| deep_prediction_diagnostics_csv | yes |" in audit_markdown
    assert "| deep_prediction_rul_bin_diagnostics_csv | yes |" in audit_markdown
    assert (
        "| deep_validation_selection_prediction_rul_bin_diagnostics_csv | yes |" in audit_markdown
    )
    assert "- None" in audit_markdown

    with result.deep_compare_csv_path.open("a", encoding="utf-8") as file:
        file.write("tampered\n")
    tampered_verification = verify_phase2_cmapss_run_manifest(result.run_manifest_path)
    assert not tampered_verification.ok
    assert any(
        "artifact deep_compare_csv has unexpected sha256" in problem
        for problem in tampered_verification.problems
    )

    result.comparison_csv_path.unlink()
    failed_verification = verify_phase2_cmapss_run_manifest(result.run_manifest_path)
    assert not failed_verification.ok
    assert any("comparison_csv is missing" in problem for problem in failed_verification.problems)
