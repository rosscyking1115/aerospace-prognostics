from __future__ import annotations

import json

import numpy as np

from aerospace_prognostics.cli import main
from aerospace_prognostics.workflows.phase2 import run_phase2_cmapss_workflow
from aerospace_prognostics.workflows.phase2_completion import (
    run_phase2_completion_audit,
    write_phase2_completion_audit_json,
    write_phase2_completion_audit_markdown,
)
from aerospace_prognostics.workflows.phase2_smap_msl import run_phase2_smap_msl_workflow
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_phase2_completion_audit_verifies_both_tracks_and_cli(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)
    _write_smap_msl_fixture(tmp_path)

    cmapss_result = run_phase2_cmapss_workflow(
        tmp_path,
        tmp_path / "phase2_cmapss",
        subsets=("FD001",),
        window_size=2,
        validation_horizon=1,
        n_regimes=1,
        models=("cnn",),
        epochs=1,
        batch_size=2,
        hidden_sizes=(4,),
    )
    smap_msl_result = run_phase2_smap_msl_workflow(
        tmp_path,
        tmp_path / "phase2_smap_msl",
        channels=("P-1",),
        window_size=2,
        hidden_size=4,
        epochs=1,
        batch_size=2,
        classical_methods=("robust_zscore",),
        robust_policy_false_alarm_budget=1.0,
        robust_policy_thresholds=(2.0, 4.0),
    )

    audit = run_phase2_completion_audit(
        cmapss_manifest=cmapss_result.run_manifest_path,
        smap_msl_manifest=smap_msl_result.run_manifest_path,
    )
    assert audit.ok
    assert audit.status == "ok"
    assert audit.cmapss.workflow == "phase2_cmapss"
    assert audit.smap_msl.workflow == "phase2_smap_msl"
    assert audit.cmapss.artifacts_checked > 0
    assert audit.smap_msl.artifacts_checked > 0
    assert audit.to_dict()["completion_gates"] == {
        "phase2_cmapss_manifest_verified": True,
        "phase2_smap_msl_manifest_verified": True,
        "phase2_evidence_bundle_ready": True,
    }

    json_path = write_phase2_completion_audit_json(
        audit,
        tmp_path / "phase2_completion_audit.json",
    )
    markdown_path = write_phase2_completion_audit_markdown(
        audit,
        tmp_path / "phase2_completion_audit.md",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["schema_version"] == "aerospace-prognostics/phase2-completion-audit/v1"
    assert payload["status"] == "ok"
    assert "# Phase 2 Completion Audit" in markdown
    assert "- Evidence bundle ready: yes" in markdown
    assert "| cmapss_sequence_models | phase2_cmapss | ok |" in markdown
    assert "| smap_msl_anomaly_baselines | phase2_smap_msl | ok |" in markdown
    assert "- None" in markdown

    exit_code = main(
        [
            "phase2-completion-audit",
            "--cmapss-manifest",
            str(cmapss_result.run_manifest_path),
            "--smap-msl-manifest",
            str(smap_msl_result.run_manifest_path),
            "--output-json",
            str(tmp_path / "cli_phase2_completion_audit.json"),
            "--output-markdown",
            str(tmp_path / "cli_phase2_completion_audit.md"),
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "audit_json=" in output
    assert "audit_markdown=" in output
    assert "status=ok" in output
    assert "cmapss_artifacts_checked=" in output
    assert "smap_msl_artifacts_checked=" in output

    with smap_msl_result.classical_csv_path.open("a", encoding="utf-8") as file:
        file.write("tampered\n")
    failed_audit = run_phase2_completion_audit(
        cmapss_manifest=cmapss_result.run_manifest_path,
        smap_msl_manifest=smap_msl_result.run_manifest_path,
    )
    assert not failed_audit.ok
    assert failed_audit.cmapss.ok
    assert not failed_audit.smap_msl.ok
    assert any(
        "artifact classical_csv has unexpected sha256" in problem
        for problem in failed_audit.smap_msl.problems
    )


def _write_smap_msl_fixture(root) -> None:
    (root / "data" / "train").mkdir(parents=True)
    (root / "data" / "test").mkdir(parents=True)
    (root / "labeled_anomalies.csv").write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                '"P-1",SMAP,"[[2, 3]]","[contextual]",6',
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
