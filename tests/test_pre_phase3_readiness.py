from __future__ import annotations

import json
from pathlib import Path

from aerospace_prognostics.cli import main
from aerospace_prognostics.workflows.pre_phase3_readiness import (
    run_pre_phase3_readiness_audit,
    write_pre_phase3_readiness_json,
    write_pre_phase3_readiness_markdown,
)


def test_pre_phase3_readiness_audit_separates_repo_gates_from_external_blockers(
    tmp_path,
) -> None:
    repo = _repo_root()

    audit = run_pre_phase3_readiness_audit(repo)

    assert audit.status == "not_ready"
    assert {gate.gate_id for gate in audit.blockers} == {
        "license_posture",
        "private_hosted_demo_url",
    }
    assert all(gate.category.startswith("external") for gate in audit.blockers)
    assert all(
        gate.status == "ok"
        for gate in audit.gates
        if gate.category == "repo_local"
    )

    json_path = write_pre_phase3_readiness_json(
        audit,
        tmp_path / "pre_phase3_readiness.json",
    )
    markdown_path = write_pre_phase3_readiness_markdown(
        audit,
        tmp_path / "pre_phase3_readiness.md",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["schema_version"] == "aerospace-prognostics/pre-phase3-readiness/v1"
    assert payload["status"] == "not_ready"
    assert payload["summary"]["blockers"] == 2
    assert "# Pre-Phase-3 Readiness Audit" in markdown
    assert "- Status: not_ready" in markdown
    assert "license_posture" in markdown
    assert "private_hosted_demo_url" in markdown


def test_pre_phase3_readiness_cli_can_pass_with_external_gate_inputs(
    tmp_path,
    capsys,
) -> None:
    output_json = tmp_path / "pre_phase3_ready.json"
    output_markdown = tmp_path / "pre_phase3_ready.md"

    exit_code = main(
        [
            "pre-phase3-readiness-audit",
            "--root",
            str(_repo_root()),
            "--hosted-demo-url",
            "https://private-demo.example.invalid",
            "--license-decision",
            "private-review-only until public launch license is chosen",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "audit_json=" in output
    assert "audit_markdown=" in output
    assert "status=ready" in output
    assert "blockers=0" in output
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "ready"
    assert "- Status: ready" in output_markdown.read_text(encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
