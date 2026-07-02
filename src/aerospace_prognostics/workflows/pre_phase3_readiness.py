"""Pre-Phase-3 readiness audit for launch and productization gates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload


@dataclass(frozen=True)
class PrePhase3ReadinessGate:
    """One gate that must be true before Phase 3 research work starts."""

    gate_id: str
    category: str
    status: str
    evidence: str
    next_action: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "category": self.category,
            "status": self.status,
            "evidence": self.evidence,
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class PrePhase3ReadinessAudit:
    """Readiness result for the work that should be done before Phase 3."""

    gates: tuple[PrePhase3ReadinessGate, ...]

    @property
    def blockers(self) -> tuple[PrePhase3ReadinessGate, ...]:
        return tuple(gate for gate in self.gates if gate.status == "blocker")

    @property
    def warnings(self) -> tuple[PrePhase3ReadinessGate, ...]:
        return tuple(gate for gate in self.gates if gate.status == "warning")

    @property
    def status(self) -> str:
        return "ready" if not self.blockers else "not_ready"

    @property
    def ok(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "aerospace-prognostics/pre-phase3-readiness/v1",
            "status": self.status,
            "summary": {
                "gates": len(self.gates),
                "ok": sum(1 for gate in self.gates if gate.status == "ok"),
                "warnings": len(self.warnings),
                "blockers": len(self.blockers),
            },
            "gates": [gate.to_dict() for gate in self.gates],
        }


def run_pre_phase3_readiness_audit(
    root: str | Path = ".",
    *,
    hosted_demo_url: str | None = None,
    hosted_demo_proof: str | Path | None = None,
    license_decision: str | None = None,
) -> PrePhase3ReadinessAudit:
    """Audit repo-local and external gates before Phase 3 starts."""

    repo_root = Path(root)
    gates = [
        _required_files_gate(
            repo_root,
            "core_docs",
            "repo_local",
            (
                "README.md",
                "docs/project_checklist.md",
                "docs/restructure_plan.md",
                "docs/product_roadmap.md",
                "docs/first_run.md",
                "docs/hosted_demo.md",
                "docs/repo_launch_strategy.md",
                "docs/public_results.md",
                "docs/public_proof_assets.md",
                "docs/pre_phase3_readiness.md",
                "docs/license_posture.md",
                "docs/private_hosting_handoff.md",
                "docs/architecture.md",
                "docs/command_catalog.md",
                "render.yaml",
            ),
            "Required planning, launch, and operator docs are tracked.",
            "Restore the missing planning or launch docs before Phase 3.",
        ),
        _required_files_gate(
            repo_root,
            "public_proof_assets",
            "repo_local",
            (
                "docs/assets/public-proof/streamlit_readonly_console.png",
                "docs/assets/public-proof/fleet_console_snapshot.svg",
                "docs/assets/public-proof/quickstart_rul_diagnostic.svg",
            ),
            "Tracked visual proof assets are present.",
            "Refresh or add the missing public proof assets.",
        ),
        _text_gate(
            repo_root / "docs" / "project_checklist.md",
            "phase2_completion_recorded",
            "repo_local",
            (
                "Combined Phase 2 completion audit",
                "Phase 3 research differentiators",
                "Keep the GitHub repository private",
            ),
            "Checklist records Phase 2 completion and keeps Phase 3 later.",
            "Update docs/project_checklist.md so it reflects current phase boundaries.",
        ),
        _text_gate(
            repo_root / "docs" / "command_catalog.md",
            "phase2_completion_command_documented",
            "repo_local",
            ("phase2-completion-audit",),
            "Command catalog includes the combined Phase 2 evidence audit.",
            "Document phase2-completion-audit in docs/command_catalog.md.",
        ),
        _text_gate(
            repo_root / "docs" / "hosted_demo.md",
            "hosted_demo_runbook",
            "repo_local",
            (
                "Dockerfile.demo",
                "AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true",
                "/_stcore/health",
                "--read-only",
                "allowlist",
            ),
            "Hosted demo runbook covers read-only image, health check, and access control.",
            "Complete the hosted demo runbook before using it as a Phase 3-ready gate.",
        ),
        _text_gate(
            repo_root / "docs" / "private_hosting_handoff.md",
            "private_hosting_handoff",
            "repo_local",
            (
                "render.yaml",
                "Cloudflare Access",
                "--hosted-demo-proof",
            ),
            "Private hosting handoff covers blueprint, access control, and proof capture.",
            "Document the private hosted-demo setup and proof-capture handoff.",
        ),
        _text_gate(
            repo_root / ".github" / "workflows" / "ci.yml",
            "hosted_demo_ci_contract",
            "repo_local",
            (
                "Build hosted demo image",
                "Verify hosted demo image contract",
                "Smoke hosted demo image",
            ),
            "CI builds and smokes the hosted demo image contract.",
            "Restore hosted-demo image contract checks in CI.",
        ),
        _text_gate(
            repo_root / ".gitignore",
            "raw_data_and_artifact_hygiene",
            "repo_local",
            ("/data/", "/artifacts/", "*.pt", "*.pth", "*.onnx"),
            "Raw telemetry, generated artifacts, and model binaries are ignored.",
            "Keep raw data, generated artifacts, and model binaries out of Git.",
        ),
        _license_gate(repo_root, license_decision),
        _hosted_demo_gate(repo_root, hosted_demo_url, hosted_demo_proof),
    ]
    return PrePhase3ReadinessAudit(gates=tuple(gates))


def write_pre_phase3_readiness_json(
    audit: PrePhase3ReadinessAudit,
    output_path: str | Path,
) -> Path:
    """Write a machine-readable pre-Phase-3 readiness audit."""

    return write_json_payload(audit.to_dict(), output_path)


def write_pre_phase3_readiness_markdown(
    audit: PrePhase3ReadinessAudit,
    output_path: str | Path,
) -> Path:
    """Write a human-readable pre-Phase-3 readiness audit."""

    path = prepare_output_path(output_path)
    lines = [
        "# Pre-Phase-3 Readiness Audit",
        "",
        f"- Status: {audit.status}",
        f"- Gates: {len(audit.gates)}",
        f"- Blockers: {len(audit.blockers)}",
        f"- Warnings: {len(audit.warnings)}",
        "",
        "## Gates",
        "",
        "| Gate | Category | Status | Evidence | Next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for gate in audit.gates:
        lines.append(
            "| "
            f"{gate.gate_id} | "
            f"{gate.category} | "
            f"{gate.status} | "
            f"{_escape_markdown_table(gate.evidence)} | "
            f"{_escape_markdown_table(gate.next_action)} |"
        )

    lines.extend(["", "## Blockers", ""])
    if audit.blockers:
        lines.extend(f"- {gate.gate_id}: {gate.next_action}" for gate in audit.blockers)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _required_files_gate(
    root: Path,
    gate_id: str,
    category: str,
    relative_paths: Iterable[str],
    ok_evidence: str,
    next_action: str,
) -> PrePhase3ReadinessGate:
    missing = [path for path in relative_paths if not (root / path).exists()]
    if missing:
        return PrePhase3ReadinessGate(
            gate_id=gate_id,
            category=category,
            status="blocker",
            evidence=f"Missing: {', '.join(missing)}",
            next_action=next_action,
        )
    return PrePhase3ReadinessGate(
        gate_id=gate_id,
        category=category,
        status="ok",
        evidence=ok_evidence,
        next_action="No action required.",
    )


def _text_gate(
    path: Path,
    gate_id: str,
    category: str,
    required_text: Iterable[str],
    ok_evidence: str,
    next_action: str,
) -> PrePhase3ReadinessGate:
    if not path.exists():
        return PrePhase3ReadinessGate(
            gate_id=gate_id,
            category=category,
            status="blocker",
            evidence=f"Missing file: {path.as_posix()}",
            next_action=next_action,
        )
    text = path.read_text(encoding="utf-8")
    missing = [item for item in required_text if item not in text]
    if missing:
        return PrePhase3ReadinessGate(
            gate_id=gate_id,
            category=category,
            status="blocker",
            evidence=f"Missing text: {', '.join(missing)}",
            next_action=next_action,
        )
    return PrePhase3ReadinessGate(
        gate_id=gate_id,
        category=category,
        status="ok",
        evidence=ok_evidence,
        next_action="No action required.",
    )


def _license_gate(root: Path, license_decision: str | None) -> PrePhase3ReadinessGate:
    license_files = ("LICENSE", "LICENSE.md", "LICENCE", "LICENCE.md")
    tracked_license = next((name for name in license_files if (root / name).exists()), None)
    if tracked_license is not None:
        return PrePhase3ReadinessGate(
            gate_id="license_posture",
            category="external_decision",
            status="ok",
            evidence=f"Tracked license file: {tracked_license}",
            next_action="No action required.",
        )
    posture_path = root / "docs" / "license_posture.md"
    if posture_path.exists():
        posture_text = posture_path.read_text(encoding="utf-8")
        required_text = (
            "private review only",
            "not currently distributed under an open-source license",
            "public-launch license",
            "UNLICENSED",
        )
        missing = [item for item in required_text if item not in posture_text]
        if not missing:
            return PrePhase3ReadinessGate(
                gate_id="license_posture",
                category="external_decision",
                status="ok",
                evidence="Tracked private-review-only license posture in docs/license_posture.md.",
                next_action="Choose and add a final license file before public launch.",
            )
    if license_decision:
        return PrePhase3ReadinessGate(
            gate_id="license_posture",
            category="external_decision",
            status="ok",
            evidence=f"License decision supplied: {license_decision}",
            next_action="Add the license file before public launch if the decision is public.",
        )
    return PrePhase3ReadinessGate(
        gate_id="license_posture",
        category="external_decision",
        status="blocker",
        evidence="No LICENSE file or explicit license decision supplied.",
        next_action="Choose the license/posture before starting Phase 3.",
    )


def _hosted_demo_gate(
    root: Path,
    hosted_demo_url: str | None,
    hosted_demo_proof: str | Path | None,
) -> PrePhase3ReadinessGate:
    if hosted_demo_url and hosted_demo_proof:
        proof_path = Path(hosted_demo_proof)
        if not proof_path.is_absolute():
            proof_path = root / proof_path
        if proof_path.exists():
            return PrePhase3ReadinessGate(
                gate_id="private_hosted_demo_url",
                category="external_deployment",
                status="ok",
                evidence=(
                    f"Private hosted demo URL supplied: {hosted_demo_url}; "
                    f"proof asset: {proof_path.as_posix()}"
                ),
                next_action="No action required.",
            )
        return PrePhase3ReadinessGate(
            gate_id="private_hosted_demo_url",
            category="external_deployment",
            status="blocker",
            evidence=f"Hosted demo proof asset not found: {proof_path.as_posix()}",
            next_action="Capture a fresh screenshot/GIF from the private hosted demo URL.",
        )
    if hosted_demo_url:
        return PrePhase3ReadinessGate(
            gate_id="private_hosted_demo_url",
            category="external_deployment",
            status="blocker",
            evidence=f"Private hosted demo URL supplied without proof: {hosted_demo_url}",
            next_action="Capture a fresh screenshot/GIF from the private hosted demo URL.",
        )
    if hosted_demo_proof:
        return PrePhase3ReadinessGate(
            gate_id="private_hosted_demo_url",
            category="external_deployment",
            status="blocker",
            evidence=f"Hosted demo proof supplied without URL: {hosted_demo_proof}",
            next_action="Create the private hosted read-only demo URL.",
        )
    return PrePhase3ReadinessGate(
        gate_id="private_hosted_demo_url",
        category="external_deployment",
        status="blocker",
        evidence="No private hosted demo URL supplied.",
        next_action="Create a private hosted read-only demo URL and capture proof from it.",
    )


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|")
