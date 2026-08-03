"""Cross-track Phase 2 completion audit helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload
from aerospace_prognostics.workflows.phase2 import (
    Phase2RunManifestVerification,
    verify_phase2_cmapss_run_manifest,
)
from aerospace_prognostics.workflows.phase2_smap_msl import (
    Phase2SmapMslRunManifestVerification,
    verify_phase2_smap_msl_run_manifest,
)


@dataclass(frozen=True)
class Phase2TrackCompletionAudit:
    """Completion status for one Phase 2 evidence track."""

    track: str
    manifest_path: Path
    workflow: str | None
    status: str
    artifacts_checked: int
    problems: tuple[str, ...]
    counts: Mapping[str, object]

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "track": self.track,
            "manifest_path": self.manifest_path.as_posix(),
            "workflow": self.workflow,
            "status": self.status,
            "artifacts_checked": self.artifacts_checked,
            "problems": list(self.problems),
            "counts": dict(self.counts),
        }


@dataclass(frozen=True)
class Phase2CompletionAudit:
    """Combined completion audit for C-MAPSS and SMAP/MSL Phase 2 tracks."""

    cmapss: Phase2TrackCompletionAudit
    smap_msl: Phase2TrackCompletionAudit

    @property
    def ok(self) -> bool:
        return self.cmapss.ok and self.smap_msl.ok

    @property
    def status(self) -> str:
        return "ok" if self.ok else "failed"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "aerospace-prognostics/phase2-completion-audit/v1",
            "status": self.status,
            "completion_gates": {
                "phase2_cmapss_manifest_verified": self.cmapss.ok,
                "phase2_smap_msl_manifest_verified": self.smap_msl.ok,
                "phase2_evidence_bundle_ready": self.ok,
            },
            "tracks": [self.cmapss.to_dict(), self.smap_msl.to_dict()],
        }


def run_phase2_completion_audit(
    *,
    cmapss_manifest: str | Path,
    smap_msl_manifest: str | Path,
    root: str | Path = ".",
) -> Phase2CompletionAudit:
    """Verify both Phase 2 track manifests and return one completion audit."""
    cmapss_verification = verify_phase2_cmapss_run_manifest(cmapss_manifest, root=root)
    smap_msl_verification = verify_phase2_smap_msl_run_manifest(smap_msl_manifest, root=root)
    return Phase2CompletionAudit(
        cmapss=_track_audit("cmapss_sequence_models", cmapss_verification),
        smap_msl=_track_audit("smap_msl_anomaly_baselines", smap_msl_verification),
    )


def write_phase2_completion_audit_json(
    audit: Phase2CompletionAudit,
    output_path: str | Path,
) -> Path:
    """Write a machine-readable Phase 2 completion audit."""
    return write_json_payload(audit.to_dict(), output_path)


def write_phase2_completion_audit_markdown(
    audit: Phase2CompletionAudit,
    output_path: str | Path,
) -> Path:
    """Write a human-readable Phase 2 completion audit."""
    path = prepare_output_path(output_path)
    lines = [
        "# Phase 2 Completion Audit",
        "",
        f"- Status: {audit.status}",
        f"- C-MAPSS manifest verified: {_yes_no(audit.cmapss.ok)}",
        f"- SMAP/MSL manifest verified: {_yes_no(audit.smap_msl.ok)}",
        f"- Evidence bundle ready: {_yes_no(audit.ok)}",
        "",
        "## Tracks",
        "",
        "| Track | Workflow | Status | Artifacts checked | Problems |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for track in (audit.cmapss, audit.smap_msl):
        lines.append(
            "| "
            f"{track.track} | "
            f"{track.workflow or 'unknown'} | "
            f"{track.status} | "
            f"{track.artifacts_checked} | "
            f"{len(track.problems)} |"
        )

    lines.extend(["", "## Counts", ""])
    for track in (audit.cmapss, audit.smap_msl):
        lines.extend([f"### {track.track}", ""])
        if track.counts:
            lines.extend(
                f"- {key}: {value}" for key, value in sorted(track.counts.items())
            )
        else:
            lines.append("- None")
        lines.append("")

    lines.extend(["## Problems", ""])
    problems = [
        f"- {track.track}: {problem}"
        for track in (audit.cmapss, audit.smap_msl)
        for problem in track.problems
    ]
    lines.extend(problems or ["- None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _track_audit(
    track: str,
    verification: Phase2RunManifestVerification | Phase2SmapMslRunManifestVerification,
) -> Phase2TrackCompletionAudit:
    payload = verification.manifest_payload or {}
    return Phase2TrackCompletionAudit(
        track=track,
        manifest_path=verification.manifest_path,
        workflow=_string_or_none(payload.get("workflow")),
        status="ok" if verification.ok else "failed",
        artifacts_checked=len(verification.checked_artifacts),
        problems=verification.problems,
        counts=_mapping_or_empty(payload.get("counts")),
    )


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _json_scalar(value) for key, value in value.items()}


def _json_scalar(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
