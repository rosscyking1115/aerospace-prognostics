"""Release provenance statements for deployment evidence bundles."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IN_TOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
SLSA_PROVENANCE_V1 = "https://slsa.dev/provenance/v1"


@dataclass(frozen=True)
class ReleaseProvenance:
    """Build provenance statement for a release-candidate bundle."""

    release_name: str
    status: str
    problems: list[str]
    statement: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable provenance document."""

        return {
            "schema_version": "aerospace-prognostics/release-provenance/v1",
            "release_name": self.release_name,
            "status": self.status,
            "problems": self.problems,
            "summary": self.summary,
            "statement": self.statement,
        }


def build_release_provenance(
    release_bundle_json: str | Path,
    *,
    repository: str | None = None,
    git_sha: str | None = None,
    git_ref: str | None = None,
    workflow: str | None = None,
    run_id: str | None = None,
    run_attempt: str | None = None,
    actor: str | None = None,
    builder_id: str | None = None,
    generated_at_utc: str | None = None,
) -> ReleaseProvenance:
    """Build an in-toto/SLSA-style provenance statement for a release bundle."""

    bundle_path = Path(release_bundle_json)
    bundle = _read_json_object(bundle_path, "release bundle")
    release_name = str(bundle.get("release_name") or bundle_path.stem)
    release_status = bundle.get("status")
    evidence = _dict_or_empty(bundle.get("evidence"))
    problems: list[str] = []
    if release_status != "ok":
        problems.append("release bundle status is not ok")

    env = os.environ
    repository_value = repository or env.get("GITHUB_REPOSITORY")
    git_sha_value = git_sha or env.get("GITHUB_SHA")
    git_ref_value = git_ref or env.get("GITHUB_REF")
    workflow_value = workflow or env.get("GITHUB_WORKFLOW")
    run_id_value = run_id or env.get("GITHUB_RUN_ID")
    run_attempt_value = run_attempt or env.get("GITHUB_RUN_ATTEMPT")
    actor_value = actor or env.get("GITHUB_ACTOR")
    generated_at = generated_at_utc or datetime.now(UTC).isoformat(timespec="seconds")
    builder = builder_id or _default_builder_id(repository_value, workflow_value)

    subjects = [_subject_from_path(bundle_path, name="release_bundle")]
    for subject_name, evidence_key in (
        ("model_artifact", "model_artifact"),
        ("metadata_json", "metadata_json"),
        ("model_card", "model_card"),
        ("promotion_report", "promotion_report"),
        ("sbom", "sbom"),
        ("container_manifest", "container_manifest"),
    ):
        subject = _subject_from_evidence(evidence, evidence_key, name=subject_name)
        if subject is not None:
            subjects.append(subject)

    if len(subjects) == 1:
        problems.append("release bundle contains no digest-addressed evidence subjects")
    if not git_sha_value:
        problems.append("git SHA is missing from provenance metadata")
    if not repository_value:
        problems.append("repository is missing from provenance metadata")

    statement = {
        "_type": IN_TOTO_STATEMENT_TYPE,
        "subject": subjects,
        "predicateType": SLSA_PROVENANCE_V1,
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/rosscyking1115/aerospace-prognostics/.github/workflows/ci.yml@v1",
                "externalParameters": {
                    "repository": repository_value,
                    "git_ref": git_ref_value,
                    "git_sha": git_sha_value,
                    "workflow": workflow_value,
                    "release_name": release_name,
                },
                "internalParameters": {
                    "release_bundle_path": str(bundle_path),
                    "container_image_ref": bundle.get("container_image_ref"),
                    "artifact_identity": _dict_or_empty(bundle.get("artifact_identity")),
                },
                "resolvedDependencies": _resolved_dependencies(
                    repository=repository_value,
                    git_sha=git_sha_value,
                ),
            },
            "runDetails": {
                "builder": {"id": builder},
                "metadata": {
                    "invocationId": _invocation_id(repository_value, run_id_value),
                    "startedOn": generated_at,
                    "finishedOn": generated_at,
                },
                "byproducts": [
                    {
                        "name": "release_bundle_status",
                        "value": str(release_status),
                    },
                    {
                        "name": "github_run_attempt",
                        "value": run_attempt_value,
                    },
                    {
                        "name": "github_actor",
                        "value": actor_value,
                    },
                ],
            },
        },
    }

    summary = {
        "subject_count": len(subjects),
        "repository": repository_value,
        "git_sha": git_sha_value,
        "git_ref": git_ref_value,
        "workflow": workflow_value,
        "run_id": run_id_value,
        "builder_id": builder,
        "predicate_type": SLSA_PROVENANCE_V1,
    }
    return ReleaseProvenance(
        release_name=release_name,
        status="ok" if not problems else "failed",
        problems=problems,
        statement=statement,
        summary=summary,
    )


def render_release_provenance_markdown(provenance: ReleaseProvenance) -> str:
    """Render release provenance as Markdown for review."""

    summary = provenance.summary
    lines = [
        "# Release Provenance",
        "",
        f"- Release: `{_markdown_inline(provenance.release_name)}`",
        f"- Status: `{_markdown_inline(provenance.status)}`",
        f"- Repository: `{_markdown_inline(summary.get('repository'))}`",
        f"- Git SHA: `{_markdown_inline(summary.get('git_sha'))}`",
        f"- Git ref: `{_markdown_inline(summary.get('git_ref'))}`",
        f"- Workflow: `{_markdown_inline(summary.get('workflow'))}`",
        f"- Run ID: `{_markdown_inline(summary.get('run_id'))}`",
        f"- Builder: `{_markdown_inline(summary.get('builder_id'))}`",
        f"- Predicate type: `{_markdown_inline(summary.get('predicate_type'))}`",
        f"- Subjects: `{_markdown_inline(summary.get('subject_count'))}`",
        "",
        "## Subjects",
        "",
        "| Name | SHA-256 |",
        "|---|---|",
    ]
    for subject in provenance.statement.get("subject", []):
        digest = _dict_or_empty(subject.get("digest"))
        subject_name = _markdown_cell(subject.get("name"))
        subject_sha256 = _markdown_cell(digest.get("sha256"))
        lines.append(
            f"| `{subject_name}` | `{subject_sha256}` |"
        )
    lines.extend(["", "## Problems", ""])
    if provenance.problems:
        lines.extend(f"- {problem}" for problem in provenance.problems)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def write_release_provenance_json(
    provenance: ReleaseProvenance,
    output_json: str | Path,
) -> Path:
    """Write release provenance JSON."""

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(provenance.to_dict(), indent=2, sort_keys=True) + "\n")
    return output_path


def write_release_provenance_markdown(
    provenance: ReleaseProvenance,
    output_markdown: str | Path,
) -> Path:
    """Write release provenance Markdown."""

    output_path = Path(output_markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_release_provenance_markdown(provenance), encoding="utf-8")
    return output_path


def _subject_from_evidence(
    evidence: dict[str, Any],
    evidence_key: str,
    *,
    name: str,
) -> dict[str, Any] | None:
    evidence_item = _dict_or_empty(evidence.get(evidence_key))
    path = evidence_item.get("path")
    digest = evidence_item.get("sha256")
    if not path or not digest:
        return None
    return {"name": f"{name}:{path}", "digest": {"sha256": str(digest)}}


def _subject_from_path(path: Path, *, name: str) -> dict[str, Any]:
    return {"name": f"{name}:{path}", "digest": {"sha256": _sha256_file(path)}}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{label} could not be read: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object: {path}")
    return payload


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolved_dependencies(
    *,
    repository: str | None,
    git_sha: str | None,
) -> list[dict[str, str]]:
    if not repository or not git_sha:
        return []
    return [
        {
            "uri": f"git+https://github.com/{repository}@{git_sha}",
            "digest": {"gitCommit": git_sha},
        }
    ]


def _default_builder_id(repository: str | None, workflow: str | None) -> str:
    repo = repository or "unknown/repository"
    workflow_name = workflow or "unknown-workflow"
    return f"https://github.com/{repo}/actions/workflows/{workflow_name}"


def _invocation_id(repository: str | None, run_id: str | None) -> str | None:
    if not repository or not run_id:
        return run_id
    return f"https://github.com/{repository}/actions/runs/{run_id}"


def _markdown_inline(value: object, *, default: str = "unknown") -> str:
    if value is None:
        return default
    return str(value).replace("`", "'")


def _markdown_cell(value: object) -> str:
    return _markdown_inline(value, default="").replace("|", "\\|")
