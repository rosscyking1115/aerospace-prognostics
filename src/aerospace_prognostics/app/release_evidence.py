"""Helpers for model-artifact release evidence rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def release_evidence_record(
    *,
    artifact_id: str,
    evidence_type: str,
    path: Path,
    payload: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Build a deterministic release-evidence row payload."""
    evidence_id = f"{evidence_type}:{artifact_id}:{_file_sha256(path)}"
    status = payload.get("status") if isinstance(payload.get("status"), str) else None
    return {
        "evidence_id": evidence_id,
        "artifact_id": artifact_id,
        "evidence_type": evidence_type,
        "source_path": str(path),
        "status": status,
        "payload_json": _json_dumps(payload),
        "created_at_utc": timestamp,
    }


def evidence_from_row(row: Any) -> dict[str, Any]:
    """Decode one SQLite release-evidence row for API/console consumers."""
    evidence = dict(row)
    evidence["payload"] = _json_loads(evidence.pop("payload_json"))
    return evidence


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _json_loads(payload: str | None) -> Any:
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}
