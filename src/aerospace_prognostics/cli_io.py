"""Shared output helpers for command-line handlers."""

from __future__ import annotations

import json
from pathlib import Path


def prepare_output_path(path: Path) -> Path:
    """Create parent directories for an output path and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json_payload(payload: object, path: Path) -> None:
    """Write a deterministic, human-readable JSON payload."""

    output_path = prepare_output_path(path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
