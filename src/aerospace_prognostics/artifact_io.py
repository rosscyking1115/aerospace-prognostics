"""Shared helpers for writing project artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def prepare_output_path(path: str | Path) -> Path:
    """Create parent directories for an output path and return the path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_json_payload(
    payload: Any,
    path: str | Path,
    *,
    default: Callable[[Any], Any] | None = None,
) -> Path:
    """Write a deterministic, human-readable JSON artifact."""
    output_path = prepare_output_path(path)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )
    return output_path
