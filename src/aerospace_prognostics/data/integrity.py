"""File integrity helpers for reproducible dataset handling."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class FileCheck:
    """Expected file metadata used to verify local dataset files."""

    path: Path
    sha256: str | None = None
    size_bytes: int | None = None


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's SHA-256 digest."""

    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(check: FileCheck, *, root: str | Path = ".") -> list[str]:
    """Return integrity problems for one file, or an empty list when it is valid."""

    full_path = Path(root) / check.path
    problems: list[str] = []
    if not full_path.exists():
        return [f"{check.path} is missing"]

    if check.size_bytes is not None and full_path.stat().st_size != check.size_bytes:
        problems.append(f"{check.path} has unexpected size")

    if check.sha256 is not None and file_sha256(full_path) != check.sha256:
        problems.append(f"{check.path} has unexpected sha256")

    return problems

