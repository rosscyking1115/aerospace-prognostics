"""Dataset manifest helpers for local provenance and integrity checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
from aerospace_prognostics.data.integrity import FileCheck, file_sha256, verify_file


@dataclass(frozen=True)
class ManifestEntry:
    """Recorded metadata for one dataset file."""

    path: str
    sha256: str
    size_bytes: int

    def to_file_check(self) -> FileCheck:
        """Convert this manifest entry into a file-integrity check."""

        return FileCheck(
            path=Path(self.path),
            sha256=self.sha256,
            size_bytes=self.size_bytes,
        )


@dataclass(frozen=True)
class DatasetManifest:
    """Local manifest for dataset provenance and reproducibility."""

    dataset: str
    source_note: str
    entries: list[ManifestEntry]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""

        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        """Write this manifest as pretty JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def build_cmapss_manifest(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    source_note: str = "NASA Prognostics Data Repository C-MAPSS local files",
) -> DatasetManifest:
    """Build a manifest for local C-MAPSS files."""

    root = Path(data_dir)
    entries: list[ManifestEntry] = []
    for relative_path in expected_cmapss_paths(subsets=subsets):
        full_path = root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"missing C-MAPSS file: {full_path}")
        entries.append(
            ManifestEntry(
                path=relative_path.as_posix(),
                sha256=file_sha256(full_path),
                size_bytes=full_path.stat().st_size,
            )
        )

    return DatasetManifest(dataset="C-MAPSS", source_note=source_note, entries=entries)


def read_manifest(path: str | Path) -> DatasetManifest:
    """Read a dataset manifest from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetManifest(
        dataset=payload["dataset"],
        source_note=payload["source_note"],
        entries=[ManifestEntry(**entry) for entry in payload["entries"]],
    )


def verify_manifest(manifest: DatasetManifest, *, root: str | Path) -> list[str]:
    """Return all verification problems for a manifest."""

    problems: list[str] = []
    for entry in manifest.entries:
        problems.extend(verify_file(entry.to_file_check(), root=root))
    return problems


def expected_cmapss_paths(*, subsets: tuple[str, ...] = CMAPSS_SUBSETS) -> list[Path]:
    """Return expected C-MAPSS relative file paths for subsets."""

    normalised_subsets = tuple(subset.upper() for subset in subsets)
    unknown = [subset for subset in normalised_subsets if subset not in CMAPSS_SUBSETS]
    if unknown:
        raise ValueError(f"unknown C-MAPSS subset(s): {unknown}")

    paths: list[Path] = []
    for subset in normalised_subsets:
        paths.extend(
            [
                Path(f"train_{subset}.txt"),
                Path(f"test_{subset}.txt"),
                Path(f"RUL_{subset}.txt"),
            ]
        )
    return paths
