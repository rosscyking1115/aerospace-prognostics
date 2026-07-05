"""ESA-ADB source manifest and local archive validation helpers."""

from __future__ import annotations

import json
from hashlib import md5
from pathlib import Path
from typing import Any

from aerospace_prognostics.artifact_io import write_json_payload

ESA_ADB_SOURCE_MANIFEST_SCHEMA = "aerospace-prognostics/esa-adb-source-manifest/v1"
ESA_ADB_ARCHIVE_VALIDATION_SCHEMA = "aerospace-prognostics/esa-adb-archive-validation/v1"

ESA_ADB_DATASET = "ESA Anomaly Dataset"
ESA_ADB_DATASET_VERSION = "v2"
ESA_ADB_DATASET_DOI = "10.5281/zenodo.15237121"
ESA_ADB_ORIGINAL_DATASET_DOI = "10.5281/zenodo.12528696"
ESA_ADB_DATASET_LICENSE = "CC BY 3.0 IGO"
ESA_ADB_SOURCE_RECORD_URL = "https://zenodo.org/records/15237121"
ESA_ADB_OFFICIAL_REPOSITORY_URL = "https://github.com/kplabs-pl/ESA-ADB"
ESA_ADB_OFFICIAL_REPOSITORY_REF = "main"
ESA_ADB_DEFAULT_ARCHIVE_DIR = "data/raw/esa_adb"
ESA_ADB_BENCHMARK_MISSIONS = ["Mission1", "Mission2"]

_ESA_ADB_SOURCE_FILES: tuple[dict[str, str], ...] = (
    {
        "mission": "Mission1",
        "file_name": "ESA-Mission1.zip",
        "default_local_path": "data/raw/esa_adb/ESA-Mission1.zip",
        "md5": "9770ad12ed730238f37c42d5c27ab436",
        "size_label": "3.8 GB",
        "zenodo_record_url": ESA_ADB_SOURCE_RECORD_URL,
        "benchmark_scope": "benchmark",
    },
    {
        "mission": "Mission2",
        "file_name": "ESA-Mission2.zip",
        "default_local_path": "data/raw/esa_adb/ESA-Mission2.zip",
        "md5": "bfc72012691427d9327eb41f726ce45e",
        "size_label": "4.1 GB",
        "zenodo_record_url": ESA_ADB_SOURCE_RECORD_URL,
        "benchmark_scope": "benchmark",
    },
    {
        "mission": "Mission3",
        "file_name": "ESA-Mission3.zip",
        "default_local_path": "data/raw/esa_adb/ESA-Mission3.zip",
        "md5": "d63943f09c81378acd9fc5e565ecc66e",
        "size_label": "3.7 GB",
        "zenodo_record_url": ESA_ADB_SOURCE_RECORD_URL,
        "benchmark_scope": "exploration_only",
    },
)


def build_esa_adb_source_manifest() -> dict[str, Any]:
    """Return the tracked source manifest for ESA Anomaly Dataset v2 archives."""

    return {
        "schema_version": ESA_ADB_SOURCE_MANIFEST_SCHEMA,
        "dataset": ESA_ADB_DATASET,
        "dataset_version": ESA_ADB_DATASET_VERSION,
        "dataset_doi": ESA_ADB_DATASET_DOI,
        "original_paper_dataset_doi": ESA_ADB_ORIGINAL_DATASET_DOI,
        "dataset_license": ESA_ADB_DATASET_LICENSE,
        "source_record_url": ESA_ADB_SOURCE_RECORD_URL,
        "official_repository_url": ESA_ADB_OFFICIAL_REPOSITORY_URL,
        "official_repository_ref": ESA_ADB_OFFICIAL_REPOSITORY_REF,
        "official_repository_commit": None,
        "default_archive_dir": ESA_ADB_DEFAULT_ARCHIVE_DIR,
        "benchmark_missions": list(ESA_ADB_BENCHMARK_MISSIONS),
        "files": [dict(entry) for entry in _ESA_ADB_SOURCE_FILES],
    }


def write_esa_adb_source_manifest(path: str | Path) -> dict[str, Any]:
    """Write the ESA-ADB source manifest as deterministic JSON."""

    manifest = build_esa_adb_source_manifest()
    write_json_payload(manifest, path)
    return manifest


def read_esa_adb_source_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate an ESA-ADB source manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != ESA_ADB_SOURCE_MANIFEST_SCHEMA:
        raise ValueError(
            "unexpected ESA-ADB source manifest schema: "
            f"{payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("files"), list):
        raise ValueError("ESA-ADB source manifest files must be a list")
    return payload


def verify_esa_adb_archives(
    archive_dir: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
    missions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Verify locally supplied ESA-ADB archives against manifest MD5 values.

    This intentionally performs no network I/O. It is the safety gate before any
    extraction, preprocessing, or benchmark work touches large raw archives.
    """

    source_manifest = manifest if manifest is not None else build_esa_adb_source_manifest()
    selected_missions = _normalise_requested_missions(source_manifest, missions)
    root = Path(archive_dir)
    problems: list[str] = []
    verified_files: list[dict[str, Any]] = []

    for entry in _files_for_missions(source_manifest, selected_missions):
        file_name = _required_string(entry, "file_name")
        expected_md5 = _required_string(entry, "md5").lower()
        path = root / file_name
        record: dict[str, Any] = {
            "mission": _required_string(entry, "mission"),
            "file_name": file_name,
            "path": str(path),
            "expected_md5": expected_md5,
            "exists": path.exists(),
            "size_bytes": None,
            "actual_md5": None,
            "md5_verified": False,
        }

        if not path.exists():
            problems.append(f"{file_name} is missing")
            verified_files.append(record)
            continue

        actual_md5 = file_md5(path)
        record["size_bytes"] = path.stat().st_size
        record["actual_md5"] = actual_md5
        record["md5_verified"] = actual_md5 == expected_md5
        if actual_md5 != expected_md5:
            problems.append(f"{file_name} has unexpected md5")
        verified_files.append(record)

    return {
        "schema_version": ESA_ADB_ARCHIVE_VALIDATION_SCHEMA,
        "dataset": source_manifest["dataset"],
        "dataset_version": source_manifest["dataset_version"],
        "dataset_doi": source_manifest["dataset_doi"],
        "archive_dir": str(root),
        "missions": list(selected_missions),
        "status": "ok" if not problems else "failed",
        "files_checked": len(verified_files),
        "files_missing": sum(1 for record in verified_files if not record["exists"]),
        "files_with_mismatches": sum(
            1 for record in verified_files if record["exists"] and not record["md5_verified"]
        ),
        "problems": problems,
        "verified_files": verified_files,
    }


def write_esa_adb_archive_validation(
    path: str | Path,
    archive_dir: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
    missions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Verify local archives and write the validation result as JSON."""

    result = verify_esa_adb_archives(archive_dir, manifest=manifest, missions=missions)
    write_json_payload(result, path)
    return result


def file_md5(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's MD5 digest for source records that publish MD5 checksums."""

    digest = md5(usedforsecurity=False)
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_requested_missions(
    manifest: dict[str, Any],
    missions: tuple[str, ...] | None,
) -> tuple[str, ...]:
    available = {
        _required_string(entry, "mission")
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
    }
    if missions is None:
        return tuple(_required_string(entry, "mission") for entry in manifest["files"])

    normalised = tuple(mission.strip() for mission in missions)
    unknown = [mission for mission in normalised if mission not in available]
    if unknown:
        raise ValueError(f"unknown ESA-ADB mission(s): {unknown}")
    return normalised


def _files_for_missions(
    manifest: dict[str, Any],
    missions: tuple[str, ...],
) -> list[dict[str, Any]]:
    mission_set = set(missions)
    return [
        entry
        for entry in manifest["files"]
        if isinstance(entry, dict) and _required_string(entry, "mission") in mission_set
    ]


def _required_string(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"ESA-ADB manifest entry field {key!r} is missing or invalid")
    return value
