from __future__ import annotations

import json
from hashlib import md5

from aerospace_prognostics.data.esa_adb import (
    build_esa_adb_source_manifest,
    read_esa_adb_source_manifest,
    verify_esa_adb_archives,
    write_esa_adb_source_manifest,
)

# Evaluator-contract behavior (label joins, event grouping, metric-input
# validation) is tested in the telemeval library, which this project now
# depends on. Only the project-specific dataset plumbing is tested here.


def test_build_esa_adb_source_manifest_records_zenodo_v2_archives() -> None:
    manifest = build_esa_adb_source_manifest()

    assert manifest["schema_version"] == "aerospace-prognostics/esa-adb-source-manifest/v1"
    assert manifest["dataset"] == "ESA Anomaly Dataset"
    assert manifest["dataset_version"] == "v2"
    assert manifest["dataset_doi"] == "10.5281/zenodo.15237121"
    assert manifest["original_paper_dataset_doi"] == "10.5281/zenodo.12528696"
    assert manifest["dataset_license"] == "CC BY 3.0 IGO"
    assert manifest["official_repository_url"] == "https://github.com/kplabs-pl/ESA-ADB"
    assert manifest["official_repository_ref"] == "main"
    assert manifest["official_repository_commit"] is None
    assert manifest["default_archive_dir"] == "data/raw/esa_adb"
    assert manifest["benchmark_missions"] == ["Mission1", "Mission2"]
    assert [entry["file_name"] for entry in manifest["files"]] == [
        "ESA-Mission1.zip",
        "ESA-Mission2.zip",
        "ESA-Mission3.zip",
    ]
    assert manifest["files"][0]["md5"] == "9770ad12ed730238f37c42d5c27ab436"
    assert manifest["files"][0]["default_local_path"] == "data/raw/esa_adb/ESA-Mission1.zip"
    assert manifest["files"][1]["md5"] == "bfc72012691427d9327eb41f726ce45e"
    assert manifest["files"][2]["md5"] == "d63943f09c81378acd9fc5e565ecc66e"


def test_write_and_read_esa_adb_source_manifest(tmp_path) -> None:
    output_path = tmp_path / "nested" / "esa_adb_source_manifest.json"

    manifest = write_esa_adb_source_manifest(output_path)
    loaded = read_esa_adb_source_manifest(output_path)

    assert output_path.exists()
    assert loaded == manifest
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_verify_esa_adb_archives_checks_selected_mission_md5(tmp_path) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive_payload = b"mission1 fixture archive\n"
    archive_path = archive_dir / "ESA-Mission1.zip"
    archive_path.write_bytes(archive_payload)
    manifest = _tiny_esa_manifest(
        md5=md5(archive_payload, usedforsecurity=False).hexdigest()
    )

    result = verify_esa_adb_archives(
        archive_dir,
        manifest=manifest,
        missions=("Mission1",),
    )

    assert result["status"] == "ok"
    assert result["files_checked"] == 1
    assert result["problems"] == []
    assert result["verified_files"][0]["file_name"] == "ESA-Mission1.zip"
    assert result["verified_files"][0]["md5_verified"] is True


def test_verify_esa_adb_archives_reports_missing_and_changed_archives(tmp_path) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    (archive_dir / "ESA-Mission1.zip").write_bytes(b"changed\n")
    manifest = _tiny_esa_manifest(
        md5=md5(b"expected\n", usedforsecurity=False).hexdigest()
    )

    changed = verify_esa_adb_archives(
        archive_dir,
        manifest=manifest,
        missions=("Mission1",),
    )
    missing = verify_esa_adb_archives(
        archive_dir,
        manifest=manifest,
        missions=("Mission2",),
    )

    assert changed["status"] == "failed"
    assert "ESA-Mission1.zip has unexpected md5" in changed["problems"]
    assert missing["status"] == "failed"
    assert "ESA-Mission2.zip is missing" in missing["problems"]


def test_read_esa_adb_source_manifest_rejects_wrong_schema(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    try:
        read_esa_adb_source_manifest(path)
    except ValueError as exc:
        assert "unexpected ESA-ADB source manifest schema" in str(exc)
    else:
        raise AssertionError("expected wrong schema to be rejected")


def _tiny_esa_manifest(*, md5: str) -> dict[str, object]:
    return {
        "schema_version": "aerospace-prognostics/esa-adb-source-manifest/v1",
        "dataset": "ESA Anomaly Dataset",
        "dataset_version": "test",
        "dataset_doi": "test-doi",
        "original_paper_dataset_doi": "paper-doi",
        "dataset_license": "CC BY 3.0 IGO",
        "source_record_url": "https://example.test/esa-adb",
        "benchmark_missions": ["Mission1", "Mission2"],
        "files": [
            {
                "mission": "Mission1",
                "file_name": "ESA-Mission1.zip",
                "md5": md5,
                "size_label": "fixture",
                "zenodo_record_url": "https://example.test/mission1",
                "benchmark_scope": "benchmark",
            },
            {
                "mission": "Mission2",
                "file_name": "ESA-Mission2.zip",
                "md5": md5,
                "size_label": "fixture",
                "zenodo_record_url": "https://example.test/mission2",
                "benchmark_scope": "benchmark",
            },
        ],
    }
