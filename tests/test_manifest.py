from __future__ import annotations

import json

import pytest

from aerospace_prognostics.data.manifest import (
    build_cmapss_manifest,
    expected_cmapss_paths,
    read_manifest,
    verify_manifest,
)
from tests.cmapss_fixtures import write_all_tiny_cmapss_subsets, write_tiny_cmapss_subset


def test_expected_cmapss_paths_returns_train_test_and_rul_files() -> None:
    paths = expected_cmapss_paths(subsets=("FD001",))

    assert [path.as_posix() for path in paths] == [
        "train_FD001.txt",
        "test_FD001.txt",
        "RUL_FD001.txt",
    ]


def test_expected_cmapss_paths_validates_subset_names() -> None:
    with pytest.raises(ValueError, match="unknown C-MAPSS"):
        expected_cmapss_paths(subsets=("FD999",))


def test_build_read_and_verify_cmapss_manifest(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    manifest = build_cmapss_manifest(tmp_path, subsets=("FD001",))
    output_path = tmp_path / "nested" / "manifest.json"

    manifest.write_json(output_path)
    loaded = read_manifest(output_path)

    assert loaded == manifest
    assert len(loaded.entries) == 3
    assert json.loads(output_path.read_text(encoding="utf-8"))["dataset"] == "C-MAPSS"
    assert verify_manifest(loaded, root=tmp_path) == []


def test_verify_manifest_reports_changed_files(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    manifest = build_cmapss_manifest(tmp_path, subsets=("FD001",))
    (tmp_path / "RUL_FD001.txt").write_text("999\n", encoding="utf-8")

    problems = verify_manifest(manifest, root=tmp_path)

    assert "RUL_FD001.txt has unexpected size" in problems
    assert "RUL_FD001.txt has unexpected sha256" in problems


def test_build_manifest_can_cover_all_subsets(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)

    manifest = build_cmapss_manifest(tmp_path)

    assert len(manifest.entries) == 12
