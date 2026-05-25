from __future__ import annotations

from aerospace_prognostics.data.integrity import FileCheck, file_sha256, verify_file


def test_file_sha256_and_verify_file(tmp_path) -> None:
    path = tmp_path / "example.txt"
    path.write_bytes(b"aerospace\n")

    digest = file_sha256(path)

    assert verify_file(FileCheck(path=path.name, sha256=digest, size_bytes=10), root=tmp_path) == []
    assert verify_file(FileCheck(path="missing.txt"), root=tmp_path) == ["missing.txt is missing"]


def test_verify_file_reports_mismatches(tmp_path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("aerospace\n", encoding="utf-8")

    problems = verify_file(FileCheck(path=path.name, sha256="0" * 64, size_bytes=1), root=tmp_path)

    assert problems == [
        "example.txt has unexpected size",
        "example.txt has unexpected sha256",
    ]
