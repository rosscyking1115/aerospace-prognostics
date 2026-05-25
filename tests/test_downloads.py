from __future__ import annotations

import json
import zipfile

from aerospace_prognostics.data.downloads import download_cmapss_dataset


def test_download_cmapss_dataset_extracts_required_files_from_zip(tmp_path) -> None:
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        for subset in ("FD001", "FD002", "FD003", "FD004"):
            archive.writestr(f"CMAPSSData/train_{subset}.txt", "1 1 0\n")
            archive.writestr(f"CMAPSSData/test_{subset}.txt", "1 1 0\n")
            archive.writestr(f"CMAPSSData/RUL_{subset}.txt", "1\n")
        archive.writestr("CMAPSSData/readme.txt", "NASA C-MAPSS")

    result = download_cmapss_dataset(
        tmp_path / "raw" / "cmapss",
        source_url=source_zip.as_uri(),
        archive_path=tmp_path / "downloads" / "cmapss.zip",
    )

    assert (result.output_dir / "train_FD001.txt").exists()
    assert (result.output_dir / "RUL_FD004.txt").exists()
    assert result.metadata_path.exists()
    assert len(result.extracted_files) == 13

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_url"] == source_zip.as_uri()


def test_download_cmapss_dataset_extracts_required_files_from_nested_zip(tmp_path) -> None:
    inner_zip = tmp_path / "CMAPSSData.zip"
    with zipfile.ZipFile(inner_zip, "w") as archive:
        for subset in ("FD001", "FD002", "FD003", "FD004"):
            archive.writestr(f"train_{subset}.txt", "1 1 0\n")
            archive.writestr(f"test_{subset}.txt", "1 1 0\n")
            archive.writestr(f"RUL_{subset}.txt", "1\n")

    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.write(inner_zip, arcname="outer/CMAPSSData.zip")

    result = download_cmapss_dataset(
        tmp_path / "raw" / "cmapss",
        source_url=source_zip.as_uri(),
        archive_path=tmp_path / "downloads" / "cmapss.zip",
    )

    assert len(result.extracted_files) == 12
    assert (result.output_dir / "train_FD004.txt").exists()


def test_download_cmapss_dataset_rejects_incomplete_zip(tmp_path) -> None:
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("train_FD001.txt", "1 1 0\n")

    try:
        download_cmapss_dataset(
            tmp_path / "raw" / "cmapss",
            source_url=source_zip.as_uri(),
            archive_path=tmp_path / "downloads" / "cmapss.zip",
        )
    except ValueError as exc:
        assert "missing required files" in str(exc)
    else:
        raise AssertionError("expected incomplete C-MAPSS zip to fail")
