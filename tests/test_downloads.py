from __future__ import annotations

import json
import urllib.error
import zipfile
from io import BytesIO

import numpy as np

from aerospace_prognostics.data.downloads import (
    download_cmapss_dataset,
    download_smap_msl_dataset,
)


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
    assert result.metadata_path.read_text(encoding="utf-8").endswith("\n")


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


def test_download_smap_msl_dataset_extracts_arrays_and_labels(tmp_path) -> None:
    source_zip = tmp_path / "smap_msl_source.zip"
    labels_csv = tmp_path / "labeled_anomalies.csv"
    labels_csv.write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                '"P-1",SMAP,"[[1, 2]]","[contextual]",3',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(
            "data/data/train/P-1.npy",
            _npy_bytes(np.array([[0.0, 0.0], [1.0, 1.0]])),
        )
        archive.writestr(
            "data/data/test/P-1.npy",
            _npy_bytes(np.array([[0.0, 0.0], [5.0, -5.0], [6.0, -6.0]])),
        )

    result = download_smap_msl_dataset(
        tmp_path / "raw" / "smap_msl",
        source_url=source_zip.as_uri(),
        labels_url=labels_csv.as_uri(),
        archive_path=tmp_path / "downloads" / "smap_msl.zip",
    )

    assert (result.output_dir / "data" / "train" / "P-1.npy").exists()
    assert (result.output_dir / "data" / "test" / "P-1.npy").exists()
    assert result.labels_path.exists()
    assert len(result.extracted_arrays) == 2
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_url"] == source_zip.as_uri()
    assert metadata["labels_url"] == labels_csv.as_uri()
    assert result.metadata_path.read_text(encoding="utf-8").endswith("\n")


def test_download_smap_msl_dataset_can_import_existing_kaggle_archive(tmp_path) -> None:
    archive_path = tmp_path / "downloads" / "smap_msl.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "data/data/train/P-1.npy",
            _npy_bytes(np.array([[0.0, 0.0], [1.0, 1.0]])),
        )
        archive.writestr(
            "data/data/test/P-1.npy",
            _npy_bytes(np.array([[0.0, 0.0], [5.0, -5.0], [6.0, -6.0]])),
        )
        archive.writestr(
            "data/data/labeled_anomalies.csv",
            "\n".join(
                [
                    "chan_id,spacecraft,anomaly_sequences,class,num_values",
                    '"P-1",SMAP,"[[1, 2]]","[contextual]",3',
                ]
            )
            + "\n",
        )

    result = download_smap_msl_dataset(
        tmp_path / "raw" / "smap_msl",
        source_url="https://example.invalid/smap_msl.zip",
        labels_url="https://example.invalid/labeled_anomalies.csv",
        archive_path=archive_path,
    )

    assert (result.output_dir / "data" / "train" / "P-1.npy").exists()
    assert result.labels_path.read_text(encoding="utf-8").startswith("chan_id,")


def test_download_smap_msl_dataset_rejects_zip_without_arrays(tmp_path) -> None:
    source_zip = tmp_path / "empty.zip"
    labels_csv = tmp_path / "labeled_anomalies.csv"
    labels_csv.write_text(
        "chan_id,spacecraft,anomaly_sequences,class,num_values\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("README.txt", "no arrays here")

    try:
        download_smap_msl_dataset(
            tmp_path / "raw" / "smap_msl",
            source_url=source_zip.as_uri(),
            labels_url=labels_csv.as_uri(),
            archive_path=tmp_path / "downloads" / "smap_msl.zip",
        )
    except ValueError as exc:
        assert "missing train/test .npy arrays" in str(exc)
    else:
        raise AssertionError("expected incomplete SMAP/MSL zip to fail")


def test_download_smap_msl_dataset_reports_kaggle_fallback_when_download_fails(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_download(url, filename):
        raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlretrieve", fail_download)
    archive_path = tmp_path / "downloads" / "smap_msl.zip"

    try:
        download_smap_msl_dataset(
            tmp_path / "raw" / "smap_msl",
            source_url="https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
            archive_path=archive_path,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "legacy public Telemanom S3 archive" in message
        assert "patrickfleith/nasa-anomaly-detection-dataset-smap-msl" in message
        assert archive_path.as_posix() in message
    else:
        raise AssertionError("expected failed SMAP/MSL download to report Kaggle fallback")


def _npy_bytes(values: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, values)
    return buffer.getvalue()
