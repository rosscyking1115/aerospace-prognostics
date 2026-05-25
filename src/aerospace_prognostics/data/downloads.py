"""Dataset download helpers."""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS

NASA_CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)


@dataclass(frozen=True)
class CmapssDownloadResult:
    """Local paths created by the C-MAPSS download helper."""

    source_url: str
    archive_path: Path
    output_dir: Path
    extracted_files: tuple[Path, ...]
    metadata_path: Path


def download_cmapss_dataset(
    output_dir: str | Path,
    *,
    source_url: str = NASA_CMAPSS_URL,
    archive_path: str | Path | None = None,
    force: bool = False,
) -> CmapssDownloadResult:
    """Download and extract the NASA C-MAPSS FD001-FD004 raw text files."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    archive = Path(archive_path) if archive_path is not None else destination / "cmapss_nasa.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)

    if archive.exists() and not force:
        raise FileExistsError(f"archive already exists: {archive}")

    urllib.request.urlretrieve(source_url, archive)

    extracted_files = _extract_cmapss_zip(archive, destination, force=force)
    metadata_path = _write_download_metadata(
        destination,
        source_url=source_url,
        archive_path=archive,
        extracted_files=extracted_files,
    )
    return CmapssDownloadResult(
        source_url=source_url,
        archive_path=archive,
        output_dir=destination,
        extracted_files=tuple(extracted_files),
        metadata_path=metadata_path,
    )


def _extract_cmapss_zip(archive: Path, destination: Path, *, force: bool) -> tuple[Path, ...]:
    required_names = {
        f"{prefix}_{subset}.txt"
        for subset in CMAPSS_SUBSETS
        for prefix in ("train", "test", "RUL")
    }

    with zipfile.ZipFile(archive) as zip_file:
        members_by_name = {Path(member.filename).name: member for member in zip_file.infolist()}
        if required_names.issubset(members_by_name):
            return _extract_members(
                zip_file,
                members_by_name,
                required_names,
                destination,
                force=force,
            )

        for nested_member in zip_file.infolist():
            if Path(nested_member.filename).suffix.lower() != ".zip":
                continue
            with zip_file.open(nested_member) as nested_source:
                nested_bytes = nested_source.read()
            with zipfile.ZipFile(BytesIO(nested_bytes)) as nested_zip:
                nested_members_by_name = {
                    Path(member.filename).name: member for member in nested_zip.infolist()
                }
                if required_names.issubset(nested_members_by_name):
                    return _extract_members(
                        nested_zip,
                        nested_members_by_name,
                        required_names,
                        destination,
                        force=force,
                    )

    missing = sorted(required_names.difference(members_by_name))
    raise ValueError(f"C-MAPSS archive is missing required files: {missing}")


def _extract_members(
    zip_file: zipfile.ZipFile,
    members_by_name: dict[str, zipfile.ZipInfo],
    required_names: set[str],
    destination: Path,
    *,
    force: bool,
) -> tuple[Path, ...]:
    extracted: list[Path] = []

    for filename in sorted(required_names):
        output_path = destination / filename
        if output_path.exists() and not force:
            raise FileExistsError(f"raw C-MAPSS file already exists: {output_path}")
        with (
            zip_file.open(members_by_name[filename]) as source,
            output_path.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        extracted.append(output_path)

    readme_member = members_by_name.get("readme.txt") or members_by_name.get("README.txt")
    if readme_member is not None:
        readme_path = destination / Path(readme_member.filename).name
        if force or not readme_path.exists():
            with (
                zip_file.open(readme_member) as source,
                readme_path.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
            extracted.append(readme_path)

    return tuple(extracted)


def _write_download_metadata(
    destination: Path,
    *,
    source_url: str,
    archive_path: Path,
    extracted_files: tuple[Path, ...],
) -> Path:
    metadata_path = destination / "cmapss_download_metadata.json"
    payload = {
        "dataset": "NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set",
        "source_url": source_url,
        "downloaded_at_utc": datetime.now(tz=UTC).isoformat(),
        "archive_path": archive_path.as_posix(),
        "extracted_files": [path.name for path in extracted_files],
        "citation": (
            "A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation "
            "Data Set, NASA Ames Prognostics Data Repository, NASA Ames, Moffett Field, CA."
        ),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def cmapss_download_result_to_dict(result: CmapssDownloadResult) -> dict[str, object]:
    """Return a JSON-serialisable representation of a download result."""

    payload = asdict(result)
    payload["archive_path"] = result.archive_path.as_posix()
    payload["output_dir"] = result.output_dir.as_posix()
    payload["extracted_files"] = [path.as_posix() for path in result.extracted_files]
    payload["metadata_path"] = result.metadata_path.as_posix()
    return payload
