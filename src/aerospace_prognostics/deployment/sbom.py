"""Software bill of materials generation from uv lockfiles."""

from __future__ import annotations

import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aerospace_prognostics.artifact_io import write_json_payload

CYCLONEDX_SPEC_VERSION = "1.6"


def build_uv_lock_cyclonedx_sbom(
    lockfile: str | Path,
    *,
    serial_number: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a CycloneDX-style dependency SBOM from a uv lockfile."""
    lock_path = Path(lockfile)
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = tuple(lock.get("package", ()))
    root_package = _root_package(packages)
    package_refs = {
        str(package["name"]).lower(): _package_bom_ref(package)
        for package in packages
        if package is not root_package
    }
    components = [
        _library_component(package)
        for package in sorted(packages, key=lambda item: str(item["name"]).lower())
        if package is not root_package
    ]
    dependencies = [_root_dependency(root_package, package_refs)]
    dependencies.extend(
        _package_dependency(package, package_refs)
        for package in sorted(packages, key=lambda item: str(item["name"]).lower())
        if package is not root_package
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_SPEC_VERSION,
        "serialNumber": serial_number or f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
            "component": _root_component(root_package),
        },
        "components": components,
        "dependencies": dependencies,
    }


def write_uv_lock_cyclonedx_sbom(lockfile: str | Path, output_json: str | Path) -> Path:
    """Write a CycloneDX-style SBOM and return the output path."""
    sbom = build_uv_lock_cyclonedx_sbom(lockfile)
    return write_json_payload(sbom, output_json)


def _root_package(packages: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    editable_packages = [
        package
        for package in packages
        if isinstance(package.get("source"), dict) and package["source"].get("editable")
    ]
    if len(editable_packages) != 1:
        raise ValueError("uv lockfile must contain exactly one editable root package")
    return editable_packages[0]


def _root_component(root_package: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "application",
        "bom-ref": _package_bom_ref(root_package),
        "name": root_package["name"],
        "version": root_package["version"],
    }


def _library_component(package: dict[str, Any]) -> dict[str, Any]:
    component: dict[str, Any] = {
        "type": "library",
        "bom-ref": _package_bom_ref(package),
        "name": package["name"],
        "version": package["version"],
        "purl": _package_purl(package),
    }
    hashes = _package_hashes(package)
    if hashes:
        component["hashes"] = hashes
    return component


def _root_dependency(
    root_package: dict[str, Any],
    package_refs: dict[str, str],
) -> dict[str, Any]:
    dependencies = [
        package_refs[dependency["name"].lower()]
        for dependency in root_package.get("dependencies", ())
        if dependency["name"].lower() in package_refs
    ]
    dependencies.extend(
        package_refs[dependency["name"].lower()]
        for dependency_group in root_package.get("dev-dependencies", {}).values()
        for dependency in dependency_group
        if dependency["name"].lower() in package_refs
    )
    return {"ref": _package_bom_ref(root_package), "dependsOn": sorted(set(dependencies))}


def _package_dependency(
    package: dict[str, Any],
    package_refs: dict[str, str],
) -> dict[str, Any]:
    dependencies = [
        package_refs[dependency["name"].lower()]
        for dependency in package.get("dependencies", ())
        if dependency["name"].lower() in package_refs
    ]
    return {"ref": _package_bom_ref(package), "dependsOn": sorted(set(dependencies))}


def _package_bom_ref(package: dict[str, Any]) -> str:
    return _package_purl(package)


def _package_purl(package: dict[str, Any]) -> str:
    return f"pkg:pypi/{_normalise_pypi_name(str(package['name']))}@{package['version']}"


def _normalise_pypi_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _package_hashes(package: dict[str, Any]) -> list[dict[str, str]]:
    hashes: list[dict[str, str]] = []
    for artifact in _distribution_artifacts(package):
        hash_value = artifact.get("hash")
        if isinstance(hash_value, str) and hash_value.startswith("sha256:"):
            hashes.append({"alg": "SHA-256", "content": hash_value.removeprefix("sha256:")})
    seen: set[str] = set()
    unique_hashes: list[dict[str, str]] = []
    for hash_row in hashes:
        content = hash_row["content"]
        if content not in seen:
            unique_hashes.append(hash_row)
            seen.add(content)
    return unique_hashes


def _distribution_artifacts(package: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    sdist = package.get("sdist")
    if isinstance(sdist, dict):
        artifacts.append(sdist)
    wheels = package.get("wheels")
    if isinstance(wheels, list):
        artifacts.extend(wheel for wheel in wheels if isinstance(wheel, dict))
    return artifacts
