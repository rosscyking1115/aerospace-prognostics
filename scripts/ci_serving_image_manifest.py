"""Generate CI release metadata for the serving Docker image."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

OCI_LABEL_KEYS = (
    "org.opencontainers.image.created",
    "org.opencontainers.image.description",
    "org.opencontainers.image.licenses",
    "org.opencontainers.image.revision",
    "org.opencontainers.image.source",
    "org.opencontainers.image.title",
    "org.opencontainers.image.version",
)


def build_serving_image_manifest(
    *,
    image: str,
    inspect_payload: list[dict[str, Any]],
    expected_revision: str | None = None,
    torch_present: bool | None = None,
) -> dict[str, Any]:
    """Build a compact, auditable serving-image manifest from Docker inspect output."""
    if len(inspect_payload) != 1:
        raise ValueError("docker inspect payload must contain exactly one image")

    image_payload = inspect_payload[0]
    config = image_payload.get("Config") or {}
    labels = config.get("Labels") or {}
    healthcheck = config.get("Healthcheck") or {}
    revision = labels.get("org.opencontainers.image.revision")
    healthcheck_test = healthcheck.get("Test") or []

    validation = {
        "has_oci_labels": all(labels.get(key) for key in OCI_LABEL_KEYS),
        "has_healthcheck": bool(healthcheck_test),
        "revision_matches_expected": (
            None if expected_revision is None else revision == expected_revision
        ),
        "torch_absent": None if torch_present is None else not torch_present,
    }

    return {
        "schema_version": "aerospace-prognostics/serving-image-manifest/v1",
        "image": image,
        "image_id": image_payload.get("Id"),
        "repo_tags": image_payload.get("RepoTags") or [],
        "created": image_payload.get("Created"),
        "labels": {key: labels.get(key) for key in OCI_LABEL_KEYS},
        "healthcheck": {
            "test": healthcheck_test,
            "interval": healthcheck.get("Interval"),
            "timeout": healthcheck.get("Timeout"),
            "start_period": healthcheck.get("StartPeriod"),
            "retries": healthcheck.get("Retries"),
        },
        "dependency_surface": {
            "torch_present": torch_present,
        },
        "validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--expected-revision")
    parser.add_argument("--check-torch-absent", action="store_true")
    args = parser.parse_args(argv)

    inspect_payload = _inspect_image(args.image)
    torch_present = _image_has_torch(args.image) if args.check_torch_absent else None
    manifest = build_serving_image_manifest(
        image=args.image,
        inspect_payload=inspect_payload,
        expected_revision=args.expected_revision,
        torch_present=torch_present,
    )
    _validate_manifest(manifest)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"serving_image_manifest={args.output_json}")
    print(f"image_id={manifest['image_id']}")
    return 0


def _inspect_image(image: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise ValueError("docker image inspect did not return a JSON list")
    return payload


def _image_has_torch(image: str) -> bool:
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            image,
            "python",
            "-c",
            (
                "import importlib.util; "
                "print('present' if importlib.util.find_spec('torch') else 'absent')"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = completed.stdout.strip()
    if result not in {"present", "absent"}:
        raise ValueError(f"unexpected torch probe output: {result!r}")
    return result == "present"


def _validate_manifest(manifest: dict[str, Any]) -> None:
    validation = manifest["validation"]
    failures = [name for name, passed in validation.items() if passed is False]
    if failures:
        raise RuntimeError(f"serving image manifest validation failed: {', '.join(failures)}")


if __name__ == "__main__":
    raise SystemExit(main())
