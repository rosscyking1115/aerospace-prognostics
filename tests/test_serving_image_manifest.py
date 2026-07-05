from __future__ import annotations

import pytest

from scripts.ci_serving_image_manifest import build_serving_image_manifest


def test_build_serving_image_manifest_records_labels_healthcheck_and_dependency_surface() -> None:
    manifest = build_serving_image_manifest(
        image="aerospace-prognostics:ci",
        inspect_payload=[_inspect_payload(revision="abc123")],
        expected_revision="abc123",
        torch_present=False,
    )

    assert manifest["schema_version"] == "aerospace-prognostics/serving-image-manifest/v1"
    assert manifest["image_id"] == "sha256:image"
    assert manifest["labels"]["org.opencontainers.image.revision"] == "abc123"
    assert manifest["healthcheck"]["test"] == [
        "CMD-SHELL",
        "python -m aerospace_prognostics.serving.healthcheck",
    ]
    assert manifest["dependency_surface"] == {"torch_present": False}
    assert manifest["validation"] == {
        "has_oci_labels": True,
        "has_healthcheck": True,
        "revision_matches_expected": True,
        "torch_absent": True,
    }


def test_build_serving_image_manifest_flags_revision_mismatch() -> None:
    manifest = build_serving_image_manifest(
        image="aerospace-prognostics:ci",
        inspect_payload=[_inspect_payload(revision="abc123")],
        expected_revision="different",
        torch_present=False,
    )

    assert manifest["validation"]["revision_matches_expected"] is False


def test_build_serving_image_manifest_requires_one_inspected_image() -> None:
    with pytest.raises(ValueError, match="exactly one image"):
        build_serving_image_manifest(image="image", inspect_payload=[])


def _inspect_payload(*, revision: str) -> dict[str, object]:
    labels = {
        "org.opencontainers.image.created": "2026-06-11T01:00:00Z",
        "org.opencontainers.image.description": "Serving image.",
        "org.opencontainers.image.licenses": "MIT",
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.source": "https://github.com/example/repo",
        "org.opencontainers.image.title": "aerospace-prognostics-serving",
        "org.opencontainers.image.version": "0.1.0",
    }
    return {
        "Id": "sha256:image",
        "RepoTags": ["aerospace-prognostics:ci"],
        "Created": "2026-06-11T01:00:00Z",
        "Config": {
            "Labels": labels,
            "Healthcheck": {
                "Test": ["CMD-SHELL", "python -m aerospace_prognostics.serving.healthcheck"],
                "Interval": 30_000_000_000,
                "Timeout": 5_000_000_000,
                "StartPeriod": 10_000_000_000,
                "Retries": 3,
            },
        },
    }
