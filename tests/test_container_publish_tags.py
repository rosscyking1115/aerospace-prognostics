from __future__ import annotations

import pytest

from scripts.ci_container_publish_tags import build_publish_tags

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_build_publish_tags_includes_sha_and_version_tag_for_git_tag() -> None:
    plan = build_publish_tags(
        image_name="ghcr.io/RossCyKing1115/Aerospace-Prognostics/Serving",
        github_sha=SHA,
        github_ref_type="tag",
        github_ref_name="v0.1.0-rc1",
    )

    assert plan["image_name"] == "ghcr.io/rosscyking1115/aerospace-prognostics/serving"
    assert plan["tags"] == ["sha-0123456789ab", "v0.1.0-rc1"]
    assert plan["published_refs"] == [
        "ghcr.io/rosscyking1115/aerospace-prognostics/serving:sha-0123456789ab",
        "ghcr.io/rosscyking1115/aerospace-prognostics/serving:v0.1.0-rc1",
    ]


def test_build_publish_tags_manual_tag_takes_precedence() -> None:
    plan = build_publish_tags(
        image_name="ghcr.io/example/project/serving",
        github_sha=SHA,
        github_ref_type="branch",
        github_ref_name="main",
        manual_tag="manual-rc",
    )

    assert plan["tags"] == ["sha-0123456789ab", "manual-rc"]


def test_build_publish_tags_omits_branch_name_without_manual_tag() -> None:
    plan = build_publish_tags(
        image_name="ghcr.io/example/project/serving",
        github_sha=SHA,
        github_ref_type="branch",
        github_ref_name="main",
    )

    assert plan["tags"] == ["sha-0123456789ab"]


def test_build_publish_tags_rejects_invalid_sha() -> None:
    with pytest.raises(ValueError, match="40-character"):
        build_publish_tags(
            image_name="ghcr.io/example/project/serving",
            github_sha="abc123",
            github_ref_type="tag",
            github_ref_name="v0.1.0",
        )


def test_build_publish_tags_rejects_invalid_manual_tag() -> None:
    with pytest.raises(ValueError, match="invalid Docker tag"):
        build_publish_tags(
            image_name="ghcr.io/example/project/serving",
            github_sha=SHA,
            github_ref_type="branch",
            github_ref_name="main",
            manual_tag="../bad",
        )
