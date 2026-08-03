"""Compute validated GHCR tags for serving-image publication."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

_IMAGE_RE = re.compile(r"^ghcr\.io/[a-z0-9][a-z0-9._/-]*/[a-z0-9][a-z0-9._/-]*$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


def build_publish_tags(
    *,
    image_name: str,
    github_sha: str,
    github_ref_type: str,
    github_ref_name: str,
    manual_tag: str | None = None,
) -> dict[str, Any]:
    """Return a compact publishing plan for the serving image."""
    normalized_image = image_name.strip().lower()
    _validate_image_name(normalized_image)

    tags = [f"sha-{_short_sha(github_sha)}"]
    ref_tag = github_ref_name.strip()
    requested_manual_tag = (manual_tag or "").strip()

    if requested_manual_tag:
        _validate_tag(requested_manual_tag)
        tags.append(requested_manual_tag)
    elif github_ref_type == "tag" and ref_tag.startswith("v"):
        _validate_tag(ref_tag)
        tags.append(ref_tag)

    deduped_tags = list(dict.fromkeys(tags))
    return {
        "image_name": normalized_image,
        "tags": deduped_tags,
        "tag_args": [arg for tag in deduped_tags for arg in ("--tag", f"{normalized_image}:{tag}")],
        "primary_ref": f"{normalized_image}:{deduped_tags[0]}",
        "published_refs": [f"{normalized_image}:{tag}" for tag in deduped_tags],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--github-sha", required=True)
    parser.add_argument("--github-ref-type", required=True)
    parser.add_argument("--github-ref-name", required=True)
    parser.add_argument("--manual-tag")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    plan = build_publish_tags(
        image_name=args.image_name,
        github_sha=args.github_sha,
        github_ref_type=args.github_ref_type,
        github_ref_name=args.github_ref_name,
        manual_tag=args.manual_tag,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"image_name={plan['image_name']}\n")
            handle.write(f"primary_ref={plan['primary_ref']}\n")
            handle.write(f"published_refs={' '.join(plan['published_refs'])}\n")
            handle.write(f"tag_args={' '.join(plan['tag_args'])}\n")

    print(json.dumps(plan, sort_keys=True))
    return 0


def _short_sha(github_sha: str) -> str:
    sha = github_sha.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("github SHA must be a 40-character hexadecimal commit SHA")
    return sha[:12]


def _validate_image_name(image_name: str) -> None:
    if not _IMAGE_RE.fullmatch(image_name):
        raise ValueError("image name must be a lower-case ghcr.io image path")


def _validate_tag(tag: str) -> None:
    if not _TAG_RE.fullmatch(tag):
        raise ValueError(f"invalid Docker tag: {tag!r}")


if __name__ == "__main__":
    raise SystemExit(main())
