"""The sdist allowlist must stay an allowlist, and may only name tracked paths.

Before this file existed the project had no sdist file selection, so hatchling packaged
the working tree minus the *repository's* `.gitignore`. Ignores that live in a
contributor's global gitignore are invisible both to `git status` and to the build
backend, so local-only files were packaged: a build from a working tree carried
`.claude/settings.local.json`, containing absolute machine paths, plus the whole
`graphify-out/` knowledge-graph cache. This package is not published anywhere, so nothing
leaked -- but the defect was invisible to every source-tree check and only appeared on
opening the built artifact.

A published release cannot be unpublished, so the control has to fail closed. Three
properties are pinned here:

1. **The selection stays an allowlist.** Deleting it silently restores the old behaviour,
   which is the regression that caused the problem in the first place.
2. **No known local-only directory is ever named**, whatever a future edit intends.
3. **Every entry names something git tracks.** An untracked path -- which is what every
   local-only file is by construction -- cannot enter the allowlist without failing here.

This does not build the sdist; it constrains what the sdist may be told to include.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory names that are local-only for at least one contributor and must never be
# named in the allowlist.
LOCAL_ONLY_MARKERS = (
    ".claude",
    ".agents",
    "graphify-out",
    ".venv",
    "artifacts",
    "data",
    "dist",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)


def _sdist_include() -> list[str]:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = config["tool"]["hatch"]["build"]["targets"]["sdist"]
    include = sdist["include"]
    assert isinstance(include, list)
    return [str(entry) for entry in include]


def _tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def test_sdist_selection_is_an_allowlist() -> None:
    include = _sdist_include()
    assert include, (
        "The sdist include allowlist is empty. Without it hatchling packages the whole "
        "working tree minus the repository .gitignore, which is how local-only files "
        "such as .claude/settings.local.json and graphify-out/ were reaching the sdist."
    )


def test_allowlist_names_no_local_only_directory() -> None:
    for entry in _sdist_include():
        parts = entry.strip("/").split("/")
        for marker in LOCAL_ONLY_MARKERS:
            assert marker not in parts, (
                f"sdist allowlist entry {entry!r} names {marker!r}, which is local-only "
                "for at least one contributor and must not be packaged."
            )


def test_every_allowlist_entry_is_tracked_by_git() -> None:
    """An untracked path cannot enter the sdist allowlist.

    Every local-only file is untracked by construction, so this is the check that stops
    the original defect rather than merely naming the directories it happened to involve.
    """
    tracked = _tracked_paths()
    for entry in _sdist_include():
        rel = entry.strip("/")
        matched = rel in tracked or any(path.startswith(f"{rel}/") for path in tracked)
        assert matched, (
            f"sdist allowlist entry {entry!r} matches no git-tracked file. Either it is "
            "stale, or it names something untracked -- which must never be packaged."
        )
