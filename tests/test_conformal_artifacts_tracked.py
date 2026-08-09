"""The conformal evidence bundle must stay git-tracked, not merely present.

The writeup claims that every conformal number traces to a committed artifact.
That claim is worth exactly as much as its enforcement: `artifacts/` is ignored
wholesale apart from a narrow negation, and a later edit to `.gitignore` --
or a `git rm --cached` -- would make the claim false without changing a single
number or breaking a single other test. This file turns that silent failure into
a red test.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFORMAL_DIRECTORY = REPOSITORY_ROOT / "artifacts" / "conformal"

REQUIRED_ARTIFACTS = (
    "cmapss_fd001_conformal.json",
    "cmapss_fd001_conformal.md",
    "cmapss_fd001_conformal_variants.csv",
    "cmapss_fd001_conformal_seed_sweep.csv",
    "cmapss_fd001_conformal_attainability.csv",
    "cmapss_fd001_conformal_alpha001.json",
    "cmapss_fd002_conformal_alpha001.json",
)


def _tracked_conformal_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "artifacts/conformal"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        Path(line).name
        for line in result.stdout.splitlines()
        if line.strip()
    }


@pytest.fixture(scope="module")
def tracked_files() -> set[str]:
    if not (REPOSITORY_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    return _tracked_conformal_files()


def test_git_tracks_the_conformal_evidence_directory(tracked_files: set[str]) -> None:
    assert tracked_files, (
        "artifacts/conformal is not tracked by git. The writeup's traceability claim "
        "is false until it is. Check the .gitignore negation with "
        "`git check-ignore -v artifacts/conformal/<file>.json`."
    )


@pytest.mark.parametrize("filename", REQUIRED_ARTIFACTS)
def test_each_cited_artifact_is_tracked(filename: str, tracked_files: set[str]) -> None:
    assert filename in tracked_files, f"{filename} is cited by the writeup but is not tracked"


@pytest.mark.parametrize("filename", REQUIRED_ARTIFACTS)
def test_each_cited_artifact_exists_on_disk(filename: str) -> None:
    assert (CONFORMAL_DIRECTORY / filename).is_file()


def test_only_text_results_are_tracked(tracked_files: set[str]) -> None:
    # Rider 3: the negation stays narrow. Model binaries, raw data, and derived
    # arrays are regenerable and stay out of git.
    allowed_suffixes = {".json", ".csv", ".md"}
    offenders = sorted(
        name for name in tracked_files if Path(name).suffix not in allowed_suffixes
    )
    assert not offenders, f"non-text artifacts are tracked: {offenders}"


def test_tracked_artifacts_stay_small(tracked_files: set[str]) -> None:
    # A committed artifact directory is only sustainable while it stays text-sized.
    oversized = sorted(
        name
        for name in tracked_files
        if (CONFORMAL_DIRECTORY / name).is_file()
        and (CONFORMAL_DIRECTORY / name).stat().st_size > 512_000
    )
    assert not oversized, f"tracked conformal artifacts exceed 500 KB: {oversized}"
