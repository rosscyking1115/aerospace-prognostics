from __future__ import annotations

from pathlib import Path


def test_demo_dockerfile_bakes_seeded_read_only_console() -> None:
    dockerfile = _repo_root().joinpath("Dockerfile.demo").read_text(encoding="utf-8")

    assert "AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true" in dockerfile
    assert "HOME=/tmp" in dockerfile
    assert "XDG_CACHE_HOME=/tmp/.cache" in dockerfile
    assert "uv run --no-sync aerospace-prognostics quickstart-cmapss-demo" in dockerfile
    assert "uv run --no-sync aerospace-prognostics app-init-db" in dockerfile
    assert "--quickstart-dir artifacts/quickstart_cmapss" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert "streamlit" in dockerfile
    assert "--server.fileWatcherType=none" in dockerfile
    assert "/_stcore/health" in dockerfile


def test_hosted_demo_docs_describe_the_read_only_image() -> None:
    docs = _repo_root().joinpath("docs", "hosted_demo.md").read_text(encoding="utf-8")

    assert "Dockerfile.demo" in docs
    assert "docker build -f Dockerfile.demo" in docs
    assert "--read-only" in docs
    assert "--tmpfs /tmp:rw,nosuid,nodev,noexec,size=128m" in docs
    assert "AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true" in docs
    assert "AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN" in docs
    assert "run evidence JSON" in docs
    assert "model-review bundles" in docs
    assert "fleet registry JSON/CSV" in docs
    assert "/_stcore/health" in docs


def test_hosted_demo_docs_do_not_imply_a_live_deployment() -> None:
    """The page must not read as though a hosted instance exists.

    A private Render service ran this console until July 2026 and was retired
    rather than repaired. Documentation that still promises a URL is a stale
    claim, and stale claims are what this repository's audit exists to catch —
    so the retraction is pinned here rather than trusted to stay written.
    """

    docs = _repo_root().joinpath("docs", "hosted_demo.md").read_text(encoding="utf-8")

    assert "There is no hosted instance of this console" in docs
    # The retired platform's blueprint is gone; nothing may point at it.
    assert "render.yaml" not in docs
    assert not _repo_root().joinpath("render.yaml").exists()


def test_demo_image_contract_script_validates_seeded_state() -> None:
    script = _repo_root().joinpath("scripts", "ci_demo_image_contract.py").read_text(
        encoding="utf-8"
    )

    assert "load_quickstart_workspace" in script
    assert "database_summary" in script
    assert "read_only=True" in script
    assert '"model_artifacts": 1' in script
    assert '"release_evidence": 5' in script
    assert "expected {key}>={expected}" in script
    assert "artifact evidence_count>=5" in script
    assert "demo_image_contract=ok" in script


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
