from __future__ import annotations

from pathlib import Path


def test_demo_dockerfile_bakes_seeded_read_only_console() -> None:
    dockerfile = _repo_root().joinpath("Dockerfile.demo").read_text(encoding="utf-8")

    assert "AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true" in dockerfile
    assert "uv run --no-sync aerospace-prognostics quickstart-cmapss-demo" in dockerfile
    assert "uv run --no-sync aerospace-prognostics app-init-db" in dockerfile
    assert "--quickstart-dir artifacts/quickstart_cmapss" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert "streamlit" in dockerfile
    assert "/_stcore/health" in dockerfile


def test_hosted_demo_docs_describe_private_read_only_deployment() -> None:
    docs = _repo_root().joinpath("docs", "hosted_demo.md").read_text(encoding="utf-8")

    assert "Dockerfile.demo" in docs
    assert "docker build -f Dockerfile.demo" in docs
    assert "AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY=true" in docs
    assert "Keep the GitHub repository private" in docs
    assert "/_stcore/health" in docs


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
