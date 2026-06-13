from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from aerospace_prognostics.cli import main


def test_quickstart_release_evidence_demo_writes_review_artifacts(tmp_path) -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "ci_release_evidence_smoke.py"
    spec = importlib.util.spec_from_file_location("ci_release_evidence_smoke", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = tmp_path / "quickstart"

    exit_code = module.run(
        root=root,
        release_name="quickstart-fd001-demo",
        repository="local/aerospace-prognostics",
        git_sha="0" * 40,
        git_ref="refs/heads/local-quickstart",
        workflow="local-quickstart",
        run_id="local",
    )

    release_bundle = json.loads(
        (root / "release" / "fd001_release_bundle.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (root / "release" / "fd001_provenance.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert release_bundle["release_name"] == "quickstart-fd001-demo"
    assert release_bundle["status"] == "ok"
    assert "dashboard_html" in release_bundle["evidence"]
    assert provenance["summary"]["workflow"] == "local-quickstart"
    assert (root / "dashboard" / "fleet_dashboard.html").exists()
    assert (root / "models" / "fd001_inspection.json").exists()
    assert (root / "models" / "fd001_model_card.md").exists()


def test_quickstart_cmapss_demo_command_writes_review_artifacts(tmp_path) -> None:
    root = tmp_path / "cli_quickstart"

    exit_code = main(
        [
            "quickstart-cmapss-demo",
            "--output-dir",
            str(root),
            "--release-name",
            "cli-quickstart-fd001-demo",
            "--repository",
            "local/aerospace-prognostics",
            "--git-sha",
            "1" * 40,
            "--git-ref",
            "refs/heads/cli-quickstart",
            "--workflow",
            "cli-quickstart",
            "--run-id",
            "cli-local",
        ]
    )

    release_bundle = json.loads(
        (root / "release" / "fd001_release_bundle.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (root / "release" / "fd001_provenance.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert release_bundle["release_name"] == "cli-quickstart-fd001-demo"
    assert release_bundle["status"] == "ok"
    assert provenance["summary"]["workflow"] == "cli-quickstart"
    assert (root / "dashboard" / "fleet_dashboard.html").exists()
    assert (root / "models" / "fd001_inspection.json").exists()
