from __future__ import annotations

import json

from aerospace_prognostics.cli import main
from aerospace_prognostics.deployment.provenance import (
    SLSA_PROVENANCE_V1,
    build_release_provenance,
    render_release_provenance_markdown,
)


def test_build_release_provenance_uses_release_bundle_subjects(tmp_path) -> None:
    bundle_json = _write_release_bundle(tmp_path)

    provenance = build_release_provenance(
        bundle_json,
        repository="example/aerospace-prognostics",
        git_sha="0123456789abcdef0123456789abcdef01234567",
        git_ref="refs/heads/main",
        workflow="CI",
        run_id="12345",
        run_attempt="1",
        actor="ci",
        generated_at_utc="2026-06-11T16:00:00+00:00",
    )
    markdown = render_release_provenance_markdown(provenance)

    assert provenance.status == "ok"
    assert provenance.problems == []
    assert provenance.statement["_type"] == "https://in-toto.io/Statement/v1"
    assert provenance.statement["predicateType"] == SLSA_PROVENANCE_V1
    assert provenance.summary["subject_count"] == 3
    assert provenance.statement["predicate"]["buildDefinition"]["externalParameters"][
        "git_sha"
    ] == "0123456789abcdef0123456789abcdef01234567"
    assert "model_artifact:artifacts/models/fd001.joblib" in {
        subject["name"] for subject in provenance.statement["subject"]
    }
    assert "# Release Provenance" in markdown
    assert "https://slsa.dev/provenance/v1" in markdown


def test_build_release_provenance_fails_without_required_source_metadata(tmp_path) -> None:
    bundle_json = _write_release_bundle(tmp_path)

    provenance = build_release_provenance(bundle_json)

    assert provenance.status == "failed"
    assert "git SHA is missing from provenance metadata" in provenance.problems
    assert "repository is missing from provenance metadata" in provenance.problems


def test_generate_release_provenance_cli_writes_json_and_markdown(tmp_path, capsys) -> None:
    bundle_json = _write_release_bundle(tmp_path)
    output_json = tmp_path / "provenance.json"
    output_markdown = tmp_path / "provenance.md"

    exit_code = main(
        [
            "generate-release-provenance",
            "--release-bundle-json",
            str(bundle_json),
            "--repository",
            "example/aerospace-prognostics",
            "--git-sha",
            "0123456789abcdef0123456789abcdef01234567",
            "--git-ref",
            "refs/tags/v0.1.0-rc1",
            "--workflow",
            "CI",
            "--run-id",
            "12345",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "status=ok" in output
    assert "subject_count=3" in output
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "aerospace-prognostics/release-provenance/v1"
    assert payload["statement"]["predicateType"] == SLSA_PROVENANCE_V1
    assert "# Release Provenance" in output_markdown.read_text(encoding="utf-8")


def _write_release_bundle(tmp_path) -> object:
    bundle_json = tmp_path / "release_bundle.json"
    bundle_json.write_text(
        json.dumps(
            {
                "schema_version": "aerospace-prognostics/cmapss-release-bundle/v1",
                "release_name": "fd001-candidate",
                "status": "ok",
                "container_image_ref": "aerospace-prognostics:ci",
                "artifact_identity": {
                    "dataset": "C-MAPSS",
                    "subset": "FD001",
                    "model_name": "hgb",
                    "artifact_id": "fd001-example",
                    "stage": "candidate",
                },
                "gates": {"promotion_report_ok": True},
                "problems": [],
                "evidence": {
                    "model_artifact": {
                        "path": "artifacts/models/fd001.joblib",
                        "sha256": "1" * 64,
                    },
                    "sbom": {
                        "path": "artifacts/sbom/cyclonedx.json",
                        "sha256": "2" * 64,
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    return bundle_json
