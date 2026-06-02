from __future__ import annotations

import json

from aerospace_prognostics.deployment.sbom import (
    CYCLONEDX_SPEC_VERSION,
    build_uv_lock_cyclonedx_sbom,
)


def test_build_uv_lock_cyclonedx_sbom_records_runtime_and_dev_dependencies(tmp_path) -> None:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        """
version = 1

[[package]]
name = "aerospace-prognostics"
version = "0.1.0"
source = { editable = "." }
dependencies = [
  { name = "fastapi" },
]

[package.dev-dependencies]
dev = [
  { name = "pytest" },
]

[[package]]
name = "fastapi"
version = "0.115.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
  { name = "starlette" },
]
wheels = [
  { url = "https://files.pythonhosted.org/packages/fastapi.whl", hash = "sha256:abc123" },
]

[[package]]
name = "pytest"
version = "8.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "starlette"
version = "0.40.0"
source = { registry = "https://pypi.org/simple" }
""",
        encoding="utf-8",
    )

    sbom = build_uv_lock_cyclonedx_sbom(
        lockfile,
        serial_number="urn:uuid:00000000-0000-0000-0000-000000000000",
        timestamp="2026-06-02T00:00:00+00:00",
    )

    component_refs = {component["name"]: component["bom-ref"] for component in sbom["components"]}
    root_dependency = next(
        dependency
        for dependency in sbom["dependencies"]
        if dependency["ref"] == "pkg:pypi/aerospace-prognostics@0.1.0"
    )
    fastapi_dependency = next(
        dependency
        for dependency in sbom["dependencies"]
        if dependency["ref"] == "pkg:pypi/fastapi@0.115.0"
    )

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == CYCLONEDX_SPEC_VERSION
    assert sbom["metadata"]["component"] == {
        "type": "application",
        "bom-ref": "pkg:pypi/aerospace-prognostics@0.1.0",
        "name": "aerospace-prognostics",
        "version": "0.1.0",
    }
    assert component_refs == {
        "fastapi": "pkg:pypi/fastapi@0.115.0",
        "pytest": "pkg:pypi/pytest@8.0.0",
        "starlette": "pkg:pypi/starlette@0.40.0",
    }
    assert root_dependency["dependsOn"] == [
        "pkg:pypi/fastapi@0.115.0",
        "pkg:pypi/pytest@8.0.0",
    ]
    assert fastapi_dependency["dependsOn"] == ["pkg:pypi/starlette@0.40.0"]
    assert sbom["components"][0]["hashes"] == [
        {
            "alg": "SHA-256",
            "content": "abc123",
        }
    ]


def test_generate_sbom_cli_writes_json(tmp_path, capsys) -> None:
    from aerospace_prognostics.cli import main

    lockfile = tmp_path / "uv.lock"
    output_json = tmp_path / "reports" / "sbom.json"
    lockfile.write_text(
        """
version = 1

[[package]]
name = "aerospace-prognostics"
version = "0.1.0"
source = { editable = "." }
dependencies = [
  { name = "fastapi" },
]

[[package]]
name = "fastapi"
version = "0.115.0"
source = { registry = "https://pypi.org/simple" }
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "generate-sbom",
            "--lockfile",
            str(lockfile),
            "--output-json",
            str(output_json),
        ]
    )

    output = capsys.readouterr().out
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "sbom_json=" in output
    assert "component_count=1" in output
    assert payload["components"][0]["name"] == "fastapi"
