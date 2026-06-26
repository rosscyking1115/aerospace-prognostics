from __future__ import annotations

import json

from aerospace_prognostics.artifact_io import prepare_output_path, write_json_payload


def test_prepare_output_path_creates_parent_directories(tmp_path) -> None:
    output_path = tmp_path / "nested" / "artifact.json"

    prepared = prepare_output_path(output_path)

    assert prepared == output_path
    assert output_path.parent.is_dir()


def test_write_json_payload_writes_sorted_pretty_json(tmp_path) -> None:
    output_path = tmp_path / "nested" / "artifact.json"

    written_path = write_json_payload({"b": 2, "a": 1}, output_path)

    assert written_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert output_path.read_text(encoding="utf-8").splitlines()[1].strip() == '"a": 1,'
