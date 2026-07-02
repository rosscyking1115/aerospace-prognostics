from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aerospace_prognostics.app.release_evidence import (
    evidence_from_row,
    release_evidence_record,
)


def test_release_evidence_record_uses_file_hash_and_status(tmp_path: Path) -> None:
    evidence_path = tmp_path / "promotion_report.json"
    evidence_path.write_text('{"status":"pass"}\n', encoding="utf-8")
    payload = {"details": {"metric": 0.42}, "status": "pass"}

    record = release_evidence_record(
        artifact_id="artifact-123",
        evidence_type="promotion_report",
        path=evidence_path,
        payload=payload,
        timestamp="2026-01-02T03:04:05+00:00",
    )

    expected_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    assert (
        record["evidence_id"]
        == f"promotion_report:artifact-123:{expected_sha}"
    )
    assert record["artifact_id"] == "artifact-123"
    assert record["evidence_type"] == "promotion_report"
    assert record["source_path"] == str(evidence_path)
    assert record["status"] == "pass"
    assert record["payload_json"] == '{"details":{"metric":0.42},"status":"pass"}'
    assert record["created_at_utc"] == "2026-01-02T03:04:05+00:00"


def test_release_evidence_record_ignores_non_string_status(tmp_path: Path) -> None:
    evidence_path = tmp_path / "metrics.json"
    evidence_path.write_text("{}", encoding="utf-8")

    record = release_evidence_record(
        artifact_id="artifact-123",
        evidence_type="metrics",
        path=evidence_path,
        payload={"status": True},
        timestamp="2026-01-02T03:04:05+00:00",
    )

    assert record["status"] is None


def test_evidence_from_row_decodes_payload_json() -> None:
    row = {
        "evidence_id": "evidence-1",
        "artifact_id": "artifact-123",
        "payload_json": json.dumps({"status": "pass"}),
    }

    evidence = evidence_from_row(row)

    assert evidence == {
        "evidence_id": "evidence-1",
        "artifact_id": "artifact-123",
        "payload": {"status": "pass"},
    }


def test_evidence_from_row_handles_bad_payload_json() -> None:
    evidence = evidence_from_row({"payload_json": "not-json"})

    assert evidence["payload"] == {}
