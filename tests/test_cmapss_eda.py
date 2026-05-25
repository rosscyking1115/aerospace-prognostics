from __future__ import annotations

import json

from aerospace_prognostics.analysis.cmapss_eda import build_cmapss_eda_report
from aerospace_prognostics.data.cmapss import load_cmapss_subset
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_build_cmapss_eda_report_summarises_sensors_and_settings(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    bundle = load_cmapss_subset(tmp_path, "FD001")

    report = build_cmapss_eda_report(bundle)

    assert report.subset == "FD001"
    assert report.train_rows == 6
    assert report.train_units == 2
    assert len(report.sensor_summaries) == 21
    assert report.sensor_summaries[0].sensor == "sensor_1"
    assert report.sensor_summaries[0].drift > 0
    assert report.operating_setting_ranges["op_setting_1"] == {"min": 0.0, "max": 0.0}


def test_cmapss_eda_report_writes_json(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    bundle = load_cmapss_subset(tmp_path, "FD001")
    report = build_cmapss_eda_report(bundle)
    output_path = tmp_path / "nested" / "eda.json"

    report.write_json(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["subset"] == "FD001"
    assert payload["sensor_summaries"][0]["sensor"] == "sensor_1"
