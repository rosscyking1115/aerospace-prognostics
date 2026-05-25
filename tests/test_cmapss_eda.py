from __future__ import annotations

import json

from aerospace_prognostics.analysis.cmapss_eda import (
    build_cmapss_eda_report,
    build_cmapss_sensor_summaries,
    select_informative_cmapss_sensors,
)
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
    assert len(report.operating_regime_clusters) == 1
    assert report.operating_regime_clusters[0].rows == 6


def test_cmapss_eda_report_writes_json(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    bundle = load_cmapss_subset(tmp_path, "FD001")
    report = build_cmapss_eda_report(bundle)
    output_path = tmp_path / "nested" / "eda.json"

    report.write_json(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["subset"] == "FD001"
    assert payload["sensor_summaries"][0]["sensor"] == "sensor_1"
    assert payload["operating_regime_clusters"][0]["cluster_id"] == 0


def test_select_informative_cmapss_sensors_filters_flat_channels(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    bundle = load_cmapss_subset(tmp_path, "FD001")
    frame = bundle.train.copy()
    frame["sensor_2"] = 7.0

    summaries = build_cmapss_sensor_summaries(frame)
    selected = select_informative_cmapss_sensors(
        summaries,
        min_abs_rul_correlation=0.05,
        min_abs_standardized_drift=0.2,
    )

    assert "sensor_1" in selected
    assert "sensor_2" not in selected


def test_build_cmapss_eda_report_clusters_operating_regimes(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    bundle = load_cmapss_subset(tmp_path, "FD001")
    bundle.train.loc[bundle.train["unit_number"] == 2, "op_setting_1"] = 10.0

    report = build_cmapss_eda_report(bundle, max_operating_regimes=2)

    assert len(report.operating_regime_clusters) == 2
    assert sorted(cluster.rows for cluster in report.operating_regime_clusters) == [3, 3]
    assert sum(cluster.fraction for cluster in report.operating_regime_clusters) == 1.0
