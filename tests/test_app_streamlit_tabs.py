from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from aerospace_prognostics.app.api_client import ApiEndpointStatus, ApiServiceStatus
from aerospace_prognostics.app.streamlit_tabs import (
    render_evidence_tab,
    render_fleet_tab,
    render_history_tab,
    render_predict_tab,
    render_registry_tab,
    render_roadmap_tab,
    render_system_tab,
)


class _FakeColumn:
    def __enter__(self) -> _FakeColumn:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def __init__(self) -> None:
        self.metrics: list[tuple[str, str]] = []

    def metric(self, label: str, value: str) -> None:
        self.metrics.append((label, value))


class _FakeExpander:
    def __enter__(self) -> _FakeExpander:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class _FakeStreamlit:
    def __init__(self) -> None:
        self.subheaders: list[str] = []
        self.infos: list[str] = []
        self.successes: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.captions: list[str] = []
        self.markdowns: list[str] = []
        self.metrics: list[tuple[str, str]] = []
        self.json_payloads: list[dict[str, Any]] = []
        self.column_groups: list[list[_FakeColumn]] = []
        self.dataframes: list[tuple[Any, dict[str, Any]]] = []
        self.bar_charts: list[tuple[Any, dict[str, Any]]] = []
        self.expanders: list[str] = []
        self.downloads: list[dict[str, Any]] = []
        self.selected_options: dict[str, Any] = {}
        self.radio_choices: dict[str, Any] = {}
        self.multiselects: list[tuple[str, list[str], list[str], str | None]] = []
        self.text_inputs: dict[str, str] = {}
        self.checkboxes: dict[str, bool] = {}
        self.widget_keys: list[str] = []
        self.buttons: dict[str, bool] = {}
        self.uploads: dict[str, Any] = {}
        self.session_state: dict[str, Any] = {}
        self.rerun_count = 0
        self.column_config = SimpleNamespace(
            CheckboxColumn=lambda *args, **kwargs: ("CheckboxColumn", args, kwargs),
            DatetimeColumn=lambda *args, **kwargs: ("DatetimeColumn", args, kwargs),
            NumberColumn=lambda *args, **kwargs: ("NumberColumn", args, kwargs),
            TextColumn=lambda *args, **kwargs: ("TextColumn", args, kwargs),
        )

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def info(self, text: str) -> None:
        self.infos.append(text)

    def success(self, text: str) -> None:
        self.successes.append(text)

    def error(self, text: str) -> None:
        self.errors.append(text)

    def warning(self, text: str) -> None:
        self.warnings.append(text)

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def markdown(self, text: str) -> None:
        self.markdowns.append(text)

    def metric(self, label: str, value: str) -> None:
        self.metrics.append((label, value))

    def json(self, payload: dict[str, Any]) -> None:
        self.json_payloads.append(payload)

    def columns(self, count: int | list[int]) -> list[_FakeColumn]:
        if isinstance(count, list):
            count = len(count)
        columns = [_FakeColumn() for _ in range(count)]
        self.column_groups.append(columns)
        return columns

    def dataframe(self, frame: Any, **kwargs: Any) -> None:
        self.dataframes.append((frame, kwargs))

    def bar_chart(self, data: Any, **kwargs: Any) -> None:
        self.bar_charts.append((data, kwargs))

    def expander(self, label: str) -> _FakeExpander:
        self.expanders.append(label)
        return _FakeExpander()

    def selectbox(self, label: str, options: list[str], **kwargs: Any) -> str:
        selected = self.selected_options.get(label)
        if selected is not None:
            return str(selected)
        index = int(kwargs.get("index", 0))
        return str(options[index])

    def radio(self, label: str, options: list[str], **kwargs: Any) -> str:
        selected = self.radio_choices.get(label)
        if selected is not None:
            return str(selected)
        index = int(kwargs.get("index", 0))
        return str(options[index])

    def multiselect(
        self,
        label: str,
        options: list[str],
        default: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        values = list(default or [])
        key = kwargs.get("key")
        if key is not None:
            self.widget_keys.append(str(key))
        self.multiselects.append((label, options, values, str(key) if key else None))
        return values

    def text_input(self, label: str, value: str = "", **_kwargs: Any) -> str:
        key = _kwargs.get("key")
        if key is not None:
            self.widget_keys.append(str(key))
        return self.text_inputs.get(label, value)

    def checkbox(self, label: str, value: bool = False, **_kwargs: Any) -> bool:
        key = _kwargs.get("key")
        if key is not None:
            self.widget_keys.append(str(key))
        return self.checkboxes.get(label, value)

    def file_uploader(self, label: str, **_kwargs: Any) -> Any:
        return self.uploads.get(label)

    def button(self, label: str, **_kwargs: Any) -> bool:
        return self.buttons.get(label, False)

    def download_button(self, label: str, **kwargs: Any) -> None:
        self.downloads.append({"label": label, **kwargs})

    def rerun(self) -> None:
        self.rerun_count += 1


def test_render_roadmap_tab_shows_engineering_roadmap() -> None:
    st = _FakeStreamlit()

    render_roadmap_tab(st)

    assert st.subheaders == ["MLOps Roadmap"]
    assert "Persist prediction runs" in st.markdowns[0]
    assert "read-only" in st.markdowns[0]


def test_render_evidence_tab_surfaces_release_evidence(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    st = _FakeStreamlit()
    workspace = SimpleNamespace(
        artifact_inspection={
            "artifact_identity": {"artifact_id": "fd001-demo"},
            "model": {"model_name": "hgb_policy"},
            "input_contract": {"feature_count": 12},
            "checks": {"passed": True},
            "uncertainty": {"method": "interval"},
        },
        release_bundle={
            "release_name": "quickstart",
            "status": "passed",
            "gates": ["inspection", "smoke"],
        },
        provenance={"summary": {"workflow": "ci"}},
        promotion_report={"status": "approved"},
    )

    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.database_summary",
        lambda *_args, **_kwargs: {"release_evidence": 5},
    )

    render_evidence_tab(
        st,
        workspace,  # type: ignore[arg-type]
        tmp_path / "app.sqlite",
        read_only=True,
        display=str,
    )

    metrics = [metric for column in st.column_groups[0] for metric in column.metrics]
    assert metrics == [
        ("Artifact", "fd001-demo"),
        ("Release", "passed"),
        ("Promotion", "approved"),
        ("DB Evidence", "5"),
    ]
    assert st.subheaders == ["Artifact Contract", "Release Evidence"]
    assert st.json_payloads[0]["model"] == {"model_name": "hgb_policy"}
    assert st.json_payloads[1]["release"]["gates"] == ["inspection", "smoke"]
    assert st.json_payloads[1]["provenance"] == {"workflow": "ci"}


def test_render_predict_tab_runs_local_artifact_and_records_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    st = _FakeStreamlit()
    st.radio_choices["Inference backend"] = "Local artifact"
    st.buttons["Run Prediction"] = True
    telemetry_path = tmp_path / "telemetry.csv"
    telemetry_path.write_text("sample", encoding="utf-8")
    model_artifact_path = tmp_path / "fd001.joblib"
    workspace = SimpleNamespace(
        telemetry_csv_path=telemetry_path,
        model_artifact_path=model_artifact_path,
    )
    telemetry = pd.DataFrame(
        [
            {"unit_number": 1, "time_in_cycles": 20, "sensor_1": 0.5},
            {"unit_number": 2, "time_in_cycles": 25, "sensor_1": 0.7},
        ]
    )
    api_status = ApiServiceStatus(
        base_url="http://127.0.0.1:8000",
        health=ApiEndpointStatus(ok=True, status_code=200, payload={"status": "ok"}),
        readiness=ApiEndpointStatus(ok=False, status_code=503, payload={}),
    )

    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs._read_telemetry_csv",
        lambda source: telemetry,
    )
    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.predict_cmapss_telemetry",
        lambda artifact_path, frame: {
            "predictions": [{"unit_number": 1, "predicted_rul": 42.0}],
            "monitoring": {"predictions": {"count": len(frame)}},
            "artifact": {"artifact_id": Path(artifact_path).stem},
        },
    )

    def record_run(
        database_path: Path,
        *,
        telemetry: pd.DataFrame,
        prediction_document: dict[str, Any],
        model_artifact_path: Path,
        source_name: str,
    ) -> str:
        assert database_path == tmp_path / "app.sqlite"
        assert len(telemetry) == 2
        assert prediction_document["artifact"]["artifact_id"] == "fd001"
        assert model_artifact_path == workspace.model_artifact_path
        assert source_name == str(telemetry_path)
        return "run-local-1"

    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.record_prediction_run",
        record_run,
    )

    render_predict_tab(
        st,
        workspace,  # type: ignore[arg-type]
        tmp_path / "app.sqlite",
        api_status,
        api_key="",
        read_only=False,
    )

    assert st.subheaders == ["Batch Prediction"]
    assert st.captions == [f"Using sample telemetry: {telemetry_path}"]
    assert st.successes == ["Stored local artifact prediction run: run-local-1"]
    assert st.session_state["selected_prediction_run_id"] == "run-local-1"
    assert st.dataframes[0][1]["width"] == "stretch"
    assert st.dataframes[1][1]["width"] == "stretch"
    assert st.json_payloads == [{"count": 2}]


def test_render_fleet_tab_surfaces_registry_and_policy(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    st = _FakeStreamlit()
    database_path = tmp_path / "app.sqlite"
    workspace = SimpleNamespace(
        dashboard_payload={
            "summary": {
                "asset_count": 2,
                "attention_required_count": 1,
                "risk_counts": {"critical": 1, "watch": 1},
            },
            "assets": [
                {
                    "priority_rank": 1,
                    "asset_id": "FD001-unit-1",
                    "risk_level": "critical",
                    "predicted_rul": 12.0,
                    "rul_interval": {"lower": 9.0, "upper": 16.0},
                    "status": "attention",
                    "attention_reasons": ["low RUL"],
                }
            ],
        }
    )
    registry_assets = [
        {
            "last_seen_at_utc": "2026-01-02T00:00:00Z",
            "asset_id": "FD001-unit-1",
            "asset_type": "turbofan",
            "domain": "turbofan",
            "source_subset": "FD001",
            "latest_risk_level": "critical",
            "priority_score": 98.0,
            "priority_band": "review",
            "priority_reasons": ["low RUL"],
            "latest_rul_prediction": 12.0,
            "latest_rul_lower": 9.0,
            "latest_rul_upper": 16.0,
            "latest_status": "attention",
            "latest_run_id": "run-1",
            "metadata": {"severity": "critical", "active": True},
            "latest_attention_reasons": ["low RUL"],
        }
    ]

    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.list_fleet_assets",
        lambda *_args, **_kwargs: registry_assets,
    )
    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.build_fleet_asset_registry_bundle",
        lambda *_args, **_kwargs: {
            "asset_count": 1,
            "priority_policy": {
                "review_queue_count": 1,
                "band_counts": {"review": 1},
            },
        },
    )
    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.build_fleet_priority_policy_validation",
        lambda *_args, **_kwargs: {"overall_status": "passed", "failed_checks": []},
    )
    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs."
        "render_fleet_priority_policy_validation_markdown",
        lambda payload: f"# Policy {payload['overall_status']}",
    )

    render_fleet_tab(
        st,
        workspace,  # type: ignore[arg-type]
        database_path,
        read_only=True,
        export_dir=tmp_path / "exports",
        upload_dir=tmp_path / "uploads",
        display=str,
        filter_options=lambda rows, key: sorted({str(row[key]) for row in rows if key in row}),
        json_download_bytes=lambda payload: repr(payload).encode("utf-8"),
    )

    assert st.subheaders == ["Persisted Asset Registry"]
    assert st.expanders == ["Spacecraft Event Ingest"]
    assert len(st.bar_charts) == 1
    assert len(st.dataframes) == 2
    assert all(call_kwargs["width"] == "stretch" for _, call_kwargs in st.dataframes)
    assert [download["label"] for download in st.downloads] == [
        "Event CSV Template",
        "Download Assets CSV",
        "Download Registry JSON",
        "Download Policy JSON",
        "Download Policy Markdown",
    ]
    metrics = [
        metric
        for column_group in st.column_groups
        for column in column_group
        for metric in column.metrics
    ]
    assert ("Assets", "2") in metrics
    assert ("Attention", "1") in metrics
    assert st.metrics == [("Policy Validation", "PASSED")]
    assert st.captions == [
        "Priority policy: 1 review-queue assets; bands {'review': 1}; validation passed"
    ]
    assert {
        "fleet_registry_risk_filter",
        "fleet_registry_domain_filter",
        "fleet_registry_status_filter",
        "fleet_registry_attention_only_filter",
    }.issubset(st.widget_keys)


def test_render_registry_tab_surfaces_artifact_review(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    st = _FakeStreamlit()
    database_path = tmp_path / "app.sqlite"
    artifact_id = "fd001-demo"

    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.list_model_artifacts",
        lambda *_args, **_kwargs: [
            {
                "created_at_utc": "2026-01-01T00:00:00Z",
                "artifact_id": artifact_id,
                "stage": "release",
                "dataset": "C-MAPSS",
                "subset": "FD001",
                "model_name": "hgb_policy",
                "evidence_count": 2,
                "prediction_run_count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.load_model_artifact",
        lambda *_args, **_kwargs: {
            "artifact": {
                "artifact_id": artifact_id,
                "artifact_sha256": "abc123",
                "artifact_path": "models/fd001.joblib",
                "schema_version": "1",
                "stage": "release",
                "dataset": "C-MAPSS",
                "subset": "FD001",
                "inspection": {
                    "model": {"model_name": "hgb_policy", "dataset": "C-MAPSS"},
                    "promotion": {"stage": "release"},
                    "checks": {"passed": True},
                    "uncertainty": {"method": "interval"},
                },
            },
            "report_card": {
                "release_status": "passed",
                "promotion_status": "approved",
                "passed_gate_count": 3,
                "gate_count": 3,
                "prediction_run_count": 1,
                "interval_availability_rate": 1.0,
                "interval_count_total": 4,
                "prediction_count_total": 4,
                "mean_interval_width": 12.25,
                "missing_interval_count": 0,
                "outcome_availability_rate": 0.5,
                "outcome_interval_coverage_rate": 0.75,
                "mean_absolute_error": 9.5,
                "outcome_count_total": 2,
                "failed_gates": [],
            },
            "release_evidence": [
                {
                    "created_at_utc": "2026-01-01T00:00:00Z",
                    "evidence_type": "inspection",
                    "status": "passed",
                    "source_path": "release.json",
                }
            ],
            "prediction_runs": [
                {
                    "created_at_utc": "2026-01-02T00:00:00Z",
                    "run_id": "run-1",
                    "source_name": "telemetry.csv",
                    "prediction_count": 4,
                    "interval_availability_rate": 1.0,
                    "mean_interval_width": 12.25,
                    "outcome_count": 2,
                    "outcome_interval_coverage_rate": 0.75,
                    "mean_absolute_error": 9.5,
                    "content_sha256": "def456",
                }
            ],
        },
    )
    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.build_model_artifact_review_bundle",
        lambda *_args, **_kwargs: {"artifact_id": artifact_id, "status": "ready"},
    )

    render_registry_tab(
        st,
        database_path,
        read_only=True,
        display=str,
        float_display=lambda value: f"{float(value):.2f}",
        percent_display=lambda value: f"{float(value) * 100:.1f}%",
        json_download_bytes=lambda payload: repr(payload).encode("utf-8"),
    )

    assert st.subheaders == [
        "Model Registry",
        "Model Report Card",
        "Artifact Record",
        "Inspection",
        "Release Evidence",
        "Prediction Usage",
    ]
    assert st.downloads[0]["label"] == "Download Review Bundle"
    assert st.downloads[0]["file_name"] == "fd001-demo_review_bundle.json"
    assert st.successes == ["All recorded gates passed."]
    assert len(st.dataframes) == 3
    assert all(call_kwargs["width"] == "stretch" for _, call_kwargs in st.dataframes)
    metrics = [
        metric
        for column_group in st.column_groups
        for column in column_group
        for metric in column.metrics
    ]
    assert ("Stage", "release") in metrics
    assert ("Release", "passed") in metrics
    assert ("Interval Availability", "100.0%") in metrics
    assert ("Observed Coverage", "75.0%") in metrics
    assert st.json_payloads[0]["mean_absolute_error"] == 9.5
    assert st.json_payloads[1]["artifact_id"] == artifact_id
    assert st.json_payloads[2]["uncertainty"] == {"method": "interval"}


def test_render_history_tab_surfaces_prediction_run_evidence(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    st = _FakeStreamlit()
    database_path = tmp_path / "app.sqlite"
    run_id = "run-1"
    runs = [
        {
            "created_at_utc": "2026-01-02T00:00:00Z",
            "run_id": run_id,
            "source_name": "telemetry.csv",
            "model_name": "hgb_policy",
            "artifact_id": "fd001-demo",
            "prediction_count": 2,
            "min_predicted_rul": 10.0,
            "mean_predicted_rul": 15.0,
            "max_predicted_rul": 20.0,
            "interval_availability_rate": 1.0,
            "mean_interval_width": 8.5,
            "outcome_count": 1,
            "outcome_interval_coverage_rate": 1.0,
            "mean_absolute_error": 3.0,
            "drift_alert_count": 0,
            "decision_status": "watch",
            "audit_event_count": 1,
        }
    ]

    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.list_prediction_runs",
        lambda *_args, **_kwargs: runs,
    )
    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.load_prediction_run",
        lambda *_args, **_kwargs: {
            "run": {
                "run_id": run_id,
                "created_at_utc": "2026-01-02T00:00:00Z",
                "source_name": "telemetry.csv",
                "content_sha256": "abc123",
                "artifact_id": "fd001-demo",
                "model_artifact_path": "models/fd001.joblib",
                "row_count": 2,
                "prediction_count": 2,
                "dataset": "C-MAPSS",
                "subset": "FD001",
                "monitoring": {"drift": {"alerts": 0}},
                "decision_status": "watch",
            },
            "predictions": [
                {
                    "asset_id": "FD001-unit-1",
                    "unit_number": 1,
                    "predicted_rul": 20.0,
                    "predicted_rul_lower": 15.0,
                    "predicted_rul_upper": 25.0,
                    "interval_method": "quantile",
                    "interval_quantile_level": 0.9,
                    "actual_rul": 18.0,
                    "signed_error": 2.0,
                    "absolute_error": 2.0,
                    "interval_covered": True,
                    "outcome_source": "ops.csv",
                }
            ],
            "audit_events": [
                {
                    "created_at_utc": "2026-01-02T00:01:00Z",
                    "event_type": "operator_decision",
                    "status": "watch",
                    "actor": "operator",
                    "note": "Monitor",
                }
            ],
        },
    )
    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.build_prediction_run_evidence",
        lambda *_args, **_kwargs: {"run_id": run_id, "status": "ready"},
    )

    render_history_tab(
        st,
        database_path,
        read_only=True,
        export_dir=tmp_path / "exports",
        display=str,
        filter_options=lambda rows, key: sorted({str(row[key]) for row in rows if key in row}),
        json_download_bytes=lambda payload: repr(payload).encode("utf-8"),
    )

    assert st.subheaders == [
        "Prediction History",
        "Run Record",
        "Monitoring",
        "Audit Log",
    ]
    assert st.session_state["selected_prediction_run_id"] == run_id
    assert len(st.dataframes) == 3
    assert all(call_kwargs["width"] == "stretch" for _, call_kwargs in st.dataframes)
    assert [download["label"] for download in st.downloads] == [
        "Download Predictions",
        "Download Outcome Template",
        "Download Evidence JSON",
    ]
    metrics = [
        metric
        for column_group in st.column_groups
        for column in column_group
        for metric in column.metrics
    ]
    assert ("Rows", "2") in metrics
    assert ("Dataset", "C-MAPSS") in metrics
    assert st.json_payloads[0]["run_id"] == run_id
    assert st.json_payloads[1] == {"drift": {"alerts": 0}}
    assert {
        "history_model_filter",
        "history_artifact_filter",
        "history_risk_filter",
        "history_decision_filter",
        "history_asset_filter",
        "history_created_from_filter",
        "history_created_to_filter",
        "history_drift_only_filter",
    }.issubset(st.widget_keys)


def test_render_system_tab_surfaces_api_and_local_state(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    st = _FakeStreamlit()
    workspace = SimpleNamespace(
        root=tmp_path / "workspace",
        model_artifact_path=tmp_path / "model.joblib",
        telemetry_csv_path=tmp_path / "telemetry.csv",
    )
    api_status = ApiServiceStatus(
        base_url="http://127.0.0.1:8000",
        health=ApiEndpointStatus(ok=True, status_code=200, payload={"status": "ok"}),
        readiness=ApiEndpointStatus(
            ok=False,
            status_code=503,
            payload={},
            error="model missing",
        ),
    )

    monkeypatch.setattr(
        "aerospace_prognostics.app.streamlit_tabs.database_summary",
        lambda *_args, **_kwargs: {
            "database_path": str(tmp_path / "app.sqlite"),
            "prediction_runs": 7,
        },
    )

    render_system_tab(
        st,
        workspace,  # type: ignore[arg-type]
        tmp_path / "app.sqlite",
        api_status,
        read_only=True,
        display=str,
        endpoint_status_label=lambda status: "up" if status.ok else "down",
    )

    metrics = [metric for column in st.column_groups[0] for metric in column.metrics]
    assert metrics == [
        ("API Health", "up"),
        ("API Ready", "down"),
        ("Model Loaded", "no"),
        ("DB Runs", "7"),
    ]
    assert st.subheaders == ["System Status", "API", "Local State"]
    assert st.json_payloads[0]["base_url"] == "http://127.0.0.1:8000"
    assert st.json_payloads[1]["database"]["prediction_runs"] == 7
