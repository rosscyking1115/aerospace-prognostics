from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aerospace_prognostics.app.api_client import ApiEndpointStatus, ApiServiceStatus
from aerospace_prognostics.app.streamlit_tabs import (
    render_evidence_tab,
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


class _FakeStreamlit:
    def __init__(self) -> None:
        self.subheaders: list[str] = []
        self.infos: list[str] = []
        self.successes: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.markdowns: list[str] = []
        self.json_payloads: list[dict[str, Any]] = []
        self.column_groups: list[list[_FakeColumn]] = []
        self.dataframes: list[tuple[Any, dict[str, Any]]] = []
        self.downloads: list[dict[str, Any]] = []
        self.selected_options: dict[str, Any] = {}
        self.column_config = SimpleNamespace(
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

    def markdown(self, text: str) -> None:
        self.markdowns.append(text)

    def json(self, payload: dict[str, Any]) -> None:
        self.json_payloads.append(payload)

    def columns(self, count: int) -> list[_FakeColumn]:
        columns = [_FakeColumn() for _ in range(count)]
        self.column_groups.append(columns)
        return columns

    def dataframe(self, frame: Any, **kwargs: Any) -> None:
        self.dataframes.append((frame, kwargs))

    def selectbox(self, label: str, options: list[str]) -> str:
        return str(self.selected_options.get(label, options[0]))

    def download_button(self, label: str, **kwargs: Any) -> None:
        self.downloads.append({"label": label, **kwargs})


def test_render_roadmap_tab_keeps_product_direction_visible() -> None:
    st = _FakeStreamlit()

    render_roadmap_tab(st)

    assert st.subheaders == ["Product Roadmap"]
    assert "Persist prediction runs" in st.markdowns[0]
    assert "hosted product surface" in st.markdowns[0]


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
