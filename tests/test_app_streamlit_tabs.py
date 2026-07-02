from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aerospace_prognostics.app.api_client import ApiEndpointStatus, ApiServiceStatus
from aerospace_prognostics.app.streamlit_tabs import (
    render_evidence_tab,
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
        self.markdowns: list[str] = []
        self.json_payloads: list[dict[str, Any]] = []
        self.column_groups: list[list[_FakeColumn]] = []

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def markdown(self, text: str) -> None:
        self.markdowns.append(text)

    def json(self, payload: dict[str, Any]) -> None:
        self.json_payloads.append(payload)

    def columns(self, count: int) -> list[_FakeColumn]:
        columns = [_FakeColumn() for _ in range(count)]
        self.column_groups.append(columns)
        return columns


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
