"""Focused Streamlit tab renderers for the PHM console."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from aerospace_prognostics.app.api_client import ApiServiceStatus
from aerospace_prognostics.app.dashboard_state import QuickstartWorkspace
from aerospace_prognostics.app.store import database_summary


def render_system_tab(
    st: Any,
    workspace: QuickstartWorkspace,
    database_path: Path,
    api_status: ApiServiceStatus,
    read_only: bool,
    *,
    display: Callable[[Any], str],
    endpoint_status_label: Callable[[Any], str],
) -> None:
    """Render API, database, and local quickstart state."""

    st.subheader("System Status")
    summary = database_summary(database_path, read_only=read_only)
    columns = st.columns(4)
    columns[0].metric("API Health", endpoint_status_label(api_status.health))
    columns[1].metric("API Ready", endpoint_status_label(api_status.readiness))
    columns[2].metric("Model Loaded", "yes" if api_status.model_loaded else "no")
    columns[3].metric("DB Runs", display(summary["prediction_runs"]))

    status_columns = st.columns(2)
    with status_columns[0]:
        st.subheader("API")
        st.json(
            {
                "base_url": api_status.base_url,
                "health": {
                    "ok": api_status.health.ok,
                    "status_code": api_status.health.status_code,
                    "payload": api_status.health.payload,
                    "error": api_status.health.error,
                },
                "readiness": {
                    "ok": api_status.readiness.ok,
                    "status_code": api_status.readiness.status_code,
                    "payload": api_status.readiness.payload,
                    "error": api_status.readiness.error,
                },
            }
        )
    with status_columns[1]:
        st.subheader("Local State")
        st.json(
            {
                "workspace": str(workspace.root),
                "database": summary,
                "model_artifact_path": str(workspace.model_artifact_path),
                "telemetry_csv_path": str(workspace.telemetry_csv_path),
            }
        )


def render_evidence_tab(
    st: Any,
    workspace: QuickstartWorkspace,
    database_path: Path,
    read_only: bool,
    *,
    display: Callable[[Any], str],
) -> None:
    """Render model artifact and release evidence from the quickstart workspace."""

    inspection = workspace.artifact_inspection or {}
    release_bundle = workspace.release_bundle or {}
    provenance = workspace.provenance or {}
    promotion_report = workspace.promotion_report or {}
    summary = database_summary(database_path, read_only=read_only)

    columns = st.columns(4)
    identity = inspection.get("artifact_identity")
    if not isinstance(identity, dict):
        identity = {}
    columns[0].metric("Artifact", display(identity.get("artifact_id")))
    columns[1].metric("Release", display(release_bundle.get("status")))
    columns[2].metric("Promotion", display(promotion_report.get("status")))
    columns[3].metric("DB Evidence", display(summary["release_evidence"]))

    evidence_columns = st.columns(2)
    with evidence_columns[0]:
        st.subheader("Artifact Contract")
        st.json(
            {
                "model": inspection.get("model"),
                "input_contract": inspection.get("input_contract"),
                "checks": inspection.get("checks"),
                "uncertainty": inspection.get("uncertainty"),
            }
        )
    with evidence_columns[1]:
        st.subheader("Release Evidence")
        st.json(
            {
                "release": {
                    "name": release_bundle.get("release_name"),
                    "status": release_bundle.get("status"),
                    "gates": release_bundle.get("gates"),
                },
                "provenance": provenance.get("summary"),
            }
        )


def render_roadmap_tab(st: Any) -> None:
    """Render the product roadmap tab."""

    st.subheader("Product Roadmap")
    st.markdown(
        """
        - Persist prediction runs, uploads, artifact records, and release evidence in SQLite.
        - Maintain a persisted fleet asset registry from stored prediction runs.
        - Add ranked spacecraft anomaly channels beside C-MAPSS engines in the same fleet view.
        - Use prediction-history filters for model, artifact, asset, risk, date,
          drift, and decisions.
        - Surface API, dashboard, mounted model storage, and database status in one console.
        - Promote the dashboard to a hosted product surface with authentication and audit logs.
        - Validate live spacecraft anomaly event prioritization with richer scenarios.
        """
    )
