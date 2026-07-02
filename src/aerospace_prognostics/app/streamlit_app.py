"""Streamlit operations console for Aerospace Prognostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from aerospace_prognostics.app.api_client import (
    ApiEndpointStatus,
    ApiServiceStatus,
    check_api_service,
)
from aerospace_prognostics.app.dashboard_state import (
    load_quickstart_workspace,
)
from aerospace_prognostics.app.store import (
    database_summary,
    initialize_app_database,
    seed_quickstart_workspace,
)
from aerospace_prognostics.app.streamlit_tabs import (
    render_evidence_tab,
    render_fleet_tab,
    render_history_tab,
    render_predict_tab,
    render_registry_tab,
    render_roadmap_tab,
    render_system_tab,
)

DEFAULT_WORKSPACE = Path("artifacts") / "quickstart_cmapss"
DEFAULT_DATABASE = Path("artifacts") / "app" / "aerospace_prognostics.sqlite"
DEFAULT_EXPORT_DIR = Path("artifacts") / "app_exports"
DEFAULT_UPLOAD_DIR = Path("artifacts") / "app_uploads"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL_ENV = "AEROSPACE_PROGNOSTICS_API_BASE_URL"
API_KEY_ENV = "AEROSPACE_PROGNOSTICS_API_KEY"
READ_ONLY_ENV = "AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY"


def main() -> None:
    """Render the Streamlit application."""

    import streamlit as st

    st.set_page_config(
        page_title="Aerospace PHM Console",
        page_icon="AP",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Aerospace PHM Console")

    with st.sidebar:
        workspace_root = Path(
            st.text_input("Artifact workspace", value=str(DEFAULT_WORKSPACE))
        )
        database_path = Path(st.text_input("SQLite database", value=str(DEFAULT_DATABASE)))
        api_base_url = st.text_input(
            "API base URL",
            value=os.getenv(API_BASE_URL_ENV, DEFAULT_API_BASE_URL),
        )
        api_key = st.text_input("API key", value=os.getenv(API_KEY_ENV, ""), type="password")
        read_only = _env_flag(READ_ONLY_ENV)
        workspace = load_quickstart_workspace(workspace_root)
        if not read_only:
            initialize_app_database(database_path)
        api_status = check_api_service(api_base_url)
        st.caption(f"Workspace: {workspace.root}")
        if read_only:
            st.warning("Read-only console mode is active.")
        if workspace.missing_paths:
            st.warning("Quickstart evidence is incomplete.")
            st.code("uv run aerospace-prognostics quickstart-cmapss-demo")
            for path in workspace.missing_paths[:6]:
                st.caption(f"Missing: {path}")
        else:
            st.success("Quickstart evidence loaded.")
            if not read_only:
                seed_quickstart_workspace(database_path, workspace)
        try:
            summary = database_summary(database_path, read_only=read_only)
        except (FileNotFoundError, RuntimeError) as exc:
            st.error(f"Database is not available: {exc}")
            return
        st.caption(f"Database: {summary['database_path']}")
        st.metric("Prediction runs", summary["prediction_runs"])
        st.metric("Stored predictions", summary["predictions"])
        st.metric("API", _api_status_label(api_status))

    if not workspace.is_ready:
        st.info("Generate the quickstart evidence bundle to activate the console.")
        return

    (
        fleet_tab,
        predict_tab,
        history_tab,
        registry_tab,
        evidence_tab,
        system_tab,
        roadmap_tab,
    ) = st.tabs(
        ["Fleet", "Predict", "History", "Registry", "Evidence", "System", "Roadmap"],
    )
    with fleet_tab:
        render_fleet_tab(
            st,
            workspace,
            database_path,
            read_only,
            export_dir=DEFAULT_EXPORT_DIR,
            upload_dir=DEFAULT_UPLOAD_DIR,
            display=_display,
            filter_options=_filter_options,
            json_download_bytes=_json_download_bytes,
        )
    with predict_tab:
        render_predict_tab(st, workspace, database_path, api_status, api_key, read_only)
    with history_tab:
        render_history_tab(
            st,
            database_path,
            read_only,
            export_dir=DEFAULT_EXPORT_DIR,
            display=_display,
            filter_options=_filter_options,
            json_download_bytes=_json_download_bytes,
        )
    with registry_tab:
        render_registry_tab(
            st,
            database_path,
            read_only,
            display=_display,
            float_display=_float_display,
            percent_display=_percent_display,
            json_download_bytes=_json_download_bytes,
        )
    with evidence_tab:
        render_evidence_tab(
            st,
            workspace,
            database_path,
            read_only,
            display=_display,
        )
    with system_tab:
        render_system_tab(
            st,
            workspace,
            database_path,
            api_status,
            read_only,
            display=_display,
            endpoint_status_label=_endpoint_status_label,
        )
    with roadmap_tab:
        render_roadmap_tab(st)


def _filter_options(rows: list[dict[str, Any]], key: str) -> list[str]:
    return sorted(
        {
            str(row.get(key))
            for row in rows
            if row.get(key) is not None and str(row.get(key)).strip()
        }
    )


def _json_download_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _float_display(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _percent_display(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return str(value)


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _api_status_label(status: ApiServiceStatus) -> str:
    if status.is_ready:
        return "ready"
    if status.is_live:
        return "live"
    return "offline"


def _endpoint_status_label(status: ApiEndpointStatus) -> str:
    if status.ok:
        return "ok"
    if status.status_code is not None:
        return str(status.status_code)
    return "offline"


if __name__ == "__main__":
    main()
