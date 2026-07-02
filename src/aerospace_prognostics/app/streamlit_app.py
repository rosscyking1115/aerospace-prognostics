"""Streamlit operations console for Aerospace Prognostics."""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from aerospace_prognostics.app.api_client import (
    ApiEndpointStatus,
    ApiRequestError,
    ApiServiceStatus,
    check_api_service,
    predict_telemetry,
)
from aerospace_prognostics.app.dashboard_state import (
    QuickstartWorkspace,
    load_quickstart_workspace,
    predict_cmapss_telemetry,
)
from aerospace_prognostics.app.store import (
    build_fleet_asset_registry_bundle,
    build_fleet_priority_policy_validation,
    database_summary,
    export_fleet_asset_registry,
    export_fleet_priority_policy_validation,
    initialize_app_database,
    inspect_anomaly_events_csv,
    list_fleet_assets,
    record_prediction_run,
    render_fleet_priority_policy_validation_markdown,
    seed_quickstart_workspace,
    sync_fleet_assets_from_anomaly_events,
    sync_fleet_assets_from_prediction_run,
)
from aerospace_prognostics.app.streamlit_tabs import (
    render_evidence_tab,
    render_history_tab,
    render_registry_tab,
    render_roadmap_tab,
    render_system_tab,
)
from aerospace_prognostics.data.cmapss import CMAPSS_COLUMNS

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
        _render_fleet_tab(st, workspace, database_path, read_only)
    with predict_tab:
        _render_predict_tab(st, workspace, database_path, api_status, api_key, read_only)
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


def _render_fleet_tab(
    st: Any,
    workspace: QuickstartWorkspace,
    database_path: Path,
    read_only: bool,
) -> None:
    payload = workspace.dashboard_payload or {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    assets = payload.get("assets") if isinstance(payload.get("assets"), list) else []
    risk_counts = summary.get("risk_counts") if isinstance(summary.get("risk_counts"), dict) else {}

    metric_columns = st.columns(4)
    metric_columns[0].metric("Assets", _display(summary.get("asset_count")))
    metric_columns[1].metric("Attention", _display(summary.get("attention_required_count")))
    metric_columns[2].metric("Critical", _display(risk_counts.get("critical")))
    metric_columns[3].metric("Watch", _display(risk_counts.get("watch")))

    assets_frame = _assets_frame(assets)
    st.dataframe(
        assets_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "priority_rank": st.column_config.NumberColumn("Priority", width="small"),
            "asset_id": st.column_config.TextColumn("Asset"),
            "risk_level": st.column_config.TextColumn("Risk"),
            "predicted_rul": st.column_config.NumberColumn("RUL", format="%.1f"),
            "rul_lower": st.column_config.NumberColumn("Lower", format="%.1f"),
            "rul_upper": st.column_config.NumberColumn("Upper", format="%.1f"),
            "status": st.column_config.TextColumn("Status"),
            "attention": st.column_config.TextColumn("Attention"),
        },
    )

    if risk_counts:
        risk_frame = pd.DataFrame(
            [{"risk_level": key, "assets": value} for key, value in risk_counts.items()]
        )
        st.bar_chart(risk_frame, x="risk_level", y="assets", horizontal=True)

    st.subheader("Persisted Asset Registry")
    registry_columns = st.columns([1, 1, 1, 1])
    with registry_columns[0]:
        if st.button("Sync Assets", disabled=read_only):
            result = sync_fleet_assets_from_prediction_run(database_path)
            st.success(
                f"Synced {result['runs_synced']} runs and refreshed "
                f"{result['updated_assets']} assets."
            )
            st.rerun()
    with st.expander("Spacecraft Event Ingest"):
        event_ingest_ready = False
        events_path: Path | None = None
        event_preview: dict[str, Any] | None = None
        event_columns = st.columns([2, 1, 1])
        with event_columns[0]:
            anomaly_events_file = st.file_uploader(
                "Anomaly Event CSV",
                type=["csv"],
                disabled=read_only,
                key="fleet-anomaly-events-upload",
            )
        with event_columns[1]:
            anomaly_source_name = st.text_input(
                "Event Source",
                value="ops_stream",
                disabled=read_only,
            )
        with event_columns[2]:
            st.download_button(
                "Event CSV Template",
                data=_anomaly_event_template_frame().to_csv(index=False).encode("utf-8"),
                file_name="spacecraft_anomaly_events_template.csv",
                mime="text/csv",
            )
        if anomaly_events_file is not None:
            try:
                events_path = _persist_uploaded_csv(
                    anomaly_events_file,
                    upload_dir=DEFAULT_UPLOAD_DIR / "anomaly_events",
                )
                event_preview = inspect_anomaly_events_csv(events_path)
            except (OSError, ValueError, EmptyDataError, ParserError) as exc:
                st.error(f"Anomaly event CSV is not ready to sync: {exc}")
            else:
                event_ingest_ready = True
                preview_columns = st.columns(4)
                preview_columns[0].metric(
                    "Events",
                    _display(event_preview["events_processed"]),
                )
                preview_columns[1].metric(
                    "Channels",
                    _display(event_preview["channels_synced"]),
                )
                preview_columns[2].metric(
                    "Active",
                    _display(event_preview["active_events"]),
                )
                preview_columns[3].metric(
                    "Threshold Crossings",
                    _display(event_preview["threshold_crossings"]),
                )
                st.dataframe(
                    _anomaly_event_preview_frame(event_preview["latest_events"]),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "event_time_utc": st.column_config.DatetimeColumn("Event Time"),
                        "spacecraft": st.column_config.TextColumn("Spacecraft"),
                        "channel_id": st.column_config.TextColumn("Channel"),
                        "severity": st.column_config.TextColumn("Severity"),
                        "active": st.column_config.CheckboxColumn("Active"),
                        "anomaly_score": st.column_config.NumberColumn(
                            "Score",
                            format="%.3f",
                        ),
                        "threshold": st.column_config.NumberColumn(
                            "Threshold",
                            format="%.3f",
                        ),
                        "model_name": st.column_config.TextColumn("Model"),
                        "source": st.column_config.TextColumn("Source"),
                    },
                )
        with event_columns[2]:
            if st.button(
                "Sync Events",
                disabled=read_only or anomaly_events_file is None or not event_ingest_ready,
            ):
                try:
                    result = sync_fleet_assets_from_anomaly_events(
                        database_path,
                        events_csv=events_path,
                        source_name=anomaly_source_name or anomaly_events_file.name,
                    )
                except (OSError, ValueError, EmptyDataError, ParserError) as exc:
                    st.error(f"Anomaly event sync failed: {exc}")
                else:
                    st.session_state["anomaly-event-sync"] = result
                    st.success(
                        f"Processed {result['events_processed']} events and refreshed "
                        f"{result['updated_assets']} assets."
                    )
                    st.rerun()
    event_sync_result = st.session_state.get("anomaly-event-sync")
    if isinstance(event_sync_result, dict):
        st.caption(
            f"Latest event sync: {event_sync_result['channels_synced']} channels "
            f"from {event_sync_result['source_name']}"
        )
    all_registry_assets = list_fleet_assets(
        database_path,
        limit=1000,
        read_only=read_only,
    )
    if not all_registry_assets:
        st.info("No persisted fleet assets yet. Run and store a prediction first.")
        return
    filter_columns = st.columns([1, 1, 1, 1])
    with filter_columns[0]:
        selected_risks = st.multiselect(
            "Risk",
            ["critical", "watch", "nominal", "unknown"],
            default=[],
        )
    with filter_columns[1]:
        selected_domains = st.multiselect(
            "Domain",
            _filter_options(all_registry_assets, "domain"),
            default=[],
        )
    with filter_columns[2]:
        selected_statuses = st.multiselect(
            "Status",
            _filter_options(all_registry_assets, "latest_status"),
            default=[],
        )
    with filter_columns[3]:
        attention_only = st.checkbox("Attention Only", value=False)
    assets_registry = list_fleet_assets(
        database_path,
        limit=100,
        risk_levels=selected_risks,
        domains=selected_domains,
        statuses=selected_statuses,
        attention_only=attention_only,
        read_only=read_only,
    )
    assets_registry_frame = _fleet_assets_frame(assets_registry)
    with registry_columns[1]:
        st.download_button(
            "Download Assets CSV",
            data=assets_registry_frame.to_csv(index=False).encode("utf-8"),
            file_name="fleet_assets.csv",
            mime="text/csv",
        )
    with registry_columns[2]:
        registry_bundle = build_fleet_asset_registry_bundle(
            database_path,
            risk_levels=selected_risks,
            domains=selected_domains,
            statuses=selected_statuses,
            attention_only=attention_only,
            read_only=read_only,
        )
        st.download_button(
            "Download Registry JSON",
            data=_json_download_bytes(registry_bundle),
            file_name="fleet_asset_registry.json",
            mime="application/json",
        )
    with registry_columns[3]:
        if st.button("Export Registry", disabled=read_only):
            result = export_fleet_asset_registry(
                database_path,
                output_dir=DEFAULT_EXPORT_DIR,
                risk_levels=selected_risks,
                domains=selected_domains,
                statuses=selected_statuses,
                attention_only=attention_only,
            )
            st.session_state["fleet-registry-export"] = result
            st.success("Fleet registry export generated.")
    policy_validation = build_fleet_priority_policy_validation(
        database_path,
        read_only=read_only,
    )
    validation_columns = st.columns([1, 1, 1, 1])
    with validation_columns[0]:
        st.download_button(
            "Download Policy JSON",
            data=_json_download_bytes(policy_validation),
            file_name="fleet_priority_policy_validation.json",
            mime="application/json",
        )
    with validation_columns[1]:
        st.download_button(
            "Download Policy Markdown",
            data=render_fleet_priority_policy_validation_markdown(
                policy_validation
            ).encode("utf-8"),
            file_name="fleet_priority_policy_validation.md",
            mime="text/markdown",
        )
    with validation_columns[2]:
        if st.button("Export Policy", disabled=read_only):
            result = export_fleet_priority_policy_validation(
                database_path,
                output_dir=DEFAULT_EXPORT_DIR,
            )
            st.session_state["fleet-priority-policy-export"] = result
            st.success("Priority policy validation export generated.")
    with validation_columns[3]:
        st.metric("Policy Validation", str(policy_validation["overall_status"]).upper())
    priority_policy = registry_bundle.get("priority_policy", {})
    if isinstance(priority_policy, dict):
        st.caption(
            "Priority policy: "
            f"{priority_policy.get('review_queue_count', 0)} review-queue assets; "
            f"bands {priority_policy.get('band_counts', {})}; "
            f"validation {policy_validation.get('overall_status')}"
        )
    if assets_registry_frame.empty:
        st.info("No fleet assets match the selected filters.")
        return
    st.dataframe(
        assets_registry_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "last_seen_at_utc": st.column_config.DatetimeColumn("Last Seen"),
            "asset_id": st.column_config.TextColumn("Asset"),
            "asset_type": st.column_config.TextColumn("Type"),
            "domain": st.column_config.TextColumn("Domain"),
            "source_subset": st.column_config.TextColumn("Subset"),
            "latest_risk_level": st.column_config.TextColumn("Risk"),
            "priority_score": st.column_config.NumberColumn("Priority", format="%.1f"),
            "priority_band": st.column_config.TextColumn("Priority Band"),
            "priority_reasons": st.column_config.TextColumn("Priority Reasons"),
            "latest_rul_prediction": st.column_config.NumberColumn("RUL", format="%.1f"),
            "latest_rul_lower": st.column_config.NumberColumn("Lower", format="%.1f"),
            "latest_rul_upper": st.column_config.NumberColumn("Upper", format="%.1f"),
            "latest_status": st.column_config.TextColumn("Status"),
            "event_time_utc": st.column_config.DatetimeColumn("Event Time"),
            "severity": st.column_config.TextColumn("Severity"),
            "active": st.column_config.CheckboxColumn("Active"),
            "anomaly_score": st.column_config.NumberColumn("Score", format="%.3f"),
            "threshold": st.column_config.NumberColumn("Threshold", format="%.3f"),
            "f1": st.column_config.NumberColumn("F1", format="%.3f"),
            "false_alarm_rate": st.column_config.NumberColumn("FAR", format="%.3f"),
            "miss_rate": st.column_config.NumberColumn("Miss", format="%.3f"),
            "predicted_positives": st.column_config.NumberColumn("Alerts"),
            "latest_run_id": st.column_config.TextColumn("Run"),
            "attention": st.column_config.TextColumn("Attention"),
        },
    )
    export_result = st.session_state.get("fleet-registry-export")
    if isinstance(export_result, dict):
        st.json(
            {
                "registry_json": export_result["registry_json"],
                "registry_sha256": export_result["registry_sha256"],
                "assets_csv": export_result["assets_csv"],
                "assets_sha256": export_result["assets_sha256"],
                "asset_count": export_result["asset_count"],
            }
        )
    policy_export_result = st.session_state.get("fleet-priority-policy-export")
    if isinstance(policy_export_result, dict):
        st.json(
            {
                "validation_json": policy_export_result["validation_json"],
                "validation_sha256": policy_export_result["validation_sha256"],
                "validation_markdown": policy_export_result["validation_markdown"],
                "markdown_sha256": policy_export_result["markdown_sha256"],
                "overall_status": policy_export_result["overall_status"],
                "failed_checks": policy_export_result["failed_checks"],
            }
        )


def _render_predict_tab(
    st: Any,
    workspace: QuickstartWorkspace,
    database_path: Path,
    api_status: ApiServiceStatus,
    api_key: str,
    read_only: bool,
) -> None:
    st.subheader("Batch Prediction")
    uploaded = st.file_uploader("Telemetry CSV", type=["csv"], disabled=read_only)
    try:
        if uploaded is None and workspace.telemetry_csv_path.exists():
            st.caption(f"Using sample telemetry: {workspace.telemetry_csv_path}")
            telemetry = _read_telemetry_csv(workspace.telemetry_csv_path)
            source_name = str(workspace.telemetry_csv_path)
        elif uploaded is not None:
            telemetry = _read_telemetry_csv(uploaded.getvalue())
            source_name = uploaded.name
        else:
            telemetry = None
            source_name = "unknown"
    except ValueError as exc:
        st.error(str(exc))
        return

    if telemetry is None:
        st.info("No telemetry loaded.")
        return

    backend_options = ["API service", "Local artifact"]
    backend_index = 0 if api_status.is_ready else 1
    backend = st.radio("Inference backend", backend_options, index=backend_index, horizontal=True)
    st.dataframe(telemetry.head(12), use_container_width=True, hide_index=True)
    if st.button("Run Prediction", type="primary", disabled=read_only):
        if backend == "API service":
            if not api_status.is_ready:
                st.error("API service is not ready.")
                return
            try:
                prediction_document = predict_telemetry(
                    api_status.base_url,
                    telemetry=_telemetry_records(telemetry),
                    api_key=api_key or None,
                )
            except ApiRequestError as exc:
                st.error(str(exc))
                if exc.payload:
                    st.json(exc.payload)
                return
            prediction_document = _with_api_artifact_metadata(prediction_document, api_status)
        else:
            prediction_document = predict_cmapss_telemetry(
                workspace.model_artifact_path,
                telemetry,
            )
        run_id = record_prediction_run(
            database_path,
            telemetry=telemetry,
            prediction_document=prediction_document,
            model_artifact_path=workspace.model_artifact_path,
            source_name=source_name,
        )
        st.session_state["selected_prediction_run_id"] = run_id
        st.success(f"Stored {backend.lower()} prediction run: {run_id}")
        predictions = pd.DataFrame(prediction_document["predictions"])
        st.dataframe(predictions, use_container_width=True, hide_index=True)
        monitoring = prediction_document.get("monitoring", {})
        if isinstance(monitoring, dict):
            prediction_summary = monitoring.get("predictions", {})
            if isinstance(prediction_summary, dict):
                st.json(prediction_summary)


def _assets_frame(assets: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        interval = asset.get("rul_interval")
        interval = interval if isinstance(interval, dict) else {}
        reasons = asset.get("attention_reasons")
        rows.append(
            {
                "priority_rank": asset.get("priority_rank"),
                "asset_id": asset.get("asset_id"),
                "risk_level": asset.get("risk_level"),
                "predicted_rul": asset.get("predicted_rul"),
                "rul_lower": interval.get("lower"),
                "rul_upper": interval.get("upper"),
                "status": asset.get("status"),
                "attention": "; ".join(str(reason) for reason in reasons or []),
            }
        )
    return pd.DataFrame(rows)


def _fleet_assets_frame(assets: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asset in assets:
        reasons = asset.get("latest_attention_reasons")
        if not isinstance(reasons, list):
            reasons = []
        metadata = asset.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        rows.append(
            {
                "last_seen_at_utc": asset.get("last_seen_at_utc"),
                "asset_id": asset.get("asset_id"),
                "asset_type": asset.get("asset_type"),
                "domain": asset.get("domain"),
                "source_subset": asset.get("source_subset"),
                "latest_risk_level": asset.get("latest_risk_level"),
                "priority_score": asset.get("priority_score"),
                "priority_band": asset.get("priority_band"),
                "priority_reasons": "; ".join(
                    str(reason) for reason in asset.get("priority_reasons") or []
                ),
                "latest_rul_prediction": asset.get("latest_rul_prediction"),
                "latest_rul_lower": asset.get("latest_rul_lower"),
                "latest_rul_upper": asset.get("latest_rul_upper"),
                "latest_status": asset.get("latest_status"),
                "latest_run_id": asset.get("latest_run_id"),
                "event_time_utc": metadata.get("event_time_utc"),
                "severity": metadata.get("severity"),
                "active": metadata.get("active"),
                "anomaly_score": metadata.get("anomaly_score"),
                "threshold": metadata.get("threshold"),
                "f1": metadata.get("f1"),
                "false_alarm_rate": metadata.get("false_alarm_rate"),
                "miss_rate": metadata.get("miss_rate"),
                "predicted_positives": metadata.get("predicted_positives"),
                "attention": "; ".join(str(reason) for reason in reasons),
            }
        )
    return pd.DataFrame(rows)


def _anomaly_event_template_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "channel_id",
            "spacecraft",
            "event_time_utc",
            "severity",
            "active",
            "anomaly_score",
            "threshold",
            "model_name",
            "source",
            "note",
        ]
    )


def _anomaly_event_preview_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "event_time_utc",
        "spacecraft",
        "channel_id",
        "severity",
        "active",
        "anomaly_score",
        "threshold",
        "model_name",
        "source",
    ]
    rows = [{column: event.get(column) for column in columns} for event in events]
    return pd.DataFrame(rows, columns=columns)


def _persist_uploaded_csv(uploaded: Any, *, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / _safe_upload_filename(getattr(uploaded, "name", "upload.csv"))
    target.write_bytes(uploaded.getvalue())
    return target


def _safe_upload_filename(name: str) -> str:
    filename = Path(str(name)).name.replace(" ", "_")
    safe = "".join(
        character
        for character in filename
        if character.isalnum() or character in {"-", "_", "."}
    )
    return safe or "upload.csv"


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


def _telemetry_records(telemetry: pd.DataFrame) -> list[dict[str, Any]]:
    sanitized = telemetry.astype(object).where(pd.notna(telemetry), None)
    return sanitized.to_dict(orient="records")


def _read_telemetry_csv(source: bytes | Path) -> pd.DataFrame:
    try:
        telemetry = (
            pd.read_csv(BytesIO(source))
            if isinstance(source, bytes)
            else pd.read_csv(source)
        )
    except (EmptyDataError, ParserError, UnicodeDecodeError) as exc:
        raise ValueError("telemetry CSV could not be read as a valid CSV") from exc
    _validate_telemetry_frame(telemetry)
    return telemetry


def _validate_telemetry_frame(telemetry: pd.DataFrame) -> None:
    if telemetry.empty:
        raise ValueError("telemetry CSV must contain at least one row")
    missing = [column for column in CMAPSS_COLUMNS if column not in telemetry.columns]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise ValueError(f"telemetry CSV is missing required columns: {preview}{suffix}")
    identity_columns = telemetry[["unit_number", "time_in_cycles"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if identity_columns.isna().any().any():
        raise ValueError("unit_number and time_in_cycles must be numeric and non-null")
    if (identity_columns["time_in_cycles"] < 1).any():
        raise ValueError("time_in_cycles values must be positive")


def _with_api_artifact_metadata(
    prediction_document: dict[str, Any],
    api_status: ApiServiceStatus,
) -> dict[str, Any]:
    if "artifact" in prediction_document:
        return prediction_document
    model = api_status.readiness.payload.get("model")
    if not isinstance(model, dict):
        return prediction_document
    enriched = dict(prediction_document)
    enriched["artifact"] = {
        "artifact_id": model.get("artifact_id"),
        "artifact_sha256": model.get("artifact_sha256"),
        "stage": model.get("stage"),
    }
    return enriched


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
