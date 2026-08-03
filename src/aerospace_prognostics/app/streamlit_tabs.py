"""Focused Streamlit tab renderers for the PHM console."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from aerospace_prognostics.app.api_client import (
    ApiRequestError,
    ApiServiceStatus,
    predict_telemetry,
)
from aerospace_prognostics.app.dashboard_state import (
    QuickstartWorkspace,
    predict_cmapss_telemetry,
)
from aerospace_prognostics.app.store import (
    build_fleet_asset_registry_bundle,
    build_fleet_priority_policy_validation,
    build_model_artifact_review_bundle,
    build_prediction_run_evidence,
    database_summary,
    export_fleet_asset_registry,
    export_fleet_priority_policy_validation,
    export_prediction_run_evidence,
    inspect_anomaly_events_csv,
    list_fleet_assets,
    list_model_artifacts,
    list_prediction_runs,
    load_model_artifact,
    load_prediction_run,
    record_prediction_outcomes,
    record_prediction_run,
    record_prediction_run_event,
    render_fleet_priority_policy_validation_markdown,
    sync_fleet_assets_from_anomaly_events,
    sync_fleet_assets_from_prediction_run,
)
from aerospace_prognostics.data.cmapss import CMAPSS_COLUMNS


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


def render_fleet_tab(
    st: Any,
    workspace: QuickstartWorkspace,
    database_path: Path,
    read_only: bool,
    *,
    export_dir: Path,
    upload_dir: Path,
    display: Callable[[Any], str],
    filter_options: Callable[[list[dict[str, Any]], str], list[str]],
    json_download_bytes: Callable[[Any], bytes],
) -> None:
    """Render the fleet priority, registry, and event-ingest operations surface."""
    payload = workspace.dashboard_payload or {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    assets = payload.get("assets") if isinstance(payload.get("assets"), list) else []
    risk_counts = summary.get("risk_counts") if isinstance(summary.get("risk_counts"), dict) else {}

    metric_columns = st.columns(4)
    metric_columns[0].metric("Assets", display(summary.get("asset_count")))
    metric_columns[1].metric("Attention", display(summary.get("attention_required_count")))
    metric_columns[2].metric("Critical", display(risk_counts.get("critical")))
    metric_columns[3].metric("Watch", display(risk_counts.get("watch")))

    assets_frame = _assets_frame(assets)
    st.dataframe(
        assets_frame,
        width="stretch",
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
                    upload_dir=upload_dir / "anomaly_events",
                )
                event_preview = inspect_anomaly_events_csv(events_path)
            except (OSError, ValueError, EmptyDataError, ParserError) as exc:
                st.error(f"Anomaly event CSV is not ready to sync: {exc}")
            else:
                event_ingest_ready = True
                preview_columns = st.columns(4)
                preview_columns[0].metric(
                    "Events",
                    display(event_preview["events_processed"]),
                )
                preview_columns[1].metric(
                    "Channels",
                    display(event_preview["channels_synced"]),
                )
                preview_columns[2].metric(
                    "Active",
                    display(event_preview["active_events"]),
                )
                preview_columns[3].metric(
                    "Threshold Crossings",
                    display(event_preview["threshold_crossings"]),
                )
                st.dataframe(
                    _anomaly_event_preview_frame(event_preview["latest_events"]),
                    width="stretch",
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
            key="fleet_registry_risk_filter",
        )
    with filter_columns[1]:
        selected_domains = st.multiselect(
            "Domain",
            filter_options(all_registry_assets, "domain"),
            default=[],
            key="fleet_registry_domain_filter",
        )
    with filter_columns[2]:
        selected_statuses = st.multiselect(
            "Status",
            filter_options(all_registry_assets, "latest_status"),
            default=[],
            key="fleet_registry_status_filter",
        )
    with filter_columns[3]:
        attention_only = st.checkbox(
            "Attention Only",
            value=False,
            key="fleet_registry_attention_only_filter",
        )
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
            data=json_download_bytes(registry_bundle),
            file_name="fleet_asset_registry.json",
            mime="application/json",
        )
    with registry_columns[3]:
        if st.button("Export Registry", disabled=read_only):
            result = export_fleet_asset_registry(
                database_path,
                output_dir=export_dir,
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
            data=json_download_bytes(policy_validation),
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
                output_dir=export_dir,
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
        width="stretch",
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


def render_predict_tab(
    st: Any,
    workspace: QuickstartWorkspace,
    database_path: Path,
    api_status: ApiServiceStatus,
    api_key: str,
    read_only: bool,
) -> None:
    """Render batch prediction upload, backend selection, and run persistence."""
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
    st.dataframe(telemetry.head(12), width="stretch", hide_index=True)
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
        st.dataframe(predictions, width="stretch", hide_index=True)
        monitoring = prediction_document.get("monitoring", {})
        if isinstance(monitoring, dict):
            prediction_summary = monitoring.get("predictions", {})
            if isinstance(prediction_summary, dict):
                st.json(prediction_summary)


def render_history_tab(
    st: Any,
    database_path: Path,
    read_only: bool,
    *,
    export_dir: Path,
    display: Callable[[Any], str],
    filter_options: Callable[[list[dict[str, Any]], str], list[str]],
    json_download_bytes: Callable[[Any], bytes],
) -> None:
    """Render persisted prediction runs, outcomes, decisions, and evidence exports."""
    st.subheader("Prediction History")
    all_runs = list_prediction_runs(database_path, limit=1000, read_only=read_only)
    if not all_runs:
        st.info("No prediction runs stored yet.")
        return

    filter_columns = st.columns([1, 1, 1, 1])
    with filter_columns[0]:
        selected_models = st.multiselect(
            "Model",
            filter_options(all_runs, "model_name"),
            default=[],
            key="history_model_filter",
        )
    with filter_columns[1]:
        selected_artifacts = st.multiselect(
            "Artifact",
            filter_options(all_runs, "artifact_id"),
            default=[],
            key="history_artifact_filter",
        )
    with filter_columns[2]:
        selected_risks = st.multiselect(
            "Risk",
            ["critical", "watch", "nominal", "unknown"],
            default=[],
            key="history_risk_filter",
        )
    with filter_columns[3]:
        selected_decisions = st.multiselect(
            "Decision",
            filter_options(all_runs, "decision_status"),
            default=[],
            key="history_decision_filter",
        )

    advanced_columns = st.columns([1, 1, 1, 1])
    with advanced_columns[0]:
        asset_filter_text = st.text_input(
            "Asset",
            value="",
            key="history_asset_filter",
        )
    with advanced_columns[1]:
        start_date = st.text_input(
            "Created From",
            value="",
            key="history_created_from_filter",
        )
    with advanced_columns[2]:
        end_date = st.text_input(
            "Created To",
            value="",
            key="history_created_to_filter",
        )
    with advanced_columns[3]:
        drift_only = st.checkbox(
            "Drift Only",
            value=False,
            key="history_drift_only_filter",
        )

    selected_assets = _csv_filter_values(asset_filter_text)
    runs = list_prediction_runs(
        database_path,
        limit=100,
        model_names=selected_models,
        artifact_ids=selected_artifacts,
        asset_ids=selected_assets,
        risk_levels=selected_risks,
        decision_statuses=selected_decisions,
        start_created_at_utc=_history_date_bound(start_date, end_of_day=False),
        end_created_at_utc=_history_date_bound(end_date, end_of_day=True),
        drift_only=drift_only,
        read_only=read_only,
    )
    if not runs:
        st.info("No prediction runs match the selected filters.")
        return

    runs_frame = _prediction_runs_frame(runs)
    st.dataframe(
        runs_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "created_at_utc": st.column_config.DatetimeColumn("Created"),
            "run_id": st.column_config.TextColumn("Run"),
            "source_name": st.column_config.TextColumn("Telemetry"),
            "model_name": st.column_config.TextColumn("Model"),
            "prediction_count": st.column_config.NumberColumn("Predictions", width="small"),
            "min_predicted_rul": st.column_config.NumberColumn("Min RUL", format="%.1f"),
            "mean_predicted_rul": st.column_config.NumberColumn("Mean RUL", format="%.1f"),
            "max_predicted_rul": st.column_config.NumberColumn("Max RUL", format="%.1f"),
            "interval_availability_rate": st.column_config.NumberColumn(
                "Intervals",
                format="%.0f%%",
            ),
            "mean_interval_width": st.column_config.NumberColumn(
                "Mean Width",
                format="%.1f",
            ),
            "outcome_count": st.column_config.NumberColumn("Outcomes", width="small"),
            "mean_absolute_error": st.column_config.NumberColumn("MAE", format="%.1f"),
            "outcome_interval_coverage_rate": st.column_config.NumberColumn(
                "Coverage",
                format="%.0f%%",
            ),
            "drift_alert_count": st.column_config.NumberColumn("Drift Alerts", width="small"),
            "decision_status": st.column_config.TextColumn("Decision"),
            "audit_event_count": st.column_config.NumberColumn("Events", width="small"),
        },
    )

    run_ids = [str(run["run_id"]) for run in runs]
    selected_default = st.session_state.get("selected_prediction_run_id")
    selected_index = run_ids.index(selected_default) if selected_default in run_ids else 0
    selected_run_id = st.selectbox("Run", run_ids, index=selected_index)
    st.session_state["selected_prediction_run_id"] = selected_run_id
    selected = load_prediction_run(database_path, selected_run_id, read_only=read_only)
    if selected is None:
        st.warning("Selected run is no longer available.")
        return

    run = selected["run"]
    predictions_frame = pd.DataFrame(selected["predictions"])
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", display(run.get("row_count")))
    metric_columns[1].metric("Predictions", display(run.get("prediction_count")))
    metric_columns[2].metric("Dataset", display(run.get("dataset")))
    metric_columns[3].metric("Subset", display(run.get("subset")))

    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.subheader("Run Record")
        st.json(
            {
                "run_id": run.get("run_id"),
                "created_at_utc": run.get("created_at_utc"),
                "source_name": run.get("source_name"),
                "content_sha256": run.get("content_sha256"),
                "artifact_id": run.get("artifact_id"),
                "model_artifact_path": run.get("model_artifact_path"),
            }
        )
    with detail_columns[1]:
        st.subheader("Monitoring")
        st.json(run.get("monitoring", {}))

    outcome_file = st.file_uploader(
        "Outcome CSV",
        type=["csv"],
        key=f"outcomes-{selected_run_id}",
        disabled=read_only,
    )
    outcome_columns = st.columns([2, 1])
    with outcome_columns[0]:
        outcome_source = st.text_input(
            "Outcome Source",
            value=outcome_file.name if outcome_file is not None else "outcomes.csv",
            disabled=read_only,
        )
    with outcome_columns[1]:
        outcome_actor = st.text_input("Outcome Actor", value="operator", disabled=read_only)
    if st.button("Attach Outcomes", disabled=read_only or outcome_file is None):
        if outcome_file is None:
            st.warning("No outcome CSV loaded.")
        else:
            try:
                outcome_frame = _read_outcome_csv(outcome_file.getvalue())
                result = record_prediction_outcomes(
                    database_path,
                    run_id=selected_run_id,
                    outcomes=outcome_frame,
                    source_name=outcome_source.strip() or outcome_file.name,
                    actor=outcome_actor.strip() or "operator",
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(f"Attached {result['outcome_count']} outcomes.")
                st.rerun()

    decision_columns = st.columns([1, 1, 2])
    with decision_columns[0]:
        decision_status = st.selectbox(
            "Decision",
            ["review_required", "accepted", "watch", "escalated", "rejected"],
            index=_decision_status_index(run.get("decision_status")),
            disabled=read_only,
        )
    with decision_columns[1]:
        decision_actor = st.text_input("Actor", value="operator", disabled=read_only)
    with decision_columns[2]:
        decision_note = st.text_input(
            "Note",
            value=str(run.get("decision_note") or ""),
            disabled=read_only,
        )
    if st.button("Record Decision", disabled=read_only):
        record_prediction_run_event(
            database_path,
            run_id=selected_run_id,
            event_type="operator_decision",
            status=decision_status,
            actor=decision_actor.strip() or "operator",
            note=decision_note.strip() or None,
            payload={"source": "streamlit_history_tab"},
        )
        st.success("Decision recorded.")
        st.rerun()

    audit_events = selected.get("audit_events", [])
    st.subheader("Audit Log")
    st.dataframe(
        _audit_events_frame(audit_events),
        width="stretch",
        hide_index=True,
        column_config={
            "created_at_utc": st.column_config.DatetimeColumn("Created"),
            "event_type": st.column_config.TextColumn("Event"),
            "status": st.column_config.TextColumn("Status"),
            "actor": st.column_config.TextColumn("Actor"),
            "note": st.column_config.TextColumn("Note"),
        },
    )

    st.dataframe(
        predictions_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "asset_id": st.column_config.TextColumn("Asset"),
            "unit_number": st.column_config.NumberColumn("Unit", width="small"),
            "predicted_rul": st.column_config.NumberColumn("RUL", format="%.1f"),
            "predicted_rul_lower": st.column_config.NumberColumn("Lower", format="%.1f"),
            "predicted_rul_upper": st.column_config.NumberColumn("Upper", format="%.1f"),
            "interval_method": st.column_config.TextColumn("Interval"),
            "interval_confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "actual_rul": st.column_config.NumberColumn("Actual", format="%.1f"),
            "signed_error": st.column_config.NumberColumn("Error", format="%.1f"),
            "absolute_error": st.column_config.NumberColumn("Abs Error", format="%.1f"),
            "interval_covered": st.column_config.CheckboxColumn("Covered"),
            "outcome_source": st.column_config.TextColumn("Outcome Source"),
        },
    )
    export_columns = st.columns([1, 1, 1, 1])
    with export_columns[0]:
        st.download_button(
            "Download Predictions",
            data=predictions_frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_run_id}_predictions.csv",
            mime="text/csv",
        )
    with export_columns[1]:
        outcome_template = _outcome_template_frame(predictions_frame)
        st.download_button(
            "Download Outcome Template",
            data=outcome_template.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_run_id}_outcomes_template.csv",
            mime="text/csv",
        )
    with export_columns[2]:
        try:
            evidence_payload = build_prediction_run_evidence(
                database_path,
                run_id=selected_run_id,
                read_only=read_only,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.download_button(
                "Download Evidence JSON",
                data=json_download_bytes(evidence_payload),
                file_name=f"{selected_run_id}_evidence.json",
                mime="application/json",
            )
    with export_columns[3]:
        if st.button("Export Run Evidence", disabled=read_only):
            try:
                result = export_prediction_run_evidence(
                    database_path,
                    run_id=selected_run_id,
                    output_dir=export_dir,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state[f"evidence-export-{selected_run_id}"] = result
                st.success("Evidence export generated.")

    export_result = st.session_state.get(f"evidence-export-{selected_run_id}")
    if isinstance(export_result, dict):
        evidence_path = Path(str(export_result["evidence_json"]))
        st.json(
            {
                "evidence_json": export_result["evidence_json"],
                "evidence_sha256": export_result["evidence_sha256"],
                "predictions_csv": export_result["predictions_csv"],
                "predictions_sha256": export_result["predictions_sha256"],
            }
        )
        st.download_button(
            "Download Exported Evidence JSON",
            data=evidence_path.read_bytes(),
            file_name=evidence_path.name,
            mime="application/json",
        )


def render_registry_tab(
    st: Any,
    database_path: Path,
    read_only: bool,
    *,
    display: Callable[[Any], str],
    float_display: Callable[[Any], str],
    percent_display: Callable[[Any], str],
    json_download_bytes: Callable[[Any], bytes],
) -> None:
    """Render the model registry and release evidence usage view."""
    st.subheader("Model Registry")
    artifacts = list_model_artifacts(database_path, limit=100, read_only=read_only)
    if not artifacts:
        st.info("No model artifacts stored yet.")
        return

    artifacts_frame = _model_artifacts_frame(artifacts)
    st.dataframe(
        artifacts_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "created_at_utc": st.column_config.DatetimeColumn("Created"),
            "artifact_id": st.column_config.TextColumn("Artifact"),
            "stage": st.column_config.TextColumn("Stage"),
            "dataset": st.column_config.TextColumn("Dataset"),
            "subset": st.column_config.TextColumn("Subset"),
            "model_name": st.column_config.TextColumn("Model"),
            "evidence_count": st.column_config.NumberColumn("Evidence", width="small"),
            "prediction_run_count": st.column_config.NumberColumn("Runs", width="small"),
            "latest_prediction_at": st.column_config.DatetimeColumn("Latest Prediction"),
        },
    )

    artifact_ids = [str(artifact["artifact_id"]) for artifact in artifacts]
    selected_artifact_id = st.selectbox("Artifact", artifact_ids)
    selected = load_model_artifact(
        database_path,
        selected_artifact_id,
        read_only=read_only,
    )
    if selected is None:
        st.warning("Selected artifact is no longer available.")
        return

    artifact = selected["artifact"]
    report_card = selected.get("report_card")
    report_card = report_card if isinstance(report_card, dict) else {}
    inspection = artifact.get("inspection")
    inspection = inspection if isinstance(inspection, dict) else {}
    model = inspection.get("model")
    model = model if isinstance(model, dict) else {}
    promotion = inspection.get("promotion")
    promotion = promotion if isinstance(promotion, dict) else {}

    metric_columns = st.columns(4)
    metric_columns[0].metric("Stage", display(artifact.get("stage") or promotion.get("stage")))
    metric_columns[1].metric("Dataset", display(artifact.get("dataset") or model.get("dataset")))
    metric_columns[2].metric("Subset", display(artifact.get("subset") or model.get("subset")))
    metric_columns[3].metric("Evidence", display(len(selected["release_evidence"])))

    try:
        review_bundle = build_model_artifact_review_bundle(
            database_path,
            artifact_id=selected_artifact_id,
            read_only=read_only,
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.download_button(
            "Download Review Bundle",
            data=json_download_bytes(review_bundle),
            file_name=f"{selected_artifact_id}_review_bundle.json",
            mime="application/json",
        )

    report_columns = st.columns(4)
    report_columns[0].metric("Release", display(report_card.get("release_status")))
    report_columns[1].metric("Promotion", display(report_card.get("promotion_status")))
    report_columns[2].metric(
        "Gates",
        f"{display(report_card.get('passed_gate_count'))}/{display(report_card.get('gate_count'))}",
    )
    report_columns[3].metric("Prediction Runs", display(report_card.get("prediction_run_count")))
    interval_columns = st.columns(4)
    interval_columns[0].metric(
        "Interval Availability",
        percent_display(report_card.get("interval_availability_rate")),
    )
    interval_columns[1].metric(
        "Interval Rows",
        f"{display(report_card.get('interval_count_total'))}/"
        f"{display(report_card.get('prediction_count_total'))}",
    )
    interval_columns[2].metric(
        "Mean Width",
        float_display(report_card.get("mean_interval_width")),
    )
    interval_columns[3].metric(
        "Missing Intervals",
        display(report_card.get("missing_interval_count")),
    )
    outcome_columns = st.columns(4)
    outcome_columns[0].metric(
        "Outcome Availability",
        percent_display(report_card.get("outcome_availability_rate")),
    )
    outcome_columns[1].metric(
        "Observed Coverage",
        percent_display(report_card.get("outcome_interval_coverage_rate")),
    )
    outcome_columns[2].metric(
        "Observed MAE",
        float_display(report_card.get("mean_absolute_error")),
    )
    outcome_columns[3].metric(
        "Outcome Rows",
        f"{display(report_card.get('outcome_count_total'))}/"
        f"{display(report_card.get('prediction_count_total'))}",
    )

    st.subheader("Model Report Card")
    card_columns = st.columns(2)
    with card_columns[0]:
        st.json(
            {
                "p95_latency_ms": report_card.get("p95_latency_ms"),
                "max_p95_latency_ms": report_card.get("max_p95_latency_ms"),
                "interval_method": report_card.get("interval_method"),
                "interval_confidence": report_card.get("interval_confidence"),
                "interval_diagnostic_kind": report_card.get("interval_diagnostic_kind"),
                "prediction_count_total": report_card.get("prediction_count_total"),
                "interval_count_total": report_card.get("interval_count_total"),
                "missing_interval_count": report_card.get("missing_interval_count"),
                "interval_availability_rate": report_card.get("interval_availability_rate"),
                "interval_complete": report_card.get("interval_complete"),
                "mean_interval_width": report_card.get("mean_interval_width"),
                "max_interval_width": report_card.get("max_interval_width"),
                "outcome_diagnostic_kind": report_card.get("outcome_diagnostic_kind"),
                "outcome_count_total": report_card.get("outcome_count_total"),
                "outcome_availability_rate": report_card.get("outcome_availability_rate"),
                "mean_absolute_error": report_card.get("mean_absolute_error"),
                "mean_signed_error": report_card.get("mean_signed_error"),
                "interval_outcome_count_total": report_card.get(
                    "interval_outcome_count_total"
                ),
                "interval_covered_count_total": report_card.get(
                    "interval_covered_count_total"
                ),
                "outcome_interval_coverage_rate": report_card.get(
                    "outcome_interval_coverage_rate"
                ),
                "provenance_workflow": report_card.get("provenance_workflow"),
                "latest_prediction_at": report_card.get("latest_prediction_at"),
            }
        )
    with card_columns[1]:
        failed_gates = report_card.get("failed_gates")
        failed_gates = failed_gates if isinstance(failed_gates, list) else []
        if failed_gates:
            st.dataframe(
                _failed_gates_frame(failed_gates),
                width="stretch",
                hide_index=True,
                column_config={"gate": st.column_config.TextColumn("Failed Gate")},
            )
        else:
            st.success("All recorded gates passed.")

    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.subheader("Artifact Record")
        st.json(
            {
                "artifact_id": artifact.get("artifact_id"),
                "artifact_sha256": artifact.get("artifact_sha256"),
                "artifact_path": artifact.get("artifact_path"),
                "schema_version": artifact.get("schema_version"),
                "created_at_utc": artifact.get("created_at_utc"),
            }
        )
    with detail_columns[1]:
        st.subheader("Inspection")
        st.json(
            {
                "model": model,
                "promotion": promotion,
                "checks": inspection.get("checks"),
                "uncertainty": inspection.get("uncertainty"),
            }
        )

    st.subheader("Release Evidence")
    st.dataframe(
        _release_evidence_frame(selected["release_evidence"]),
        width="stretch",
        hide_index=True,
        column_config={
            "created_at_utc": st.column_config.DatetimeColumn("Created"),
            "evidence_type": st.column_config.TextColumn("Type"),
            "status": st.column_config.TextColumn("Status"),
            "source_path": st.column_config.TextColumn("Source"),
        },
    )

    st.subheader("Prediction Usage")
    usage_frame = _artifact_prediction_runs_frame(selected["prediction_runs"])
    if usage_frame.empty:
        st.info("No prediction runs recorded for this artifact.")
    else:
        st.dataframe(
            usage_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "created_at_utc": st.column_config.DatetimeColumn("Created"),
                "run_id": st.column_config.TextColumn("Run"),
                "source_name": st.column_config.TextColumn("Telemetry"),
                "prediction_count": st.column_config.NumberColumn("Predictions", width="small"),
                "interval_availability_rate": st.column_config.NumberColumn(
                    "Intervals",
                    format="%.0f%%",
                ),
                "mean_interval_width": st.column_config.NumberColumn(
                    "Mean Width",
                    format="%.1f",
                ),
                "outcome_count": st.column_config.NumberColumn("Outcomes", width="small"),
                "outcome_interval_coverage_rate": st.column_config.NumberColumn(
                    "Coverage",
                    format="%.0f%%",
                ),
                "mean_absolute_error": st.column_config.NumberColumn("MAE", format="%.1f"),
                "content_sha256": st.column_config.TextColumn("Input SHA-256"),
            },
        )


def render_roadmap_tab(st: Any) -> None:
    """Render the MLOps engineering roadmap tab."""
    st.subheader("MLOps Roadmap")
    st.markdown(
        """
        - Persist prediction runs, uploads, artifact records, and release evidence in SQLite.
        - Maintain a persisted fleet asset registry from stored prediction runs.
        - Add ranked spacecraft anomaly channels beside C-MAPSS engines in the same fleet view.
        - Use prediction-history filters for model, artifact, asset, risk, date,
          drift, and decisions.
        - Surface API, dashboard, mounted model storage, and database status in one console.
        - Keep the console read-only when hosted so evidence stays inspectable but immutable.
        - Validate live spacecraft anomaly event prioritization with richer scenarios.
        """
    )


def _model_artifacts_frame(artifacts: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "created_at_utc",
        "artifact_id",
        "stage",
        "dataset",
        "subset",
        "model_name",
        "schema_version",
        "evidence_count",
        "prediction_run_count",
        "latest_prediction_at",
    ]
    return pd.DataFrame(artifacts).reindex(columns=columns)


def _release_evidence_frame(evidence: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["created_at_utc", "evidence_type", "status", "source_path"]
    return pd.DataFrame(evidence).reindex(columns=columns)


def _artifact_prediction_runs_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "created_at_utc",
        "run_id",
        "source_name",
        "prediction_count",
        "interval_availability_rate",
        "mean_interval_width",
        "outcome_count",
        "outcome_interval_coverage_rate",
        "mean_absolute_error",
        "content_sha256",
    ]
    frame = pd.DataFrame(runs).reindex(columns=columns)
    return _percent_frame_columns(
        frame,
        ["interval_availability_rate", "outcome_interval_coverage_rate"],
    )


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


def _prediction_runs_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "created_at_utc",
        "run_id",
        "source_name",
        "model_name",
        "prediction_count",
        "min_predicted_rul",
        "mean_predicted_rul",
        "max_predicted_rul",
        "interval_availability_rate",
        "mean_interval_width",
        "outcome_count",
        "outcome_interval_coverage_rate",
        "mean_absolute_error",
        "drift_alert_count",
        "decision_status",
        "audit_event_count",
    ]
    frame = pd.DataFrame(runs).reindex(columns=columns)
    return _percent_frame_columns(
        frame,
        ["interval_availability_rate", "outcome_interval_coverage_rate"],
    )


def _outcome_template_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    template = predictions.reindex(columns=["unit_number"]).copy()
    template["actual_rul"] = ""
    return template


def _failed_gates_frame(failed_gates: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([{"gate": str(gate)} for gate in failed_gates])


def _audit_events_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["created_at_utc", "event_type", "status", "actor", "note"]
    return pd.DataFrame(events).reindex(columns=columns)


def _decision_status_index(status: Any) -> int:
    options = ["review_required", "accepted", "watch", "escalated", "rejected"]
    if status in options:
        return options.index(status)
    return 0


def _csv_filter_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _history_date_bound(value: str, *, end_of_day: bool) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    if "T" in normalized:
        return normalized
    suffix = "T23:59:59.999999+00:00" if end_of_day else "T00:00:00+00:00"
    return f"{normalized}{suffix}"


def _read_outcome_csv(source: bytes | Path) -> pd.DataFrame:
    try:
        outcomes = (
            pd.read_csv(BytesIO(source))
            if isinstance(source, bytes)
            else pd.read_csv(source)
        )
    except (EmptyDataError, ParserError, UnicodeDecodeError) as exc:
        raise ValueError("outcome CSV could not be read as a valid CSV") from exc
    _validate_outcome_frame(outcomes)
    return outcomes


def _validate_outcome_frame(outcomes: pd.DataFrame) -> None:
    if outcomes.empty:
        raise ValueError("outcome CSV must contain at least one row")
    required_columns = ("unit_number", "actual_rul")
    missing = [column for column in required_columns if column not in outcomes.columns]
    if missing:
        raise ValueError(f"outcome CSV is missing required columns: {', '.join(missing)}")
    numeric_columns = outcomes[list(required_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric_columns.isna().any().any():
        raise ValueError("unit_number and actual_rul must be numeric and non-null")
    if (numeric_columns["unit_number"] % 1 != 0).any():
        raise ValueError("unit_number values must be whole numbers")
    if (numeric_columns["actual_rul"] < 0).any():
        raise ValueError("actual_rul values must be nonnegative")


def _percent_frame_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce") * 100
    return result
