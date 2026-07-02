"""Focused Streamlit tab renderers for the PHM console."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from aerospace_prognostics.app.api_client import ApiServiceStatus
from aerospace_prognostics.app.dashboard_state import QuickstartWorkspace
from aerospace_prognostics.app.store import (
    build_model_artifact_review_bundle,
    build_prediction_run_evidence,
    database_summary,
    export_prediction_run_evidence,
    list_model_artifacts,
    list_prediction_runs,
    load_model_artifact,
    load_prediction_run,
    record_prediction_outcomes,
    record_prediction_run_event,
)


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
        )
    with filter_columns[1]:
        selected_artifacts = st.multiselect(
            "Artifact",
            filter_options(all_runs, "artifact_id"),
            default=[],
        )
    with filter_columns[2]:
        selected_risks = st.multiselect(
            "Risk",
            ["critical", "watch", "nominal", "unknown"],
            default=[],
        )
    with filter_columns[3]:
        selected_decisions = st.multiselect(
            "Decision",
            filter_options(all_runs, "decision_status"),
            default=[],
        )

    advanced_columns = st.columns([1, 1, 1, 1])
    with advanced_columns[0]:
        asset_filter_text = st.text_input("Asset", value="")
    with advanced_columns[1]:
        start_date = st.text_input("Created From", value="")
    with advanced_columns[2]:
        end_date = st.text_input("Created To", value="")
    with advanced_columns[3]:
        drift_only = st.checkbox("Drift Only", value=False)

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
