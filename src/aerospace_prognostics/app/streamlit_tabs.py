"""Focused Streamlit tab renderers for the PHM console."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from aerospace_prognostics.app.api_client import ApiServiceStatus
from aerospace_prognostics.app.dashboard_state import QuickstartWorkspace
from aerospace_prognostics.app.store import (
    build_model_artifact_review_bundle,
    database_summary,
    list_model_artifacts,
    load_model_artifact,
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


def _failed_gates_frame(failed_gates: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([{"gate": str(gate)} for gate in failed_gates])


def _percent_frame_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce") * 100
    return result
