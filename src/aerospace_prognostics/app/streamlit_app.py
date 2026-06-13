"""Streamlit operations console for Aerospace Prognostics."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from aerospace_prognostics.app.dashboard_state import (
    QuickstartWorkspace,
    load_quickstart_workspace,
    predict_cmapss_telemetry,
)
from aerospace_prognostics.app.store import (
    database_summary,
    initialize_app_database,
    list_prediction_runs,
    load_prediction_run,
    record_prediction_run,
    seed_quickstart_workspace,
)

DEFAULT_WORKSPACE = Path("artifacts") / "quickstart_cmapss"
DEFAULT_DATABASE = Path("artifacts") / "app" / "aerospace_prognostics.sqlite"


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
        workspace = load_quickstart_workspace(workspace_root)
        initialize_app_database(database_path)
        st.caption(f"Workspace: {workspace.root}")
        if workspace.missing_paths:
            st.warning("Quickstart evidence is incomplete.")
            st.code("uv run aerospace-prognostics quickstart-cmapss-demo")
            for path in workspace.missing_paths[:6]:
                st.caption(f"Missing: {path}")
        else:
            st.success("Quickstart evidence loaded.")
            seed_quickstart_workspace(database_path, workspace)
        summary = database_summary(database_path)
        st.caption(f"Database: {summary['database_path']}")
        st.metric("Prediction runs", summary["prediction_runs"])
        st.metric("Stored predictions", summary["predictions"])

    if not workspace.is_ready:
        st.info("Generate the quickstart evidence bundle to activate the console.")
        return

    fleet_tab, predict_tab, history_tab, evidence_tab, roadmap_tab = st.tabs(
        ["Fleet", "Predict", "History", "Evidence", "Roadmap"]
    )
    with fleet_tab:
        _render_fleet_tab(st, workspace)
    with predict_tab:
        _render_predict_tab(st, workspace, database_path)
    with history_tab:
        _render_history_tab(st, database_path)
    with evidence_tab:
        _render_evidence_tab(st, workspace, database_path)
    with roadmap_tab:
        _render_roadmap_tab(st)


def _render_fleet_tab(st: Any, workspace: QuickstartWorkspace) -> None:
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


def _render_predict_tab(st: Any, workspace: QuickstartWorkspace, database_path: Path) -> None:
    st.subheader("Batch Prediction")
    uploaded = st.file_uploader("Telemetry CSV", type=["csv"])
    if uploaded is None and workspace.telemetry_csv_path.exists():
        st.caption(f"Using sample telemetry: {workspace.telemetry_csv_path}")
        telemetry = pd.read_csv(workspace.telemetry_csv_path)
        source_name = str(workspace.telemetry_csv_path)
    elif uploaded is not None:
        telemetry = pd.read_csv(BytesIO(uploaded.getvalue()))
        source_name = uploaded.name
    else:
        telemetry = None
        source_name = "unknown"

    if telemetry is None:
        st.info("No telemetry loaded.")
        return

    st.dataframe(telemetry.head(12), use_container_width=True, hide_index=True)
    if st.button("Run Prediction", type="primary"):
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
        st.success(f"Stored prediction run: {run_id}")
        predictions = pd.DataFrame(prediction_document["predictions"])
        st.dataframe(predictions, use_container_width=True, hide_index=True)
        monitoring = prediction_document.get("monitoring", {})
        if isinstance(monitoring, dict):
            prediction_summary = monitoring.get("predictions", {})
            if isinstance(prediction_summary, dict):
                st.json(prediction_summary)


def _render_history_tab(st: Any, database_path: Path) -> None:
    st.subheader("Prediction History")
    runs = list_prediction_runs(database_path, limit=100)
    if not runs:
        st.info("No prediction runs stored yet.")
        return

    runs_frame = _prediction_runs_frame(runs)
    st.dataframe(
        runs_frame,
        use_container_width=True,
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
            "drift_alert_count": st.column_config.NumberColumn("Drift Alerts", width="small"),
        },
    )

    run_ids = [str(run["run_id"]) for run in runs]
    selected_default = st.session_state.get("selected_prediction_run_id")
    selected_index = run_ids.index(selected_default) if selected_default in run_ids else 0
    selected_run_id = st.selectbox("Run", run_ids, index=selected_index)
    st.session_state["selected_prediction_run_id"] = selected_run_id
    selected = load_prediction_run(database_path, selected_run_id)
    if selected is None:
        st.warning("Selected run is no longer available.")
        return

    run = selected["run"]
    predictions_frame = pd.DataFrame(selected["predictions"])
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", _display(run.get("row_count")))
    metric_columns[1].metric("Predictions", _display(run.get("prediction_count")))
    metric_columns[2].metric("Dataset", _display(run.get("dataset")))
    metric_columns[3].metric("Subset", _display(run.get("subset")))

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

    st.dataframe(
        predictions_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "asset_id": st.column_config.TextColumn("Asset"),
            "unit_number": st.column_config.NumberColumn("Unit", width="small"),
            "predicted_rul": st.column_config.NumberColumn("RUL", format="%.1f"),
            "predicted_rul_lower": st.column_config.NumberColumn("Lower", format="%.1f"),
            "predicted_rul_upper": st.column_config.NumberColumn("Upper", format="%.1f"),
            "interval_method": st.column_config.TextColumn("Interval"),
            "interval_confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
        },
    )
    st.download_button(
        "Download Predictions",
        data=predictions_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_run_id}_predictions.csv",
        mime="text/csv",
    )


def _render_evidence_tab(st: Any, workspace: QuickstartWorkspace, database_path: Path) -> None:
    inspection = workspace.artifact_inspection or {}
    release_bundle = workspace.release_bundle or {}
    provenance = workspace.provenance or {}
    promotion_report = workspace.promotion_report or {}
    summary = database_summary(database_path)

    columns = st.columns(4)
    identity = inspection.get("artifact_identity")
    if not isinstance(identity, dict):
        identity = {}
    columns[0].metric("Artifact", _display(identity.get("artifact_id")))
    columns[1].metric("Release", _display(release_bundle.get("status")))
    columns[2].metric("Promotion", _display(promotion_report.get("status")))
    columns[3].metric("DB Evidence", _display(summary["release_evidence"]))

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


def _render_roadmap_tab(st: Any) -> None:
    st.subheader("Product Roadmap")
    st.markdown(
        """
        - Persist prediction runs, uploads, artifact records, and release evidence in SQLite.
        - Make prediction history filterable by model, asset, risk, date, and drift status.
        - Add Docker Compose for FastAPI, dashboard, mounted model storage, and the database.
        - Promote the dashboard to a hosted product surface with authentication and audit logs.
        - Add spacecraft anomaly assets beside C-MAPSS engines in the same fleet view.
        """
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
        "drift_alert_count",
    ]
    return pd.DataFrame(runs).reindex(columns=columns)


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


if __name__ == "__main__":
    main()
