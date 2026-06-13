from __future__ import annotations

from aerospace_prognostics.app.dashboard_state import (
    load_quickstart_workspace,
    predict_cmapss_telemetry,
)
from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.deployment.artifacts import (
    save_cmapss_model_artifact,
    train_cmapss_hgb_policy_artifact,
)
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_load_quickstart_workspace_reports_missing_artifacts(tmp_path) -> None:
    workspace = load_quickstart_workspace(tmp_path / "missing")

    assert workspace.is_ready is False
    assert workspace.dashboard_payload is None
    assert workspace.artifact_inspection is None
    assert workspace.model_artifact_path.name == "fd001.joblib"
    assert len(workspace.missing_paths) >= 5


def test_predict_cmapss_telemetry_returns_predictions_and_monitoring(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1)
    artifact_path = save_cmapss_model_artifact(
        packaged.artifact,
        tmp_path / "models" / "fd001.joblib",
    )
    telemetry = load_cmapss_subset(tmp_path, "FD001").test

    prediction_document = predict_cmapss_telemetry(artifact_path, telemetry)

    assert prediction_document["dataset"] == "C-MAPSS"
    assert prediction_document["subset"] == "FD001"
    assert prediction_document["artifact"]["artifact_id"].startswith("fd001-")
    assert len(prediction_document["predictions"]) == 2
    assert prediction_document["predictions"][0]["predicted_rul_lower"] is not None
    assert prediction_document["monitoring"]["predictions"]["count"] == 2
