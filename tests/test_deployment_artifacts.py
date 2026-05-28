from __future__ import annotations

from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.deployment.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    load_cmapss_model_artifact,
    save_cmapss_model_artifact,
    train_cmapss_hgb_policy_artifact,
)
from tests.cmapss_fixtures import write_tiny_cmapss_subset


def test_train_save_load_and_predict_cmapss_hgb_policy_artifact(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    packaged = train_cmapss_hgb_policy_artifact(
        tmp_path,
        "FD001",
        n_regimes=1,
    )
    artifact_path = tmp_path / "models" / "fd001.joblib"

    save_cmapss_model_artifact(packaged.artifact, artifact_path)
    loaded = load_cmapss_model_artifact(artifact_path)
    bundle = load_cmapss_subset(tmp_path, "FD001")
    predictions = loaded.predict_from_frame(bundle.test)

    assert artifact_path.exists()
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert loaded.subset == "FD001"
    assert loaded.feature_policy == "regime_engineered"
    assert packaged.result.rmse >= 0
    assert [prediction.unit_number for prediction in predictions] == [1, 2]
    assert all(0 <= prediction.predicted_rul <= loaded.rul_cap for prediction in predictions)


def test_cmapss_hgb_policy_artifact_rejects_missing_columns(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact = train_cmapss_hgb_policy_artifact(tmp_path, "FD001", n_regimes=1).artifact
    bundle = load_cmapss_subset(tmp_path, "FD001")
    malformed = bundle.test.drop(columns=["sensor_1"])

    try:
        artifact.predict_from_frame(malformed)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("expected missing-column validation error")
