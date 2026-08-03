"""State loading helpers for the interactive fleet dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aerospace_prognostics.deployment.artifacts import load_cmapss_model_artifact


@dataclass(frozen=True)
class QuickstartWorkspace:
    """Resolved quickstart artifact paths and loaded JSON documents."""

    root: Path
    model_artifact_path: Path
    telemetry_csv_path: Path
    dashboard_payload_path: Path
    artifact_inspection_path: Path
    release_bundle_path: Path
    provenance_path: Path
    promotion_report_path: Path
    dashboard_payload: dict[str, Any] | None
    artifact_inspection: dict[str, Any] | None
    release_bundle: dict[str, Any] | None
    provenance: dict[str, Any] | None
    promotion_report: dict[str, Any] | None
    missing_paths: tuple[Path, ...]

    @property
    def is_ready(self) -> bool:
        """Return True when the app has enough quickstart evidence to render."""
        return self.dashboard_payload is not None and self.model_artifact_path.exists()


def load_quickstart_workspace(root: str | Path) -> QuickstartWorkspace:
    """Load the default product-demo workspace produced by quickstart-cmapss-demo."""
    root_path = Path(root)
    model_artifact_path = root_path / "models" / "fd001.joblib"
    telemetry_csv_path = root_path / "predictions" / "fd001_input.csv"
    dashboard_payload_path = root_path / "dashboard" / "fleet_payload.json"
    artifact_inspection_path = root_path / "models" / "fd001_inspection.json"
    release_bundle_path = root_path / "release" / "fd001_release_bundle.json"
    provenance_path = root_path / "release" / "fd001_provenance.json"
    promotion_report_path = root_path / "models" / "fd001_promotion.json"
    required_paths = (
        model_artifact_path,
        telemetry_csv_path,
        dashboard_payload_path,
        artifact_inspection_path,
        release_bundle_path,
        provenance_path,
        promotion_report_path,
    )
    return QuickstartWorkspace(
        root=root_path,
        model_artifact_path=model_artifact_path,
        telemetry_csv_path=telemetry_csv_path,
        dashboard_payload_path=dashboard_payload_path,
        artifact_inspection_path=artifact_inspection_path,
        release_bundle_path=release_bundle_path,
        provenance_path=provenance_path,
        promotion_report_path=promotion_report_path,
        dashboard_payload=_read_optional_json(dashboard_payload_path),
        artifact_inspection=_read_optional_json(artifact_inspection_path),
        release_bundle=_read_optional_json(release_bundle_path),
        provenance=_read_optional_json(provenance_path),
        promotion_report=_read_optional_json(promotion_report_path),
        missing_paths=tuple(path for path in required_paths if not path.exists()),
    )


def predict_cmapss_telemetry(
    model_artifact_path: str | Path,
    telemetry: pd.DataFrame,
) -> dict[str, Any]:
    """Run deployed C-MAPSS predictions and return the app prediction document."""
    artifact = load_cmapss_model_artifact(model_artifact_path)
    predictions = artifact.predict_from_frame(telemetry)
    monitoring = artifact.monitoring_summary(telemetry, predictions)
    return {
        "dataset": artifact.dataset,
        "subset": artifact.subset,
        "model_name": artifact.model_name,
        "rul_cap": artifact.rul_cap,
        "predictions": [prediction.to_dict() for prediction in predictions],
        "monitoring": monitoring,
        "artifact": {
            "schema_version": artifact.schema_version,
            "artifact_id": artifact.promotion_metadata.get("artifact_id"),
            "stage": artifact.promotion_metadata.get("stage"),
        },
    }


def read_json_document(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json_document(path)
