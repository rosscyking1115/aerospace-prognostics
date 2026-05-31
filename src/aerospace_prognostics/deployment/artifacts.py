"""Model artifact packaging for deployment-oriented C-MAPSS inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aerospace_prognostics.data.cmapss import (
    CMAPSS_COLUMNS,
    load_cmapss_subset,
)
from aerospace_prognostics.evaluation import RegressionRunResult
from aerospace_prognostics.experiments.cmapss_baseline import (
    CMAPSS_ENGINEERED_DEFAULT_WINDOWS,
    CMAPSS_HGB_PARAM_GRID,
    CMAPSS_VALIDATION_SELECTED_FEATURES,
    CMAPSS_VALIDATION_SELECTED_HGB_PARAMS,
)
from aerospace_prognostics.features import (
    OperatingRegimeFeatureTransformer,
    engineered_cycle_feature_table,
    engineered_last_cycle_feature_table,
)
from aerospace_prognostics.metrics import nasa_rul_score, rmse
from aerospace_prognostics.models.baselines import hist_gradient_boosting_rul
from aerospace_prognostics.preprocessing import FeatureStandardizer

ARTIFACT_SCHEMA_VERSION = "1.1"
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = {"1.0", ARTIFACT_SCHEMA_VERSION}


@dataclass(frozen=True)
class CmapssArtifactValidation:
    """Deployment validation report for a packaged C-MAPSS artifact."""

    artifact_path: str
    metadata_json_path: str | None
    input_csv_path: str | None
    status: str
    checks: dict[str, bool]
    problems: list[str]
    artifact_identity: dict[str, Any] = field(default_factory=dict)
    prediction_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable validation report."""

        return {
            "artifact_path": self.artifact_path,
            "metadata_json_path": self.metadata_json_path,
            "input_csv_path": self.input_csv_path,
            "status": self.status,
            "checks": self.checks,
            "problems": self.problems,
            "artifact_identity": self.artifact_identity,
            "prediction_count": self.prediction_count,
        }


@dataclass(frozen=True)
class CmapssPrediction:
    """One deployed C-MAPSS RUL prediction."""

    unit_number: int
    predicted_rul: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class CmapssHgbPolicyModelArtifact:
    """Packaged validation-selected HGB policy model and preprocessing state."""

    schema_version: str
    dataset: str
    subset: str
    model_name: str
    feature_policy: str
    hgb_policy: str
    rolling_window: int
    rul_cap: int
    random_state: int
    standardize: bool
    input_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    model: Any
    standardizer: FeatureStandardizer | None = None
    regime_transformer: OperatingRegimeFeatureTransformer | None = None
    reference_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    promotion_metadata: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Return deployment metadata without binary model objects."""

        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "subset": self.subset,
            "model_name": self.model_name,
            "feature_policy": self.feature_policy,
            "hgb_policy": self.hgb_policy,
            "rolling_window": self.rolling_window,
            "rul_cap": self.rul_cap,
            "random_state": self.random_state,
            "standardize": self.standardize,
            "input_columns": list(self.input_columns),
            "feature_columns": list(self.feature_columns),
            "reference_stats_columns": sorted(self._reference_stats),
            "promotion": self._promotion_metadata,
        }

    def predict_from_frame(self, frame: pd.DataFrame) -> list[CmapssPrediction]:
        """Predict capped RUL from one or more raw C-MAPSS telemetry histories."""

        _validate_inference_frame(frame, self.input_columns)
        features = self._build_features(frame)
        if self.standardizer is not None:
            features = self.standardizer.transform_features(features)
        predictions = np.clip(self.model.predict(features), 0.0, float(self.rul_cap))
        unit_numbers = (
            frame.sort_values(["unit_number", "time_in_cycles"])
            .groupby("unit_number", as_index=False)
            .tail(1)
            .sort_values("unit_number")["unit_number"]
            .astype(int)
            .tolist()
        )
        return [
            CmapssPrediction(unit_number=unit, predicted_rul=float(prediction))
            for unit, prediction in zip(unit_numbers, predictions, strict=True)
        ]

    def monitoring_summary(
        self,
        frame: pd.DataFrame,
        predictions: list[CmapssPrediction],
        *,
        drift_threshold: float = 3.0,
    ) -> dict[str, Any]:
        """Summarise request telemetry drift and prediction distribution."""

        _validate_inference_frame(frame, self.input_columns)
        telemetry_summary = _telemetry_drift_summary(
            frame,
            self._reference_stats,
            drift_threshold=drift_threshold,
        )
        prediction_values = [prediction.predicted_rul for prediction in predictions]
        return {
            "telemetry": telemetry_summary,
            "predictions": _numeric_distribution(prediction_values),
        }

    @property
    def _reference_stats(self) -> dict[str, dict[str, float]]:
        return getattr(self, "reference_stats", {})

    @property
    def _promotion_metadata(self) -> dict[str, Any]:
        return getattr(self, "promotion_metadata", {})

    def _build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.feature_policy == "engineered":
            return engineered_last_cycle_feature_table(
                frame,
                rolling_window=self.rolling_window,
            )
        if self.feature_policy == "regime_engineered":
            if self.regime_transformer is None:
                raise ValueError("regime_engineered artifacts require a regime transformer")
            return self.regime_transformer.transform_engineered_last_cycle_frame(
                frame,
                rolling_window=self.rolling_window,
            )
        raise ValueError(f"unsupported feature policy: {self.feature_policy}")


@dataclass(frozen=True)
class PackagedCmapssModel:
    """A trained artifact plus its official-test evaluation result."""

    artifact: CmapssHgbPolicyModelArtifact
    result: RegressionRunResult


def train_cmapss_hgb_policy_artifact(
    data_dir: str | Path,
    subset: str,
    *,
    rul_cap: int = 125,
    random_state: int = 42,
    n_regimes: int = 6,
    standardize: bool = True,
) -> PackagedCmapssModel:
    """Train the validation-selected HGB policy and return a deployable artifact."""

    normalised_subset = subset.upper()
    rolling_window = CMAPSS_ENGINEERED_DEFAULT_WINDOWS[normalised_subset]
    feature_policy = CMAPSS_VALIDATION_SELECTED_FEATURES[normalised_subset]
    hgb_policy = CMAPSS_VALIDATION_SELECTED_HGB_PARAMS[normalised_subset]
    hgb_params = _hgb_params_by_label()[hgb_policy]

    bundle = load_cmapss_subset(data_dir, normalised_subset, rul_cap=rul_cap)
    regime_transformer = None
    if feature_policy == "engineered":
        train_features, train_target = engineered_cycle_feature_table(
            bundle.train,
            rolling_window=rolling_window,
        )
        test_features = engineered_last_cycle_feature_table(
            bundle.test,
            rolling_window=rolling_window,
        )
        model_prefix = f"hist_gradient_boosting_engineered_w{rolling_window}"
    elif feature_policy == "regime_engineered":
        regime_transformer = OperatingRegimeFeatureTransformer.fit(
            bundle.train,
            n_regimes=n_regimes,
            random_state=random_state,
        )
        train_features = regime_transformer.transform_engineered_frame(
            bundle.train,
            rolling_window=rolling_window,
        )
        train_target = bundle.train.loc[train_features.index, "rul_capped"].copy()
        test_features = regime_transformer.transform_engineered_last_cycle_frame(
            bundle.test,
            rolling_window=rolling_window,
        )
        model_prefix = (
            f"hist_gradient_boosting_regime_engineered_w{rolling_window}"
            f"_r{regime_transformer.n_regimes}"
        )
    else:
        raise ValueError("feature policy must be 'engineered' or 'regime_engineered'")

    standardizer = None
    if standardize:
        standardizer = FeatureStandardizer.fit(
            train_features,
            feature_columns=list(train_features.columns),
        )
        train_features = standardizer.transform_features(train_features)
        test_features = standardizer.transform_features(test_features)

    params = {key: value for key, value in hgb_params.items() if key != "label"}
    model = hist_gradient_boosting_rul(random_state=random_state, **params)
    model.fit(train_features, train_target)
    predictions = np.clip(model.predict(test_features), 0.0, float(rul_cap))
    model_name = f"{model_prefix}_{hgb_policy}"

    result = RegressionRunResult(
        dataset="C-MAPSS",
        subset=bundle.subset,
        model_name=model_name,
        rmse=rmse(bundle.test_rul, predictions),
        nasa_score=nasa_rul_score(bundle.test_rul, predictions),
        train_rows=len(bundle.train),
        train_units=bundle.train["unit_number"].nunique(),
        test_rows=len(bundle.test),
        test_units=bundle.test["unit_number"].nunique(),
        test_rul_values=len(bundle.test_rul),
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
    )
    promotion_metadata = _promotion_metadata(
        dataset="C-MAPSS",
        subset=bundle.subset,
        model_name=model_name,
        feature_policy=feature_policy,
        hgb_policy=hgb_policy,
        rolling_window=rolling_window,
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
        result=result,
    )

    artifact = CmapssHgbPolicyModelArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        dataset="C-MAPSS",
        subset=bundle.subset,
        model_name=model_name,
        feature_policy=feature_policy,
        hgb_policy=hgb_policy,
        rolling_window=rolling_window,
        rul_cap=rul_cap,
        random_state=random_state,
        standardize=standardize,
        input_columns=tuple(CMAPSS_COLUMNS),
        feature_columns=tuple(train_features.columns),
        model=model,
        standardizer=standardizer,
        regime_transformer=regime_transformer,
        reference_stats=_reference_stats(bundle.train, tuple(CMAPSS_COLUMNS[1:])),
        promotion_metadata=promotion_metadata,
    )
    return PackagedCmapssModel(artifact=artifact, result=result)


def save_cmapss_model_artifact(
    artifact: CmapssHgbPolicyModelArtifact,
    path: str | Path,
) -> Path:
    """Persist a deployable model artifact with joblib."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return output_path


def load_cmapss_model_artifact(path: str | Path) -> CmapssHgbPolicyModelArtifact:
    """Load and validate a packaged C-MAPSS model artifact."""

    artifact = joblib.load(Path(path))
    if not isinstance(artifact, CmapssHgbPolicyModelArtifact):
        raise TypeError("artifact is not a CmapssHgbPolicyModelArtifact")
    if artifact.schema_version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(
            "unsupported artifact schema version "
            f"{artifact.schema_version!r}; expected one of "
            f"{sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)!r}"
        )
    if not hasattr(artifact, "reference_stats"):
        object.__setattr__(artifact, "reference_stats", {})
    if not hasattr(artifact, "promotion_metadata"):
        object.__setattr__(artifact, "promotion_metadata", {})
    return artifact


def validate_cmapss_model_artifact(
    artifact_path: str | Path,
    *,
    metadata_json: str | Path | None = None,
    input_csv: str | Path | None = None,
) -> CmapssArtifactValidation:
    """Validate that a packaged artifact is loadable and promotion-ready."""

    artifact_file = Path(artifact_path)
    metadata_file = Path(metadata_json) if metadata_json is not None else None
    input_file = Path(input_csv) if input_csv is not None else None
    checks = {
        "artifact_exists": artifact_file.exists(),
        "artifact_loads": False,
        "schema_version_supported": False,
        "promotion_metadata_present": False,
    }
    problems: list[str] = []
    artifact: CmapssHgbPolicyModelArtifact | None = None
    prediction_count: int | None = None

    if not checks["artifact_exists"]:
        problems.append(f"artifact does not exist: {artifact_file}")
    else:
        try:
            artifact = load_cmapss_model_artifact(artifact_file)
        except (OSError, TypeError, ValueError) as exc:
            problems.append(f"artifact failed to load: {exc}")
        else:
            checks["artifact_loads"] = True
            checks["schema_version_supported"] = True
            checks["promotion_metadata_present"] = _has_required_promotion_metadata(artifact)
            if not checks["promotion_metadata_present"]:
                problems.append("artifact is missing required promotion metadata")

    artifact_identity = _artifact_identity(artifact) if artifact is not None else {}

    if metadata_file is not None:
        checks["metadata_json_matches"] = False
        if artifact is None:
            problems.append("metadata JSON cannot be checked because artifact did not load")
        elif not metadata_file.exists():
            problems.append(f"metadata JSON does not exist: {metadata_file}")
        else:
            try:
                metadata_payload = json.loads(metadata_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append(f"metadata JSON is not valid JSON: {exc}")
            else:
                metadata_problems = _metadata_json_mismatches(artifact, metadata_payload)
                if metadata_problems:
                    problems.extend(metadata_problems)
                else:
                    checks["metadata_json_matches"] = True

    if input_file is not None:
        checks["prediction_smoke"] = False
        if artifact is None:
            problems.append("prediction smoke cannot run because artifact did not load")
        elif not input_file.exists():
            problems.append(f"input CSV does not exist: {input_file}")
        else:
            try:
                telemetry = pd.read_csv(input_file)
                predictions = artifact.predict_from_frame(telemetry)
            except (OSError, ValueError) as exc:
                problems.append(f"prediction smoke failed: {exc}")
            else:
                prediction_count = len(predictions)
                checks["prediction_smoke"] = prediction_count > 0
                if prediction_count == 0:
                    problems.append("prediction smoke produced no predictions")

    return CmapssArtifactValidation(
        artifact_path=str(artifact_file),
        metadata_json_path=str(metadata_file) if metadata_file is not None else None,
        input_csv_path=str(input_file) if input_file is not None else None,
        status="ok" if not problems and all(checks.values()) else "failed",
        checks=checks,
        problems=problems,
        artifact_identity=artifact_identity,
        prediction_count=prediction_count,
    )


def _validate_inference_frame(frame: pd.DataFrame, input_columns: tuple[str, ...]) -> None:
    if frame.empty:
        raise ValueError("telemetry frame must contain at least one row")
    missing = [column for column in input_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"telemetry frame is missing columns: {missing}")
    if frame["unit_number"].isna().any() or frame["time_in_cycles"].isna().any():
        raise ValueError("unit_number and time_in_cycles cannot contain null values")
    if (frame["time_in_cycles"] < 1).any():
        raise ValueError("time_in_cycles values must be positive")


def _reference_stats(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        stats[column] = {
            "count": float(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return stats


def _telemetry_drift_summary(
    frame: pd.DataFrame,
    reference_stats: dict[str, dict[str, float]],
    *,
    drift_threshold: float,
) -> dict[str, Any]:
    columns: dict[str, dict[str, float | None]] = {}
    alert_columns: list[str] = []
    max_shift: float | None = None
    for column, reference in sorted(reference_stats.items()):
        values = pd.to_numeric(frame[column], errors="coerce")
        request_mean = float(values.mean())
        request_std = float(values.std(ddof=0))
        reference_std = reference["std"]
        standardized_shift = (
            abs(request_mean - reference["mean"]) / reference_std if reference_std > 0 else None
        )
        if standardized_shift is not None:
            max_shift = (
                standardized_shift if max_shift is None else max(max_shift, standardized_shift)
            )
            if standardized_shift >= drift_threshold:
                alert_columns.append(column)
        columns[column] = {
            "reference_mean": reference["mean"],
            "reference_std": reference_std,
            "request_mean": request_mean,
            "request_std": request_std,
            "standardized_abs_mean_shift": standardized_shift,
        }
    return {
        "reference_columns": len(reference_stats),
        "drift_threshold": drift_threshold,
        "alert_column_count": len(alert_columns),
        "alert_columns": alert_columns,
        "max_standardized_abs_mean_shift": max_shift,
        "columns": columns,
    }


def _numeric_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    series = pd.Series(values, dtype=float)
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _has_required_promotion_metadata(artifact: CmapssHgbPolicyModelArtifact) -> bool:
    promotion = artifact.promotion_metadata
    return all(
        [
            bool(promotion.get("artifact_id")),
            bool(promotion.get("stage")),
            isinstance(promotion.get("identity"), dict),
            isinstance(promotion.get("rollback"), dict),
        ]
    )


def _artifact_identity(artifact: CmapssHgbPolicyModelArtifact | None) -> dict[str, Any]:
    if artifact is None:
        return {}
    return {
        "schema_version": artifact.schema_version,
        "dataset": artifact.dataset,
        "subset": artifact.subset,
        "model_name": artifact.model_name,
        "artifact_id": artifact.promotion_metadata.get("artifact_id"),
        "stage": artifact.promotion_metadata.get("stage"),
    }


def _metadata_json_mismatches(
    artifact: CmapssHgbPolicyModelArtifact,
    metadata_payload: Any,
) -> list[str]:
    if not isinstance(metadata_payload, dict):
        return ["metadata JSON root must be an object"]
    artifact_metadata = metadata_payload.get("artifact")
    if not isinstance(artifact_metadata, dict):
        return ["metadata JSON must contain an artifact object"]

    problems: list[str] = []
    expected = artifact.metadata()
    for key in ["schema_version", "dataset", "subset", "model_name"]:
        if artifact_metadata.get(key) != expected[key]:
            problems.append(
                f"metadata {key} mismatch: expected {expected[key]!r}, "
                f"got {artifact_metadata.get(key)!r}"
            )

    metadata_promotion = artifact_metadata.get("promotion")
    if not isinstance(metadata_promotion, dict):
        problems.append("metadata promotion block must be an object")
    elif metadata_promotion.get("artifact_id") != artifact.promotion_metadata.get("artifact_id"):
        problems.append(
            "metadata artifact_id mismatch: expected "
            f"{artifact.promotion_metadata.get('artifact_id')!r}, "
            f"got {metadata_promotion.get('artifact_id')!r}"
        )
    return problems


def _promotion_metadata(
    *,
    dataset: str,
    subset: str,
    model_name: str,
    feature_policy: str,
    hgb_policy: str,
    rolling_window: int,
    rul_cap: int,
    random_state: int,
    standardize: bool,
    result: RegressionRunResult,
) -> dict[str, Any]:
    created_at_utc = datetime.now(UTC).isoformat(timespec="seconds")
    identity = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset": dataset,
        "subset": subset,
        "model_name": model_name,
        "feature_policy": feature_policy,
        "hgb_policy": hgb_policy,
        "rolling_window": rolling_window,
        "rul_cap": rul_cap,
        "random_state": random_state,
        "standardize": standardize,
        "official_test_rmse": round(result.rmse, 12),
        "official_test_nasa_score": round(result.nasa_score, 12),
        "train_rows": result.train_rows,
        "train_units": result.train_units,
        "test_rows": result.test_rows,
        "test_units": result.test_units,
        "test_rul_values": result.test_rul_values,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "artifact_id": f"{subset.lower()}-{digest[:12]}",
        "stage": "candidate",
        "created_at_utc": created_at_utc,
        "selection_source": "validation-selected HGB policy",
        "promotion_gate": "official-test metrics reviewed and serving smoke checks green",
        "rollback": {
            "strategy": "restore the previous promoted artifact path or container environment",
            "requires_retraining": False,
        },
        "identity": identity,
    }


def _hgb_params_by_label() -> dict[str, dict[str, float | int | str]]:
    return {str(params["label"]): params for params in CMAPSS_HGB_PARAM_GRID}
