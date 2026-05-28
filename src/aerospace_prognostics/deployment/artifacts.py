"""Model artifact packaging for deployment-oriented C-MAPSS inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

ARTIFACT_SCHEMA_VERSION = "1.0"


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
    )
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
    if artifact.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported artifact schema version "
            f"{artifact.schema_version!r}; expected {ARTIFACT_SCHEMA_VERSION!r}"
        )
    return artifact


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


def _hgb_params_by_label() -> dict[str, dict[str, float | int | str]]:
    return {str(params["label"]): params for params in CMAPSS_HGB_PARAM_GRID}
