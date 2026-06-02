"""Model artifact packaging for deployment-oriented C-MAPSS inference."""

from __future__ import annotations

import hashlib
import json
import time
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
class CmapssArtifactBenchmark:
    """Inference benchmark report for a packaged C-MAPSS artifact."""

    artifact_path: str
    input_csv_path: str
    status: str
    runs: int
    warmup_runs: int
    input_rows: int
    prediction_count: int
    model_size_bytes: int
    latency_ms: dict[str, float]
    max_p95_latency_ms: float | None = None
    problems: list[str] = field(default_factory=list)
    artifact_identity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable benchmark report."""

        return {
            "artifact_path": self.artifact_path,
            "input_csv_path": self.input_csv_path,
            "status": self.status,
            "runs": self.runs,
            "warmup_runs": self.warmup_runs,
            "input_rows": self.input_rows,
            "prediction_count": self.prediction_count,
            "model_size_bytes": self.model_size_bytes,
            "latency_ms": self.latency_ms,
            "max_p95_latency_ms": self.max_p95_latency_ms,
            "problems": self.problems,
            "artifact_identity": self.artifact_identity,
        }


@dataclass(frozen=True)
class CmapssPromotionReport:
    """Promotion-gate evidence report for a packaged C-MAPSS artifact."""

    validation_json_path: str
    benchmark_json_path: str
    status: str
    gates: dict[str, bool]
    problems: list[str]
    artifact_identity: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    model_card_markdown_path: str | None = None
    sbom_json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable promotion report."""

        return {
            "validation_json_path": self.validation_json_path,
            "benchmark_json_path": self.benchmark_json_path,
            "model_card_markdown_path": self.model_card_markdown_path,
            "sbom_json_path": self.sbom_json_path,
            "status": self.status,
            "gates": self.gates,
            "problems": self.problems,
            "artifact_identity": self.artifact_identity,
            "evidence": self.evidence,
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


def benchmark_cmapss_model_artifact(
    artifact_path: str | Path,
    input_csv: str | Path,
    *,
    runs: int = 20,
    warmup_runs: int = 3,
    max_p95_latency_ms: float | None = None,
) -> CmapssArtifactBenchmark:
    """Benchmark batch inference latency for a packaged C-MAPSS artifact."""

    if runs < 1:
        raise ValueError("runs must be at least 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be greater than or equal to 0")
    if max_p95_latency_ms is not None and max_p95_latency_ms <= 0:
        raise ValueError("max_p95_latency_ms must be positive when provided")

    artifact_file = Path(artifact_path)
    input_file = Path(input_csv)
    artifact = load_cmapss_model_artifact(artifact_file)
    telemetry = pd.read_csv(input_file)

    predictions: list[CmapssPrediction] = []
    for _ in range(warmup_runs):
        predictions = artifact.predict_from_frame(telemetry)

    latencies_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        predictions = artifact.predict_from_frame(telemetry)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    latency_summary = _latency_distribution(latencies_ms)
    problems: list[str] = []
    if max_p95_latency_ms is not None and latency_summary["p95"] > max_p95_latency_ms:
        problems.append(
            "p95 latency exceeded budget: "
            f"{latency_summary['p95']:.6f} ms > {max_p95_latency_ms:.6f} ms"
        )

    return CmapssArtifactBenchmark(
        artifact_path=str(artifact_file),
        input_csv_path=str(input_file),
        status="ok" if not problems else "failed",
        runs=runs,
        warmup_runs=warmup_runs,
        input_rows=len(telemetry),
        prediction_count=len(predictions),
        model_size_bytes=artifact_file.stat().st_size,
        latency_ms=latency_summary,
        max_p95_latency_ms=max_p95_latency_ms,
        problems=problems,
        artifact_identity=_artifact_identity(artifact),
    )


def build_cmapss_promotion_report(
    validation_json: str | Path,
    benchmark_json: str | Path,
    *,
    model_card_markdown: str | Path | None = None,
    sbom_json: str | Path | None = None,
) -> CmapssPromotionReport:
    """Combine deployment evidence into a promotion-gate report."""

    validation_file = Path(validation_json)
    benchmark_file = Path(benchmark_json)
    model_card_file = Path(model_card_markdown) if model_card_markdown is not None else None
    sbom_file = Path(sbom_json) if sbom_json is not None else None

    validation = _read_json_object(validation_file, "validation report")
    benchmark = _read_json_object(benchmark_file, "benchmark report")

    validation_problems = _string_list(validation.get("problems"))
    benchmark_problems = _string_list(benchmark.get("problems"))
    gates = {
        "artifact_validation": validation.get("status") == "ok",
        "latency_benchmark": benchmark.get("status") == "ok",
    }
    problems: list[str] = []
    if not gates["artifact_validation"]:
        problems.append("artifact validation gate failed")
    if validation_problems:
        problems.extend(f"validation: {problem}" for problem in validation_problems)
    if not gates["latency_benchmark"]:
        problems.append("latency benchmark gate failed")
    if benchmark_problems:
        problems.extend(f"benchmark: {problem}" for problem in benchmark_problems)

    validation_identity = _dict_or_empty(validation.get("artifact_identity"))
    benchmark_identity = _dict_or_empty(benchmark.get("artifact_identity"))
    artifact_identity = validation_identity or benchmark_identity
    validation_artifact_id = _artifact_id(validation_identity)
    benchmark_artifact_id = _artifact_id(benchmark_identity)
    gates["artifact_identity_match"] = (
        validation_artifact_id is not None and validation_artifact_id == benchmark_artifact_id
    )
    if not gates["artifact_identity_match"]:
        problems.append(
            "artifact identity mismatch between validation and benchmark evidence: "
            f"{validation_artifact_id!r} != {benchmark_artifact_id!r}"
        )

    evidence: dict[str, Any] = {
        "validation": {
            "status": validation.get("status"),
            "checks": _dict_or_empty(validation.get("checks")),
            "prediction_count": validation.get("prediction_count"),
        },
        "benchmark": {
            "status": benchmark.get("status"),
            "runs": benchmark.get("runs"),
            "warmup_runs": benchmark.get("warmup_runs"),
            "model_size_bytes": benchmark.get("model_size_bytes"),
            "prediction_count": benchmark.get("prediction_count"),
            "latency_ms": _dict_or_empty(benchmark.get("latency_ms")),
            "max_p95_latency_ms": benchmark.get("max_p95_latency_ms"),
        },
    }

    if model_card_file is not None:
        gates["model_card_present"] = model_card_file.exists()
        if gates["model_card_present"]:
            model_card_text = model_card_file.read_text(encoding="utf-8")
            gates["model_card_has_required_sections"] = all(
                section in model_card_text
                for section in (
                    "# C-MAPSS Deployment Model Card",
                    "## Intended Use",
                    "## Performance",
                    "## Inference Contract",
                    "## Monitoring",
                    "## Limitations",
                    "## Rollback",
                )
            )
            evidence["model_card"] = {
                "path": str(model_card_file),
                "bytes": model_card_file.stat().st_size,
            }
            if not gates["model_card_has_required_sections"]:
                problems.append("model card is missing required deployment sections")
        else:
            gates["model_card_has_required_sections"] = False
            problems.append(f"model card markdown does not exist: {model_card_file}")

    if sbom_file is not None:
        gates["sbom_present"] = sbom_file.exists()
        if gates["sbom_present"]:
            sbom = _read_json_object(sbom_file, "SBOM")
            components = sbom.get("components", [])
            component_count = len(components) if isinstance(components, list) else 0
            gates["sbom_cyclonedx"] = sbom.get("bomFormat") == "CycloneDX"
            gates["sbom_has_components"] = component_count > 0
            evidence["sbom"] = {
                "path": str(sbom_file),
                "bom_format": sbom.get("bomFormat"),
                "spec_version": sbom.get("specVersion"),
                "component_count": component_count,
            }
            if not gates["sbom_cyclonedx"]:
                problems.append("SBOM is not CycloneDX formatted")
            if not gates["sbom_has_components"]:
                problems.append("SBOM contains no dependency components")
        else:
            gates["sbom_cyclonedx"] = False
            gates["sbom_has_components"] = False
            problems.append(f"SBOM JSON does not exist: {sbom_file}")

    return CmapssPromotionReport(
        validation_json_path=str(validation_file),
        benchmark_json_path=str(benchmark_file),
        model_card_markdown_path=str(model_card_file) if model_card_file is not None else None,
        sbom_json_path=str(sbom_file) if sbom_file is not None else None,
        status="ok" if not problems and all(gates.values()) else "failed",
        gates=gates,
        problems=problems,
        artifact_identity=artifact_identity,
        evidence=evidence,
    )


def render_cmapss_promotion_report_markdown(report: CmapssPromotionReport) -> str:
    """Render a markdown promotion-gate report."""

    identity = report.artifact_identity
    benchmark = report.evidence.get("benchmark", {})
    latency = benchmark.get("latency_ms", {}) if isinstance(benchmark, dict) else {}
    lines = [
        "# C-MAPSS Promotion Report",
        "",
        f"- Status: `{_markdown_inline(report.status)}`",
        f"- Artifact ID: `{_markdown_inline(identity.get('artifact_id'))}`",
        f"- Dataset: `{_markdown_inline(identity.get('dataset'))}`",
        f"- Subset: `{_markdown_inline(identity.get('subset'))}`",
        f"- Stage: `{_markdown_inline(identity.get('stage'))}`",
        "",
        "## Gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{_markdown_cell(name)}` | {passed} |"
        for name, passed in sorted(report.gates.items())
    )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Validation JSON: `{_markdown_inline(report.validation_json_path)}`",
            f"- Benchmark JSON: `{_markdown_inline(report.benchmark_json_path)}`",
            f"- Model card: `{_markdown_inline(report.model_card_markdown_path)}`",
            f"- SBOM JSON: `{_markdown_inline(report.sbom_json_path)}`",
            f"- Benchmark p95 latency ms: `{_markdown_inline(latency.get('p95'))}`",
            (
                "- Benchmark model size bytes: "
                f"`{_markdown_inline(benchmark.get('model_size_bytes'))}`"
            ),
            "",
            "## Problems",
            "",
        ]
    )
    if report.problems:
        lines.extend(f"- {problem}" for problem in report.problems)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def write_cmapss_promotion_report_markdown(
    report: CmapssPromotionReport,
    output_markdown: str | Path,
) -> Path:
    """Write a markdown promotion-gate report."""

    output_path = Path(output_markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_cmapss_promotion_report_markdown(report), encoding="utf-8")
    return output_path


def render_cmapss_model_card_markdown(
    artifact: CmapssHgbPolicyModelArtifact,
    result: RegressionRunResult,
) -> str:
    """Render a human-readable model card for a packaged C-MAPSS artifact."""

    promotion = artifact.promotion_metadata
    identity = promotion.get("identity", {})
    rollback = promotion.get("rollback", {})
    reference_column_count = len(artifact.reference_stats)
    lines = [
        "# C-MAPSS Deployment Model Card",
        "",
        "## Overview",
        "",
        f"- Dataset: `{_markdown_inline(artifact.dataset)}`",
        f"- Subset: `{_markdown_inline(artifact.subset)}`",
        f"- Model: `{_markdown_inline(artifact.model_name)}`",
        f"- Artifact ID: `{_markdown_inline(promotion.get('artifact_id'))}`",
        f"- Stage: `{_markdown_inline(promotion.get('stage'))}`",
        f"- Schema version: `{_markdown_inline(artifact.schema_version)}`",
        f"- Created at UTC: `{_markdown_inline(promotion.get('created_at_utc'))}`",
        "",
        "## Intended Use",
        "",
        (
            "This artifact predicts capped Remaining Useful Life for NASA C-MAPSS "
            "turbofan telemetry. It is a portfolio deployment candidate for local "
            "batch inference and FastAPI serving, not a certified aviation system."
        ),
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Official-test RMSE | {result.rmse:.6f} |",
        f"| Official-test NASA score | {result.nasa_score:.6f} |",
        f"| Train rows | {result.train_rows} |",
        f"| Train units | {result.train_units} |",
        f"| Test rows | {result.test_rows} |",
        f"| Test units | {result.test_units} |",
        "",
        "## Model And Feature Policy",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Feature policy | `{_markdown_cell(artifact.feature_policy)}` |",
        f"| HGB policy | `{_markdown_cell(artifact.hgb_policy)}` |",
        f"| Rolling window | {artifact.rolling_window} |",
        f"| RUL cap | {artifact.rul_cap} |",
        f"| Standardized | {artifact.standardize} |",
        f"| Random state | {artifact.random_state} |",
        "",
        "## Inference Contract",
        "",
        (
            "`POST /predict` expects raw C-MAPSS rows with the packaged artifact input "
            "schema. Rows are grouped by `unit_number`, and one prediction is returned "
            "per unit using that unit's latest `time_in_cycles` row."
        ),
        "",
        f"- Input columns: {len(artifact.input_columns)}",
        f"- Feature columns after preprocessing: {len(artifact.feature_columns)}",
        f"- Prediction bounds: `[0, {artifact.rul_cap}]`",
        f"- Required first columns: `{_markdown_inline(', '.join(artifact.input_columns[:5]))}`",
        "",
        "## Monitoring",
        "",
        (
            "Serving responses include telemetry mean-shift drift summaries and "
            "prediction-distribution summaries. The artifact stores train-fit reference "
            f"statistics for {reference_column_count} telemetry columns."
        ),
        "",
        "## Promotion",
        "",
        f"- Selection source: `{_markdown_inline(promotion.get('selection_source'))}`",
        f"- Promotion gate: `{_markdown_inline(promotion.get('promotion_gate'))}`",
        f"- Identity official-test RMSE: `{_markdown_inline(identity.get('official_test_rmse'))}`",
        (
            "- Identity official-test NASA score: "
            f"`{_markdown_inline(identity.get('official_test_nasa_score'))}`"
        ),
        "",
        "## Limitations",
        "",
        "- C-MAPSS is simulated benchmark telemetry, not operational fleet data.",
        "- RUL is capped for training and serving, so early-life absolute RUL is compressed.",
        "- This artifact does not provide prediction intervals or certification evidence.",
        "- Public deployment still requires TLS termination, secret rotation, and audit logging.",
        "",
        "## Rollback",
        "",
        f"- Strategy: `{_markdown_inline(rollback.get('strategy'))}`",
        f"- Requires retraining: `{_markdown_inline(rollback.get('requires_retraining'))}`",
        "",
    ]
    return "\n".join(lines)


def write_cmapss_model_card_markdown(
    artifact: CmapssHgbPolicyModelArtifact,
    result: RegressionRunResult,
    output_markdown: str | Path,
) -> Path:
    """Write a model-card markdown artifact for a packaged C-MAPSS model."""

    output_path = Path(output_markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_cmapss_model_card_markdown(artifact, result),
        encoding="utf-8",
    )
    return output_path


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


def _latency_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values must contain at least one latency")
    sorted_values = sorted(values)
    return {
        "min": float(sorted_values[0]),
        "mean": float(sum(sorted_values) / len(sorted_values)),
        "p50": float(_nearest_rank_percentile(sorted_values, 0.50)),
        "p95": float(_nearest_rank_percentile(sorted_values, 0.95)),
        "max": float(sorted_values[-1]),
    }


def _nearest_rank_percentile(sorted_values: list[float], percentile: float) -> float:
    index = max(0, min(len(sorted_values) - 1, int(percentile * len(sorted_values) + 0.999) - 1))
    return sorted_values[index]


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{label} could not be read: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object: {path}")
    return payload


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _artifact_id(identity: dict[str, Any]) -> str | None:
    value = identity.get("artifact_id")
    return str(value) if value is not None else None


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


def _markdown_inline(value: object, *, default: str = "unknown") -> str:
    if value is None:
        return default
    return str(value).replace("`", "'")


def _markdown_cell(value: object) -> str:
    return _markdown_inline(value, default="").replace("|", "\\|")
