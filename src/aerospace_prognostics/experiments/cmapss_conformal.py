"""Unit-grouped split conformal prediction study for C-MAPSS RUL.

The study exists to answer one question honestly: does a distribution-free
interval around this repository's C-MAPSS point predictor actually attain its
nominal coverage on the official test set, and at what width?

It runs the same trained model through three calibration designs that differ
only in how the calibration scores are drawn, so the cost of each modelling
shortcut is visible as a number rather than argued about:

``matched_unit_grouped``
    One score per calibration unit, taken at a truncation point drawn at
    random from that unit's life. This is the only design whose split is
    exchangeable in the sense the conformal guarantee requires, and the only
    one whose coverage number should be quoted.
``pooled_within_cap``
    Every calibration cycle inside the training RUL cap treated as its own
    independent draw. Same RUL population as above, so the difference between
    the two isolates the cost of ignoring within-unit dependence.
``pooled_full_trajectory``
    Every calibration cycle, the naive practice. Differs from
    ``pooled_within_cap`` by including the long early-life plateau that no test
    unit is scored on, so the difference between those two isolates the
    calibration-to-test covariate shift.

A constant-median predictor is carried through the identical pipeline as a
control. It should reach nominal coverage and be marked uninformative; a
framework in which it looks like a success is measuring nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aerospace_prognostics.data.cmapss import load_cmapss_subset
from aerospace_prognostics.experiments.cmapss_baseline import (
    CMAPSS_ENGINEERED_DEFAULT_WINDOWS,
    CMAPSS_HGB_PARAM_GRID,
    CMAPSS_VALIDATION_SELECTED_FEATURES,
    CMAPSS_VALIDATION_SELECTED_HGB_PARAMS,
)
from aerospace_prognostics.features import (
    OperatingRegimeFeatureTransformer,
    engineered_feature_frame,
    engineered_last_cycle_feature_table,
)
from aerospace_prognostics.metrics import rmse
from aerospace_prognostics.models.baselines import hist_gradient_boosting_rul
from aerospace_prognostics.preprocessing import FeatureStandardizer
from aerospace_prognostics.uncertainty.conformal import (
    ConformalCoverageWidth,
    SplitConformalInterval,
    UnitPrediction,
    UnitScore,
    evaluate_conformal_intervals,
    fit_split_conformal_interval,
    label_only_reference_value,
    label_only_reference_width,
    minimum_calibration_size,
    require_disjoint_units,
)

if TYPE_CHECKING:
    import pandas as pd

CALIBRATION_DESIGNS = (
    "matched_unit_grouped",
    "pooled_within_cap",
    "pooled_full_trajectory",
)


def unit_id(source: str, unit: int | str) -> str:
    """Return a unit identifier namespaced by the file it came from.

    C-MAPSS numbers train and test engines independently: ``train_FD001`` unit 10
    and ``test_FD001`` unit 10 are different physical engines with unrelated
    trajectories. Without the namespace the disjointness guard reads the two as
    the same unit and refuses every legitimate split -- which is what it did the
    first time this study was run. Namespacing keeps the guard sharp where it
    matters, inside one source, instead of blunting it.
    """
    return f"{source}:{unit}"


@dataclass(frozen=True)
class CalibrationScoreSummary:
    """Shape of the calibration score population a design drew from."""

    design: str
    unit_grouped: bool
    score_regime: str
    score_count: int
    calibration_units: int
    mean_score: float
    median_score: float
    mean_calibration_rul: float
    median_calibration_rul: float

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Return a flat serialisable row."""
        return {
            "design": self.design,
            "unit_grouped": self.unit_grouped,
            "score_regime": self.score_regime,
            "score_count": self.score_count,
            "calibration_units": self.calibration_units,
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "mean_calibration_rul": self.mean_calibration_rul,
            "median_calibration_rul": self.median_calibration_rul,
        }


@dataclass(frozen=True)
class CmapssConformalVariant:
    """One calibration design, its fitted interval, and its measured coverage/width."""

    design: str
    predictor: str
    calibration: CalibrationScoreSummary
    interval: SplitConformalInterval
    evaluation: ConformalCoverageWidth

    def to_row(self) -> dict[str, float | int | str | bool]:
        """Return one flat row joining the design, the interval, and the measurement."""
        return {
            "design": self.design,
            "predictor": self.predictor,
            "unit_grouped": self.calibration.unit_grouped,
            "score_regime": self.calibration.score_regime,
            "calibration_scores": self.calibration.score_count,
            "calibration_units": self.calibration.calibration_units,
            "alpha": self.interval.alpha,
            "nominal_coverage": self.interval.nominal_coverage,
            "quantile_rank": self.interval.quantile_rank,
            "interval_radius": self.interval.radius,
            "interval_width": self.interval.width,
            "empirical_coverage": self.evaluation.empirical_coverage,
            "coverage_gap": (
                self.evaluation.empirical_coverage - self.interval.nominal_coverage
            ),
            "evaluation_units": self.evaluation.evaluation_units,
            "evaluation_points": self.evaluation.evaluation_points,
            "uncovered_points": self.evaluation.uncovered_points,
            "is_finite": self.evaluation.is_finite,
            "is_informative": self.evaluation.is_informative,
            "uninformative_reason": self.evaluation.uninformative_reason,
            "reference_width": self.evaluation.reference_width,
        }


@dataclass(frozen=True)
class CmapssConformalPopulation:
    """Exactly which rows, units, and cycles every number in this run describes."""

    subset: str
    model_name: str
    alpha: float
    rul_cap: int
    random_state: int
    train_units_total: int
    proper_train_units: int
    proper_train_rows: int
    calibration_units: int
    calibration_unit_numbers: tuple[int, ...]
    calibration_cycles_available: int
    evaluation_units: int
    evaluation_rows: int
    evaluation_source: str
    test_rmse: float
    test_units_above_rul_cap: int
    minimum_calibration_units_for_alpha: int
    calibration_units_sufficient: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Return a flat serialisable row, with the unit list rendered as text."""
        return {
            "subset": self.subset,
            "model_name": self.model_name,
            "alpha": self.alpha,
            "nominal_coverage": 1.0 - self.alpha,
            "rul_cap": self.rul_cap,
            "random_state": self.random_state,
            "train_units_total": self.train_units_total,
            "proper_train_units": self.proper_train_units,
            "proper_train_rows": self.proper_train_rows,
            "calibration_units": self.calibration_units,
            "calibration_unit_numbers": ",".join(
                str(unit) for unit in self.calibration_unit_numbers
            ),
            "calibration_cycles_available": self.calibration_cycles_available,
            "evaluation_units": self.evaluation_units,
            "evaluation_rows": self.evaluation_rows,
            "evaluation_source": self.evaluation_source,
            "test_rmse": self.test_rmse,
            "test_units_above_rul_cap": self.test_units_above_rul_cap,
            "minimum_calibration_units_for_alpha": self.minimum_calibration_units_for_alpha,
            "calibration_units_sufficient": self.calibration_units_sufficient,
        }


@dataclass(frozen=True)
class CmapssConformalSubgroup:
    """Coverage and width of the primary interval on a named slice of the test set."""

    subgroup: str
    definition: str
    evaluation: ConformalCoverageWidth

    def to_row(self) -> dict[str, float | int | str | bool]:
        """Return one flat row for the subgroup measurement."""
        return {
            "subgroup": self.subgroup,
            "definition": self.definition,
            "evaluation_units": self.evaluation.evaluation_units,
            "empirical_coverage": self.evaluation.empirical_coverage,
            "nominal_coverage": self.evaluation.nominal_coverage,
            "uncovered_points": self.evaluation.uncovered_points,
            "interval_width": self.evaluation.mean_interval_width,
        }


@dataclass(frozen=True)
class CmapssConformalStudy:
    """Every variant of one conformal run, plus the population they were measured on."""

    population: CmapssConformalPopulation
    variants: tuple[CmapssConformalVariant, ...]
    subgroups: tuple[CmapssConformalSubgroup, ...]
    label_only_reference_width: float

    def to_dict(self) -> dict[str, object]:
        """Return a nested serialisable payload for a JSON artifact."""
        return {
            "population": self.population.to_dict(),
            "label_only_reference_width": self.label_only_reference_width,
            "variants": [variant.to_row() for variant in self.variants],
            "calibration_score_summaries": [
                variant.calibration.to_dict() for variant in self.variants
            ],
            "primary_design_subgroups": [subgroup.to_row() for subgroup in self.subgroups],
        }


def run_cmapss_conformal_study(
    data_dir: str | Path,
    subset: str,
    *,
    alpha: float = 0.10,
    calibration_unit_count: int = 30,
    random_state: int = 42,
    rul_cap: int = 125,
    rolling_window: int | None = None,
    n_regimes: int = 6,
    min_calibration_history: int = 20,
    standardize: bool = True,
) -> CmapssConformalStudy:
    """Fit and evaluate unit-grouped conformal intervals for one C-MAPSS subset.

    The calibration units are held out of training entirely. A unit that
    contributed to the fit is not exchangeable with a test unit, so calibrating
    on training residuals would understate the radius no matter how the split is
    grouped.
    """
    normalised = subset.upper()
    window = rolling_window or CMAPSS_ENGINEERED_DEFAULT_WINDOWS[normalised]
    feature_policy = CMAPSS_VALIDATION_SELECTED_FEATURES[normalised]
    hgb_params = {
        params["label"]: params for params in CMAPSS_HGB_PARAM_GRID
    }[CMAPSS_VALIDATION_SELECTED_HGB_PARAMS[normalised]]

    bundle = load_cmapss_subset(data_dir, normalised, rul_cap=rul_cap)
    train_units = sorted(int(unit) for unit in bundle.train["unit_number"].unique())
    if calibration_unit_count >= len(train_units):
        raise ValueError(
            f"calibration_unit_count must leave units to train on; "
            f"{normalised} has {len(train_units)} training units"
        )
    shuffled = list(train_units)
    random.Random(random_state).shuffle(shuffled)
    calibration_units = tuple(sorted(shuffled[:calibration_unit_count]))
    proper_train_units = tuple(sorted(shuffled[calibration_unit_count:]))
    require_disjoint_units(
        (str(unit) for unit in calibration_units),
        (str(unit) for unit in proper_train_units),
    )

    proper_train = bundle.train.loc[
        bundle.train["unit_number"].isin(proper_train_units)
    ].copy()
    calibration_frame = bundle.train.loc[
        bundle.train["unit_number"].isin(calibration_units)
    ].copy()

    model, standardizer, transformer = _fit_predictor(
        proper_train,
        feature_policy=feature_policy,
        rolling_window=window,
        hgb_params=hgb_params,
        n_regimes=n_regimes,
        random_state=random_state,
        standardize=standardize,
    )

    calibration_features = _feature_frame(
        calibration_frame,
        feature_policy=feature_policy,
        rolling_window=window,
        transformer=transformer,
    )
    calibration_context = calibration_frame.loc[
        calibration_features.index, ["unit_number", "time_in_cycles", "rul"]
    ].copy()
    calibration_context["predicted"] = model.predict(
        standardizer.transform_features(calibration_features)
        if standardizer is not None
        else calibration_features
    )
    calibration_context["absolute_error"] = (
        calibration_context["predicted"] - calibration_context["rul"]
    ).abs()

    test_features = _last_cycle_feature_frame(
        bundle.test,
        feature_policy=feature_policy,
        rolling_window=window,
        transformer=transformer,
    )
    test_predictions = model.predict(
        standardizer.transform_features(test_features)
        if standardizer is not None
        else test_features
    )
    test_units = sorted(int(unit) for unit in bundle.test["unit_number"].unique())
    evaluation = tuple(
        UnitPrediction(
            unit=unit_id("test", unit),
            predicted=float(prediction),
            actual=float(actual),
        )
        for unit, prediction, actual in zip(
            test_units,
            test_predictions,
            bundle.test_rul,
            strict=True,
        )
    )

    matched = _matched_calibration_rows(
        calibration_context,
        rul_cap=rul_cap,
        min_calibration_history=min_calibration_history,
        random_state=random_state,
    )
    within_cap = calibration_context.loc[calibration_context["rul"] <= rul_cap]
    designs: dict[str, pd.DataFrame] = {
        "matched_unit_grouped": matched,
        "pooled_within_cap": within_cap,
        "pooled_full_trajectory": calibration_context,
    }

    reference_width = label_only_reference_width(
        [float(value) for value in matched["rul"]],
        alpha=alpha,
    )

    variants: list[CmapssConformalVariant] = []
    for design, rows in designs.items():
        variants.append(
            _build_variant(
                design=design,
                predictor="hist_gradient_boosting",
                rows=rows,
                calibration_unit_count=len(calibration_units),
                evaluation=evaluation,
                alpha=alpha,
                reference_width=reference_width,
            )
        )
    variants.append(
        _constant_predictor_control(
            matched,
            evaluation=evaluation,
            alpha=alpha,
            reference_width=reference_width,
            calibration_unit_count=len(calibration_units),
        )
    )

    population = CmapssConformalPopulation(
        subset=normalised,
        model_name=_model_name(feature_policy, window, hgb_params, transformer),
        alpha=alpha,
        rul_cap=rul_cap,
        random_state=random_state,
        train_units_total=len(train_units),
        proper_train_units=len(proper_train_units),
        proper_train_rows=len(proper_train),
        calibration_units=len(calibration_units),
        calibration_unit_numbers=calibration_units,
        calibration_cycles_available=len(calibration_context),
        evaluation_units=len(evaluation),
        evaluation_rows=len(evaluation),
        evaluation_source="official C-MAPSS test set, one final-window row per unit",
        test_rmse=rmse(bundle.test_rul, test_predictions),
        test_units_above_rul_cap=int(sum(1 for value in bundle.test_rul if value > rul_cap)),
        minimum_calibration_units_for_alpha=minimum_calibration_size(alpha),
        calibration_units_sufficient=(
            len(calibration_units) >= minimum_calibration_size(alpha)
        ),
    )

    return CmapssConformalStudy(
        population=population,
        variants=tuple(variants),
        subgroups=_primary_design_subgroups(
            variants[0],
            evaluation=evaluation,
            rul_cap=rul_cap,
            reference_width=reference_width,
        ),
        label_only_reference_width=reference_width,
    )


def _primary_design_subgroups(
    primary: CmapssConformalVariant,
    *,
    evaluation: tuple[UnitPrediction, ...],
    rul_cap: int,
    reference_width: float,
) -> tuple[CmapssConformalSubgroup, ...]:
    """Split the primary measurement by whether the truth is inside the model's range.

    A test unit whose true RUL exceeds the training cap is asking the model for a
    number it was trained never to emit. Conformal prediction cannot repair that:
    the interval is centred on a prediction that is capped, so the truth can sit
    outside it however wide the interval is. Reporting the two slices separately
    keeps a marginal coverage figure from hiding a structural miss.
    """
    slices = (
        (
            "within_training_cap",
            f"official test units with true RUL <= {rul_cap}",
            [item for item in evaluation if item.actual <= rul_cap],
        ),
        (
            "above_training_cap",
            f"official test units with true RUL > {rul_cap}",
            [item for item in evaluation if item.actual > rul_cap],
        ),
    )
    return tuple(
        CmapssConformalSubgroup(
            subgroup=name,
            definition=definition,
            evaluation=evaluate_conformal_intervals(
                tuple(items),
                primary.interval,
                reference_width=reference_width,
            ),
        )
        for name, definition, items in slices
        if items
    )


@dataclass(frozen=True)
class CmapssConformalSeedSummary:
    """Coverage and width of one calibration design across repeated unit splits."""

    design: str
    seeds: int
    mean_empirical_coverage: float
    min_empirical_coverage: float
    max_empirical_coverage: float
    mean_interval_width: float
    min_interval_width: float
    max_interval_width: float
    splits_meeting_nominal: int
    mean_coverage_meets_nominal: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Return a flat serialisable row."""
        return {
            "design": self.design,
            "seeds": self.seeds,
            "mean_empirical_coverage": self.mean_empirical_coverage,
            "min_empirical_coverage": self.min_empirical_coverage,
            "max_empirical_coverage": self.max_empirical_coverage,
            "mean_interval_width": self.mean_interval_width,
            "min_interval_width": self.min_interval_width,
            "max_interval_width": self.max_interval_width,
            "splits_meeting_nominal": self.splits_meeting_nominal,
            "mean_coverage_meets_nominal": self.mean_coverage_meets_nominal,
        }


@dataclass(frozen=True)
class CmapssConformalSeedSweep:
    """Repeated-split evidence for a coverage claim that is only true on average."""

    subset: str
    alpha: float
    calibration_unit_count: int
    seeds: tuple[int, ...]
    summaries: tuple[CmapssConformalSeedSummary, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a nested serialisable payload."""
        return {
            "subset": self.subset,
            "alpha": self.alpha,
            "nominal_coverage": 1.0 - self.alpha,
            "calibration_unit_count": self.calibration_unit_count,
            "seeds": list(self.seeds),
            "summaries": [summary.to_dict() for summary in self.summaries],
        }


def run_cmapss_conformal_seed_sweep(
    data_dir: str | Path,
    subset: str,
    *,
    alpha: float = 0.10,
    calibration_unit_count: int = 30,
    seeds: tuple[int, ...] = (11, 23, 42, 57, 71, 89, 103, 127, 151, 173),
    **study_kwargs: object,
) -> CmapssConformalSeedSweep:
    """Re-run the study across unit splits, because the guarantee is an average.

    Conformal coverage is a statement about repeated exchangeable draws, not
    about the split in front of you. A single split can land well above or well
    below nominal without anything being wrong, so a single coverage number is
    not evidence that the method works -- nor, on its own, evidence that it
    failed. Each seed reshuffles which units calibrate and which train, and
    retrains from scratch.
    """
    if not seeds:
        raise ValueError("seeds must contain at least one value")

    per_design: dict[str, list[tuple[float, float]]] = {}
    for seed in seeds:
        study = run_cmapss_conformal_study(
            data_dir,
            subset,
            alpha=alpha,
            calibration_unit_count=calibration_unit_count,
            random_state=seed,
            **study_kwargs,  # type: ignore[arg-type]
        )
        for variant in study.variants:
            per_design.setdefault(variant.design, []).append(
                (
                    variant.evaluation.empirical_coverage,
                    variant.evaluation.mean_interval_width,
                )
            )

    nominal = 1.0 - alpha
    summaries = []
    for design, measurements in per_design.items():
        coverages = [coverage for coverage, _ in measurements]
        widths = [width for _, width in measurements]
        mean_coverage = sum(coverages) / len(coverages)
        summaries.append(
            CmapssConformalSeedSummary(
                design=design,
                seeds=len(measurements),
                mean_empirical_coverage=mean_coverage,
                min_empirical_coverage=min(coverages),
                max_empirical_coverage=max(coverages),
                mean_interval_width=sum(widths) / len(widths),
                min_interval_width=min(widths),
                max_interval_width=max(widths),
                splits_meeting_nominal=sum(1 for value in coverages if value >= nominal),
                mean_coverage_meets_nominal=mean_coverage >= nominal,
            )
        )

    return CmapssConformalSeedSweep(
        subset=subset.upper(),
        alpha=alpha,
        calibration_unit_count=calibration_unit_count,
        seeds=tuple(seeds),
        summaries=tuple(summaries),
    )


def build_attainability_table(
    alphas: tuple[float, ...] = (0.20, 0.10, 0.05, 0.02, 0.01),
    *,
    available_units_by_subset: dict[str, int] | None = None,
) -> list[dict[str, float | int | str | bool]]:
    """Derive, per confidence level, how many calibration units are needed.

    The rank a split conformal interval uses is ``ceil((n + 1)(1 - alpha))``, and
    there is no ``(n + 1)``-th order statistic among ``n`` scores. Requiring the
    rank to be attainable gives ``n >= 1/alpha - 1``. The consequence is a
    property of the dataset: with a fixed fleet, high confidence levels are
    arithmetically out of reach at the unit level however good the model is,
    because calibration units also have to be units the model was not trained on.
    """
    unit_counts = available_units_by_subset or {}
    rows: list[dict[str, float | int | str | bool]] = []
    for alpha in alphas:
        required = minimum_calibration_size(alpha)
        row: dict[str, float | int | str | bool] = {
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "minimum_calibration_units": required,
        }
        for subset, total in sorted(unit_counts.items()):
            # `rank_attainable` is arithmetic only: it says a finite radius exists,
            # not that the result is worth having. At alpha=0.01 FD001 clears it with
            # exactly one training unit left over, which is why the remaining-units
            # column is reported beside it rather than collapsed into a verdict.
            row[f"{subset}_training_units"] = total
            row[f"{subset}_units_left_to_train_on"] = total - required
            row[f"{subset}_rank_attainable"] = total - required >= 1
        rows.append(row)
    return rows


def _build_variant(
    *,
    design: str,
    predictor: str,
    rows: pd.DataFrame,
    calibration_unit_count: int,
    evaluation: tuple[UnitPrediction, ...],
    alpha: float,
    reference_width: float,
    error_column: str = "absolute_error",
) -> CmapssConformalVariant:
    unit_grouped = design == "matched_unit_grouped" or design.startswith("control_")
    score_regime = "final_window" if unit_grouped else "full_trajectory"
    scores = tuple(
        UnitScore(
            unit=(
                unit_id("train", int(row.unit_number))
                if unit_grouped
                # The pooled designs are the naive practice written out honestly:
                # treating each cycle as its own exchangeable draw is exactly the
                # same thing as pretending every cycle came from a different engine.
                else f"{unit_id('train', int(row.unit_number))}@{int(row.time_in_cycles)}"
            ),
            score=float(getattr(row, error_column)),
        )
        for row in rows.itertuples()
    )
    interval = fit_split_conformal_interval(scores, alpha=alpha, score_regime=score_regime)
    summary = CalibrationScoreSummary(
        design=design,
        unit_grouped=unit_grouped,
        score_regime=score_regime,
        score_count=len(scores),
        calibration_units=calibration_unit_count,
        mean_score=float(rows[error_column].mean()),
        median_score=float(rows[error_column].median()),
        mean_calibration_rul=float(rows["rul"].mean()),
        median_calibration_rul=float(rows["rul"].median()),
    )
    return CmapssConformalVariant(
        design=design,
        predictor=predictor,
        calibration=summary,
        interval=interval,
        evaluation=evaluate_conformal_intervals(
            evaluation,
            interval,
            reference_width=reference_width,
        ),
    )


def _constant_predictor_control(
    matched: pd.DataFrame,
    *,
    evaluation: tuple[UnitPrediction, ...],
    alpha: float,
    reference_width: float,
    calibration_unit_count: int,
) -> CmapssConformalVariant:
    """Carry a predictor that ignores its inputs through the identical pipeline.

    This is the control for the direction coverage cannot see. The constant
    median predictor reaches nominal coverage by widening, so a report that
    quoted coverage alone would score it a success. Reported with its width, it
    is marked uninformative.
    """
    constant = label_only_reference_value([float(value) for value in matched["rul"]])
    control_rows = matched.copy()
    control_rows["control_absolute_error"] = (control_rows["rul"] - constant).abs()
    control_evaluation = tuple(
        UnitPrediction(unit=prediction.unit, predicted=constant, actual=prediction.actual)
        for prediction in evaluation
    )
    return _build_variant(
        design="control_constant_predictor",
        predictor=f"constant_median_{constant:g}",
        rows=control_rows,
        calibration_unit_count=calibration_unit_count,
        evaluation=control_evaluation,
        alpha=alpha,
        reference_width=reference_width,
        error_column="control_absolute_error",
    )


def _matched_calibration_rows(
    calibration_context: pd.DataFrame,
    *,
    rul_cap: int,
    min_calibration_history: int,
    random_state: int,
) -> pd.DataFrame:
    """Draw one truncation point per calibration unit, mimicking test construction.

    C-MAPSS test trajectories are truncated at an unknown point before failure
    and scored once, at that final window. The calibration set is built the same
    way: one cycle per unit, drawn uniformly at random from the cycles that
    could plausibly have been a test truncation point.

    Two eligibility rules apply, both stated because both are assumptions rather
    than facts about the data. Cycles whose true RUL exceeds the training cap are
    excluded, because the model's target is capped there and it cannot express a
    larger value -- this is a property of the predictor, not of the test labels.
    Cycles before ``min_calibration_history`` are excluded so that a calibration
    unit has a comparable amount of observed history to a test unit. The
    truncation distribution is uniform by choice; NASA's is unknown, and any
    remaining difference between it and this one is residual covariate shift that
    the reported calibration and test RUL distributions make visible.
    """
    import pandas as pd

    eligible = calibration_context.loc[
        (calibration_context["rul"] <= rul_cap)
        & (calibration_context["time_in_cycles"] >= min_calibration_history)
    ]
    if eligible.empty:
        raise ValueError("no calibration cycles satisfy the truncation eligibility rules")

    rng = random.Random(random_state)
    picked = []
    for unit in sorted(int(unit) for unit in eligible["unit_number"].unique()):
        unit_rows = eligible.loc[eligible["unit_number"] == unit]
        picked.append(unit_rows.iloc[rng.randrange(len(unit_rows))])
    return pd.DataFrame(picked)


def _fit_predictor(
    proper_train: pd.DataFrame,
    *,
    feature_policy: str,
    rolling_window: int,
    hgb_params: dict[str, float | int | str],
    n_regimes: int,
    random_state: int,
    standardize: bool,
) -> tuple[object, FeatureStandardizer | None, OperatingRegimeFeatureTransformer | None]:
    transformer: OperatingRegimeFeatureTransformer | None = None
    if feature_policy == "regime_engineered":
        transformer = OperatingRegimeFeatureTransformer.fit(
            proper_train,
            n_regimes=n_regimes,
            random_state=random_state,
        )
    elif feature_policy != "engineered":
        raise ValueError("feature policy must be 'engineered' or 'regime_engineered'")

    features = _feature_frame(
        proper_train,
        feature_policy=feature_policy,
        rolling_window=rolling_window,
        transformer=transformer,
    )
    target = proper_train.loc[features.index, "rul_capped"].copy()

    standardizer = None
    if standardize:
        standardizer = FeatureStandardizer.fit(
            features,
            feature_columns=list(features.columns),
        )
        features = standardizer.transform_features(features)

    params = {key: value for key, value in hgb_params.items() if key != "label"}
    model = hist_gradient_boosting_rul(random_state=random_state, **params)
    model.fit(features, target)
    return model, standardizer, transformer


def _feature_frame(
    frame: pd.DataFrame,
    *,
    feature_policy: str,
    rolling_window: int,
    transformer: OperatingRegimeFeatureTransformer | None,
) -> pd.DataFrame:
    if transformer is not None:
        return transformer.transform_engineered_frame(frame, rolling_window=rolling_window)
    return engineered_feature_frame(frame, rolling_window=rolling_window)


def _last_cycle_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_policy: str,
    rolling_window: int,
    transformer: OperatingRegimeFeatureTransformer | None,
) -> pd.DataFrame:
    if transformer is not None:
        return transformer.transform_engineered_last_cycle_frame(
            frame,
            rolling_window=rolling_window,
        )
    return engineered_last_cycle_feature_table(frame, rolling_window=rolling_window)


def _model_name(
    feature_policy: str,
    rolling_window: int,
    hgb_params: dict[str, float | int | str],
    transformer: OperatingRegimeFeatureTransformer | None,
) -> str:
    label = hgb_params.get("label", "candidate")
    if transformer is not None:
        return (
            f"hist_gradient_boosting_regime_engineered_w{rolling_window}"
            f"_r{transformer.n_regimes}_{label}"
        )
    return f"hist_gradient_boosting_engineered_w{rolling_window}_{label}"
