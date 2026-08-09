"""Split conformal prediction intervals with unit-grouped calibration.

WHAT THIS GUARANTEES
--------------------
Split conformal prediction converts any point predictor into an interval
predictor with a finite-sample, distribution-free marginal coverage guarantee:
for a fresh calibration/test draw, the interval covers the truth with
probability at least ``1 - alpha``. Nothing is assumed about the predictor, the
noise, or the shape of the residual distribution.

EXCHANGEABILITY -- THE ASSUMPTION THAT DECIDES WHETHER ANY OF THIS IS TRUE
--------------------------------------------------------------------------
The guarantee holds if and only if the calibration scores and the test score
are EXCHANGEABLE: their joint distribution is unchanged by permutation. It is
not an assumption about independence, and it is not a technicality. It is the
whole proof. If it fails, the printed coverage number is still a number, but it
no longer estimates anything.

On run-to-failure fleet data such as C-MAPSS, exchangeability fails at the
sample level for two distinct reasons, and only the first is widely noticed:

1. WITHIN-UNIT DEPENDENCE. Consecutive cycles of one engine share the same
   degradation trajectory, the same unit-specific manufacturing offset, and
   very nearly the same sensor noise realisation. Residuals along a trajectory
   are strongly correlated. Splitting rows at random puts cycles of the same
   engine into both calibration and test, so the "test" residual is partly
   predicted by calibration residuals it is not independent of. Coverage then
   looks better than the method can honestly promise, because the calibration
   set has effectively seen the test unit. This module makes that failure
   impossible to reach by accident: calibration is grouped by unit, one score
   per unit, and `require_disjoint_units` raises if a unit appears on both
   sides of the split.

2. COVARIATE SHIFT BETWEEN CALIBRATION AND TEST REGIMES. Grouping by unit is
   necessary and NOT sufficient. C-MAPSS test trajectories are truncated at an
   unknown point before failure and scored once, at that final window. If
   calibration scores are pooled over whole trajectories, they are drawn from a
   different population -- every RUL level, including easy high-RUL cycles far
   from failure -- than the test scores, which come only from final windows.
   Two populations that differ in distribution are not exchangeable with each
   other however carefully they are grouped. `score_regime` records which
   population a calibration set was drawn from so that the mismatch is visible
   in the artifact rather than buried in the code.

Other things that would violate it, none of which this module can detect:
different fault modes between calibration and deployment fleets, a model
retrained between calibration and use, operating-condition drift, and any
selection of the calibration units that depends on their outcomes.

COVERAGE AND WIDTH ARE ONE MEASUREMENT
--------------------------------------
Coverage alone is satisfiable by absence: a predictor emitting infinite
intervals covers 100% of everything and has said nothing. Width alone is
satisfiable by a confidently wrong model. `ConformalCoverageWidth` therefore
carries both, and `evaluate_conformal_intervals` is the only way to obtain
either -- there is deliberately no function in this module that returns a
coverage number on its own.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

SCORE_REGIMES = ("final_window", "full_trajectory")
"""Which population a calibration score was drawn from.

``final_window``   one score per unit, taken at that unit's last observed
                   cycle, matching how C-MAPSS test units are scored.
``full_trajectory`` scores pooled over all cycles of a unit. Diagnostic only:
                   the calibration population then differs from the test
                   population even when the split is unit-grouped.
"""

# ``ceil`` on a product of floats is not safe here. ``20 * (1 - 0.05)`` evaluates
# to 19.000000000000004, whose ceiling is 20 rather than 19 -- which would move the
# attainability threshold by one unit and silently widen every interval at that
# boundary. Rounding to a fixed number of decimals first is exact for the
# confidence levels anyone states in practice, and leaves genuinely fractional
# targets such as 18.9 untouched.
_RANK_DECIMALS = 10


@dataclass(frozen=True)
class UnitScore:
    """One nonconformity score, attributed to the unit that produced it."""

    unit: str
    score: float


@dataclass(frozen=True)
class UnitPrediction:
    """One point prediction and its truth, attributed to a unit."""

    unit: str
    predicted: float
    actual: float


@dataclass(frozen=True)
class SplitConformalInterval:
    """A fitted symmetric split-conformal radius and the split that produced it."""

    alpha: float
    nominal_coverage: float
    calibration_size: int
    quantile_rank: int
    minimum_calibration_size: int
    radius: float
    is_finite: bool
    score_regime: str
    calibration_units: tuple[str, ...]

    @property
    def width(self) -> float:
        """Return the two-sided interval width."""
        return 2.0 * self.radius

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Return a flat serialisable row, with the unit list rendered as text."""
        payload: dict[str, float | int | str | bool] = dict(asdict(self))
        payload["calibration_units"] = ",".join(self.calibration_units)
        payload["width"] = self.width
        return payload


@dataclass(frozen=True)
class ConformalCoverageWidth:
    """Empirical coverage and interval width, with the population they describe.

    These are reported as one object on purpose. A coverage figure quoted
    without its width, or without the population it was measured on, is not
    evidence -- see the module docstring.
    """

    alpha: float
    nominal_coverage: float
    empirical_coverage: float
    evaluation_units: int
    evaluation_points: int
    covered_points: int
    uncovered_points: int
    mean_interval_width: float
    median_interval_width: float
    worst_unit_coverage: float
    mean_unit_coverage: float
    score_regime: str
    calibration_size: int
    is_finite: bool
    is_informative: bool
    uninformative_reason: str
    reference_width: float

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Return a flat serialisable row."""
        return dict(asdict(self))


def conformal_quantile_rank(calibration_size: int, *, alpha: float) -> int:
    """Return the order statistic index used by split conformal prediction.

    The rank is ``ceil((n + 1) * (1 - alpha))``. The ``n + 1`` is what makes the
    guarantee finite-sample rather than asymptotic: it accounts for the test
    point, which is exchangeable with the calibration points and so could have
    landed anywhere among them. A plain empirical quantile of the calibration
    residuals omits it, and undercovers slightly at every sample size.
    """
    _require_valid_alpha(alpha)
    if calibration_size < 1:
        raise ValueError("calibration_size must be at least 1")
    target = round((calibration_size + 1) * (1.0 - alpha), _RANK_DECIMALS)
    return math.ceil(target)


def minimum_calibration_size(alpha: float) -> int:
    """Return the smallest calibration size at which ``1 - alpha`` is attainable.

    The rank is bounded by the calibration size: with ``n`` scores there is no
    ``(n + 1)``-th order statistic, so a rank above ``n`` means the only honest
    interval is an infinite one. Solving ``ceil((n + 1)(1 - alpha)) <= n`` gives
    ``n >= 1/alpha - 1``; at 99% confidence that is 99 calibration units, at 95%
    it is 19. This is a property of the confidence level and the number of
    exchangeable units available, not of any model or of any particular split.
    """
    _require_valid_alpha(alpha)
    candidate = max(1, math.floor(1.0 / alpha) - 2)
    while conformal_quantile_rank(candidate, alpha=alpha) > candidate:
        candidate += 1
    return candidate


def require_disjoint_units(
    calibration_units: Iterable[str],
    evaluation_units: Iterable[str],
) -> None:
    """Raise if any unit appears on both sides of the split.

    A shared unit is the leak described in the module docstring, and it is
    invisible in the results: it inflates coverage rather than throwing. Making
    it an error is the only reliable way to keep it out.
    """
    overlap = sorted(set(calibration_units) & set(evaluation_units))
    if overlap:
        raise ValueError(
            "calibration and evaluation units must be disjoint; these appear in both: "
            + ", ".join(str(unit) for unit in overlap)
        )


def fit_split_conformal_interval(
    scores: Sequence[UnitScore],
    *,
    alpha: float,
    score_regime: str = "final_window",
) -> SplitConformalInterval:
    """Fit a symmetric conformal radius from one nonconformity score per unit.

    Exactly one score per unit is required. Passing several scores from the same
    unit would count one engine's trajectory as several exchangeable draws,
    inflating the effective sample size and shrinking the radius on evidence
    that is not there.
    """
    _require_valid_alpha(alpha)
    if score_regime not in SCORE_REGIMES:
        raise ValueError(f"score_regime must be one of {SCORE_REGIMES}")
    if not scores:
        raise ValueError("at least one calibration score is required")

    units = [str(score.unit) for score in scores]
    duplicates = sorted({unit for unit in units if units.count(unit) > 1})
    if duplicates:
        raise ValueError(
            "split conformal calibration takes one calibration score per unit; "
            "these units appear more than once: " + ", ".join(duplicates)
        )
    values = [float(score.score) for score in scores]
    if any(value < 0.0 for value in values):
        raise ValueError("nonconformity scores must be non-negative")

    calibration_size = len(values)
    rank = conformal_quantile_rank(calibration_size, alpha=alpha)
    radius = math.inf if rank > calibration_size else sorted(values)[rank - 1]

    return SplitConformalInterval(
        alpha=alpha,
        nominal_coverage=1.0 - alpha,
        calibration_size=calibration_size,
        quantile_rank=rank,
        minimum_calibration_size=minimum_calibration_size(alpha),
        radius=radius,
        is_finite=math.isfinite(radius),
        score_regime=score_regime,
        calibration_units=tuple(units),
    )


def evaluate_conformal_intervals(
    predictions: Sequence[UnitPrediction],
    interval: SplitConformalInterval,
    *,
    reference_width: float | None = None,
) -> ConformalCoverageWidth:
    """Measure coverage and width together on a held-out population.

    Args:
        predictions: Held-out point predictions, attributed to their units. No
            unit may also appear in the interval's calibration set.
        interval: The fitted conformal radius to apply.
        reference_width: Width of the interval a label-only predictor achieves
            at the same confidence, from `label_only_reference_width`. Supplying
            it turns "is this informative?" from a judgement into a check: an
            interval no narrower than the one you get from the label
            distribution alone has added nothing to what was already known.
    """
    if not predictions:
        raise ValueError("at least one evaluation prediction is required")
    require_disjoint_units(
        interval.calibration_units,
        (str(prediction.unit) for prediction in predictions),
    )

    radius = interval.radius
    covered_by_unit: dict[str, list[bool]] = {}
    for prediction in predictions:
        error = abs(float(prediction.actual) - float(prediction.predicted))
        covered_by_unit.setdefault(str(prediction.unit), []).append(error <= radius)

    flags = [flag for unit_flags in covered_by_unit.values() for flag in unit_flags]
    unit_coverages = [
        sum(unit_flags) / len(unit_flags) for unit_flags in covered_by_unit.values()
    ]
    width = interval.width
    is_informative, reason = _informativeness(width, reference_width)

    return ConformalCoverageWidth(
        alpha=interval.alpha,
        nominal_coverage=interval.nominal_coverage,
        empirical_coverage=sum(flags) / len(flags),
        evaluation_units=len(covered_by_unit),
        evaluation_points=len(flags),
        covered_points=sum(flags),
        uncovered_points=len(flags) - sum(flags),
        mean_interval_width=width,
        median_interval_width=width,
        worst_unit_coverage=min(unit_coverages),
        mean_unit_coverage=sum(unit_coverages) / len(unit_coverages),
        score_regime=interval.score_regime,
        calibration_size=interval.calibration_size,
        is_finite=interval.is_finite,
        is_informative=is_informative,
        uninformative_reason=reason,
        reference_width=math.inf if reference_width is None else float(reference_width),
    )


def label_only_reference_value(targets: Sequence[float]) -> float:
    """Return the constant a predictor that ignores its inputs would emit.

    Exported so that a caller building an explicit constant-predictor control
    uses the same constant as the reference width it will be compared against.
    Two different median conventions would make the control disagree with its
    own reference by a few cycles, for no reason a reader could follow.
    """
    if not targets:
        raise ValueError("at least one target is required")
    values = sorted(float(target) for target in targets)
    return values[len(values) // 2]


def label_only_reference_width(targets: Sequence[float], *, alpha: float) -> float:
    """Return the conformal interval width of a predictor that ignores its inputs.

    This is the floor any real model has to beat. It applies the same conformal
    machinery to the constant median predictor, so it answers "how wide would
    the interval be if the model contributed nothing at all?" -- the honest
    reference for whether a width is informative, and one that adapts to the
    label distribution instead of hard-coding a threshold.
    """
    values = [float(target) for target in targets]
    median = label_only_reference_value(values)
    interval = fit_split_conformal_interval(
        tuple(
            UnitScore(unit=f"reference_{index}", score=abs(value - median))
            for index, value in enumerate(values)
        ),
        alpha=alpha,
    )
    return interval.width


def _informativeness(width: float, reference_width: float | None) -> tuple[bool, str]:
    if not math.isfinite(width):
        return False, "infinite interval width"
    if reference_width is not None and width >= float(reference_width):
        return False, "no narrower than the label-only reference interval"
    return True, ""


def _require_valid_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
