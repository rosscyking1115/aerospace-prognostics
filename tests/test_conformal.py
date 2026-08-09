"""Tests for unit-grouped split conformal prediction."""

from __future__ import annotations

import math
import random

import pytest

from aerospace_prognostics.uncertainty.conformal import (
    ConformalCoverageWidth,
    SplitConformalInterval,
    UnitPrediction,
    UnitScore,
    conformal_quantile_rank,
    evaluate_conformal_intervals,
    fit_split_conformal_interval,
    label_only_reference_width,
    minimum_calibration_size,
    require_disjoint_units,
)


def _scores(values: dict[str, float]) -> tuple[UnitScore, ...]:
    return tuple(UnitScore(unit=unit, score=score) for unit, score in values.items())


class TestConformalQuantileRank:
    """The finite-sample rank is the whole reason conformal differs from a quantile."""

    def test_rank_is_ceiling_of_n_plus_one_times_confidence(self) -> None:
        assert conformal_quantile_rank(20, alpha=0.05) == 20
        assert conformal_quantile_rank(20, alpha=0.10) == 19
        assert conformal_quantile_rank(99, alpha=0.01) == 99

    def test_rank_can_exceed_the_calibration_size(self) -> None:
        assert conformal_quantile_rank(20, alpha=0.01) == 21

    def test_rank_rejects_degenerate_inputs(self) -> None:
        with pytest.raises(ValueError):
            conformal_quantile_rank(0, alpha=0.1)
        with pytest.raises(ValueError):
            conformal_quantile_rank(10, alpha=0.0)
        with pytest.raises(ValueError):
            conformal_quantile_rank(10, alpha=1.0)


class TestMinimumCalibrationSize:
    """The attainability threshold is derived, not asserted."""

    @pytest.mark.parametrize(
        ("alpha", "expected"),
        [(0.10, 9), (0.05, 19), (0.02, 49), (0.01, 99)],
    )
    def test_threshold_matches_the_closed_form(self, alpha: float, expected: int) -> None:
        assert minimum_calibration_size(alpha) == expected

    @pytest.mark.parametrize("alpha", [0.10, 0.05, 0.02, 0.01])
    def test_threshold_is_exactly_where_the_rank_becomes_attainable(self, alpha: float) -> None:
        threshold = minimum_calibration_size(alpha)
        assert conformal_quantile_rank(threshold, alpha=alpha) <= threshold
        assert conformal_quantile_rank(threshold - 1, alpha=alpha) > threshold - 1


class TestFitSplitConformalInterval:
    """Calibration is by unit, and the unattainable case is explicit, not silent."""

    def test_radius_is_the_rank_order_statistic(self) -> None:
        interval = fit_split_conformal_interval(
            _scores({f"u{index}": float(index) for index in range(1, 21)}),
            alpha=0.10,
        )
        assert interval.quantile_rank == 19
        assert interval.radius == pytest.approx(19.0)
        assert interval.is_finite is True
        assert interval.calibration_size == 20

    def test_unattainable_confidence_returns_an_infinite_radius(self) -> None:
        interval = fit_split_conformal_interval(
            _scores({f"u{index}": float(index) for index in range(1, 21)}),
            alpha=0.01,
        )
        assert interval.quantile_rank == 21
        assert math.isinf(interval.radius)
        assert interval.is_finite is False
        assert interval.minimum_calibration_size == 99

    def test_duplicate_units_are_rejected(self) -> None:
        duplicated = (UnitScore(unit="u1", score=1.0), UnitScore(unit="u1", score=2.0))
        with pytest.raises(ValueError, match="one calibration score per unit"):
            fit_split_conformal_interval(duplicated, alpha=0.1)

    def test_empty_calibration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            fit_split_conformal_interval((), alpha=0.1)

    def test_negative_scores_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            fit_split_conformal_interval(_scores({"u1": -1.0}), alpha=0.1)

    def test_score_regime_is_recorded_on_the_interval(self) -> None:
        interval = fit_split_conformal_interval(
            _scores({"u1": 1.0, "u2": 2.0}),
            alpha=0.5,
            score_regime="full_trajectory",
        )
        assert interval.score_regime == "full_trajectory"

    def test_unknown_score_regime_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="score_regime"):
            fit_split_conformal_interval(_scores({"u1": 1.0}), alpha=0.5, score_regime="whatever")


class TestRequireDisjointUnits:
    """A unit on both sides of the split is the failure this whole module exists to stop."""

    def test_overlapping_units_raise(self) -> None:
        with pytest.raises(ValueError, match="appear in both"):
            require_disjoint_units(("u1", "u2"), ("u2", "u3"))

    def test_disjoint_units_pass(self) -> None:
        require_disjoint_units(("u1", "u2"), ("u3", "u4"))

    def test_the_overlapping_units_are_named_in_the_error(self) -> None:
        with pytest.raises(ValueError, match="u2, u3"):
            require_disjoint_units(("u1", "u2", "u3"), ("u2", "u3", "u4"))


class TestEvaluateConformalIntervals:
    """Coverage and width are one object, because coverage alone is satisfiable by absence."""

    def test_coverage_and_width_are_reported_together(self) -> None:
        interval = fit_split_conformal_interval(_scores({"c1": 10.0, "c2": 10.0}), alpha=0.5)
        predictions = (
            UnitPrediction(unit="t1", predicted=100.0, actual=105.0),
            UnitPrediction(unit="t2", predicted=100.0, actual=130.0),
        )
        result = evaluate_conformal_intervals(predictions, interval)

        assert isinstance(result, ConformalCoverageWidth)
        assert result.empirical_coverage == pytest.approx(0.5)
        assert result.mean_interval_width == pytest.approx(20.0)
        assert result.covered_points == 1
        assert result.uncovered_points == 1
        assert result.evaluation_units == 2
        assert result.evaluation_points == 2

    def test_calibration_units_may_not_reappear_in_the_evaluation_set(self) -> None:
        interval = fit_split_conformal_interval(_scores({"u1": 1.0, "u2": 1.0}), alpha=0.5)
        predictions = (UnitPrediction(unit="u1", predicted=1.0, actual=1.0),)
        with pytest.raises(ValueError, match="appear in both"):
            evaluate_conformal_intervals(predictions, interval)

    def test_per_unit_coverage_is_reported_for_multi_point_units(self) -> None:
        interval = fit_split_conformal_interval(_scores({"c1": 5.0, "c2": 5.0}), alpha=0.5)
        predictions = (
            UnitPrediction(unit="t1", predicted=0.0, actual=0.0),
            UnitPrediction(unit="t1", predicted=0.0, actual=1.0),
            UnitPrediction(unit="t2", predicted=0.0, actual=0.0),
            UnitPrediction(unit="t2", predicted=0.0, actual=99.0),
        )
        result = evaluate_conformal_intervals(predictions, interval)

        assert result.empirical_coverage == pytest.approx(0.75)
        assert result.worst_unit_coverage == pytest.approx(0.5)
        assert result.mean_unit_coverage == pytest.approx(0.75)

    def test_empty_evaluation_set_is_rejected(self) -> None:
        interval = fit_split_conformal_interval(_scores({"c1": 1.0}), alpha=0.5)
        with pytest.raises(ValueError, match="at least one"):
            evaluate_conformal_intervals((), interval)


class TestBothDirectionControls:
    """Coverage is satisfiable by absence; width alone says nothing. Test the pair."""

    def test_an_infinite_interval_covers_everything_and_is_marked_uninformative(self) -> None:
        interval = fit_split_conformal_interval(_scores({"c1": 1.0, "c2": 2.0}), alpha=0.01)
        predictions = tuple(
            UnitPrediction(unit=f"t{index}", predicted=0.0, actual=float(index))
            for index in range(10)
        )
        result = evaluate_conformal_intervals(predictions, interval)

        assert result.empirical_coverage == pytest.approx(1.0)
        assert math.isinf(result.mean_interval_width)
        assert result.is_informative is False
        assert result.uninformative_reason == "infinite interval width"

    # Both controls use 1000 calibration and 1000 evaluation units. At n=100 the
    # coverage of a single split has a standard deviation near 0.03, so a control
    # asserting a threshold would be deciding on a coin flip: seed 20260809 drew a
    # calibration half whose residuals were milder than its evaluation half (mean
    # 3.63 against 4.16), and a correct implementation returned 0.81 against a
    # nominal 0.90. That is the finite-sample variability the guarantee explicitly
    # allows -- it is a fact about one draw, not a defect -- but a control has to
    # separate right from wrong, so it is given enough units to do so.
    _CONTROL_UNITS = 1000

    def test_a_constant_predictor_reaches_coverage_only_by_being_uninformative(self) -> None:
        rng = random.Random(20260809)
        targets = {
            f"unit{index}": rng.uniform(0.0, 125.0) for index in range(2 * self._CONTROL_UNITS)
        }
        calibration_units = list(targets)[: self._CONTROL_UNITS]
        calibration_targets = [targets[unit] for unit in calibration_units]
        constant = sorted(calibration_targets)[len(calibration_targets) // 2]

        interval = fit_split_conformal_interval(
            _scores({unit: abs(targets[unit] - constant) for unit in calibration_units}),
            alpha=0.10,
        )
        predictions = tuple(
            UnitPrediction(unit=unit, predicted=constant, actual=targets[unit])
            for unit in list(targets)[self._CONTROL_UNITS :]
        )
        reference = label_only_reference_width(calibration_targets, alpha=0.10)
        result = evaluate_conformal_intervals(predictions, interval, reference_width=reference)

        assert result.empirical_coverage >= 0.87
        assert result.is_informative is False
        assert result.uninformative_reason == "no narrower than the label-only reference interval"

    def test_a_known_good_predictor_is_not_rejected(self) -> None:
        rng = random.Random(20260809)
        targets = {
            f"unit{index}": rng.uniform(0.0, 125.0) for index in range(2 * self._CONTROL_UNITS)
        }
        noise = {unit: rng.gauss(0.0, 5.0) for unit in targets}
        calibration_units = list(targets)[: self._CONTROL_UNITS]

        interval = fit_split_conformal_interval(
            _scores({unit: abs(noise[unit]) for unit in calibration_units}),
            alpha=0.10,
        )
        predictions = tuple(
            UnitPrediction(unit=unit, predicted=targets[unit] + noise[unit], actual=targets[unit])
            for unit in list(targets)[self._CONTROL_UNITS :]
        )
        reference = label_only_reference_width(
            [targets[unit] for unit in calibration_units],
            alpha=0.10,
        )
        result = evaluate_conformal_intervals(predictions, interval, reference_width=reference)

        assert result.empirical_coverage >= 0.87
        assert result.is_informative is True
        assert result.uninformative_reason == ""
        assert result.mean_interval_width < 40.0


class TestLabelOnlyReferenceWidth:
    """The reference is the width you get from the labels alone, at the same confidence."""

    def test_reference_is_the_conformal_width_of_the_median_predictor(self) -> None:
        targets = [float(value) for value in range(1, 21)]
        reference = label_only_reference_width(targets, alpha=0.10)
        median = sorted(targets)[len(targets) // 2]
        expected = fit_split_conformal_interval(
            _scores({f"u{index}": abs(value - median) for index, value in enumerate(targets)}),
            alpha=0.10,
        )
        assert reference == pytest.approx(2.0 * expected.radius)

    def test_reference_is_infinite_when_the_confidence_is_unattainable(self) -> None:
        assert math.isinf(label_only_reference_width([1.0, 2.0, 3.0], alpha=0.01))


class TestMarginalCoverageGuarantee:
    """The guarantee is distribution-free but only over repeated exchangeable draws."""

    def test_coverage_holds_on_average_over_exchangeable_resplits(self) -> None:
        rng = random.Random(7)
        alpha = 0.10
        coverages: list[float] = []
        for _ in range(200):
            residuals = [abs(rng.gauss(0.0, 10.0)) for _ in range(60)]
            calibration = _scores(
                {f"c{index}": value for index, value in enumerate(residuals[:40])}
            )
            interval = fit_split_conformal_interval(calibration, alpha=alpha)
            predictions = tuple(
                UnitPrediction(unit=f"t{index}", predicted=0.0, actual=value)
                for index, value in enumerate(residuals[40:])
            )
            coverages.append(evaluate_conformal_intervals(predictions, interval).empirical_coverage)

        mean_coverage = sum(coverages) / len(coverages)
        assert mean_coverage >= 1.0 - alpha


class TestSerialisation:
    """Every reported number has to survive the trip to a committed artifact."""

    def test_interval_round_trips_to_a_flat_dict(self) -> None:
        interval = fit_split_conformal_interval(_scores({"c1": 1.0, "c2": 2.0}), alpha=0.5)
        payload = interval.to_dict()
        assert payload["alpha"] == 0.5
        assert payload["calibration_size"] == 2
        assert isinstance(payload["calibration_units"], str)
        assert isinstance(interval, SplitConformalInterval)

    def test_coverage_width_round_trips_to_a_flat_dict(self) -> None:
        interval = fit_split_conformal_interval(_scores({"c1": 1.0, "c2": 2.0}), alpha=0.5)
        predictions = (UnitPrediction(unit="t1", predicted=0.0, actual=0.5),)
        payload = evaluate_conformal_intervals(predictions, interval).to_dict()
        assert payload["empirical_coverage"] == pytest.approx(1.0)
        assert payload["mean_interval_width"] == pytest.approx(4.0)
        assert "nominal_coverage" in payload
