"""Tests for the C-MAPSS unit-grouped conformal prediction study."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from aerospace_prognostics.experiments.cmapss_conformal import (
    build_attainability_table,
    run_cmapss_conformal_seed_sweep,
    run_cmapss_conformal_study,
    unit_id,
)
from aerospace_prognostics.reports.cmapss_conformal import write_cmapss_conformal_evidence
from tests.cmapss_fixtures import write_discriminating_cmapss_subset

STUDY_KWARGS = {
    "alpha": 0.20,
    "calibration_unit_count": 4,
    "rul_cap": 30,
    "min_calibration_history": 5,
}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("cmapss_conformal")
    write_discriminating_cmapss_subset(directory)
    return directory


@pytest.fixture(scope="module")
def study(data_dir: Path):
    return run_cmapss_conformal_study(data_dir, "FD001", **STUDY_KWARGS)


class TestCalibrationSplit:
    """Calibration units are held out of training, and never shared with it."""

    def test_calibration_units_are_excluded_from_training(self, study) -> None:
        population = study.population
        assert population.calibration_units == 4
        assert population.proper_train_units == population.train_units_total - 4
        assert population.proper_train_units + population.calibration_units == 12

    def test_calibration_unit_numbers_are_recorded(self, study) -> None:
        assert len(study.population.calibration_unit_numbers) == 4
        assert len(set(study.population.calibration_unit_numbers)) == 4

    def test_a_calibration_set_larger_than_the_fleet_is_rejected(self, data_dir: Path) -> None:
        with pytest.raises(ValueError, match="leave units to train on"):
            run_cmapss_conformal_study(
                data_dir,
                "FD001",
                **{**STUDY_KWARGS, "calibration_unit_count": 12},
            )

    def test_train_and_test_unit_numbers_are_namespaced(self) -> None:
        # C-MAPSS numbers train and test engines independently, so bare numbers
        # would collide and the disjointness guard would reject every split.
        assert unit_id("train", 10) != unit_id("test", 10)


class TestCalibrationDesigns:
    """Each design differs from the next by exactly one modelling shortcut."""

    def test_all_designs_and_the_control_are_reported(self, study) -> None:
        designs = [variant.design for variant in study.variants]
        assert designs == [
            "matched_unit_grouped",
            "pooled_within_cap",
            "pooled_full_trajectory",
            "control_constant_predictor",
        ]

    def test_the_matched_design_draws_one_score_per_unit(self, study) -> None:
        matched = study.variants[0]
        assert matched.calibration.unit_grouped is True
        assert matched.calibration.score_count == study.population.calibration_units
        assert matched.calibration.score_regime == "final_window"

    def test_the_pooled_designs_are_not_unit_grouped(self, study) -> None:
        for variant in study.variants[1:3]:
            assert variant.calibration.unit_grouped is False
            assert variant.calibration.score_count > study.population.calibration_units
            assert variant.calibration.score_regime == "full_trajectory"

    def test_full_trajectory_pooling_draws_from_a_higher_rul_population(self, study) -> None:
        # This is the covariate shift: pooling the whole trajectory pulls in the
        # early-life cycles that no test unit is ever scored on.
        within_cap = study.variants[1].calibration
        full = study.variants[2].calibration
        assert full.score_count > within_cap.score_count
        assert full.mean_calibration_rul > within_cap.mean_calibration_rul


class TestCoverageAndWidthArePaired:
    """No design reports a coverage number without the width beside it."""

    def test_every_variant_row_carries_both(self, study) -> None:
        for variant in study.variants:
            row = variant.to_row()
            assert "empirical_coverage" in row
            assert "interval_width" in row
            assert row["evaluation_units"] == study.population.evaluation_units

    def test_the_constant_predictor_control_is_marked_uninformative(self, study) -> None:
        control = study.variants[-1]
        assert control.design == "control_constant_predictor"
        assert control.evaluation.is_informative is False
        assert control.evaluation.mean_interval_width >= study.label_only_reference_width

    def test_subgroups_split_the_primary_measurement_by_the_training_cap(self, study) -> None:
        names = [subgroup.subgroup for subgroup in study.subgroups]
        assert names
        assert set(names) <= {"within_training_cap", "above_training_cap"}
        total = sum(subgroup.evaluation.evaluation_units for subgroup in study.subgroups)
        assert total == study.population.evaluation_units


class TestAttainability:
    """The confidence threshold is derived from the rank, not asserted."""

    def test_thresholds_follow_the_closed_form(self) -> None:
        rows = {row["alpha"]: row for row in build_attainability_table()}
        assert rows[0.10]["minimum_calibration_units"] == 9
        assert rows[0.05]["minimum_calibration_units"] == 19
        assert rows[0.01]["minimum_calibration_units"] == 99

    def test_fleet_size_is_reported_beside_the_threshold(self) -> None:
        rows = {
            row["alpha"]: row
            for row in build_attainability_table(available_units_by_subset={"FD001": 100})
        }
        assert rows[0.01]["FD001_units_left_to_train_on"] == 1
        assert rows[0.01]["FD001_rank_attainable"] is True

    def test_an_unattainable_confidence_yields_an_infinite_interval(self, data_dir: Path) -> None:
        study = run_cmapss_conformal_study(
            data_dir,
            "FD001",
            **{**STUDY_KWARGS, "alpha": 0.01},
        )
        primary = study.variants[0]
        assert primary.interval.is_finite is False
        assert math.isinf(primary.evaluation.mean_interval_width)
        assert primary.evaluation.empirical_coverage == pytest.approx(1.0)
        assert primary.evaluation.is_informative is False
        assert study.population.calibration_units_sufficient is False


class TestSeedSweep:
    """Coverage is an average over splits, so the evidence has to be too."""

    def test_the_sweep_summarises_every_design_across_seeds(self, data_dir: Path) -> None:
        sweep = run_cmapss_conformal_seed_sweep(
            data_dir,
            "FD001",
            seeds=(11, 42, 71),
            **STUDY_KWARGS,
        )
        assert sweep.seeds == (11, 42, 71)
        assert len(sweep.summaries) == 4
        for summary in sweep.summaries:
            assert summary.seeds == 3
            # Tolerance, not sloppiness: when every split returns the same coverage
            # the mean of three equal floats can land one ulp above their maximum.
            assert summary.min_empirical_coverage <= summary.mean_empirical_coverage + 1e-9
            assert summary.mean_empirical_coverage <= summary.max_empirical_coverage + 1e-9

    def test_an_empty_seed_list_is_rejected(self, data_dir: Path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            run_cmapss_conformal_seed_sweep(data_dir, "FD001", seeds=(), **STUDY_KWARGS)


class TestEvidenceArtifacts:
    """The bundle has to be readable by something other than this process."""

    def test_the_bundle_writes_json_csv_and_markdown(self, study, tmp_path: Path) -> None:
        evidence = write_cmapss_conformal_evidence(study, output_directory=tmp_path)
        assert evidence.json_path.exists()
        assert evidence.variants_csv_path.exists()
        assert evidence.markdown_path.exists()

        payload = json.loads(evidence.json_path.read_text(encoding="utf-8"))
        assert payload["population"]["subset"] == "FD001"
        assert len(payload["variants"]) == 4

    def test_infinite_widths_survive_as_valid_json(self, data_dir: Path, tmp_path: Path) -> None:
        unattainable = run_cmapss_conformal_study(
            data_dir,
            "FD001",
            **{**STUDY_KWARGS, "alpha": 0.01},
        )
        evidence = write_cmapss_conformal_evidence(unattainable, output_directory=tmp_path)
        text = evidence.json_path.read_text(encoding="utf-8")
        assert "Infinity" not in text
        payload = json.loads(text)
        assert payload["variants"][0]["interval_width"] == "inf"

    def test_the_markdown_states_coverage_and_width_in_the_same_table(
        self,
        study,
        tmp_path: Path,
    ) -> None:
        evidence = write_cmapss_conformal_evidence(study, output_directory=tmp_path)
        markdown = evidence.markdown_path.read_text(encoding="utf-8")
        header = next(line for line in markdown.splitlines() if line.startswith("| Design |"))
        assert "Coverage" in header
        assert "Width" in header
