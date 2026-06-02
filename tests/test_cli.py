from __future__ import annotations

import csv
import json
import zipfile
from io import BytesIO

import numpy as np

from aerospace_prognostics.cli import main
from aerospace_prognostics.data.cmapss import read_cmapss_frame
from aerospace_prognostics.evaluation import RegressionRunResult, write_results_csv
from tests.cmapss_fixtures import write_all_tiny_cmapss_subsets, write_tiny_cmapss_subset


def test_cmapss_summary_command_prints_dataset_shape(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)

    exit_code = main(["cmapss-summary", "--data-dir", str(tmp_path), "--subset", "FD001"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "subset=FD001" in output
    assert "train_rows=6 train_units=2" in output
    assert "test_rows=4 test_units=2" in output
    assert "test_rul_values=2" in output


def test_cmapss_baseline_command_writes_json_result(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)
    output_path = tmp_path / "result.json"

    exit_code = main(
        [
            "cmapss-baseline",
            "--data-dir",
            str(tmp_path),
            "--subset",
            "FD001",
            "--output-json",
            str(output_path),
            "--standardize",
        ]
    )

    terminal_output = capsys.readouterr().out
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "model=hist_gradient_boosting" in terminal_output
    assert "standardize=True" in terminal_output
    assert result["dataset"] == "C-MAPSS"
    assert result["subset"] == "FD001"
    assert result["train_rows"] == 6
    assert result["test_units"] == 2
    assert result["standardize"] is True


def test_cmapss_baseline_all_command_writes_result_tables(tmp_path, capsys) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    json_path = tmp_path / "nested" / "results.json"
    csv_path = tmp_path / "nested" / "results.csv"

    exit_code = main(
        [
            "cmapss-baseline-all",
            "--data-dir",
            str(tmp_path),
            "--standardize",
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "subset,model,standardize,rmse,nasa_score" in terminal_output
    assert "FD004,hist_gradient_boosting,True" in terminal_output
    assert json_path.exists()
    assert csv_path.exists()


def test_cmapss_engineered_baseline_all_command_writes_result_tables(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "engineered_results.csv"

    exit_code = main(
        [
            "cmapss-engineered-baseline-all",
            "--data-dir",
            str(tmp_path),
            "--rolling-window",
            "2",
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "FD004,hist_gradient_boosting_engineered_w2,True" in terminal_output
    assert csv_path.exists()


def test_cmapss_engineered_window_sweep_command_writes_result_tables(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "sweep_results.csv"

    exit_code = main(
        [
            "cmapss-engineered-window-sweep",
            "--data-dir",
            str(tmp_path),
            "--rolling-windows",
            "2",
            "3",
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "FD004,hist_gradient_boosting_engineered_w3,True" in terminal_output
    assert csv_path.exists()


def test_cmapss_engineered_best_baseline_all_command_writes_result_tables(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "best_results.csv"

    exit_code = main(
        [
            "cmapss-engineered-best-baseline-all",
            "--data-dir",
            str(tmp_path),
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "rolling_windows=FD001:10,FD002:3,FD003:5,FD004:3" in terminal_output
    assert "FD004,hist_gradient_boosting_engineered_w3,True" in terminal_output
    assert csv_path.exists()


def test_cmapss_regime_engineered_best_baseline_all_command_writes_result_tables(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "regime_results.csv"

    exit_code = main(
        [
            "cmapss-regime-engineered-best-baseline-all",
            "--data-dir",
            str(tmp_path),
            "--n-regimes",
            "2",
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "max_regimes=2" in terminal_output
    assert "FD004,hist_gradient_boosting_regime_engineered_w3_r1,True" in terminal_output
    assert csv_path.exists()


def test_cmapss_validation_selected_baseline_all_command_writes_result_tables(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "validation_selected_results.csv"

    exit_code = main(
        [
            "cmapss-validation-selected-baseline-all",
            "--data-dir",
            str(tmp_path),
            "--n-regimes",
            "2",
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "feature_policy=FD001:regime_engineered,FD002:engineered" in terminal_output
    assert "FD001,hist_gradient_boosting_regime_engineered_w10_r1,True" in terminal_output
    assert "FD002,hist_gradient_boosting_engineered_w3,True" in terminal_output
    assert csv_path.exists()


def test_cmapss_hgb_policy_baseline_all_command_writes_result_tables(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "hgb_policy_results.csv"

    exit_code = main(
        [
            "cmapss-hgb-policy-baseline-all",
            "--data-dir",
            str(tmp_path),
            "--n-regimes",
            "2",
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "hgb_policy=FD001:default,FD002:slow_regularized" in terminal_output
    assert "FD001,hist_gradient_boosting_regime_engineered_w10_r1_default,True" in terminal_output
    assert "FD002,hist_gradient_boosting_engineered_w3_slow_regularized,True" in terminal_output
    assert csv_path.exists()


def test_cmapss_validate_feature_candidates_command_writes_result_tables(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "validation_results.csv"

    exit_code = main(
        [
            "cmapss-validate-feature-candidates",
            "--data-dir",
            str(tmp_path),
            "--subsets",
            "FD001",
            "--validation-horizon",
            "1",
            "--n-regimes",
            "2",
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "validation_horizon=1" in terminal_output
    assert "FD001,hist_gradient_boosting_engineered_w10,True" in terminal_output
    assert "FD001,hist_gradient_boosting_regime_engineered_w10_r1,True" in terminal_output
    assert "selected_by_nasa=FD001:" in terminal_output
    assert csv_path.exists()


def test_cmapss_validate_feature_candidates_repeated_command_writes_aggregate_table(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "validation_aggregate.csv"

    exit_code = main(
        [
            "cmapss-validate-feature-candidates-repeated",
            "--data-dir",
            str(tmp_path),
            "--subsets",
            "FD001",
            "--random-states",
            "1",
            "2",
            "--validation-horizons",
            "1",
            "--n-regimes",
            "2",
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "validation_horizons=1" in terminal_output
    assert "random_states=1,2" in terminal_output
    assert "FD001,hist_gradient_boosting_engineered_w10,True,2" in terminal_output
    assert "selected_by_mean_nasa=FD001:" in terminal_output
    assert csv_path.exists()


def test_cmapss_validate_hgb_grid_command_writes_result_table(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "hgb_grid.csv"

    exit_code = main(
        [
            "cmapss-validate-hgb-grid",
            "--data-dir",
            str(tmp_path),
            "--subsets",
            "FD001",
            "--validation-horizon",
            "1",
            "--n-regimes",
            "2",
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "param_grid=default,slow_regularized,shallow_fast" in terminal_output
    assert "FD001,hist_gradient_boosting_regime_engineered_w10_r1_default,True" in terminal_output
    assert "selected_by_nasa=FD001:" in terminal_output
    assert csv_path.exists()


def test_cmapss_validate_sensor_filters_command_writes_result_table(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    csv_path = tmp_path / "nested" / "sensor_filters.csv"

    exit_code = main(
        [
            "cmapss-validate-sensor-filters",
            "--data-dir",
            str(tmp_path),
            "--subsets",
            "FD001",
            "--validation-horizon",
            "1",
            "--n-regimes",
            "2",
            "--output-csv",
            str(csv_path),
        ]
    )

    terminal_output = capsys.readouterr().out

    assert exit_code == 0
    assert "sensor_filter_candidates=all_sensors,eda_filtered" in terminal_output
    assert "min_abs_standardized_drift=0.2" in terminal_output
    assert "FD001,hist_gradient_boosting_regime_engineered_w10_r1_default,True" in terminal_output
    assert (
        "FD001,hist_gradient_boosting_regime_engineered_w10_r1_default_eda_filtered,True"
        in terminal_output
    )
    assert "selected_by_nasa=FD001:" in terminal_output
    assert csv_path.exists()


def test_cmapss_eda_command_writes_json_report(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)
    output_path = tmp_path / "eda" / "fd001.json"

    exit_code = main(
        [
            "cmapss-eda",
            "--data-dir",
            str(tmp_path),
            "--subset",
            "FD001",
            "--output-json",
            str(output_path),
        ]
    )

    terminal_output = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "subset=FD001" in terminal_output
    assert "largest_abs_drift_sensor=sensor_1" in terminal_output
    assert report["train_rows"] == 6


def test_cmapss_manifest_and_verify_commands(tmp_path, capsys) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    manifest_path = tmp_path / "artifacts" / "manifest.json"

    manifest_exit = main(
        [
            "cmapss-manifest",
            "--data-dir",
            str(tmp_path),
            "--output-json",
            str(manifest_path),
        ]
    )
    manifest_output = capsys.readouterr().out

    verify_exit = main(
        [
            "cmapss-verify",
            "--data-dir",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
        ]
    )
    verify_output = capsys.readouterr().out

    assert manifest_exit == 0
    assert "files=12" in manifest_output
    assert verify_exit == 0
    assert "status=ok" in verify_output


def test_cmapss_verify_command_returns_failure_for_mismatch(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    main(
        [
            "cmapss-manifest",
            "--data-dir",
            str(tmp_path),
            "--subsets",
            "FD001",
            "--output-json",
            str(manifest_path),
        ]
    )
    capsys.readouterr()
    (tmp_path / "RUL_FD001.txt").write_text("999\n", encoding="utf-8")

    exit_code = main(
        [
            "cmapss-verify",
            "--data-dir",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "status=failed" in output
    assert "problem=RUL_FD001.txt has unexpected sha256" in output


def test_phase1_cmapss_command_runs_full_workflow(tmp_path, capsys) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    exit_code = main(
        [
            "phase1-cmapss",
            "--data-dir",
            str(tmp_path),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "eda_reports=4" in output
    assert "hgb_policy_csv=" in output
    assert "sensor_filter_csv=" in output
    assert (artifact_dir / "phase1_summary.md").exists()
    assert (artifact_dir / "results" / "cmapss_hgb_policy_baseline.csv").exists()
    assert (artifact_dir / "results" / "cmapss_validation_sensor_filters.csv").exists()


def test_phase2_cmapss_command_runs_sequence_model_workflow(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact_dir = tmp_path / "phase2"

    exit_code = main(
        [
            "phase2-cmapss",
            "--data-dir",
            str(tmp_path),
            "--artifact-dir",
            str(artifact_dir),
            "--subsets",
            "FD001",
            "--window-size",
            "2",
            "--validation-horizon",
            "1",
            "--n-regimes",
            "1",
            "--models",
            "cnn",
            "tcn",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--training-loss",
            "mse_nasa_blend_w0p001",
            "--hidden-sizes",
            "4",
            "--tcn-levels",
            "1",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "sequence_exports=1" in output
    assert "deep_results=2" in output
    assert "comparison_rows=3" in output
    assert "deep_predictions_csv=" in output
    assert "deep_validation_selection_predictions_csv=" in output
    assert "deep_prediction_diagnostics_csv=" in output
    assert "deep_validation_selection_prediction_diagnostics_csv=" in output
    assert "deep_prediction_rul_bin_diagnostics_csv=" in output
    assert "deep_validation_selection_prediction_rul_bin_diagnostics_csv=" in output
    assert "deep_prediction_monotonicity_diagnostics_csv=" in output
    assert "deep_validation_selection_prediction_monotonicity_diagnostics_csv=" in output
    assert "deep_prediction_unit_diagnostics_csv=" in output
    assert "deep_validation_selection_prediction_unit_diagnostics_csv=" in output
    assert "deep_prediction_diagnostics_markdown=" in output
    assert "deep_validation_selection_prediction_diagnostics_markdown=" in output
    assert "run_manifest=" in output
    assert (artifact_dir / "phase2_summary.md").exists()
    assert (artifact_dir / "phase2_run_manifest.json").exists()
    assert (artifact_dir / "results" / "cmapss_deep_compare.csv").exists()
    assert (artifact_dir / "results" / "cmapss_deep_predictions.csv").exists()
    assert (artifact_dir / "results" / "cmapss_deep_validation_selection_predictions.csv").exists()
    assert (artifact_dir / "results" / "cmapss_deep_prediction_diagnostics.csv").exists()
    assert (
        artifact_dir / "results" / "cmapss_deep_validation_selection_prediction_diagnostics.csv"
    ).exists()
    assert (artifact_dir / "results" / "cmapss_deep_prediction_rul_bins.csv").exists()
    assert (
        artifact_dir / "results" / "cmapss_deep_validation_selection_prediction_rul_bins.csv"
    ).exists()
    assert (artifact_dir / "results" / "cmapss_deep_prediction_monotonicity.csv").exists()
    assert (
        artifact_dir / "results" / "cmapss_deep_validation_selection_prediction_monotonicity.csv"
    ).exists()
    assert (
        artifact_dir / "results" / "cmapss_deep_prediction_unit_diagnostics.csv"
    ).exists()
    assert (
        artifact_dir
        / "results"
        / "cmapss_deep_validation_selection_prediction_unit_diagnostics.csv"
    ).exists()
    assert (artifact_dir / "results" / "cmapss_deep_prediction_diagnostics.md").exists()
    assert (
        artifact_dir / "results" / "cmapss_deep_validation_selection_prediction_diagnostics.md"
    ).exists()
    assert (artifact_dir / "results" / "cmapss_phase2_model_comparison.md").exists()
    manifest = json.loads(
        (artifact_dir / "phase2_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["parameters"]["training_loss"] == "mse_nasa_blend_w0p001"
    summary_markdown = (artifact_dir / "phase2_summary.md").read_text(encoding="utf-8")
    assert "- Training loss: mse_nasa_blend_w0p001" in summary_markdown

    verify_exit_code = main(
        [
            "phase2-cmapss-verify-manifest",
            "--manifest",
            str(artifact_dir / "phase2_run_manifest.json"),
            "--output-markdown",
            str(artifact_dir / "phase2_manifest_audit.md"),
        ]
    )
    verify_output = capsys.readouterr().out

    assert verify_exit_code == 0
    assert "audit_markdown=" in verify_output
    assert "status=ok" in verify_output
    assert "artifacts_checked=25" in verify_output
    audit_markdown = (artifact_dir / "phase2_manifest_audit.md").read_text(encoding="utf-8")
    assert "# Phase 2 C-MAPSS Manifest Audit" in audit_markdown
    assert "- Status: ok" in audit_markdown
    assert "- Training loss: mse_nasa_blend_w0p001" in audit_markdown


def test_cmapss_export_sequences_command_writes_sequence_artifacts(tmp_path, capsys) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    output_dir = tmp_path / "sequence_exports"

    exit_code = main(
        [
            "cmapss-export-sequences",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--subsets",
            "FD001",
            "--window-size",
            "2",
            "--validation-horizon",
            "1",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "window_size=2" in output
    assert (
        "FD001,train_windows=2,validation_windows=1,"
        "validation_selection_windows=1,test_windows=2"
    ) in output
    assert (output_dir / "fd001" / "train_sequences.npz").exists()
    assert (output_dir / "fd001" / "validation_selection_sequences.npz").exists()
    assert (output_dir / "fd001" / "metadata.json").exists()


def test_cmapss_cnn_baseline_command_writes_result_tables(tmp_path, capsys) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    sequence_dir = tmp_path / "sequences"
    output_csv = tmp_path / "results" / "cnn.csv"
    history_json = tmp_path / "results" / "cnn_history.json"
    main(
        [
            "cmapss-export-sequences",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--window-size",
            "2",
            "--validation-horizon",
            "1",
        ]
    )
    capsys.readouterr()

    exit_code = main(
        [
            "cmapss-cnn-baseline",
            "--sequence-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-channels",
            "4",
            "--training-loss",
            "nasa_surrogate",
            "--output-csv",
            str(output_csv),
            "--history-json",
            str(history_json),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "epochs=1" in output
    assert "training_loss=nasa_surrogate" in output
    assert "checkpoint_policy=validation_nasa" in output
    assert "FD001,cnn_1d_w2_e1_c4" in output
    assert "_loss_nasa_surrogate_" in output
    assert "selected_epochs=FD001:1" in output
    assert output_csv.exists()
    assert history_json.exists()


def test_cmapss_lstm_baseline_command_writes_result_tables(tmp_path, capsys) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    sequence_dir = tmp_path / "sequences"
    output_csv = tmp_path / "results" / "lstm.csv"
    history_json = tmp_path / "results" / "lstm_history.json"
    main(
        [
            "cmapss-export-sequences",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--window-size",
            "2",
            "--validation-horizon",
            "1",
        ]
    )
    capsys.readouterr()

    exit_code = main(
        [
            "cmapss-lstm-baseline",
            "--sequence-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-size",
            "4",
            "--bidirectional",
            "--output-csv",
            str(output_csv),
            "--history-json",
            str(history_json),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "epochs=1" in output
    assert "bidirectional=True" in output
    assert "checkpoint_policy=validation_nasa" in output
    assert "FD001,bilstm_w2_e1_h4_l1" in output
    assert "selected_epochs=FD001:1" in output
    assert output_csv.exists()
    assert history_json.exists()


def test_cmapss_tcn_baseline_command_writes_result_tables(tmp_path, capsys) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    sequence_dir = tmp_path / "sequences"
    output_csv = tmp_path / "results" / "tcn.csv"
    history_json = tmp_path / "results" / "tcn_history.json"
    main(
        [
            "cmapss-export-sequences",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--window-size",
            "2",
            "--validation-horizon",
            "1",
        ]
    )
    capsys.readouterr()

    exit_code = main(
        [
            "cmapss-tcn-baseline",
            "--sequence-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-channels",
            "4",
            "--num-levels",
            "1",
            "--output-csv",
            str(output_csv),
            "--history-json",
            str(history_json),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "epochs=1" in output
    assert "num_levels=1" in output
    assert "checkpoint_policy=validation_nasa" in output
    assert "FD001,tcn_w2_e1_c4_l1_k3" in output
    assert "selected_epochs=FD001:1" in output
    assert output_csv.exists()
    assert history_json.exists()


def test_cmapss_transformer_baseline_command_writes_result_tables(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    sequence_dir = tmp_path / "sequences"
    output_csv = tmp_path / "results" / "transformer.csv"
    history_json = tmp_path / "results" / "transformer_history.json"
    main(
        [
            "cmapss-export-sequences",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--window-size",
            "2",
            "--validation-horizon",
            "1",
        ]
    )
    capsys.readouterr()

    exit_code = main(
        [
            "cmapss-transformer-baseline",
            "--sequence-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--d-model",
            "4",
            "--num-heads",
            "2",
            "--num-layers",
            "1",
            "--dim-feedforward",
            "8",
            "--output-csv",
            str(output_csv),
            "--history-json",
            str(history_json),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "epochs=1" in output
    assert "d_model=4" in output
    assert "num_heads=2" in output
    assert "checkpoint_policy=validation_nasa" in output
    assert "FD001,transformer_w2_e1_d4_h2_l1_ff8" in output
    assert "selected_epochs=FD001:1" in output
    assert output_csv.exists()
    assert history_json.exists()


def test_cmapss_deep_baseline_compare_command_writes_result_tables(
    tmp_path,
    capsys,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    sequence_dir = tmp_path / "sequences"
    output_csv = tmp_path / "results" / "deep_compare.csv"
    main(
        [
            "cmapss-export-sequences",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--window-size",
            "2",
            "--validation-horizon",
            "1",
        ]
    )
    capsys.readouterr()

    exit_code = main(
        [
            "cmapss-deep-baseline-compare",
            "--sequence-dir",
            str(sequence_dir),
            "--subsets",
            "FD001",
            "--models",
            "cnn",
            "tcn",
            "transformer",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-sizes",
            "4",
            "--tcn-levels",
            "1",
            "--tcn-normalization",
            "layer_norm",
            "--tcn-weight-norm",
            "--tcn-pooling",
            "mean",
            "--transformer-heads",
            "2",
            "--transformer-dim-feedforward",
            "8",
            "--output-csv",
            str(output_csv),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "models=cnn,tcn,transformer" in output
    assert "hidden_sizes=4" in output
    assert "tcn_normalization=layer_norm" in output
    assert "tcn_weight_norm=True" in output
    assert "tcn_pooling=mean" in output
    assert "compare_cnn_h4_lr0p001" in output
    assert "compare_tcn_h4_lr0p001" in output
    assert "normlayer_norm_wn_poolmean" in output
    assert "compare_transformer_h4_lr0p001" in output
    assert "selected_by_nasa=FD001:" in output
    assert output_csv.exists()


def test_cmapss_calibrate_deep_predictions_command_writes_reports(
    tmp_path,
    capsys,
) -> None:
    calibration_csv = tmp_path / "results" / "validation_predictions.csv"
    predictions_csv = tmp_path / "results" / "official_predictions.csv"
    output_csv = tmp_path / "reports" / "official_predictions_calibrated.csv"
    output_calibration_csv = tmp_path / "reports" / "calibration.csv"
    output_diagnostics_csv = tmp_path / "reports" / "diagnostics.csv"
    output_rul_bins_csv = tmp_path / "reports" / "diagnostics_by_rul_bin.csv"
    output_unit_diagnostics_csv = tmp_path / "reports" / "diagnostics_by_unit.csv"
    output_markdown = tmp_path / "reports" / "diagnostics.md"
    _write_cli_predictions(
        calibration_csv,
        [
            _cli_prediction("FD001", "transformer", 1, 10.0, 5.0),
            _cli_prediction("FD001", "transformer", 2, 30.0, 15.0),
        ],
    )
    _write_cli_predictions(
        predictions_csv,
        [
            _cli_prediction("FD001", "transformer", 3, 20.0, 8.0),
        ],
    )

    exit_code = main(
        [
            "cmapss-calibrate-deep-predictions",
            "--calibration-csv",
            str(calibration_csv),
            "--predictions-csv",
            str(predictions_csv),
            "--output-csv",
            str(output_csv),
            "--output-calibration-csv",
            str(output_calibration_csv),
            "--output-diagnostics-csv",
            str(output_diagnostics_csv),
            "--output-rul-bins-csv",
            str(output_rul_bins_csv),
            "--output-unit-diagnostics-csv",
            str(output_unit_diagnostics_csv),
            "--output-markdown",
            str(output_markdown),
        ]
    )

    output = capsys.readouterr().out
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))
    assert exit_code == 0
    assert "calibration_groups=1" in output
    assert "calibrated_prediction_rows=1" in output
    assert "calibration=FD001:transformer:rows=2:intercept=0:slope=2" in output
    assert "calibration_csv=" in output
    assert "unit_diagnostics_csv=" in output
    assert rows[0]["calibration_method"] == "validation_affine"
    assert float(rows[0]["predicted_rul"]) == 16.0
    assert output_calibration_csv.exists()
    assert output_diagnostics_csv.exists()
    assert output_rul_bins_csv.exists()
    assert output_unit_diagnostics_csv.exists()
    assert output_markdown.exists()


def test_cmapss_calibrate_deep_predictions_command_supports_predicted_bin_residual(
    tmp_path,
    capsys,
) -> None:
    calibration_csv = tmp_path / "results" / "validation_predictions.csv"
    predictions_csv = tmp_path / "results" / "official_predictions.csv"
    output_csv = tmp_path / "reports" / "official_predictions_calibrated.csv"
    output_calibration_csv = tmp_path / "reports" / "calibration.csv"
    _write_cli_predictions(
        calibration_csv,
        [
            _cli_prediction("FD001", "transformer", 1, 10.0, 20.0),
            _cli_prediction("FD001", "transformer", 2, 55.0, 45.0),
            _cli_prediction("FD001", "transformer", 3, 65.0, 80.0),
        ],
    )
    _write_cli_predictions(
        predictions_csv,
        [
            _cli_prediction("FD001", "transformer", 4, 30.0, 45.0),
        ],
    )

    exit_code = main(
        [
            "cmapss-calibrate-deep-predictions",
            "--method",
            "predicted_bin_residual",
            "--shrinkage-strength",
            "0",
            "--calibration-csv",
            str(calibration_csv),
            "--predictions-csv",
            str(predictions_csv),
            "--output-csv",
            str(output_csv),
            "--output-calibration-csv",
            str(output_calibration_csv),
        ]
    )

    output = capsys.readouterr().out
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))
    assert exit_code == 0
    assert "calibration_method=predicted_bin_residual" in output
    assert "bin=31-60" in output
    assert rows[0]["calibration_method"] == "validation_predicted_bin_residual"
    assert rows[0]["calibration_predicted_rul_bin"] == "31-60"
    assert float(rows[0]["predicted_rul"]) == 55.0
    assert output_calibration_csv.exists()


def test_cmapss_calibrate_deep_predictions_command_supports_nasa_shift(
    tmp_path,
    capsys,
) -> None:
    calibration_csv = tmp_path / "results" / "validation_predictions.csv"
    predictions_csv = tmp_path / "results" / "official_predictions.csv"
    output_csv = tmp_path / "reports" / "official_predictions_calibrated.csv"
    output_calibration_csv = tmp_path / "reports" / "calibration.csv"
    _write_cli_predictions(
        calibration_csv,
        [
            _cli_prediction("FD001", "transformer", 1, 30.0, 45.0),
            _cli_prediction("FD001", "transformer", 2, 35.0, 45.0),
            _cli_prediction("FD001", "transformer", 3, 40.0, 45.0),
        ],
    )
    _write_cli_predictions(
        predictions_csv,
        [
            _cli_prediction("FD001", "transformer", 4, 30.0, 45.0),
        ],
    )

    exit_code = main(
        [
            "cmapss-calibrate-deep-predictions",
            "--method",
            "predicted_bin_nasa_shift",
            "--shrinkage-strength",
            "0",
            "--nasa-shift-max",
            "20",
            "--nasa-shift-step",
            "5",
            "--calibration-csv",
            str(calibration_csv),
            "--predictions-csv",
            str(predictions_csv),
            "--output-csv",
            str(output_csv),
            "--output-calibration-csv",
            str(output_calibration_csv),
        ]
    )

    output = capsys.readouterr().out
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))
    assert exit_code == 0
    assert "calibration_method=predicted_bin_nasa_shift" in output
    assert "bin=31-60" in output
    assert rows[0]["calibration_method"] == "validation_predicted_bin_nasa_shift"
    assert rows[0]["calibration_predicted_rul_bin"] == "31-60"
    assert float(rows[0]["predicted_rul"]) < 45.0
    assert output_calibration_csv.exists()


def test_cmapss_compare_rul_results_command_writes_report_tables(
    tmp_path,
    capsys,
) -> None:
    baseline_csv = tmp_path / "results" / "hgb.csv"
    candidate_csv = tmp_path / "results" / "deep.csv"
    prediction_csv = tmp_path / "results" / "calibrated_predictions.csv"
    output_csv = tmp_path / "reports" / "comparison.csv"
    output_markdown = tmp_path / "reports" / "comparison.md"
    write_results_csv(
        [_cli_result("FD001", "hist_gradient_boosting_policy", 13.0, 250.0)],
        baseline_csv,
    )
    write_results_csv(
        [
            _cli_result("FD001", "cnn", 22.0, 400.0),
            _cli_result("FD001", "tcn", 12.5, 240.0),
        ],
        candidate_csv,
    )
    _write_cli_predictions(
        prediction_csv,
        [
            _cli_prediction("FD001", "transformer", 1, 100.0, 95.0),
            _cli_prediction("FD001", "transformer", 2, 120.0, 119.0),
        ],
    )

    exit_code = main(
        [
            "cmapss-compare-rul-results",
            "--baseline-csv",
            str(baseline_csv),
            "--candidate-csv",
            str(candidate_csv),
            "--prediction-csv",
            str(prediction_csv),
            "--prediction-model-suffixes",
            "calibrated",
            "--output-csv",
            str(output_csv),
            "--output-markdown",
            str(output_markdown),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "rows=4" in output
    assert "subsets=FD001" in output
    assert "best_by_nasa=FD001:phase2_predictions:transformer_calibrated" in output
    assert "FD001,1,phase2_predictions,transformer_calibrated" in output
    assert output_csv.exists()
    assert output_markdown.exists()


def test_telemetry_robust_zscore_baseline_command_writes_outputs(tmp_path, capsys) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    output_json = tmp_path / "results" / "anomaly.json"
    predictions_csv = tmp_path / "results" / "predictions.csv"
    train_csv.write_text(
        "\n".join(
            [
                "timestamp,bus_voltage,thermal_zone",
                "0,0.0,10.0",
                "1,0.1,10.1",
                "2,-0.1,9.9",
                "3,0.0,10.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    test_csv.write_text(
        "\n".join(
            [
                "timestamp,bus_voltage,thermal_zone,label",
                "0,0.0,10.0,0",
                "1,8.0,10.0,1",
                "2,0.0,25.0,1",
                "3,0.1,10.1,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "telemetry-robust-zscore-baseline",
            "--train-csv",
            str(train_csv),
            "--test-csv",
            str(test_csv),
            "--label-column",
            "label",
            "--feature-columns",
            "bus_voltage",
            "thermal_zone",
            "--output-json",
            str(output_json),
            "--predictions-csv",
            str(predictions_csv),
        ]
    )
    output = capsys.readouterr().out
    result = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "features=2" in output
    assert "test_rows=4" in output
    assert "f1=1.000000" in output
    assert "point_adjusted_f1=1.000000" in output
    assert result["metrics"]["true_positives"] == 2
    assert result["predictions"] == [0, 1, 1, 0]
    assert predictions_csv.exists()


def test_telemetry_classical_anomaly_baselines_command_writes_comparison_outputs(
    tmp_path,
    capsys,
) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    output_json = tmp_path / "results" / "classical.json"
    output_csv = tmp_path / "results" / "classical.csv"
    predictions_csv = tmp_path / "results" / "classical_predictions.csv"
    train_csv.write_text(
        "\n".join(
            [
                "timestamp,bus_voltage,thermal_zone",
                "0,-0.2,-0.2",
                "1,-0.1,-0.1",
                "2,0.0,0.0",
                "3,0.1,0.1",
                "4,0.2,0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    test_csv.write_text(
        "\n".join(
            [
                "timestamp,bus_voltage,thermal_zone,label",
                "0,0.0,0.0,0",
                "1,8.0,-8.0,1",
                "2,0.1,0.1,0",
                "3,-7.0,7.0,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "telemetry-classical-anomaly-baselines",
            "--train-csv",
            str(train_csv),
            "--test-csv",
            str(test_csv),
            "--label-column",
            "label",
            "--feature-columns",
            "bus_voltage",
            "thermal_zone",
            "--pca-components",
            "1",
            "--pca-threshold-quantile",
            "0.95",
            "--isolation-contamination",
            "0.2",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--predictions-csv",
            str(predictions_csv),
        ]
    )
    output = capsys.readouterr().out
    result = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "model,precision,recall,f1,point_adjusted_f1,false_alarm_rate" in output
    assert "robust_zscore," in output
    assert "pca_reconstruction," in output
    assert "isolation_forest," in output
    assert [row["model_name"] for row in result] == [
        "robust_zscore",
        "pca_reconstruction",
        "isolation_forest",
    ]
    assert output_csv.exists()
    assert predictions_csv.exists()


def test_smap_msl_summary_command_prints_dataset_counts(tmp_path, capsys) -> None:
    _write_cli_smap_msl_channel(tmp_path)

    exit_code = main(["smap-msl-summary", "--data-dir", str(tmp_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "channels=1" in output
    assert "anomaly_sequences=1" in output
    assert "spacecraft=SMAP:1" in output


def test_smap_msl_select_channels_command_writes_benchmark_list(tmp_path, capsys) -> None:
    _write_cli_smap_msl_selection_fixture(tmp_path)
    output_json = tmp_path / "selection" / "channels.json"
    output_csv = tmp_path / "selection" / "channels.csv"

    exit_code = main(
        [
            "smap-msl-select-channels",
            "--data-dir",
            str(tmp_path),
            "--count",
            "3",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "selected_channels=3" in output
    assert "channels=M-1 P-1 M-2" in output
    assert "rank,channel_id,spacecraft,anomaly_sequences" in output
    assert [row["channel_id"] for row in payload] == ["M-1", "P-1", "M-2"]
    assert output_csv.exists()


def test_smap_msl_export_channel_csv_command_writes_baseline_inputs(tmp_path, capsys) -> None:
    _write_cli_smap_msl_channel(tmp_path)
    output_dir = tmp_path / "exports"
    metadata_json = tmp_path / "exports" / "p1.json"

    exit_code = main(
        [
            "smap-msl-export-channel-csv",
            "--data-dir",
            str(tmp_path),
            "--channel-id",
            "P-1",
            "--output-dir",
            str(output_dir),
            "--metadata-json",
            str(metadata_json),
        ]
    )
    output = capsys.readouterr().out
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "channel_id=P-1" in output
    assert "train_rows=4" in output
    assert "test_rows=5" in output
    assert (output_dir / "P-1" / "train.csv").exists()
    assert (output_dir / "P-1" / "test.csv").exists()
    assert metadata["feature_names"] == ["feature_0", "feature_1"]


def test_smap_msl_classical_baselines_command_writes_outputs(tmp_path, capsys) -> None:
    _write_cli_smap_msl_channel(tmp_path)
    output_json = tmp_path / "results" / "smap.json"
    output_csv = tmp_path / "results" / "smap.csv"

    exit_code = main(
        [
            "smap-msl-classical-baselines",
            "--data-dir",
            str(tmp_path),
            "--channels",
            "P-1",
            "--methods",
            "robust_zscore",
            "pca_reconstruction",
            "--pca-components",
            "1",
            "--pca-threshold-quantile",
            "0.95",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "channels=1" in output
    assert "runs=2" in output
    assert "channel_id,spacecraft,model,precision,recall,f1" in output
    assert "P-1,SMAP,robust_zscore" in output
    assert "P-1,SMAP,pca_reconstruction" in output
    assert [row["model_name"] for row in payload] == ["robust_zscore", "pca_reconstruction"]
    assert output_csv.exists()


def test_smap_msl_robust_threshold_sweep_command_writes_outputs(tmp_path, capsys) -> None:
    _write_cli_smap_msl_channel(tmp_path)
    output_csv = tmp_path / "results" / "robust_sweep.csv"
    aggregate_csv = tmp_path / "results" / "robust_sweep_aggregate.csv"
    operating_point_csv = tmp_path / "results" / "robust_operating_points.csv"
    policy_csv = tmp_path / "results" / "robust_policy.csv"

    exit_code = main(
        [
            "smap-msl-robust-threshold-sweep",
            "--data-dir",
            str(tmp_path),
            "--channels",
            "P-1",
            "--thresholds",
            "2.0",
            "4.0",
            "--output-csv",
            str(output_csv),
            "--aggregate-csv",
            str(aggregate_csv),
            "--false-alarm-budget",
            "1.0",
            "--selection-group",
            "global",
            "--operating-point-csv",
            str(operating_point_csv),
            "--policy-csv",
            str(policy_csv),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "channels=1" in output
    assert "thresholds=2,4" in output
    assert "runs=2" in output
    assert "threshold,channels,wins_by_f1" in output
    assert "false_alarm_budget=1" in output
    assert "policy_runs=1" in output
    assert "scope,group,false_alarm_budget,selected_threshold" in output
    assert output_csv.exists()
    assert aggregate_csv.exists()
    assert operating_point_csv.exists()
    assert policy_csv.exists()


def test_smap_msl_lstm_forecast_baseline_command_writes_outputs(tmp_path, capsys) -> None:
    _write_cli_smap_msl_channel(tmp_path)
    output_json = tmp_path / "results" / "smap_lstm.json"
    output_csv = tmp_path / "results" / "smap_lstm.csv"

    exit_code = main(
        [
            "smap-msl-lstm-forecast-baseline",
            "--data-dir",
            str(tmp_path),
            "--channels",
            "P-1",
            "--window-size",
            "2",
            "--hidden-size",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "channels=1" in output
    assert "runs=1" in output
    assert "channel_id,spacecraft,model,epochs,final_train_loss" in output
    assert "P-1,SMAP,lstm_forecast_robust_threshold" in output
    assert payload[0]["model_name"] == "lstm_forecast_robust_threshold"
    assert payload[0]["history"][0]["epoch"] == 1
    assert output_csv.exists()


def test_smap_msl_lstm_forecast_baseline_command_supports_dynamic_threshold(
    tmp_path,
    capsys,
) -> None:
    _write_cli_smap_msl_channel(tmp_path)

    exit_code = main(
        [
            "smap-msl-lstm-forecast-baseline",
            "--data-dir",
            str(tmp_path),
            "--channels",
            "P-1",
            "--window-size",
            "2",
            "--hidden-size",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--threshold-method",
            "dynamic",
            "--dynamic-batch-size",
            "2",
            "--dynamic-window-batches",
            "1",
            "--dynamic-error-buffer",
            "0",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "P-1,SMAP,lstm_forecast_dynamic_threshold" in output


def test_phase2_smap_msl_command_runs_anomaly_workflow(tmp_path, capsys) -> None:
    _write_cli_smap_msl_channel(tmp_path)
    artifact_dir = tmp_path / "phase2_smap_msl"

    exit_code = main(
        [
            "phase2-smap-msl",
            "--data-dir",
            str(tmp_path),
            "--artifact-dir",
            str(artifact_dir),
            "--channels",
            "P-1",
            "--window-size",
            "2",
            "--hidden-size",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--classical-methods",
            "robust_zscore",
            "pca_reconstruction",
            "--pca-components",
            "1",
            "--pca-threshold-quantile",
            "0.95",
            "--dynamic-batch-size",
            "2",
            "--dynamic-window-batches",
            "1",
            "--dynamic-error-buffer",
            "0",
            "--robust-policy-false-alarm-budget",
            "1.0",
            "--robust-policy-thresholds",
            "2.0",
            "4.0",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "classical_runs=2" in output
    assert "lstm_robust_runs=1" in output
    assert "lstm_dynamic_runs=1" in output
    assert "robust_threshold_policy_runs=1" in output
    assert "comparison_rows=5" in output
    assert "robust_threshold_policy_csv=" in output
    assert "run_manifest=" in output
    assert (artifact_dir / "results" / "smap_msl_anomaly_model_comparison.csv").exists()
    assert (artifact_dir / "results" / "smap_msl_anomaly_model_comparison.md").exists()
    assert (artifact_dir / "results" / "smap_msl_robust_threshold_policy.csv").exists()
    assert (artifact_dir / "phase2_smap_msl_summary.md").exists()
    assert (artifact_dir / "phase2_smap_msl_run_manifest.json").exists()

    verify_exit_code = main(
        [
            "phase2-smap-msl-verify-manifest",
            "--manifest",
            str(artifact_dir / "phase2_smap_msl_run_manifest.json"),
            "--output-markdown",
            str(artifact_dir / "phase2_smap_msl_manifest_audit.md"),
        ]
    )
    verify_output = capsys.readouterr().out

    assert verify_exit_code == 0
    assert "audit_markdown=" in verify_output
    assert "status=ok" in verify_output
    assert "artifacts_checked=18" in verify_output
    audit_markdown = (artifact_dir / "phase2_smap_msl_manifest_audit.md").read_text(
        encoding="utf-8"
    )
    assert "# Phase 2 SMAP/MSL Manifest Audit" in audit_markdown
    assert "- Status: ok" in audit_markdown


def test_smap_msl_compare_anomaly_results_command_writes_report_tables(
    tmp_path,
    capsys,
) -> None:
    classical_csv = tmp_path / "results" / "classical.csv"
    forecast_csv = tmp_path / "results" / "forecast.csv"
    output_csv = tmp_path / "reports" / "comparison.csv"
    output_markdown = tmp_path / "reports" / "comparison.md"
    _write_cli_anomaly_result_csv(
        classical_csv,
        [
            _cli_anomaly_result_row("P-1", "SMAP", "robust_zscore", f1=0.40),
            _cli_anomaly_result_row("P-1", "SMAP", "pca_reconstruction", f1=0.60),
        ],
    )
    _write_cli_anomaly_result_csv(
        forecast_csv,
        [_cli_anomaly_result_row("P-1", "SMAP", "lstm_forecast_dynamic_threshold", f1=0.55)],
    )

    exit_code = main(
        [
            "smap-msl-compare-anomaly-results",
            "--result-csv",
            str(classical_csv),
            str(forecast_csv),
            "--source-labels",
            "classical",
            "lstm",
            "--output-csv",
            str(output_csv),
            "--output-markdown",
            str(output_markdown),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "channels=1" in output
    assert "rows=3" in output
    assert "P-1,SMAP,1,classical,pca_reconstruction" in output
    assert "P-1,SMAP,2,lstm,lstm_forecast_dynamic_threshold" in output
    assert output_csv.exists()
    assert "# Telemetry Anomaly Model Comparison" in output_markdown.read_text(encoding="utf-8")


def test_cmapss_package_and_predict_artifact_commands(tmp_path, capsys) -> None:
    write_tiny_cmapss_subset(tmp_path)
    artifact_path = tmp_path / "models" / "fd001.joblib"
    metadata_json = tmp_path / "models" / "fd001_metadata.json"
    model_card_markdown = tmp_path / "models" / "fd001_model_card.md"
    prediction_json = tmp_path / "predictions" / "fd001.json"
    benchmark_json = tmp_path / "models" / "fd001_benchmark.json"
    promotion_json = tmp_path / "models" / "fd001_promotion.json"
    promotion_markdown = tmp_path / "models" / "fd001_promotion.md"
    sbom_json = tmp_path / "sbom" / "cyclonedx.json"
    input_csv = tmp_path / "test_input.csv"
    read_cmapss_frame(tmp_path / "test_FD001.txt").to_csv(input_csv, index=False)

    package_exit_code = main(
        [
            "cmapss-package-hgb-policy",
            "--data-dir",
            str(tmp_path),
            "--subset",
            "FD001",
            "--output-path",
            str(artifact_path),
            "--metadata-json",
            str(metadata_json),
            "--model-card-markdown",
            str(model_card_markdown),
            "--n-regimes",
            "1",
        ]
    )
    package_output = capsys.readouterr().out

    assert package_exit_code == 0
    assert "model=hist_gradient_boosting_regime_engineered_w10_r1_default" in package_output
    assert "artifact_id=fd001-" in package_output
    assert "model_card_markdown=" in package_output
    assert artifact_path.exists()
    assert metadata_json.exists()
    assert model_card_markdown.exists()
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    assert metadata["artifact"]["promotion"]["stage"] == "candidate"
    assert metadata["artifact"]["promotion"]["rollback"]["requires_retraining"] is False
    model_card = model_card_markdown.read_text(encoding="utf-8")
    assert "# C-MAPSS Deployment Model Card" in model_card
    assert "## Limitations" in model_card

    predict_exit_code = main(
        [
            "cmapss-predict-artifact",
            "--model-artifact",
            str(artifact_path),
            "--input-csv",
            str(input_csv),
            "--output-json",
            str(prediction_json),
        ]
    )
    predict_output = capsys.readouterr().out

    assert predict_exit_code == 0
    assert "predictions=2" in predict_output
    assert prediction_json.exists()

    benchmark_exit_code = main(
        [
            "cmapss-benchmark-artifact",
            "--model-artifact",
            str(artifact_path),
            "--input-csv",
            str(input_csv),
            "--runs",
            "2",
            "--warmup-runs",
            "1",
            "--max-p95-latency-ms",
            "10000",
            "--output-json",
            str(benchmark_json),
        ]
    )
    benchmark_output = capsys.readouterr().out

    assert benchmark_exit_code == 0
    assert "status=ok" in benchmark_output
    assert "model_size_bytes=" in benchmark_output
    assert "latency_p95_ms=" in benchmark_output
    benchmark = json.loads(benchmark_json.read_text(encoding="utf-8"))
    assert benchmark["status"] == "ok"
    assert benchmark["prediction_count"] == 2
    assert benchmark["latency_ms"]["p95"] >= 0.0

    validation_json = tmp_path / "models" / "fd001_validation.json"
    validate_exit_code = main(
        [
            "cmapss-validate-artifact",
            "--model-artifact",
            str(artifact_path),
            "--metadata-json",
            str(metadata_json),
            "--input-csv",
            str(input_csv),
            "--output-json",
            str(validation_json),
        ]
    )
    validate_output = capsys.readouterr().out

    assert validate_exit_code == 0
    assert "status=ok" in validate_output
    assert "artifact_id=fd001-" in validate_output
    assert "prediction_count=2" in validate_output
    validation = json.loads(validation_json.read_text(encoding="utf-8"))
    assert validation["status"] == "ok"
    assert validation["checks"]["metadata_json_matches"] is True
    assert validation["checks"]["prediction_smoke"] is True

    sbom_json.parent.mkdir(parents=True)
    sbom_json.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "components": [{"type": "library", "name": "numpy", "version": "1.0.0"}],
            },
        ),
        encoding="utf-8",
    )
    promotion_exit_code = main(
        [
            "cmapss-promotion-report",
            "--validation-json",
            str(validation_json),
            "--benchmark-json",
            str(benchmark_json),
            "--model-card-markdown",
            str(model_card_markdown),
            "--sbom-json",
            str(sbom_json),
            "--output-json",
            str(promotion_json),
            "--output-markdown",
            str(promotion_markdown),
        ]
    )
    promotion_output = capsys.readouterr().out

    assert promotion_exit_code == 0
    assert "status=ok" in promotion_output
    assert "artifact_id=fd001-" in promotion_output
    assert "gates_passed=8" in promotion_output
    promotion = json.loads(promotion_json.read_text(encoding="utf-8"))
    assert promotion["status"] == "ok"
    assert promotion["gates"]["sbom_cyclonedx"] is True
    assert "# C-MAPSS Promotion Report" in promotion_markdown.read_text(encoding="utf-8")


def test_cmapss_download_command_extracts_archive(tmp_path, capsys) -> None:
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        for subset in ("FD001", "FD002", "FD003", "FD004"):
            archive.writestr(f"nested/train_{subset}.txt", "1 1 0\n")
            archive.writestr(f"nested/test_{subset}.txt", "1 1 0\n")
            archive.writestr(f"nested/RUL_{subset}.txt", "1\n")

    output_dir = tmp_path / "raw" / "cmapss"
    exit_code = main(
        [
            "cmapss-download",
            "--output-dir",
            str(output_dir),
            "--archive-path",
            str(tmp_path / "downloads" / "cmapss.zip"),
            "--source-url",
            source_zip.as_uri(),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "files=12" in output
    assert (output_dir / "train_FD001.txt").exists()


def test_smap_msl_download_command_extracts_archive(tmp_path, capsys) -> None:
    source_zip = tmp_path / "smap_msl_source.zip"
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                '"P-1",SMAP,"[[1, 2]]","[contextual]",3',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("data/data/train/P-1.npy", _cli_npy_bytes(np.array([[0.0], [1.0]])))
        archive.writestr(
            "data/data/test/P-1.npy",
            _cli_npy_bytes(np.array([[0.0], [5.0], [6.0]])),
        )
    output_dir = tmp_path / "raw" / "smap_msl"

    exit_code = main(
        [
            "smap-msl-download",
            "--output-dir",
            str(output_dir),
            "--archive-path",
            str(tmp_path / "downloads" / "smap_msl.zip"),
            "--source-url",
            source_zip.as_uri(),
            "--labels-url",
            labels_csv.as_uri(),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "arrays=2" in output
    assert (output_dir / "labeled_anomalies.csv").exists()
    assert (output_dir / "data" / "train" / "P-1.npy").exists()
    assert (output_dir / "data" / "test" / "P-1.npy").exists()


def test_smap_msl_download_command_reports_actionable_download_failure(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    def fail_download(*args, **kwargs):
        raise RuntimeError("Could not download SMAP/MSL data\nDownload the Kaggle dataset")

    monkeypatch.setattr("aerospace_prognostics.cli.download_smap_msl_dataset", fail_download)

    exit_code = main(
        [
            "smap-msl-download",
            "--output-dir",
            str(tmp_path / "raw" / "smap_msl"),
            "--archive-path",
            str(tmp_path / "downloads" / "smap_msl.zip"),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "status=failed" in output
    assert "problem=Could not download SMAP/MSL data" in output
    assert "problem=Download the Kaggle dataset" in output


def _cli_result(
    subset: str,
    model_name: str,
    rmse: float,
    nasa_score: float,
) -> RegressionRunResult:
    return RegressionRunResult(
        dataset="C-MAPSS",
        subset=subset,
        model_name=model_name,
        rmse=rmse,
        nasa_score=nasa_score,
        train_rows=10,
        train_units=2,
        test_rows=4,
        test_units=2,
        test_rul_values=2,
        rul_cap=125,
        random_state=42,
        standardize=True,
    )


def _cli_prediction(
    subset: str,
    model_name: str,
    unit_number: int,
    actual_rul: float,
    predicted_rul: float,
) -> dict[str, str | int | float]:
    error = predicted_rul - actual_rul
    return {
        "dataset": "C-MAPSS-sequence",
        "prediction_split": "validation_selection",
        "subset": subset,
        "model_name": model_name,
        "selected_epoch": 1,
        "unit_number": unit_number,
        "end_cycle": 10 + unit_number,
        "actual_rul": actual_rul,
        "predicted_rul": predicted_rul,
        "error": error,
        "absolute_error": abs(error),
        "late_error": max(error, 0.0),
        "early_error": max(-error, 0.0),
    }


def _write_cli_predictions(
    path,
    rows: list[dict[str, str | int | float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_cli_smap_msl_channel(root) -> None:
    (root / "data" / "train").mkdir(parents=True)
    (root / "data" / "test").mkdir(parents=True)
    (root / "labeled_anomalies.csv").write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                '"P-1",SMAP,"[[1, 2]]","[contextual]",5',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(root / "data" / "train" / "P-1.npy", np.array([[0, 0], [1, 1], [2, 2], [3, 3]]))
    np.save(
        root / "data" / "test" / "P-1.npy",
        np.array([[0, 0], [6, -6], [7, -7], [1, 1], [10, -10]]),
    )


def _write_cli_smap_msl_selection_fixture(root) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("labeled_anomalies.csv").write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                '"P-1",SMAP,"[[1, 2]]","[contextual]",5',
                '"P-2",SMAP,"[[1, 1]]","[point]",5',
                '"M-1",MSL,"[[3, 5]]","[contextual]",8',
                '"M-2",MSL,"[[0, 0], [2, 2]]","[point, point]",6',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_cli_anomaly_result_csv(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cli_anomaly_result_row(
    channel_id: str,
    spacecraft: str,
    model_name: str,
    *,
    f1: float,
) -> dict[str, object]:
    return {
        "channel_id": channel_id,
        "spacecraft": spacecraft,
        "model_name": model_name,
        "train_rows": 5,
        "test_rows": 6,
        "feature_count": 2,
        "anomaly_sequences": 1,
        "anomaly_points": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": f1,
        "point_adjusted_f1": 0.5,
        "false_alarm_rate": 0.1,
        "miss_rate": 0.5,
        "support": 2,
        "predicted_positives": 2,
    }


def _cli_npy_bytes(values: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, values)
    return buffer.getvalue()
