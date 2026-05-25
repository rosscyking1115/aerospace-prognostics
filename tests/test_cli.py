from __future__ import annotations

import json
import zipfile

from aerospace_prognostics.cli import main
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
    assert (artifact_dir / "phase1_summary.md").exists()


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
