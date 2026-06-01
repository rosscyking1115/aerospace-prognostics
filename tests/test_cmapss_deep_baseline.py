from __future__ import annotations

import csv

import pytest
import torch

from aerospace_prognostics.experiments.cmapss_deep_baseline import (
    CmapssLstmRegressor,
    CmapssOneDimensionalCnn,
    CmapssResidualCnnRegressor,
    CmapssTemporalConvolutionalRegressor,
    CmapssTransformerRegressor,
    _deep_training_loss,
    run_all_cmapss_cnn_baselines,
    run_all_cmapss_lstm_baselines,
    run_all_cmapss_tcn_baselines,
    run_all_cmapss_transformer_baselines,
    run_cmapss_cnn_baseline,
    run_cmapss_cnn_baseline_run,
    run_cmapss_deep_baseline_comparison,
    run_cmapss_deep_baseline_comparison_runs,
    run_cmapss_lstm_baseline,
    run_cmapss_lstm_baseline_run,
    run_cmapss_residual_cnn_baseline_run,
    run_cmapss_tcn_baseline,
    run_cmapss_tcn_baseline_run,
    run_cmapss_transformer_baseline,
    run_cmapss_transformer_baseline_run,
    write_cmapss_deep_predictions_csv,
)
from aerospace_prognostics.sequence_exports import export_cmapss_sequence_splits
from tests.cmapss_fixtures import write_all_tiny_cmapss_subsets, write_tiny_cmapss_subset


def test_cmapss_1d_cnn_forward_returns_batch_predictions() -> None:
    model = CmapssOneDimensionalCnn(feature_count=3, hidden_channels=4)

    predictions = model(torch.zeros((2, 5, 3), dtype=torch.float32))

    assert predictions.shape == (2,)


def test_cmapss_residual_cnn_forward_returns_batch_predictions() -> None:
    model = CmapssResidualCnnRegressor(
        feature_count=3,
        hidden_channels=4,
        num_blocks=2,
    )

    predictions = model(torch.zeros((2, 5, 3), dtype=torch.float32))

    assert predictions.shape == (2,)


def test_cmapss_lstm_forward_returns_batch_predictions() -> None:
    model = CmapssLstmRegressor(feature_count=3, hidden_size=4, bidirectional=True)

    predictions = model(torch.zeros((2, 5, 3), dtype=torch.float32))

    assert predictions.shape == (2,)


def test_cmapss_tcn_forward_returns_batch_predictions() -> None:
    model = CmapssTemporalConvolutionalRegressor(
        feature_count=3,
        hidden_channels=4,
        num_levels=2,
    )

    predictions = model(torch.zeros((2, 5, 3), dtype=torch.float32))

    assert predictions.shape == (2,)


def test_cmapss_transformer_forward_returns_batch_predictions() -> None:
    model = CmapssTransformerRegressor(
        feature_count=3,
        sequence_length=5,
        d_model=4,
        num_heads=2,
        num_layers=1,
        dim_feedforward=8,
    )

    predictions = model(torch.zeros((2, 5, 3), dtype=torch.float32))

    assert predictions.shape == (2,)


def test_nasa_surrogate_training_loss_penalizes_late_errors_more() -> None:
    target = torch.tensor([100.0])
    late_loss = _deep_training_loss(
        torch.tensor([110.0]),
        target,
        training_loss="nasa_surrogate",
    )
    early_loss = _deep_training_loss(
        torch.tensor([90.0]),
        target,
        training_loss="nasa_surrogate",
    )
    mse_loss = _deep_training_loss(
        torch.tensor([110.0]),
        target,
        training_loss="mse",
    )

    assert late_loss > early_loss
    assert float(mse_loss) == pytest.approx(100.0)


def test_mse_nasa_blend_training_loss_keeps_mse_scale() -> None:
    target = torch.tensor([100.0])
    pure_nasa_loss = _deep_training_loss(
        torch.tensor([110.0]),
        target,
        training_loss="nasa_surrogate",
    )
    blended_loss = _deep_training_loss(
        torch.tensor([110.0]),
        target,
        training_loss="mse_nasa_blend_w0p001",
    )

    assert float(blended_loss) == pytest.approx(100.0 + (0.001 * float(pure_nasa_loss)))


def test_asymmetric_mse_training_loss_penalizes_late_errors_with_mse_scale() -> None:
    target = torch.tensor([100.0])
    late_loss = _deep_training_loss(
        torch.tensor([110.0]),
        target,
        training_loss="asymmetric_mse_late_w2",
    )
    early_loss = _deep_training_loss(
        torch.tensor([90.0]),
        target,
        training_loss="asymmetric_mse_late_w2",
    )

    assert float(late_loss) == pytest.approx(200.0)
    assert float(early_loss) == pytest.approx(100.0)


def test_run_cmapss_cnn_baseline_returns_structured_result(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    result = run_cmapss_cnn_baseline(
        sequence_dir,
        "FD001",
        epochs=1,
        batch_size=2,
        hidden_channels=4,
    )

    assert result.dataset == "C-MAPSS-sequence"
    assert result.subset == "FD001"
    assert result.model_name.startswith("cnn_1d_w2_e1_c4")
    assert result.train_rows == 2
    assert result.test_rul_values == 2
    assert result.rmse >= 0
    assert result.nasa_score >= 0


def test_run_cmapss_lstm_baseline_returns_structured_result(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    result = run_cmapss_lstm_baseline(
        sequence_dir,
        "FD001",
        epochs=1,
        batch_size=2,
        hidden_size=4,
    )

    assert result.dataset == "C-MAPSS-sequence"
    assert result.subset == "FD001"
    assert result.model_name.startswith("lstm_w2_e1_h4_l1")
    assert result.train_rows == 2
    assert result.test_rul_values == 2
    assert result.rmse >= 0
    assert result.nasa_score >= 0


def test_run_cmapss_tcn_baseline_returns_structured_result(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    result = run_cmapss_tcn_baseline(
        sequence_dir,
        "FD001",
        epochs=1,
        batch_size=2,
        hidden_channels=4,
        num_levels=1,
    )

    assert result.dataset == "C-MAPSS-sequence"
    assert result.subset == "FD001"
    assert result.model_name.startswith("tcn_w2_e1_c4_l1_k3")
    assert result.train_rows == 2
    assert result.test_rul_values == 2
    assert result.rmse >= 0
    assert result.nasa_score >= 0


def test_run_cmapss_transformer_baseline_returns_structured_result(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    result = run_cmapss_transformer_baseline(
        sequence_dir,
        "FD001",
        epochs=1,
        batch_size=2,
        d_model=4,
        num_heads=2,
        num_layers=1,
        dim_feedforward=8,
    )

    assert result.dataset == "C-MAPSS-sequence"
    assert result.subset == "FD001"
    assert result.model_name.startswith("transformer_w2_e1_d4_h2_l1_ff8")
    assert result.train_rows == 2
    assert result.test_rul_values == 2
    assert result.rmse >= 0
    assert result.nasa_score >= 0


def test_run_cmapss_cnn_baseline_run_tracks_history_and_selected_epoch(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_cnn_baseline_run(
        sequence_dir,
        "FD001",
        epochs=2,
        batch_size=2,
        hidden_channels=4,
    )

    assert len(run.history) == 2
    assert run.selected_epoch in {1, 2}
    assert f"_best_e{run.selected_epoch}_" in run.result.model_name
    assert all(epoch.train_loss >= 0 for epoch in run.history)
    assert len(run.predictions) == run.result.test_rul_values
    assert all(prediction.absolute_error >= 0 for prediction in run.predictions)
    assert all(prediction.late_error >= 0 for prediction in run.predictions)
    assert all(prediction.early_error >= 0 for prediction in run.predictions)
    assert run.validation_selection_predictions
    assert all(
        prediction.absolute_error >= 0 for prediction in run.validation_selection_predictions
    )


def test_run_cmapss_residual_cnn_baseline_run_tracks_history(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_residual_cnn_baseline_run(
        sequence_dir,
        "FD001",
        epochs=1,
        batch_size=2,
        hidden_channels=4,
        num_blocks=2,
    )

    assert len(run.history) == 1
    assert run.result.model_name.startswith("rescnn_w2_e1_c4_b2_k3")
    assert len(run.predictions) == run.result.test_rul_values


def test_run_cmapss_cnn_baseline_run_supports_nasa_surrogate_loss(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_cnn_baseline_run(
        sequence_dir,
        "FD001",
        epochs=1,
        batch_size=2,
        training_loss="nasa_surrogate",
        hidden_channels=4,
    )

    assert "_loss_nasa_surrogate_" in run.result.model_name
    assert run.history[0].train_loss >= 0


def test_run_cmapss_cnn_baseline_run_supports_mse_nasa_blend_loss(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_cnn_baseline_run(
        sequence_dir,
        "FD001",
        epochs=1,
        batch_size=2,
        training_loss="mse_nasa_blend_w0p001",
        hidden_channels=4,
    )

    assert "_loss_mse_nasa_blend_w0p001_" in run.result.model_name
    assert run.history[0].train_loss >= 0


def test_run_cmapss_lstm_baseline_run_supports_bidirectional_checkpointing(
    tmp_path,
) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_lstm_baseline_run(
        sequence_dir,
        "FD001",
        epochs=2,
        batch_size=2,
        hidden_size=4,
        bidirectional=True,
    )

    assert len(run.history) == 2
    assert run.selected_epoch in {1, 2}
    assert run.result.model_name.startswith("bilstm_w2_e2_h4_l1")
    assert f"_best_e{run.selected_epoch}_" in run.result.model_name


def test_run_cmapss_tcn_baseline_run_tracks_history_and_selected_epoch(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_tcn_baseline_run(
        sequence_dir,
        "FD001",
        epochs=2,
        batch_size=2,
        hidden_channels=4,
        num_levels=1,
    )

    assert len(run.history) == 2
    assert run.selected_epoch in {1, 2}
    assert run.result.model_name.startswith("tcn_w2_e2_c4_l1_k3")
    assert f"_best_e{run.selected_epoch}_" in run.result.model_name


def test_run_cmapss_transformer_baseline_run_tracks_history_and_selected_epoch(
    tmp_path,
) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_transformer_baseline_run(
        sequence_dir,
        "FD001",
        epochs=2,
        batch_size=2,
        d_model=4,
        num_heads=2,
        num_layers=1,
        dim_feedforward=8,
    )

    assert len(run.history) == 2
    assert run.selected_epoch in {1, 2}
    assert run.result.model_name.startswith("transformer_w2_e2_d4_h2_l1_ff8")
    assert f"_best_e{run.selected_epoch}_" in run.result.model_name


def test_run_cmapss_cnn_baseline_run_can_select_final_checkpoint(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_cnn_baseline_run(
        sequence_dir,
        "FD001",
        epochs=2,
        batch_size=2,
        hidden_channels=4,
        checkpoint_policy="final",
    )

    assert run.selected_epoch == 2
    assert "_final_e2_" in run.result.model_name


def test_run_cmapss_lstm_baseline_run_can_select_final_checkpoint(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_lstm_baseline_run(
        sequence_dir,
        "FD001",
        epochs=2,
        batch_size=2,
        hidden_size=4,
        checkpoint_policy="final",
    )

    assert run.selected_epoch == 2
    assert "_final_e2_" in run.result.model_name


def test_run_cmapss_tcn_baseline_run_can_select_final_checkpoint(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_tcn_baseline_run(
        sequence_dir,
        "FD001",
        epochs=2,
        batch_size=2,
        hidden_channels=4,
        num_levels=1,
        checkpoint_policy="final",
    )

    assert run.selected_epoch == 2
    assert "_final_e2_" in run.result.model_name


def test_run_cmapss_transformer_baseline_run_can_select_final_checkpoint(
    tmp_path,
) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    run = run_cmapss_transformer_baseline_run(
        sequence_dir,
        "FD001",
        epochs=2,
        batch_size=2,
        d_model=4,
        num_heads=2,
        num_layers=1,
        dim_feedforward=8,
        checkpoint_policy="final",
    )

    assert run.selected_epoch == 2
    assert "_final_e2_" in run.result.model_name


def test_run_all_cmapss_cnn_baselines_returns_requested_subsets(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    sequence_dir = tmp_path / "sequences"
    for subset in ("FD001", "FD002"):
        export_cmapss_sequence_splits(
            tmp_path,
            sequence_dir,
            subset,
            window_size=2,
            validation_fraction=0.5,
            validation_horizon=1,
        )

    results = run_all_cmapss_cnn_baselines(
        sequence_dir,
        subsets=("FD001", "FD002"),
        epochs=1,
        batch_size=2,
        hidden_channels=4,
    )

    assert [result.subset for result in results] == ["FD001", "FD002"]


def test_run_all_cmapss_lstm_baselines_returns_requested_subsets(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    sequence_dir = tmp_path / "sequences"
    for subset in ("FD001", "FD002"):
        export_cmapss_sequence_splits(
            tmp_path,
            sequence_dir,
            subset,
            window_size=2,
            validation_fraction=0.5,
            validation_horizon=1,
        )

    results = run_all_cmapss_lstm_baselines(
        sequence_dir,
        subsets=("FD001", "FD002"),
        epochs=1,
        batch_size=2,
        hidden_size=4,
    )

    assert [result.subset for result in results] == ["FD001", "FD002"]


def test_run_all_cmapss_tcn_baselines_returns_requested_subsets(tmp_path) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    sequence_dir = tmp_path / "sequences"
    for subset in ("FD001", "FD002"):
        export_cmapss_sequence_splits(
            tmp_path,
            sequence_dir,
            subset,
            window_size=2,
            validation_fraction=0.5,
            validation_horizon=1,
        )

    results = run_all_cmapss_tcn_baselines(
        sequence_dir,
        subsets=("FD001", "FD002"),
        epochs=1,
        batch_size=2,
        hidden_channels=4,
        num_levels=1,
    )

    assert [result.subset for result in results] == ["FD001", "FD002"]


def test_run_all_cmapss_transformer_baselines_returns_requested_subsets(
    tmp_path,
) -> None:
    write_all_tiny_cmapss_subsets(tmp_path)
    sequence_dir = tmp_path / "sequences"
    for subset in ("FD001", "FD002"):
        export_cmapss_sequence_splits(
            tmp_path,
            sequence_dir,
            subset,
            window_size=2,
            validation_fraction=0.5,
            validation_horizon=1,
        )

    results = run_all_cmapss_transformer_baselines(
        sequence_dir,
        subsets=("FD001", "FD002"),
        epochs=1,
        batch_size=2,
        d_model=4,
        num_heads=2,
        num_layers=1,
        dim_feedforward=8,
    )

    assert [result.subset for result in results] == ["FD001", "FD002"]


def test_run_cmapss_deep_baseline_comparison_labels_candidates(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )

    results = run_cmapss_deep_baseline_comparison(
        sequence_dir,
        subsets=("FD001",),
        models=("cnn", "rescnn", "tcn", "transformer"),
        epochs=1,
        batch_size=2,
        learning_rates=(1e-3,),
        hidden_sizes=(4,),
        tcn_levels=1,
        transformer_heads=2,
        transformer_dim_feedforward=8,
    )

    assert len(results) == 4
    assert {result.subset for result in results} == {"FD001"}
    assert all(result.model_name.startswith("compare_") for result in results)
    assert any("compare_cnn_h4_lr0p001" in result.model_name for result in results)
    assert any("compare_rescnn_h4_lr0p001" in result.model_name for result in results)
    assert any("compare_tcn_h4_lr0p001" in result.model_name for result in results)
    assert any(
        "compare_transformer_h4_lr0p001" in result.model_name for result in results
    )


def test_write_cmapss_deep_predictions_csv_records_unit_diagnostics(tmp_path) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )
    output_csv = tmp_path / "predictions" / "deep_predictions.csv"

    runs = run_cmapss_deep_baseline_comparison_runs(
        sequence_dir,
        subsets=("FD001",),
        models=("cnn",),
        epochs=1,
        batch_size=2,
        learning_rates=(1e-3,),
        hidden_sizes=(4,),
    )
    write_cmapss_deep_predictions_csv(runs, output_csv)

    with output_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == runs[0].result.test_rul_values
    assert rows[0]["prediction_split"] == "official_test"
    assert rows[0]["model_name"].startswith("compare_cnn_h4_lr0p001")
    assert rows[0]["unit_number"]
    assert rows[0]["end_cycle"]
    assert rows[0]["actual_rul"]
    assert rows[0]["predicted_rul"]
    assert rows[0]["absolute_error"]


def test_write_cmapss_deep_predictions_csv_records_validation_selection_rows(
    tmp_path,
) -> None:
    write_tiny_cmapss_subset(tmp_path)
    sequence_dir = tmp_path / "sequences"
    export_cmapss_sequence_splits(
        tmp_path,
        sequence_dir,
        "FD001",
        window_size=2,
        validation_fraction=0.5,
        validation_horizon=1,
    )
    output_csv = tmp_path / "predictions" / "validation_selection_predictions.csv"

    runs = run_cmapss_deep_baseline_comparison_runs(
        sequence_dir,
        subsets=("FD001",),
        models=("cnn",),
        epochs=1,
        batch_size=2,
        learning_rates=(1e-3,),
        hidden_sizes=(4,),
    )
    write_cmapss_deep_predictions_csv(
        runs,
        output_csv,
        prediction_split="validation_selection",
    )

    with output_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == len(runs[0].validation_selection_predictions)
    assert rows[0]["prediction_split"] == "validation_selection"
    assert rows[0]["end_cycle"]
