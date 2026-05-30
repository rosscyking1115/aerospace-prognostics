from __future__ import annotations

import torch

from aerospace_prognostics.experiments.cmapss_deep_baseline import (
    CmapssLstmRegressor,
    CmapssOneDimensionalCnn,
    CmapssTemporalConvolutionalRegressor,
    run_all_cmapss_cnn_baselines,
    run_all_cmapss_lstm_baselines,
    run_all_cmapss_tcn_baselines,
    run_cmapss_cnn_baseline,
    run_cmapss_cnn_baseline_run,
    run_cmapss_lstm_baseline,
    run_cmapss_lstm_baseline_run,
    run_cmapss_tcn_baseline,
    run_cmapss_tcn_baseline_run,
)
from aerospace_prognostics.sequence_exports import export_cmapss_sequence_splits
from tests.cmapss_fixtures import write_all_tiny_cmapss_subsets, write_tiny_cmapss_subset


def test_cmapss_1d_cnn_forward_returns_batch_predictions() -> None:
    model = CmapssOneDimensionalCnn(feature_count=3, hidden_channels=4)

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
