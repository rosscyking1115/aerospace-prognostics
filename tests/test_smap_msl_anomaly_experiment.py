from __future__ import annotations

import json

import numpy as np
import pytest

from aerospace_prognostics.experiments.smap_msl_anomaly import (
    run_smap_msl_classical_baselines,
    write_smap_msl_classical_baselines_csv,
    write_smap_msl_classical_baselines_json,
)


def test_run_smap_msl_classical_baselines_scores_selected_channels(tmp_path) -> None:
    _write_multi_channel_smap_msl(tmp_path)

    runs = run_smap_msl_classical_baselines(
        tmp_path,
        channels=("P-1",),
        methods=("robust_zscore", "pca_reconstruction"),
        pca_components=1,
        pca_threshold_quantile=0.95,
    )

    assert [run.channel_id for run in runs] == ["P-1", "P-1"]
    assert [run.model_name for run in runs] == ["robust_zscore", "pca_reconstruction"]
    assert all(run.spacecraft == "SMAP" for run in runs)
    assert all(run.anomaly_points == 2 for run in runs)
    assert runs[0].metrics.support == 2


def test_run_smap_msl_classical_baselines_respects_max_channels(tmp_path) -> None:
    _write_multi_channel_smap_msl(tmp_path)

    runs = run_smap_msl_classical_baselines(
        tmp_path,
        max_channels=1,
        methods=("robust_zscore",),
    )

    assert [run.channel_id for run in runs] == ["P-1"]


def test_run_smap_msl_classical_baselines_rejects_missing_channel(tmp_path) -> None:
    _write_multi_channel_smap_msl(tmp_path)

    with pytest.raises(ValueError, match="channels not found"):
        run_smap_msl_classical_baselines(tmp_path, channels=("missing",))


def test_write_smap_msl_classical_baseline_outputs(tmp_path) -> None:
    _write_multi_channel_smap_msl(tmp_path)
    runs = run_smap_msl_classical_baselines(
        tmp_path,
        channels=("P-1",),
        methods=("robust_zscore",),
    )
    json_path = tmp_path / "results" / "runs.json"
    csv_path = tmp_path / "results" / "runs.csv"

    write_smap_msl_classical_baselines_json(runs, json_path)
    write_smap_msl_classical_baselines_csv(runs, csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    csv_text = csv_path.read_text(encoding="utf-8")
    assert payload[0]["channel_id"] == "P-1"
    assert payload[0]["model_name"] == "robust_zscore"
    assert "channel_id,spacecraft,model_name" in csv_text
    assert "P-1,SMAP,robust_zscore" in csv_text


def _write_multi_channel_smap_msl(root) -> None:
    (root / "data" / "train").mkdir(parents=True)
    (root / "data" / "test").mkdir(parents=True)
    (root / "labeled_anomalies.csv").write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                '"P-1",SMAP,"[[1, 2]]","[contextual]",5',
                '"C-1",MSL,"[[3, 4]]","[point]",5',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for channel_id in ("P-1", "C-1"):
        np.save(
            root / "data" / "train" / f"{channel_id}.npy",
            np.array([[-0.2, -0.2], [-0.1, -0.1], [0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
        )
        np.save(
            root / "data" / "test" / f"{channel_id}.npy",
            np.array([[0.0, 0.0], [8.0, -8.0], [7.0, -7.0], [0.1, 0.1], [-7.0, 7.0]]),
        )
