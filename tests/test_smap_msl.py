from __future__ import annotations

import json

import numpy as np
import pytest

from aerospace_prognostics.data.smap_msl import (
    build_smap_msl_label_vector,
    export_smap_msl_channel_csv,
    load_smap_msl_channel,
    read_smap_msl_labels,
)


def test_read_smap_msl_labels_parses_telemanom_metadata(tmp_path) -> None:
    _write_tiny_smap_msl_channel(tmp_path)

    labels = read_smap_msl_labels(tmp_path)

    assert len(labels) == 1
    assert labels[0].channel_id == "P-1"
    assert labels[0].spacecraft == "SMAP"
    assert labels[0].anomaly_sequences == ((1, 2), (4, 4))
    assert labels[0].anomaly_classes == ("contextual", "point")
    assert labels[0].num_values == 5


def test_load_smap_msl_channel_builds_test_labels(tmp_path) -> None:
    _write_tiny_smap_msl_channel(tmp_path)

    channel = load_smap_msl_channel(tmp_path, "P-1")

    assert channel.train_values.shape == (4, 2)
    assert channel.test_values.shape == (5, 2)
    assert channel.test_labels.tolist() == [0, 1, 1, 0, 1]
    assert channel.feature_names == ("feature_0", "feature_1")


def test_export_smap_msl_channel_csv_writes_generic_baseline_inputs(tmp_path) -> None:
    _write_tiny_smap_msl_channel(tmp_path)
    output_dir = tmp_path / "exports"

    export = export_smap_msl_channel_csv(tmp_path, "P-1", output_dir)

    train_text = export.train_csv.read_text(encoding="utf-8")
    test_text = export.test_csv.read_text(encoding="utf-8")
    assert export.train_rows == 4
    assert export.test_rows == 5
    assert "timestep,feature_0,feature_1" in train_text
    assert "timestep,feature_0,feature_1,label" in test_text
    assert "4,10,-10,1" in test_text


def test_build_smap_msl_label_vector_rejects_out_of_range_interval() -> None:
    with pytest.raises(ValueError, match="outside test length"):
        build_smap_msl_label_vector(3, ((1, 3),), channel_id="P-1")


def _write_tiny_smap_msl_channel(root) -> None:
    (root / "data" / "train").mkdir(parents=True)
    (root / "data" / "test").mkdir(parents=True)
    (root / "labeled_anomalies.csv").write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                '"P-1",SMAP,"[[1, 2], [4, 4]]","[contextual, point]",5',
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


def test_export_metadata_is_json_serializable(tmp_path) -> None:
    _write_tiny_smap_msl_channel(tmp_path)
    export = export_smap_msl_channel_csv(tmp_path, "P-1", tmp_path / "exports")

    assert json.dumps(export.to_dict())
