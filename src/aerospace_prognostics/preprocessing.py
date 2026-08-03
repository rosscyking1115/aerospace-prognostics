"""Feature normalisation utilities fitted on training data only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import StandardScaler

from aerospace_prognostics.features import cmapss_feature_columns
from aerospace_prognostics.sequences import SequenceWindowDataset

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class FeatureStandardizer:
    """Standardise C-MAPSS feature columns using train-only statistics."""

    feature_columns: tuple[str, ...]
    scaler: StandardScaler

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
    ) -> FeatureStandardizer:
        """Fit a standardizer from a training frame."""
        columns = tuple(feature_columns or cmapss_feature_columns())
        _validate_columns(frame, columns)
        scaler = StandardScaler()
        scaler.fit(frame.loc[:, columns].to_numpy())
        return cls(feature_columns=columns, scaler=scaler)

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with feature columns standardised."""
        _validate_columns(frame, self.feature_columns)
        result = frame.copy()
        feature_values = result.loc[:, self.feature_columns].to_numpy()
        transformed = self.scaler.transform(feature_values)
        return result.assign(**dict(zip(self.feature_columns, transformed.T, strict=True)))

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardise a feature table and preserve its index/columns."""
        _validate_columns(features, self.feature_columns)
        transformed = self.scaler.transform(features.loc[:, self.feature_columns].to_numpy())
        return features.assign(**dict(zip(self.feature_columns, transformed.T, strict=True)))

    def transform_windows(self, windows: np.ndarray) -> np.ndarray:
        """Standardise a 3D `(samples, timesteps, features)` window tensor."""
        if windows.ndim != 3:
            raise ValueError("windows must have shape (samples, timesteps, features)")
        if windows.shape[-1] != len(self.feature_columns):
            raise ValueError(
                "windows feature dimension does not match fitted feature columns "
                f"({windows.shape[-1]} != {len(self.feature_columns)})"
            )
        if windows.shape[0] == 0:
            return windows.astype(np.float32, copy=True)

        sample_count, timestep_count, feature_count = windows.shape
        flat = windows.reshape(sample_count * timestep_count, feature_count)
        transformed = self.scaler.transform(flat)
        return transformed.reshape(sample_count, timestep_count, feature_count).astype(np.float32)

    def transform_sequence_dataset(self, dataset: SequenceWindowDataset) -> SequenceWindowDataset:
        """Return a copy of a sequence dataset with normalised windows."""
        if dataset.feature_columns != self.feature_columns:
            raise ValueError("dataset feature columns do not match fitted feature columns")
        return SequenceWindowDataset(
            windows=self.transform_windows(dataset.windows),
            targets=dataset.targets.copy() if dataset.targets is not None else None,
            unit_numbers=dataset.unit_numbers.copy(),
            end_cycles=dataset.end_cycles.copy(),
            feature_columns=dataset.feature_columns,
        )


def _validate_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
