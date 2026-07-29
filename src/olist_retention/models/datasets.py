"""Load feature snapshots into pandas for scikit-learn.

The Spark job writes one parquet snapshot per cutoff date. Training reads
the earlier snapshot, evaluation reads the later one, which gives a real
out-of-time test: the model is judged on customers scored three months
after the data it learned from, the way it would be used in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from olist_retention.feature_names import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    LABEL_COLUMN,
)


@dataclass(frozen=True)
class ModelData:
    """Features, labels and identifiers for one cutoff snapshot."""

    cutoff: date
    ids: pd.Series
    X: pd.DataFrame
    y: np.ndarray

    @property
    def base_rate(self) -> float:
        return float(self.y.mean())

    def __len__(self) -> int:
        return len(self.y)


def snapshot_path(features_dir: Path, cutoff: date) -> Path:
    return features_dir / f"cutoff={cutoff.isoformat()}"


def load_snapshot(features_dir: Path, cutoff: date) -> ModelData:
    """Read one feature snapshot and split it into X, y and ids."""
    path = snapshot_path(features_dir, cutoff)
    if not path.exists():
        raise FileNotFoundError(
            f"No feature snapshot at {path}. Run 'olist-retention pipeline' first."
        )
    frame = pd.read_parquet(path)

    missing = [c for c in (*ALL_FEATURES, LABEL_COLUMN, ID_COLUMN) if c not in frame.columns]
    if missing:
        raise ValueError(f"Snapshot {path} is missing columns: {missing}")

    X = frame[list(ALL_FEATURES)].copy()
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype("object")
    y = frame[LABEL_COLUMN].to_numpy(dtype=int)
    return ModelData(cutoff=cutoff, ids=frame[ID_COLUMN], X=X, y=y)
