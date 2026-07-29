"""Load the persisted model artefact and score feature rows."""

from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from olist_retention.feature_names import ALL_FEATURES, CATEGORICAL_FEATURES
from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_ARTEFACT_VERSION = 1


class ModelNotReadyError(RuntimeError):
    """The model artefact is missing or unusable."""


class ModelScorer:
    """Thread-safe lazy loader around the joblib artefact.

    Loading is deferred to first use so the process can start (and report
    itself unhealthy in a useful way) even when the artefact has not been
    mounted yet.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._lock = threading.Lock()
        self._pipeline = None
        self._card: dict | None = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            if not self.model_path.is_file():
                raise ModelNotReadyError(
                    f"Model artefact not found at {self.model_path}. "
                    "Train one with 'olist-retention train' or mount it into the container."
                )
            artefact = joblib.load(self.model_path)
            version = artefact.get("model_card", {}).get("artefact_version")
            if version != SUPPORTED_ARTEFACT_VERSION:
                raise ModelNotReadyError(
                    f"Artefact version {version!r} is not supported "
                    f"(expected {SUPPORTED_ARTEFACT_VERSION})"
                )
            self._pipeline = artefact["pipeline"]
            self._card = artefact["model_card"]
            logger.info(
                "Loaded %s trained %s",
                self._card.get("model_name"),
                self._card.get("trained_at_utc"),
            )

    def card(self) -> dict:
        self._ensure_loaded()
        assert self._card is not None
        return self._card

    def score(self, rows: list[dict]) -> np.ndarray:
        """Return repeat purchase probabilities for feature dictionaries."""
        self._ensure_loaded()
        frame = pd.DataFrame(rows, columns=list(ALL_FEATURES))
        for column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].astype("object")
        assert self._pipeline is not None
        return self._pipeline.predict_proba(frame)[:, 1]
