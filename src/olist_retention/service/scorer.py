"""Load the persisted model artefact and score feature rows.

The artefact can come from the local filesystem or from S3. Loading is
deferred to first use either way, so the process starts, answers a
liveness probe and reports a useful reason on the readiness probe even
when the model is missing or the bucket is unreachable.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from olist_retention.feature_names import ALL_FEATURES, CATEGORICAL_FEATURES
from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_ARTEFACT_VERSION = 2


class ModelNotReadyError(RuntimeError):
    """The model artefact is missing or unusable."""


class ModelScorer:
    """Thread-safe lazy loader around the joblib artefact."""

    def __init__(self, model_path: Path, s3_uri: str | None = None) -> None:
        self.model_path = model_path
        self.s3_uri = s3_uri or os.environ.get("OLIST_MODEL_S3_URI") or None
        self._lock = threading.Lock()
        self._pipeline = None
        self._card: dict | None = None

    def _download(self) -> None:
        """Pull the artefact from S3, with its digest checked on arrival."""
        from olist_retention.artefacts.s3 import ArtefactError, ArtefactStore, parse_uri

        bucket, _ = parse_uri(self.s3_uri or "")
        try:
            store = ArtefactStore(bucket=bucket, region=os.environ.get("AWS_REGION"))
            store.fetch(self.s3_uri or "", self.model_path)
        except ArtefactError as error:
            raise ModelNotReadyError(str(error)) from error

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            if self.s3_uri and not self.model_path.is_file():
                self._download()
            if not self.model_path.is_file():
                raise ModelNotReadyError(
                    f"Model artefact not found at {self.model_path}. "
                    "Train one with 'olist-retention train', mount it into the container, "
                    "or set OLIST_MODEL_S3_URI."
                )
            artefact = joblib.load(self.model_path)
            version = artefact.get("model_card", {}).get("artefact_version")
            if version != SUPPORTED_ARTEFACT_VERSION:
                raise ModelNotReadyError(
                    f"Artefact version {version!r} is not supported "
                    f"(expected {SUPPORTED_ARTEFACT_VERSION}). Version 2 carries the "
                    "reference distribution the drift monitor needs; retrain to produce one."
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

    def reference_distribution(self) -> dict[str, Any] | None:
        """The training baseline the drift monitor compares against."""
        try:
            return self.card().get("reference_distribution")
        except ModelNotReadyError:
            return None

    def score(self, rows: list[dict]) -> np.ndarray:
        """Return repeat purchase probabilities for feature dictionaries."""
        self._ensure_loaded()
        frame = pd.DataFrame(rows, columns=list(ALL_FEATURES))
        for column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].astype("object")
        assert self._pipeline is not None
        return self._pipeline.predict_proba(frame)[:, 1]
