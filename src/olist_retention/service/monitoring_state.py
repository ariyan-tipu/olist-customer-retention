"""Holds the service's monitoring objects and binds them to the model.

The drift monitor needs the reference distribution, which only exists
once the artefact has loaded, and the artefact loads on first use. This
keeps that ordering in one place so the endpoints do not each have to
remember it.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from olist_retention.logging_utils import get_logger
from olist_retention.monitoring.drift import DriftMonitor
from olist_retention.monitoring.telemetry import LatencyWindow
from olist_retention.service.scorer import ModelScorer

logger = get_logger(__name__)


class MonitoringState:
    def __init__(self, scorer: ModelScorer, window: int = 2000) -> None:
        self.scorer = scorer
        self.latency = LatencyWindow(window=window)
        self.window = window
        self._drift: DriftMonitor | None = None
        self._lock = threading.Lock()

    @property
    def drift(self) -> DriftMonitor:
        if self._drift is None:
            with self._lock:
                if self._drift is None:
                    self._drift = DriftMonitor(
                        self.scorer.reference_distribution(), window=self.window
                    )
        return self._drift

    def record(self, rows: list[dict[str, Any]], duration_ms: float, model_name: str) -> None:
        self.latency.observe(duration_ms)
        self.drift.observe(rows)
        # One structured line per request. Whatever collects logs on the
        # instance can parse it without a regular expression, and the
        # fields match the names used by the metrics endpoint.
        logger.info(
            json.dumps(
                {
                    "event": "score",
                    "rows": len(rows),
                    "duration_ms": round(duration_ms, 3),
                    "model_name": model_name,
                    "ts": time.time(),
                }
            )
        )

    def snapshots(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.latency.snapshot(), self.drift.snapshot()
