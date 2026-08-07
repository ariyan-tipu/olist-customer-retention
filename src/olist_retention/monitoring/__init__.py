"""Serving-time monitoring: request latency and input feature drift.

Kept apart from the model package because these run in the scoring
container, which ships without Spark and without the training code.
"""

from __future__ import annotations

from olist_retention.monitoring.drift import (
    DriftMonitor,
    build_reference,
)
from olist_retention.monitoring.telemetry import LatencyWindow, prometheus_text

__all__ = [
    "DriftMonitor",
    "LatencyWindow",
    "build_reference",
    "prometheus_text",
]
