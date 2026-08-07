"""Latency accounting and the scrape endpoint's output format.

Percentiles come from a bounded window that is sorted on demand. An
exact number over the last few thousand requests is easier to defend in
a conversation than an approximate one from a streaming estimator, and
at this request volume the sort costs nothing worth measuring.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np


class LatencyWindow:
    """Bounded window of scoring durations in milliseconds."""

    def __init__(self, window: int = 2000) -> None:
        self.window = window
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._count = 0
        self._total_ms = 0.0

    def observe(self, duration_ms: float) -> None:
        with self._lock:
            self._samples.append(duration_ms)
            self._count += 1
            self._total_ms += duration_ms

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            count = self._count
            total = self._total_ms
        if not samples:
            return {"requests_total": 0, "window_requests": 0}
        ordered = np.sort(np.asarray(samples, dtype=float))
        return {
            "requests_total": count,
            "window_requests": len(ordered),
            "mean_ms": round(total / count, 3),
            "p50_ms": round(float(np.percentile(ordered, 50)), 3),
            "p95_ms": round(float(np.percentile(ordered, 95)), 3),
            "p99_ms": round(float(np.percentile(ordered, 99)), 3),
            "max_ms": round(float(ordered[-1]), 3),
        }


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def prometheus_text(latency: dict[str, Any], drift: dict[str, Any], model_name: str) -> str:
    """Render both snapshots in the Prometheus text exposition format.

    That format is the one every monitoring stack can already scrape, so
    the service can be pointed at whatever an employer runs without a
    translation layer sitting in between.
    """
    # Requests and rows are counted separately on purpose: one HTTP call
    # can carry a batch of customers, so latency belongs to the request
    # and drift belongs to the rows inside it.
    lines: list[str] = [
        "# HELP olist_score_requests_total Scoring requests served since process start.",
        "# TYPE olist_score_requests_total counter",
        f"olist_score_requests_total {float(latency.get('requests_total', 0))}",
        "# HELP olist_score_rows_total Customer rows scored since process start.",
        "# TYPE olist_score_rows_total counter",
        f"olist_score_rows_total {float(drift.get('rows_scored_total', 0))}",
        "# HELP olist_score_latency_milliseconds Scoring latency over the recent window.",
        "# TYPE olist_score_latency_milliseconds gauge",
    ]
    for quantile in ("p50", "p95", "p99"):
        key = f"{quantile}_ms"
        if key in latency:
            lines.append(
                f'olist_score_latency_milliseconds{{quantile="{quantile}"}} {latency[key]}'
            )

    if drift.get("status") == "ok":
        lines.extend(
            [
                "# HELP olist_feature_drift_psi Population stability index per feature.",
                "# TYPE olist_feature_drift_psi gauge",
            ]
        )
        for item in drift.get("features", []):
            lines.append(
                f'olist_feature_drift_psi{{feature="{_escape(item["feature"])}"}} {item["psi"]}'
            )
        lines.extend(
            [
                "# HELP olist_feature_drift_psi_max Largest index across monitored features.",
                "# TYPE olist_feature_drift_psi_max gauge",
                f"olist_feature_drift_psi_max {float(drift.get('max_psi', 0.0))}",
            ]
        )

    lines.extend(
        [
            "# HELP olist_model_info Loaded model, as a label on a constant.",
            "# TYPE olist_model_info gauge",
            f'olist_model_info{{model_name="{_escape(model_name)}"}} 1',
        ]
    )
    return "\n".join(lines) + "\n"
