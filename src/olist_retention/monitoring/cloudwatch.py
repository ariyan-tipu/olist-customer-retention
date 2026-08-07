"""Ship a small set of metrics to CloudWatch from the running service.

Six metrics, published once a minute. The number is chosen rather than
arrived at: CloudWatch's always-free allowance covers ten custom metrics
and a million API calls a month, so six metrics at one call a minute
stays inside it with room to spare and the monitoring cannot become a
line on a bill.

Everything here degrades quietly. A laptop run has no credentials and no
CloudWatch, and the service must still start and serve; the publisher
logs once that it is disabled and then stays out of the way.
"""

from __future__ import annotations

import threading
from typing import Any

from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)

NAMESPACE = "OlistRetention"
DEFAULT_INTERVAL_SECONDS = 60


class CloudWatchPublisher:
    """Background thread that pushes latency and drift gauges."""

    def __init__(
        self,
        namespace: str = NAMESPACE,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        service_name: str = "olist-scoring",
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.namespace = namespace
        self.interval_seconds = interval_seconds
        self.service_name = service_name
        self._client = client or self._build_client(region)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @staticmethod
    def _build_client(region: str | None) -> Any | None:
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, NoCredentialsError
        except ImportError:
            logger.info("boto3 is absent, so CloudWatch publishing is off")
            return None
        try:
            client = boto3.client("cloudwatch", region_name=region)
            # Touching the credential chain now turns a permanent
            # misconfiguration into one log line at boot instead of a
            # failure every minute for the life of the process.
            boto3.Session().get_credentials()
        except (BotoCoreError, NoCredentialsError) as error:
            logger.info("CloudWatch publishing is off: %s", error)
            return None
        return client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _datum(self, name: str, value: float, unit: str) -> dict[str, Any]:
        return {
            "MetricName": name,
            "Value": float(value),
            "Unit": unit,
            "Dimensions": [{"Name": "ServiceName", "Value": self.service_name}],
        }

    def build_payload(self, latency: dict[str, Any], drift: dict[str, Any]) -> list[dict[str, Any]]:
        payload = [
            self._datum("RequestsServed", latency.get("requests_total", 0), "Count"),
            self._datum("RowsScored", drift.get("rows_scored_total", 0), "Count"),
        ]
        for quantile in ("p50", "p95", "p99"):
            key = f"{quantile}_ms"
            if key in latency:
                payload.append(
                    self._datum(f"Latency{quantile.upper()}", latency[key], "Milliseconds")
                )
        if drift.get("status") == "ok":
            payload.append(self._datum("MaxFeaturePSI", drift.get("max_psi", 0.0), "None"))
            payload.append(
                self._datum(
                    "MaterialDriftFeatures", len(drift.get("material_features", [])), "Count"
                )
            )
        return payload

    def publish_once(self, latency: dict[str, Any], drift: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        payload = self.build_payload(latency, drift)
        if not payload:
            return False
        try:
            self._client.put_metric_data(Namespace=self.namespace, MetricData=payload)
        except Exception as error:  # noqa: BLE001 - telemetry must never break serving
            logger.warning("CloudWatch publish failed: %s", error)
            return False
        return True

    def start(self, collect: Any) -> None:
        """Begin publishing. `collect` returns the latency and drift pair."""
        if not self.enabled or self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(self.interval_seconds):
                latency, drift = collect()
                self.publish_once(latency, drift)

        self._thread = threading.Thread(target=loop, name="cloudwatch-publisher", daemon=True)
        self._thread.start()
        logger.info(
            "Publishing %s metrics to CloudWatch every %ds",
            self.namespace,
            self.interval_seconds,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
