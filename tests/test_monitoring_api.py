"""The service reports on itself: latency, drift and the scrape format.

A small artefact is built here rather than reused from the Spark
fixtures, so the monitoring surface is covered on a machine with no JDK.
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from olist_retention.feature_names import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from olist_retention.models.train import ARTEFACT_VERSION, build_preprocessor
from olist_retention.monitoring.cloudwatch import CloudWatchPublisher
from olist_retention.monitoring.drift import build_reference
from olist_retention.service.app import create_app

REGIONS = ("sudeste", "nordeste", "sul")


def _training_frame(rng: np.random.Generator, n: int, recency_centre: float = 90.0) -> pd.DataFrame:
    data = {name: rng.gamma(2.0, 5.0, n) for name in NUMERIC_FEATURES}
    data["recency_days"] = rng.normal(recency_centre, 20.0, n)
    data["customer_region"] = rng.choice(REGIONS, n)
    data["top_category"] = rng.choice(("cama_mesa_banho", "beleza_saude"), n)
    return pd.DataFrame(data)[list(ALL_FEATURES)]


@pytest.fixture(scope="module")
def artefact(tmp_path_factory):
    rng = np.random.default_rng(0)
    frame = _training_frame(rng, 1500)
    labels = rng.binomial(1, 0.01, len(frame))

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    ).fit(frame, labels)

    path = tmp_path_factory.mktemp("artefact") / "churn_model.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_card": {
                "artefact_version": ARTEFACT_VERSION,
                "model_name": "logistic_regression",
                "trained_at_utc": "2026-08-07T00:00:00+00:00",
                "sklearn_version": "1.9.0",
                "train_cutoff": "2018-01-01",
                "test_cutoff": "2018-04-01",
                "horizon_days": 180,
                "holdout_metrics": {"roc_auc": 0.61},
                "reference_distribution": build_reference(
                    frame, NUMERIC_FEATURES, CATEGORICAL_FEATURES
                ),
            },
        },
        path,
    )
    return path


def _score_many(client: TestClient, frame: pd.DataFrame, chunk: int = 100) -> None:
    records = frame.to_dict("records")
    for start in range(0, len(records), chunk):
        payload = {"customers": records[start : start + chunk]}
        assert client.post("/v1/score", json=payload).status_code == 200


def test_monitoring_reports_latency_after_scoring(artefact):
    client = TestClient(create_app(model_path=str(artefact)))
    rng = np.random.default_rng(1)
    _score_many(client, _training_frame(rng, 400))

    payload = client.get("/v1/monitoring").json()
    # Four batched calls carrying a hundred customers each: latency is
    # counted per request, drift per row.
    assert payload["latency"]["requests_total"] == 4
    assert payload["drift"]["rows_scored_total"] == 400
    assert payload["latency"]["p50_ms"] > 0.0
    assert payload["latency"]["p99_ms"] >= payload["latency"]["p50_ms"]


def test_steady_traffic_reports_no_material_drift(artefact):
    client = TestClient(create_app(model_path=str(artefact)))
    rng = np.random.default_rng(2)
    _score_many(client, _training_frame(rng, 1200))

    drift = client.get("/v1/monitoring").json()["drift"]
    assert drift["status"] == "ok"
    assert drift["material_features"] == []


def test_shifted_traffic_is_reported_as_material_drift(artefact):
    client = TestClient(create_app(model_path=str(artefact)))
    rng = np.random.default_rng(3)
    _score_many(client, _training_frame(rng, 1200, recency_centre=260.0))

    drift = client.get("/v1/monitoring").json()["drift"]
    assert drift["max_psi_feature"] == "recency_days"
    assert "recency_days" in drift["material_features"]


def test_the_metrics_endpoint_speaks_prometheus(artefact):
    client = TestClient(create_app(model_path=str(artefact)))
    rng = np.random.default_rng(4)
    _score_many(client, _training_frame(rng, 300))

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    body = response.text
    assert "# TYPE olist_score_rows_total counter" in body
    assert "olist_score_requests_total 3.0" in body
    assert "olist_score_rows_total 300.0" in body
    assert 'olist_score_latency_milliseconds{quantile="p99"}' in body
    assert 'olist_model_info{model_name="logistic_regression"} 1' in body
    for line in body.splitlines():
        if not line.startswith("#"):
            assert len(line.rsplit(" ", 1)) == 2


def test_metrics_answer_before_a_model_is_loaded(tmp_path):
    client = TestClient(create_app(model_path=str(tmp_path / "absent.joblib")))
    response = client.get("/metrics")
    assert response.status_code == 200
    assert 'model_name="unloaded"' in response.text


def test_a_version_one_artefact_is_refused(tmp_path):
    path = tmp_path / "old.joblib"
    joblib.dump({"pipeline": None, "model_card": {"artefact_version": 1}}, path)
    client = TestClient(create_app(model_path=str(path)))

    response = client.get("/readyz")
    assert response.status_code == 503
    assert "reference distribution" in response.json()["detail"]


def test_each_score_writes_one_structured_log_line(artefact):
    """One machine-readable line per request, so log collection needs no regex.

    A handler is attached by hand instead of using caplog, because the
    package logger does not propagate to root and caplog's own plumbing
    then reports the same record more than once.
    """
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _Capture()
    logger = logging.getLogger("olist_retention.service.monitoring_state")
    logger.addHandler(handler)
    try:
        client = TestClient(create_app(model_path=str(artefact)))
        client.post("/v1/score", json={"customers": [{"recency_days": 30.0}]})
    finally:
        logger.removeHandler(handler)

    events = [json.loads(line) for line in captured if line.startswith("{")]
    assert len(events) == 1
    assert events[0]["event"] == "score"
    assert events[0]["rows"] == 1
    assert events[0]["model_name"] == "logistic_regression"
    assert events[0]["duration_ms"] > 0.0


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def put_metric_data(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_cloudwatch_stays_inside_the_always_free_metric_allowance(artefact):
    """Ten custom metrics are free forever; this must not exceed that."""
    client = TestClient(create_app(model_path=str(artefact)))
    rng = np.random.default_rng(5)
    _score_many(client, _training_frame(rng, 300))

    payload = client.get("/v1/monitoring").json()
    latency, drift = payload["latency"], payload["drift"]
    recorder = _RecordingClient()
    publisher = CloudWatchPublisher(client=recorder)

    assert publisher.publish_once(latency, drift) is True
    names = {item["MetricName"] for item in recorder.calls[0]["MetricData"]}
    assert len(names) <= 10
    assert {"RowsScored", "LatencyP99", "MaxFeaturePSI"} <= names


def test_a_failing_cloudwatch_call_never_breaks_scoring():
    class Broken:
        def put_metric_data(self, **kwargs):
            raise RuntimeError("throttled")

    publisher = CloudWatchPublisher(client=Broken())
    assert publisher.publish_once({"requests_total": 1, "p50_ms": 1.0}, {"status": "ok"}) is False
