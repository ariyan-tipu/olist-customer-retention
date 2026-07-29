"""The scoring API: contract, readiness behaviour and real scoring."""

from __future__ import annotations

from fastapi.testclient import TestClient

from olist_retention.feature_names import ALL_FEATURES
from olist_retention.service.app import create_app
from olist_retention.service.schemas import CustomerFeatures


def test_api_schema_cannot_drift_from_the_pipeline():
    """The explicit pydantic fields must mirror the feature list exactly."""
    assert set(CustomerFeatures.model_fields) == set(ALL_FEATURES)


def test_liveness_works_without_a_model(tmp_path):
    client = TestClient(create_app(model_path=str(tmp_path / "absent.joblib")))
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_and_scoring_fail_clearly_without_a_model(tmp_path):
    client = TestClient(create_app(model_path=str(tmp_path / "absent.joblib")))
    assert client.get("/readyz").status_code == 503
    response = client.post("/v1/score", json={"customers": [{}]})
    assert response.status_code == 503
    assert "artefact" in response.json()["detail"].lower()


def test_scoring_round_trip(trained_model):
    client = TestClient(create_app(model_path=str(trained_model.model_path)))

    assert client.get("/readyz").status_code == 200

    info = client.get("/v1/model")
    assert info.status_code == 200
    assert info.json()["model_name"] == trained_model.winner

    request = {
        "customers": [
            {
                "recency_days": 30,
                "frequency": 2,
                "monetary_total": 250.0,
                "avg_review_score": 5,
                "late_delivery_rate": 0.0,
                "customer_region": "sudeste",
            },
            {"recency_days": 200, "frequency": 1, "avg_review_score": 1},
        ]
    }
    response = client.post("/v1/score", json=request)
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == trained_model.winner
    assert len(payload["scores"]) == 2
    for entry in payload["scores"]:
        assert 0.0 <= entry["repeat_probability"] <= 1.0


def test_unknown_fields_are_rejected(trained_model):
    client = TestClient(create_app(model_path=str(trained_model.model_path)))
    response = client.post("/v1/score", json={"customers": [{"not_a_feature": 1.0}]})
    assert response.status_code == 422


def test_empty_batch_is_rejected(trained_model):
    client = TestClient(create_app(model_path=str(trained_model.model_path)))
    assert client.post("/v1/score", json={"customers": []}).status_code == 422
