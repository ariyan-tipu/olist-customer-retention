"""End-to-end training run on the synthetic snapshots."""

from __future__ import annotations

import json

import joblib

from olist_retention.feature_names import ALL_FEATURES
from olist_retention.models.train import ARTEFACT_VERSION


def test_training_produces_a_complete_result(trained_model):
    assert trained_model.winner in {"logistic_regression", "gradient_boosting"}

    metrics = trained_model.metrics
    for model_name in ("logistic_regression", "gradient_boosting"):
        assert metrics["cross_validation"][model_name]["average_precision_mean"] > 0
        holdout = metrics["holdout"][model_name]
        assert 0 < holdout["roc_auc"] <= 1
        assert 0 < holdout["pr_auc"] <= 1
    assert metrics["n_train"] > 0 and metrics["n_test"] > 0


def test_artefact_round_trips(trained_model):
    artefact = joblib.load(trained_model.model_path)
    assert artefact["model_card"]["artefact_version"] == ARTEFACT_VERSION
    assert artefact["model_card"]["model_name"] == trained_model.winner

    pipeline = artefact["pipeline"]
    assert hasattr(pipeline, "predict_proba")


def test_metrics_file_is_written_and_readable(trained_model):
    payload = json.loads(trained_model.metrics_path.read_text(encoding="utf-8"))
    assert payload["winner"] == trained_model.winner
    importance = payload["permutation_importance_top15"]
    assert 0 < len(importance) <= 15
    named = {row["feature"] for row in importance}
    assert named.issubset(set(ALL_FEATURES))
