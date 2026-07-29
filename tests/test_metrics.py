"""Metric helpers on small arrays where the answer is computable by hand."""

from __future__ import annotations

import numpy as np
import pytest

from olist_retention.models.metrics import classification_metrics, targeting_at_share


def test_targeting_at_share_exact_values():
    # Ten customers, three positives; scores rank one positive first.
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

    # Reported values are rounded to four decimals, hence the absolute
    # tolerance against the exact fractions.
    result = targeting_at_share(y, scores, share=0.10)
    assert result["precision_at_share"] == pytest.approx(1.0, abs=1e-3)
    assert result["recall_at_share"] == pytest.approx(1 / 3, abs=1e-3)

    result_half = targeting_at_share(y, scores, share=0.50)
    assert result_half["precision_at_share"] == pytest.approx(2 / 5, abs=1e-3)
    assert result_half["recall_at_share"] == pytest.approx(2 / 3, abs=1e-3)
    # Base rate is 0.3, so precision 0.4 is a lift of 4/3.
    assert result_half["lift_at_share"] == pytest.approx(1.33, abs=0.01)


def test_classification_metrics_on_a_perfect_ranker():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    metrics = classification_metrics(y, scores)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["base_rate"] == pytest.approx(0.4)
    assert metrics["n"] == 5


def test_share_bounds_are_validated():
    y = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    with pytest.raises(ValueError):
        targeting_at_share(y, scores, share=1.5)
