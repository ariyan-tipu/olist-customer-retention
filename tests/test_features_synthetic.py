"""Feature snapshots on the synthetic environment: shape and hygiene."""

from __future__ import annotations

import pandas as pd

from olist_retention.feature_names import ALL_FEATURES
from olist_retention.models.datasets import load_snapshot
from tests.conftest import TEST_CUTOFF, TRAIN_CUTOFF


def test_snapshots_load_with_expected_columns(synthetic_snapshots):
    for cutoff in (TRAIN_CUTOFF, TEST_CUTOFF):
        data = load_snapshot(synthetic_snapshots.features_dir, cutoff)
        assert set(ALL_FEATURES).issubset(data.X.columns)
        assert len(data) > 50
        assert 0 < data.base_rate < 1


def test_feature_hygiene(synthetic_snapshots):
    data = load_snapshot(synthetic_snapshots.features_dir, TEST_CUTOFF)

    assert data.ids.is_unique
    assert set(pd.unique(data.y)) <= {0, 1}
    assert (data.X["frequency"] >= 1).all()
    assert data.X["customer_region"].notna().all()
    assert data.X["recency_days"].ge(0).all()
    # Rates are genuinely rates.
    for column in ("late_delivery_rate", "share_low_reviews", "share_credit_card"):
        values = data.X[column].dropna()
        assert values.between(0, 1).all(), column


def test_later_cutoff_sees_more_customers(synthetic_snapshots):
    early = load_snapshot(synthetic_snapshots.features_dir, TRAIN_CUTOFF)
    late = load_snapshot(synthetic_snapshots.features_dir, TEST_CUTOFF)
    assert len(late) > len(early)
