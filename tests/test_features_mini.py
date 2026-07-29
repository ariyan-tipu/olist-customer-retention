"""Point-in-time correctness of the feature builder, pinned exactly.

The handcrafted dataset in ``mini_data`` makes every expected value
computable by hand. If any of these fail, the feature table is leaking
future information or losing history, which no aggregate metric on real
data would reveal.
"""

from __future__ import annotations

import math

import pytest


def test_cancelled_only_customer_is_excluded(mini_features):
    assert "CU_D" not in mini_features.index


def test_expected_cohort(mini_features):
    assert set(mini_features.index) == {"CU_A", "CU_B", "CU_C", "CU_E", "CU_F", "CU_G"}


def test_labels_respect_the_horizon_window(mini_features):
    labels = mini_features["label_repeat"].to_dict()
    assert labels == {
        "CU_A": 1,  # repeats 2018-02-15, inside the window
        "CU_B": 0,
        "CU_C": 0,
        "CU_E": 0,  # repeats 2018-08-01, after the 2018-06-30 window end
        "CU_F": 1,
        "CU_G": 0,
    }


def test_rfm_counts_only_pre_cutoff_purchases(mini_features):
    row = mini_features.loc["CU_A"]
    assert row["frequency"] == 1
    assert row["recency_days"] == 52
    assert row["tenure_days"] == 52
    assert row["monetary_total"] == pytest.approx(120.0)

    row_g = mini_features.loc["CU_G"]
    assert row_g["frequency"] == 2
    assert row_g["monetary_total"] == pytest.approx(240.0)


def test_delivery_still_in_transit_at_cutoff_is_unknown(mini_features):
    """CU_C ordered before the cutoff but delivery arrived after it."""
    row = mini_features.loc["CU_C"]
    assert row["frequency"] == 1
    assert math.isnan(row["n_delivered_orders"])
    assert math.isnan(row["avg_delivery_delay_days"])
    assert math.isnan(row["late_delivery_rate"])


def test_review_written_after_cutoff_is_unknown(mini_features):
    row = mini_features.loc["CU_C"]
    assert math.isnan(row["avg_review_score"])
    assert math.isnan(row["n_reviews"])


def test_delivery_aggregates(mini_features):
    row_b = mini_features.loc["CU_B"]
    assert row_b["avg_delivery_delay_days"] == pytest.approx(10.0)
    assert row_b["late_delivery_rate"] == pytest.approx(1.0)

    row_g = mini_features.loc["CU_G"]
    assert row_g["avg_delivery_delay_days"] == pytest.approx(-5.0)
    assert row_g["late_delivery_rate"] == pytest.approx(0.5)
    assert row_g["max_delivery_delay_days"] == pytest.approx(5.0)


def test_latest_of_multiple_reviews_wins(mini_features):
    row = mini_features.loc["CU_F"]
    assert row["avg_review_score"] == pytest.approx(4.0)
    assert row["n_reviews"] == 1
    assert row["share_low_reviews"] == pytest.approx(0.0)


def test_basket_features(mini_features):
    row = mini_features.loc["CU_F"]
    assert row["avg_items_per_order"] == pytest.approx(2.0)
    assert row["n_distinct_categories"] == 2
    # Tie on item counts breaks alphabetically.
    assert row["top_category"] == "telephony"
    assert row["avg_freight_share"] == pytest.approx(20.0 / 140.0)


def test_regions_are_mapped(mini_features):
    assert mini_features.loc["CU_A", "customer_region"] == "sudeste"
    assert mini_features.loc["CU_B", "customer_region"] == "nordeste"
