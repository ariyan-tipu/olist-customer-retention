"""The feature contract shared by the pipeline, the models and the API.

This module has no third party imports on purpose. The Spark pipeline,
the scikit-learn training code and the scoring service all need the same
column names, but the scoring container ships without Spark, so the
contract cannot live in a module that imports pyspark.
"""

from __future__ import annotations

ID_COLUMN = "customer_unique_id"
LABEL_COLUMN = "label_repeat"
CUTOFF_COLUMN = "cutoff"

NUMERIC_FEATURES: tuple[str, ...] = (
    "recency_days",
    "tenure_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "avg_installments",
    "share_credit_card",
    "n_payment_methods",
    "n_delivered_orders",
    "avg_delivery_delay_days",
    "max_delivery_delay_days",
    "late_delivery_rate",
    "avg_transit_days",
    "avg_review_score",
    "min_review_score",
    "n_reviews",
    "share_low_reviews",
    "has_comment_rate",
    "avg_items_per_order",
    "avg_basket_price",
    "avg_freight_share",
    "n_distinct_categories",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("customer_region", "top_category")

ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
