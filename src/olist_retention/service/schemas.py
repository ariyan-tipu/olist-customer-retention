"""Request and response contracts for the scoring API.

The feature fields are written out explicitly rather than generated from
the pipeline constants, because an API schema should be readable on its
own. A unit test pins this class to the pipeline's feature list, so the
two cannot drift apart silently.

Every feature is optional: the model's imputers were fitted to handle
missing values, and a scoring service that rejects the rows the model can
handle would be refusing work for no reason.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """One customer's features at the scoring date."""

    model_config = ConfigDict(extra="forbid")

    recency_days: float | None = None
    tenure_days: float | None = None
    frequency: float | None = None
    monetary_total: float | None = None
    monetary_avg: float | None = None
    avg_installments: float | None = None
    share_credit_card: float | None = None
    n_payment_methods: float | None = None
    n_delivered_orders: float | None = None
    avg_delivery_delay_days: float | None = None
    max_delivery_delay_days: float | None = None
    late_delivery_rate: float | None = None
    avg_transit_days: float | None = None
    avg_review_score: float | None = None
    min_review_score: float | None = None
    n_reviews: float | None = None
    share_low_reviews: float | None = None
    has_comment_rate: float | None = None
    avg_items_per_order: float | None = None
    avg_basket_price: float | None = None
    avg_freight_share: float | None = None
    n_distinct_categories: float | None = None
    customer_region: str | None = None
    top_category: str | None = None


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class CustomerScore(BaseModel):
    repeat_probability: float


class ScoreResponse(BaseModel):
    model_name: str
    scores: list[CustomerScore]


class ModelInfo(BaseModel):
    """Subset of the model card the service exposes."""

    model_name: str
    trained_at_utc: str
    sklearn_version: str
    train_cutoff: str
    horizon_days: int
    holdout_metrics: dict
