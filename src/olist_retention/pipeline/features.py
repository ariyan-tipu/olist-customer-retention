"""Point-in-time customer features and repeat purchase labels.

Everything here is anchored to a cutoff date. Features may only use
information available at the cutoff; the label is any further purchase in
the ``horizon_days`` window after it. Getting this boundary right is most
of the value of the module, so the rules are spelled out:

* Purchases count towards recency, frequency and monetary features when
  they were placed before the cutoff.
* Delivery features additionally require the parcel to have arrived by the
  cutoff. An order still in transit has no known delay yet.
* Review features require the review to have been written by the cutoff.
* The label window is strictly after the cutoff: ``(cutoff, cutoff + H]``.

One disclosed approximation: the snapshot stores only each order's final
status, not its history, so cancelled orders are excluded from features
even where the cancellation happened after the cutoff. Cancellations are
about one percent of orders, and the alternative (event sourcing) is not
possible from this dataset.

Missing values are left as nulls on purpose. How to impute them is a
modelling decision that belongs inside the scikit-learn pipeline, where it
is fitted on training folds only, rather than being baked invisibly into
the feature store.

The grain is ``customer_unique_id`` throughout. Joining on ``customer_id``
looks right and silently produces an order-level analysis, because Olist
issues a fresh ``customer_id`` for every order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from olist_retention.config import DataConfig
from olist_retention.feature_names import (
    ALL_FEATURES,
    CUTOFF_COLUMN,
    ID_COLUMN,
    LABEL_COLUMN,
)
from olist_retention.logging_utils import get_logger
from olist_retention.pipeline.geo import region_for
from olist_retention.pipeline.session import timestamp_literal
from olist_retention.pipeline.staging import read_staged

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureSummary:
    cutoff: str
    horizon_days: int
    n_customers: int
    n_positive: int
    base_rate: float

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2) + "\n", encoding="utf-8")


class FeatureBuilder:
    """Build the customer feature table for one cutoff date."""

    def __init__(
        self,
        spark: SparkSession,
        data_config: DataConfig,
        cutoff: date,
        horizon_days: int,
    ) -> None:
        self.spark = spark
        self.config = data_config
        self.cutoff = cutoff
        self.horizon_days = horizon_days
        self._cutoff_ts = datetime.combine(cutoff, time.min)
        self._horizon_end_ts = self._cutoff_ts + timedelta(days=horizon_days)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> tuple[Path, FeatureSummary]:
        """Build, persist and summarise the feature table."""
        features = self.build()

        target = self.config.features_dir / f"cutoff={self.cutoff.isoformat()}"
        # A few tens of thousands of rows: one parquet file reads faster
        # than a directory of shards. Partitioning pays above this scale,
        # not at it.
        features.coalesce(1).write.mode("overwrite").parquet(str(target))

        counts = features.agg(
            F.count("*").alias("n"), F.sum(LABEL_COLUMN).alias("positives")
        ).first()
        summary = FeatureSummary(
            cutoff=self.cutoff.isoformat(),
            horizon_days=self.horizon_days,
            n_customers=int(counts["n"]),
            n_positive=int(counts["positives"]),
            base_rate=round(float(counts["positives"]) / float(counts["n"]), 6),
        )
        summary.write(self.config.outputs_dir / f"features_summary_{self.cutoff.isoformat()}.json")
        logger.info(
            "Features at %s: %d customers, base repeat rate %.3f%%",
            self.cutoff,
            summary.n_customers,
            100 * summary.base_rate,
        )
        return target, summary

    def build(self) -> DataFrame:
        """Return the feature DataFrame without writing it."""
        orders = self._orders_with_customer()

        # The spine is reused by every aggregate below, so one deliberate
        # materialisation avoids re-reading and re-joining staged tables
        # each time an action fires. Everything else stays lazy.
        purchases = self._purchases_before_cutoff(orders).cache()

        spine = purchases.select(ID_COLUMN).distinct()
        rfm = self._rfm(purchases)
        payment = self._payment_features(purchases)
        delivery = self._delivery_features(purchases)
        reviews = self._review_features(purchases)
        basket = self._basket_features(purchases)
        geography = self._geography(purchases)
        labels = self._labels(orders)

        features = (
            spine.join(rfm, ID_COLUMN, "left")
            .join(payment, ID_COLUMN, "left")
            .join(delivery, ID_COLUMN, "left")
            .join(reviews, ID_COLUMN, "left")
            .join(basket, ID_COLUMN, "left")
            .join(geography, ID_COLUMN, "left")
            .join(labels, ID_COLUMN, "left")
            .withColumn(LABEL_COLUMN, F.coalesce(F.col(LABEL_COLUMN), F.lit(0)))
            .withColumn(CUTOFF_COLUMN, F.lit(self.cutoff.isoformat()))
        )
        ordered = [ID_COLUMN, CUTOFF_COLUMN, *ALL_FEATURES, LABEL_COLUMN]
        return features.select(*ordered)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _orders_with_customer(self) -> DataFrame:
        orders = read_staged(self.spark, self.config.staging_dir, "orders")
        customers = read_staged(self.spark, self.config.staging_dir, "customers").select(
            "customer_id", ID_COLUMN, "customer_state"
        )
        # orders and customers are the same size (one customer row per
        # order), so neither side is broadcast; this join shuffles once
        # and everything after it is keyed by customer_unique_id.
        return orders.join(customers, "customer_id", "inner")

    def _purchases_before_cutoff(self, orders: DataFrame) -> DataFrame:
        return orders.where(
            (F.col("order_purchase_timestamp") < timestamp_literal(self._cutoff_ts))
            & (F.col("order_status") != "canceled")
        )

    def _rfm(self, purchases: DataFrame) -> DataFrame:
        cutoff = timestamp_literal(self._cutoff_ts)
        return purchases.groupBy(ID_COLUMN).agg(
            F.datediff(cutoff, F.max("order_purchase_timestamp")).alias("recency_days"),
            F.datediff(cutoff, F.min("order_purchase_timestamp")).alias("tenure_days"),
            F.count("order_id").alias("frequency"),
        )

    def _payment_features(self, purchases: DataFrame) -> DataFrame:
        payments = read_staged(self.spark, self.config.staging_dir, "order_payments")
        per_order = payments.groupBy("order_id").agg(
            F.sum("payment_value").alias("order_value"),
            F.max("payment_installments").alias("order_installments"),
            F.max((F.col("payment_type") == "credit_card").cast("int")).alias("used_credit"),
        )
        joined = purchases.select(ID_COLUMN, "order_id").join(per_order, "order_id", "left")
        per_order_types = purchases.select(ID_COLUMN, "order_id").join(
            payments.select("order_id", "payment_type"), "order_id", "left"
        )
        methods = per_order_types.groupBy(ID_COLUMN).agg(
            F.countDistinct("payment_type").alias("n_payment_methods")
        )
        aggregated = joined.groupBy(ID_COLUMN).agg(
            F.sum("order_value").alias("monetary_total"),
            F.avg("order_value").alias("monetary_avg"),
            F.avg("order_installments").alias("avg_installments"),
            F.avg("used_credit").alias("share_credit_card"),
        )
        return aggregated.join(methods, ID_COLUMN, "left")

    def _delivery_features(self, purchases: DataFrame) -> DataFrame:
        # Only deliveries that had happened by the cutoff are knowable.
        delivered = purchases.where(
            F.col("order_delivered_customer_date").isNotNull()
            & (F.col("order_delivered_customer_date") <= timestamp_literal(self._cutoff_ts))
        ).withColumns(
            {
                "delay_days": F.datediff(
                    "order_delivered_customer_date", "order_estimated_delivery_date"
                ),
                "transit_days": F.datediff(
                    "order_delivered_customer_date", "order_purchase_timestamp"
                ),
            }
        )
        return delivered.groupBy(ID_COLUMN).agg(
            F.count("order_id").alias("n_delivered_orders"),
            F.avg("delay_days").alias("avg_delivery_delay_days"),
            F.max("delay_days").alias("max_delivery_delay_days"),
            F.avg((F.col("delay_days") > 0).cast("double")).alias("late_delivery_rate"),
            F.avg("transit_days").alias("avg_transit_days"),
        )

    def _review_features(self, purchases: DataFrame) -> DataFrame:
        reviews = read_staged(self.spark, self.config.staging_dir, "order_reviews").where(
            F.col("review_creation_date") <= timestamp_literal(self._cutoff_ts)
        )
        # Some orders carry several reviews; the latest one written by the
        # cutoff wins. review_id breaks exact-timestamp ties determinately.
        latest = Window.partitionBy("order_id").orderBy(
            F.desc("review_creation_date"), F.desc("review_id")
        )
        deduped = (
            reviews.withColumn("rn", F.row_number().over(latest)).where(F.col("rn") == 1).drop("rn")
        )
        joined = purchases.select(ID_COLUMN, "order_id").join(deduped, "order_id", "inner")
        return joined.groupBy(ID_COLUMN).agg(
            F.avg("review_score").alias("avg_review_score"),
            F.min("review_score").alias("min_review_score"),
            F.count("review_id").alias("n_reviews"),
            F.avg((F.col("review_score") <= 2).cast("double")).alias("share_low_reviews"),
            F.avg(F.col("review_comment_message").isNotNull().cast("double")).alias(
                "has_comment_rate"
            ),
        )

    def _basket_features(self, purchases: DataFrame) -> DataFrame:
        items = read_staged(self.spark, self.config.staging_dir, "order_items")
        products = read_staged(self.spark, self.config.staging_dir, "products").select(
            "product_id", "product_category_name"
        )
        translation = read_staged(self.spark, self.config.staging_dir, "category_translation")

        # Products (~33k rows) and the category translation (71 rows) are
        # small enough to ship to every executor, turning both joins into
        # map-side hash joins with no shuffle of the item rows. Spark's
        # auto-broadcast threshold would usually spot this, but making it
        # explicit documents the intent and survives config drift.
        catalogue = F.broadcast(
            products.join(F.broadcast(translation), "product_category_name", "left").select(
                "product_id",
                F.coalesce("product_category_name_english", "product_category_name").alias(
                    "category"
                ),
            )
        )

        order_items = (
            purchases.select(ID_COLUMN, "order_id")
            .join(items, "order_id", "inner")
            .join(catalogue, "product_id", "left")
        )

        per_order = order_items.groupBy(ID_COLUMN, "order_id").agg(
            F.count("order_item_id").alias("n_items"),
            F.sum("price").alias("basket_price"),
            F.sum("freight_value").alias("basket_freight"),
        )
        per_customer = per_order.groupBy(ID_COLUMN).agg(
            F.avg("n_items").alias("avg_items_per_order"),
            F.avg("basket_price").alias("avg_basket_price"),
            F.avg(
                F.col("basket_freight") / (F.col("basket_price") + F.col("basket_freight"))
            ).alias("avg_freight_share"),
        )

        category_counts = order_items.groupBy(ID_COLUMN, "category").agg(F.count("*").alias("n"))
        top_window = Window.partitionBy(ID_COLUMN).orderBy(F.desc("n"), F.asc("category"))
        top_category = (
            category_counts.withColumn("rn", F.row_number().over(top_window))
            .where(F.col("rn") == 1)
            .select(ID_COLUMN, F.col("category").alias("top_category"))
        )
        distinct_categories = category_counts.groupBy(ID_COLUMN).agg(
            F.countDistinct("category").alias("n_distinct_categories")
        )
        return per_customer.join(top_category, ID_COLUMN, "left").join(
            distinct_categories, ID_COLUMN, "left"
        )

    def _geography(self, purchases: DataFrame) -> DataFrame:
        latest = Window.partitionBy(ID_COLUMN).orderBy(
            F.desc("order_purchase_timestamp"), F.desc("order_id")
        )
        return (
            purchases.withColumn("rn", F.row_number().over(latest))
            .where(F.col("rn") == 1)
            .select(ID_COLUMN, region_for(F.col("customer_state")).alias("customer_region"))
        )

    def _labels(self, orders: DataFrame) -> DataFrame:
        window_orders = orders.where(
            (F.col("order_purchase_timestamp") > timestamp_literal(self._cutoff_ts))
            & (F.col("order_purchase_timestamp") <= timestamp_literal(self._horizon_end_ts))
            & (F.col("order_status") != "canceled")
        )
        return window_orders.select(ID_COLUMN).distinct().withColumn(LABEL_COLUMN, F.lit(1))
