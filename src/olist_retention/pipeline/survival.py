"""Export the survival analysis dataset: time from first delivery to the
next purchase, with explicit right-censoring.

Each customer whose first order was delivered contributes one spell:

* origin: the delivery timestamp of their first order,
* event: their next purchase (any non-cancelled order) after the origin,
* censoring: customers with no further purchase are censored at the last
  timestamp observed in the dataset, not treated as churned. Deleting or
  mislabelling them is the classic mistake this analysis exists to avoid;
  around ninety-seven percent of Olist customers are censored, so getting
  this wrong changes every estimate.

Covariates describe the first order experience (delivery delay, review
score, basket) because the question is explanatory: how does that first
experience shape the hazard of ever coming back?

A small wrinkle: a handful of customers place their second order
before the first one arrives. Those purchases precede the origin, so the
spell measures the next purchase after delivery, and the count of such
customers is recorded in the metadata rather than swept away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from olist_retention.config import DataConfig
from olist_retention.feature_names import ID_COLUMN
from olist_retention.logging_utils import get_logger
from olist_retention.pipeline.geo import region_for
from olist_retention.pipeline.session import timestamp_literal
from olist_retention.pipeline.staging import read_staged

logger = get_logger(__name__)

# Same-day repurchases produce zero-length spells that a hazard model
# cannot use; half a day is the smallest defensible duration for them.
_MIN_SPELL_DAYS = 0.5

COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "time_days",
    "event",
    "delivery_delay_days",
    "late_delivery",
    "review_score",
    "order_value",
    "n_items",
    "installments",
    "customer_region",
)


@dataclass(frozen=True)
class SurvivalMetadata:
    observation_end: str
    n_spells: int
    n_events: int
    censoring_rate: float
    pre_delivery_repurchasers: int

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2) + "\n", encoding="utf-8")


class SurvivalDatasetBuilder:
    """Build and export the censored time-to-next-purchase dataset."""

    def __init__(self, spark: SparkSession, data_config: DataConfig, observation_end: date) -> None:
        self.spark = spark
        self.config = data_config
        self._end_ts = datetime.combine(observation_end, time.max)
        self.observation_end = observation_end

    def run(self) -> tuple[Path, SurvivalMetadata]:
        spells = self.build().cache()

        stats = spells.agg(
            F.count("*").alias("n"),
            F.sum("event").alias("events"),
        ).first()
        early = self._pre_delivery_repurchasers()

        out_dir = self.config.outputs_dir / "survival"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "survival_dataset.csv"
        # One small CSV is the contract with R. The hop goes through
        # parquet and pyarrow rather than toPandas(): it keeps a columnar
        # artefact alongside the CSV and avoids the Spark to pandas
        # conversion layer, which still lags behind pandas 3.
        parquet_dir = out_dir / "spells.parquet"
        spells.coalesce(1).write.mode("overwrite").parquet(str(parquet_dir))
        pd.read_parquet(parquet_dir).to_csv(csv_path, index=False)

        metadata = SurvivalMetadata(
            observation_end=self.observation_end.isoformat(),
            n_spells=int(stats["n"]),
            n_events=int(stats["events"]),
            censoring_rate=round(1.0 - float(stats["events"]) / float(stats["n"]), 6),
            pre_delivery_repurchasers=early,
        )
        metadata.write(out_dir / "survival_metadata.json")
        logger.info(
            "Survival dataset: %d spells, %d events, censoring rate %.1f%%",
            metadata.n_spells,
            metadata.n_events,
            100 * metadata.censoring_rate,
        )
        return csv_path, metadata

    # ------------------------------------------------------------------

    def _orders_with_customer(self) -> DataFrame:
        orders = read_staged(self.spark, self.config.staging_dir, "orders")
        customers = read_staged(self.spark, self.config.staging_dir, "customers").select(
            "customer_id", ID_COLUMN, "customer_state"
        )
        return orders.join(customers, "customer_id", "inner").where(
            F.col("order_purchase_timestamp") <= timestamp_literal(self._end_ts)
        )

    def _first_delivered_order(self, orders: DataFrame) -> DataFrame:
        delivered = orders.where(
            (F.col("order_status") == "delivered")
            & F.col("order_delivered_customer_date").isNotNull()
        )
        first = Window.partitionBy(ID_COLUMN).orderBy(
            F.asc("order_purchase_timestamp"), F.asc("order_id")
        )
        return (
            delivered.withColumn("rn", F.row_number().over(first))
            .where(F.col("rn") == 1)
            .select(
                ID_COLUMN,
                F.col("order_id").alias("first_order_id"),
                F.col("order_purchase_timestamp").alias("first_purchase_ts"),
                F.col("order_delivered_customer_date").alias("origin_ts"),
                F.datediff("order_delivered_customer_date", "order_estimated_delivery_date").alias(
                    "delivery_delay_days"
                ),
                region_for(F.col("customer_state")).alias("customer_region"),
            )
            .withColumn("late_delivery", (F.col("delivery_delay_days") > 0).cast("int"))
        )

    def build(self) -> DataFrame:
        orders = self._orders_with_customer()
        first = self._first_delivered_order(orders)

        next_purchase = (
            orders.where(F.col("order_status") != "canceled")
            .select(ID_COLUMN, "order_purchase_timestamp")
            .join(first.select(ID_COLUMN, "origin_ts"), ID_COLUMN, "inner")
            .where(F.col("order_purchase_timestamp") > F.col("origin_ts"))
            .groupBy(ID_COLUMN)
            .agg(F.min("order_purchase_timestamp").alias("event_ts"))
        )

        spells = first.join(next_purchase, ID_COLUMN, "left").withColumns(
            {
                "event": F.col("event_ts").isNotNull().cast("int"),
                "end_ts": F.coalesce(F.col("event_ts"), timestamp_literal(self._end_ts)),
            }
        )
        spells = spells.withColumn(
            "time_days",
            F.greatest(
                (F.unix_timestamp("end_ts") - F.unix_timestamp("origin_ts")).cast("double")
                / F.lit(86400.0),
                F.lit(_MIN_SPELL_DAYS),
            ),
        )

        covariates = self._first_order_covariates(first)
        return spells.join(covariates, "first_order_id", "left").select(
            ID_COLUMN,
            F.round("time_days", 3).alias("time_days"),
            "event",
            "delivery_delay_days",
            "late_delivery",
            "review_score",
            F.round("order_value", 2).alias("order_value"),
            "n_items",
            "installments",
            "customer_region",
        )

    def _first_order_covariates(self, first: DataFrame) -> DataFrame:
        payments = read_staged(self.spark, self.config.staging_dir, "order_payments")
        items = read_staged(self.spark, self.config.staging_dir, "order_items")
        reviews = read_staged(self.spark, self.config.staging_dir, "order_reviews")

        first_ids = F.broadcast(first.select("first_order_id"))

        payment = (
            payments.join(first_ids, payments.order_id == first_ids.first_order_id, "inner")
            .groupBy("first_order_id")
            .agg(
                F.sum("payment_value").alias("order_value"),
                F.max("payment_installments").alias("installments"),
            )
        )
        basket = (
            items.join(first_ids, items.order_id == first_ids.first_order_id, "inner")
            .groupBy("first_order_id")
            .agg(F.count("order_item_id").alias("n_items"))
        )

        latest = Window.partitionBy("order_id").orderBy(
            F.desc("review_creation_date"), F.desc("review_id")
        )
        review = (
            reviews.withColumn("rn", F.row_number().over(latest))
            .where(F.col("rn") == 1)
            .join(first_ids, reviews.order_id == first_ids.first_order_id, "inner")
            .select("first_order_id", "review_score")
        )

        return payment.join(basket, "first_order_id", "outer").join(
            review, "first_order_id", "outer"
        )

    def _pre_delivery_repurchasers(self) -> int:
        orders = self._orders_with_customer()
        first = self._first_delivered_order(orders)
        return (
            orders.where(F.col("order_status") != "canceled")
            .join(first, ID_COLUMN, "inner")
            .where(
                (F.col("order_purchase_timestamp") > F.col("first_purchase_ts"))
                & (F.col("order_purchase_timestamp") <= F.col("origin_ts"))
            )
            .select(ID_COLUMN)
            .distinct()
            .count()
        )
