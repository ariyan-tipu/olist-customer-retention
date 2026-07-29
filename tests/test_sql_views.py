"""Execute the SQL view definitions for real against Spark.

The file in sql/ is the same one an analyst would run on Databricks; here
each statement executes against temporary views over the staged synthetic
tables, so a broken view fails the build rather than a dashboard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from olist_retention.data.schemas import TABLES
from tests.conftest import TEST_CUTOFF

SQL_PATH = Path(__file__).parents[1] / "sql" / "feature_views.sql"

VIEW_NAMES = (
    "v_orders_enriched",
    "v_monthly_cohort_retention",
    "v_delivery_performance_monthly",
    "v_repeat_by_first_review",
    "v_feature_summary_by_region",
)


@pytest.fixture(scope="module")
def sql_views(spark, synthetic_snapshots):
    for name in TABLES:
        frame = spark.read.parquet(str(synthetic_snapshots.staging_dir / name))
        frame.createOrReplaceTempView(name)
    features = spark.read.parquet(
        str(synthetic_snapshots.features_dir / f"cutoff={TEST_CUTOFF.isoformat()}")
    )
    features.createOrReplaceTempView("customer_features")

    executed = 0
    for statement in SQL_PATH.read_text(encoding="utf-8").split(";"):
        # A chunk may open with comment lines; it still executes as long
        # as any real SQL follows them.
        body = "\n".join(
            line for line in statement.splitlines() if not line.strip().startswith("--")
        ).strip()
        if body:
            spark.sql(statement)
            executed += 1
    assert executed == len(VIEW_NAMES)
    return spark


def test_every_view_is_queryable(sql_views):
    for name in VIEW_NAMES:
        assert sql_views.table(name).count() > 0, name


def test_cohort_retention_shape(sql_views):
    rows = sql_views.table("v_monthly_cohort_retention").collect()
    assert all(row["months_since_first"] >= 0 for row in rows)
    assert all(row["active_customers"] > 0 for row in rows)
    # Month zero must exist for every cohort (the cohort itself).
    cohorts = {row["cohort_month"] for row in rows}
    month_zero = {row["cohort_month"] for row in rows if row["months_since_first"] == 0}
    assert cohorts == month_zero


def test_rates_are_rates(sql_views):
    for row in sql_views.table("v_repeat_by_first_review").collect():
        assert 0.0 <= row["repeat_rate"] <= 1.0
    for row in sql_views.table("v_delivery_performance_monthly").collect():
        assert 0.0 <= row["late_rate"] <= 1.0
    for row in sql_views.table("v_feature_summary_by_region").collect():
        assert 0.0 <= row["repeat_rate"] <= 1.0
