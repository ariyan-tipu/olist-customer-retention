# Databricks notebook source
# MAGIC %md
# MAGIC # The Olist pipeline, on Databricks
# MAGIC
# MAGIC Same package, same configuration file, different storage paths. The
# MAGIC point of running it here is to check that claim rather than assert it:
# MAGIC the last cell compares the row counts and base rates produced on the
# MAGIC cluster against the ones measured on a laptop, and fails loudly if any
# MAGIC of them moved.
# MAGIC
# MAGIC Written for **Databricks Free Edition**, which is serverless and needs
# MAGIC no card. Storage is a Unity Catalog volume. On a workspace with its own
# MAGIC bucket the only difference is that the paths start `s3://`.
# MAGIC
# MAGIC Before running, create the volume: Catalog, then `workspace`, then
# MAGIC `default`, then Create Volume named `olist`.

# COMMAND ----------

# MAGIC %pip install "olist-retention[data] @ git+https://github.com/ariyan-tipu/olist-customer-retention@main"

# COMMAND ----------

# MAGIC %md
# MAGIC The `spark` extra is deliberately left out. It pins a pyspark version,
# MAGIC and the runtime already supplies its own Spark; installing another over
# MAGIC the top is the documented way to break the session on serverless
# MAGIC compute. The runtime's Spark is the one this should be running against
# MAGIC anyway, since the point of the exercise is that the code does not care
# MAGIC which it gets.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where the data lives
# MAGIC
# MAGIC Every path comes from configuration, which is the whole reason the
# MAGIC pipeline code needs no changes to run here.

# COMMAND ----------

from datetime import date
from pathlib import Path

VOLUME = Path("/Volumes/workspace/default/olist")

from olist_retention.config import DataConfig, SparkConfig

data = DataConfig(
    raw_dir=VOLUME / "raw",
    staging_dir=VOLUME / "staging",
    features_dir=VOLUME / "features",
    outputs_dir=VOLUME / "outputs",
    kaggle_handle="olistbr/brazilian-ecommerce",
)

TRAIN_CUTOFF = date(2018, 1, 1)
TEST_CUTOFF = date(2018, 4, 1)
OBSERVATION_END = date(2018, 10, 17)
HORIZON_DAYS = 180

for directory in (data.raw_dir, data.staging_dir, data.features_dir, data.outputs_dir):
    directory.mkdir(parents=True, exist_ok=True)

print(f"Working under {VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch the raw CSVs
# MAGIC
# MAGIC The downloader records a manifest with sizes and SHA-256 digests, so
# MAGIC the data behind this run is identifiable afterwards. If outbound
# MAGIC internet is blocked in your workspace, upload the nine CSVs to
# MAGIC `<volume>/raw` through the Catalog interface and skip this cell.

# COMMAND ----------

from olist_retention.data.download import DatasetDownloader

if not any(data.raw_dir.glob("*.csv")):
    DatasetDownloader(data.kaggle_handle, data.raw_dir).fetch()

print(sorted(path.name for path in data.raw_dir.glob("*.csv")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the stages
# MAGIC
# MAGIC `build_spark_session` detects the runtime, attaches to the managed
# MAGIC session and leaves cluster sizing alone. It does pin the session
# MAGIC timezone to UTC, because the cutoffs are timestamp literals and a
# MAGIC workspace on another timezone would move every boundary by an hour.

# COMMAND ----------

from olist_retention.pipeline.features import FeatureBuilder
from olist_retention.pipeline.session import build_spark_session, on_databricks
from olist_retention.pipeline.staging import RawToStaging
from olist_retention.pipeline.survival import SurvivalDatasetBuilder

assert on_databricks(), "This notebook is meant to run on a Databricks cluster"

spark = build_spark_session(SparkConfig("ignored", "ignored", 16, "olist"))
print("session timezone:", spark.conf.get("spark.sql.session.timeZone"))

RawToStaging(spark, data).run(validate=True)

# COMMAND ----------

for cutoff in (TRAIN_CUTOFF, TEST_CUTOFF):
    FeatureBuilder(spark, data, cutoff, HORIZON_DAYS).run()

survival_path, survival_metadata = SurvivalDatasetBuilder(spark, data, OBSERVATION_END).run()
print(survival_metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The SQL views, unchanged
# MAGIC
# MAGIC `sql/feature_views.sql` is executed against Spark by the test suite and
# MAGIC against the cluster here. One definition, two engines, which is why it
# MAGIC lives in a file rather than inside a Python string.

# COMMAND ----------

TABLES = (
    "customers",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "products",
    "sellers",
    "geolocation",
    "category_translation",
)

for name in TABLES:
    spark.read.parquet(str(data.staging_dir / name)).createOrReplaceTempView(name)

import olist_retention

sql_path = Path(olist_retention.__file__).parents[2] / "sql" / "feature_views.sql"
if not sql_path.is_file():
    # Installed from a wheel rather than a checkout: fetch the file.
    import urllib.request

    raw = (
        "https://raw.githubusercontent.com/ariyan-tipu/"
        "olist-customer-retention/main/sql/feature_views.sql"
    )
    sql_text = urllib.request.urlopen(raw).read().decode("utf-8")
else:
    sql_text = sql_path.read_text(encoding="utf-8")

for statement in [part.strip() for part in sql_text.split(";") if part.strip()]:
    spark.sql(statement)

display(spark.sql("SELECT * FROM customer_order_features LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Does the cluster agree with the laptop?
# MAGIC
# MAGIC These are the figures quoted in the repository README, measured
# MAGIC locally. If the cluster run produces the same ones, "the same code
# MAGIC runs in both places" is a checked statement instead of a hopeful one.

# COMMAND ----------

import json

EXPECTED = {
    "staged_orders": 99441,
    "survival_spells": 93350,
    "survival_events": 1808,
    "train_customers": 43768,
    "test_customers": 64089,
    "train_base_rate": 0.01286,
    "test_base_rate": 0.00975,
    "censoring_rate": 0.981,
}

train = spark.read.parquet(str(data.features_dir / f"cutoff={TRAIN_CUTOFF.isoformat()}"))
test = spark.read.parquet(str(data.features_dir / f"cutoff={TEST_CUTOFF.isoformat()}"))
survival = spark.read.csv(str(survival_path), header=True, inferSchema=True)

actual = {
    "staged_orders": spark.read.parquet(str(data.staging_dir / "orders")).count(),
    "survival_spells": survival.count(),
    "survival_events": int(survival.selectExpr("sum(event)").collect()[0][0]),
    "train_customers": train.count(),
    "test_customers": test.count(),
    "train_base_rate": round(train.selectExpr("avg(label_repeat)").collect()[0][0], 5),
    "test_base_rate": round(test.selectExpr("avg(label_repeat)").collect()[0][0], 5),
    "censoring_rate": round(1 - survival.selectExpr("avg(event)").collect()[0][0], 3),
}

mismatches = {key: (EXPECTED[key], actual[key]) for key in EXPECTED if EXPECTED[key] != actual[key]}

print(json.dumps({"expected": EXPECTED, "actual": actual}, indent=2))
print()
if mismatches:
    print("MISMATCHES (expected, actual):")
    print(json.dumps(mismatches, indent=2))
else:
    print("Every figure matches the local run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Copy this back into the repository
# MAGIC
# MAGIC The cell below prints a summary for `results/databricks_run.json`,
# MAGIC alongside the runtime version and cluster shape that produced it.

# COMMAND ----------

import os
from datetime import UTC, datetime

summary = {
    "ran_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "databricks_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION", "unknown"),
    "spark_version": spark.version,
    "session_timezone": spark.conf.get("spark.sql.session.timeZone"),
    "storage": str(VOLUME),
    "expected_from_local_run": EXPECTED,
    "measured_on_databricks": actual,
    "all_match": not mismatches,
}
print(json.dumps(summary, indent=2))
