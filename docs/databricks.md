# Running the pipeline on Databricks

The pipeline is written so that the laptop run and the cloud run are the
same code with different configuration. `build_spark_session` attaches to
the Databricks-managed session when it detects the runtime and only
builds its own local session otherwise, and every read and write goes
through paths taken from `conf/pipeline.yaml`, so pointing the pipeline
at cloud storage is a configuration change, not a code change.

The layout maps onto the usual medallion layering:

| Layer  | Contents                                   | Local default    | Cloud example                   |
| ------ | ------------------------------------------ | ---------------- | ------------------------------- |
| Bronze | raw Olist CSVs plus provenance manifest    | `data/raw`       | `s3://<bucket>/olist/raw`       |
| Silver | typed, validated parquet per table         | `data/staging`   | `s3://<bucket>/olist/staging`   |
| Gold   | feature snapshots and the survival export  | `data/features`  | `s3://<bucket>/olist/features`  |

## Free Edition walkthrough (no card required)

Databricks Free Edition gives a serverless workspace at no cost, which
keeps this project's zero-cost rule intact. Storage there is a managed
Unity Catalog volume rather than your own S3 bucket; the S3 story below
is identical apart from the path prefix.

1. Sign up at databricks.com for Free Edition and create a workspace.
2. Create a volume for the project, for example
   `/Volumes/workspace/default/olist`.
3. Upload the nine raw CSVs to `<volume>/raw` (download them locally
   first with `olist-retention download`, or run the downloader in a
   notebook cell).
4. Build the wheel locally and upload it, then install it in a notebook:

   ```
   py -m build --wheel
   %pip install /Volumes/workspace/default/olist/olist_retention-0.1.0-py3-none-any.whl
   ```

5. Point the configuration at the volume and run the stages in a
   notebook cell:

   ```python
   from datetime import date
   from pathlib import Path

   from olist_retention.config import DataConfig
   from olist_retention.pipeline.session import build_spark_session
   from olist_retention.pipeline.staging import RawToStaging
   from olist_retention.pipeline.features import FeatureBuilder
   from olist_retention.pipeline.survival import SurvivalDatasetBuilder
   from olist_retention.config import SparkConfig

   base = Path("/Volumes/workspace/default/olist")
   data = DataConfig(
       raw_dir=base / "raw",
       staging_dir=base / "staging",
       features_dir=base / "features",
       outputs_dir=base / "outputs",
       kaggle_handle="olistbr/brazilian-ecommerce",
   )
   spark = build_spark_session(SparkConfig("ignored", "ignored", 16, "olist"))

   RawToStaging(spark, data).run()
   for cutoff in (date(2018, 1, 1), date(2018, 4, 1)):
       FeatureBuilder(spark, data, cutoff, horizon_days=180).run()
   SurvivalDatasetBuilder(spark, data, date(2018, 10, 17)).run()
   ```

6. Register the staged parquet as tables and run `sql/feature_views.sql`
   in the SQL editor (replace the temporary view names with table
   references, or simply create the temp views in a notebook first):

   ```python
   for name in ("customers", "orders", "order_items", "order_payments",
                "order_reviews", "products", "sellers", "geolocation",
                "category_translation"):
       spark.read.parquet(f"{base}/staging/{name}").createOrReplaceTempView(name)
   ```

The same view definitions the test suite executes against Spark run
unchanged here; that equivalence is the reason the SQL lives in one file.

## Using S3 directly (paid AWS workspace)

On a standard Databricks-on-AWS workspace the only change from the
walkthrough above is the paths. Grant the cluster access to the bucket
through an instance profile or a Unity Catalog external location, then
set the four directories to `s3://` URIs. Spark resolves the scheme
through the Hadoop S3A connector; none of the pipeline code refers to a
filesystem directly.

Two sizing notes worth carrying over from the local configuration:

* `spark.sql.shuffle.partitions` stays small. The dataset is ~100k
  orders; the 200-partition default exists for clusters processing far
  more data and only adds scheduling overhead here.
* The feature snapshots are written as single parquet files on purpose.
  At this scale one file per snapshot reads faster than a directory of
  shards; partitioning earns its keep at orders of magnitude more data.

## Why the laptop run stays the default

Everything in the README's results section is measured by the local run
and re-verified in CI, neither of which needs an account with anyone.
Databricks is the deployment story, not a dependency: the code is
structured so an interviewer can ask "what changes at scale" and the
answer is configuration, cluster sizing and nothing else.
