# Running the pipeline on Databricks

The laptop run and the cluster run are the same code with different
configuration. `build_spark_session` attaches to the Databricks-managed
session when it detects the runtime, every read and write goes through
paths taken from `conf/pipeline.yaml`, and pointing the pipeline at
cloud storage is therefore a configuration change.

That is a claim, and a claim in a README is worth little, so the
notebook at
[`notebooks/databricks/01_pipeline_on_databricks.py`](../notebooks/databricks/01_pipeline_on_databricks.py)
ends by comparing what the cluster produced against the figures measured
locally. If a row count or a base rate moved, it prints the mismatch.

## It has been run, and it matched

Executed on Databricks Free Edition serverless compute on 2026-08-07.
All eight figures came back identical to the laptop run, and
[`results/databricks_run.json`](../results/databricks_run.json) carries
the full record.

| Figure | Laptop | Cluster |
| --- | --- | --- |
| Staged orders | 99,441 | 99,441 |
| Survival spells | 93,350 | 93,350 |
| Survival events | 1,808 | 1,808 |
| Train customers | 43,768 | 43,768 |
| Test customers | 64,089 | 64,089 |
| Train base rate | 1.286% | 1.286% |
| Test base rate | 0.975% | 0.975% |
| Censoring rate | 98.1% | 98.1% |

All five SQL views built and returned rows. The regional summary splits
the test cohort into five regions that sum to 64,089, which is the test
snapshot exactly, so the view agrees with the snapshot it reads from.

Two platform differences turned up, and both are the interesting part of
the exercise:

* Serverless compute rejects `PERSIST TABLE` outright. Caching in this
  pipeline only ever changed how long a job took, so `maybe_cache` now
  asks, accepts a refusal and logs it. The numbers above were produced
  with no caching at all, which is the proof that it was an optimisation.
* The managed session does not default to UTC. Since every cutoff is a
  timestamp literal parsed in the session timezone, the pipeline pins it.
  Without that pin the boundaries would have moved and this table would
  not have matched.

| Layer  | Contents                                  | Local default   | Cloud example                 |
| ------ | ----------------------------------------- | --------------- | ----------------------------- |
| Bronze | raw Olist CSVs plus provenance manifest   | `data/raw`      | `s3://<bucket>/olist/raw`     |
| Silver | typed, validated parquet per table        | `data/staging`  | `s3://<bucket>/olist/staging` |
| Gold   | feature snapshots and the survival export | `data/features` | `s3://<bucket>/olist/features` |

## Free Edition walkthrough

Databricks Free Edition gives a serverless workspace at no cost and asks
for no card, which keeps the project's zero-cost rule intact. Storage is
a managed Unity Catalog volume; on a workspace with its own bucket the
only difference is the path prefix.

1. Sign up for Free Edition and let it create a workspace.
2. Open Catalog, expand `workspace` and its `default` schema,
   then create a volume named `olist`. The path comes out as
   `/Volumes/workspace/default/olist`.
3. Import `notebooks/databricks/01_pipeline_on_databricks.py`. Databricks
   reads that source format directly, so the file stays reviewable in a
   pull request instead of arriving as a wall of JSON.
4. Run the cells in order. The first installs the package straight from
   GitHub, so there is no wheel to build or upload.

The notebook stages the nine raw tables, builds both feature snapshots
and the survival export, registers the staged parquet as temporary views
and executes `sql/feature_views.sql` unchanged. That file is the same one
the test suite runs against a local Spark session, which is why the SQL
lives in a file rather than inside a Python string.

## The timezone detail worth knowing about

Every cutoff and censoring boundary in this pipeline is a timestamp
literal, parsed in the Spark session timezone. The local session pins
that to UTC. The managed session does not: it follows the workspace, so
a workspace on London time would move every boundary by an hour across a
daylight saving change and quietly produce different numbers from the
same code.

`build_spark_session` now pins the session timezone on the Databricks
path as well, and warns loudly if the runtime refuses. The equivalent bug
was caught locally by a handcrafted censoring test whose window straddled
a British summer time change, and it would have been invisible on a
UTC-only machine.

## Using S3 directly

On a standard Databricks-on-AWS workspace the only change is the paths.
Grant the cluster access to the bucket through an instance profile or a
Unity Catalog external location, then set the four directories to `s3://`
URIs. Spark resolves the scheme through the Hadoop S3A connector, and no
pipeline code refers to a filesystem directly.

Two sizing notes carry over from the local configuration:

* `spark.sql.shuffle.partitions` stays small. The dataset is about a
  hundred thousand orders; the 200-partition default exists for clusters
  processing far more than that and only adds scheduling overhead here.
* The feature snapshots are written as single parquet files on purpose.
  At this scale one file per snapshot reads faster than a directory of
  shards. Partitioning earns its keep at orders of magnitude more data.

## Why the laptop run stays the default

Everything in the README's results section is measured by the local run
and re-verified in CI, neither of which needs an account with anyone.
Databricks answers a different question, which is what changes when the
data outgrows one machine, and the answer is configuration and cluster
sizing.
