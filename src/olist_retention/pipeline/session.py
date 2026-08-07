"""Create or attach the Spark session for local and Databricks runs.

The same entry point serves both environments. On Databricks the runtime
already owns the session and cluster sizing, so the function attaches to it
and leaves the configuration alone. Locally it builds a session sized for a
laptop: two cores, a small driver heap, no web UI, and a shuffle partition
count matched to the data volume rather than the cluster-scale default.
"""

from __future__ import annotations

import os
from datetime import datetime

from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as F

from olist_retention.config import SparkConfig
from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)


def timestamp_literal(value: datetime) -> Column:
    """A timestamp literal that means the same thing on every machine.

    Passing a naive Python datetime through ``F.lit`` converts it via the
    JVM's default timezone, so two literals in one query can shift by an
    hour when they straddle a daylight saving change on the developer's
    machine. Rendering the value as a string and parsing it with
    ``to_timestamp`` keeps the interpretation in the session timezone
    (pinned to UTC here), identical on a laptop, in CI and on Databricks.
    A test with a censoring window across the change caught this.
    """
    return F.to_timestamp(F.lit(value.strftime("%Y-%m-%d %H:%M:%S")))


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def build_spark_session(config: SparkConfig) -> SparkSession:
    """Return an active Spark session appropriate for the environment."""
    if on_databricks():
        logger.info("Databricks runtime detected, attaching to the managed session")
        session = SparkSession.builder.getOrCreate()
        # Cluster sizing is the runtime's business and is left alone. The
        # session timezone is not: every cutoff and censoring boundary is
        # a timestamp literal parsed in it, so a workspace defaulting to
        # anything other than UTC would silently move the boundaries and
        # produce different numbers from the same code. It is pinned here
        # for the same reason it is pinned in the local branch below.
        try:
            session.conf.set("spark.sql.session.timeZone", "UTC")
        except Exception as error:  # noqa: BLE001 - depends on the runtime
            logger.warning(
                "Could not pin the session timezone to UTC (%s). Cutoff boundaries "
                "may differ from a local run.",
                error,
            )
        return session

    session = (
        SparkSession.builder.master(config.master)
        .appName(config.app_name)
        .config("spark.driver.memory", config.driver_memory)
        # Deliberate departure from the default of 200: with ~100k orders a
        # shuffle produces hundreds of near-empty tasks whose scheduling
        # overhead exceeds the work. Sized to the data, not the default.
        .config("spark.sql.shuffle.partitions", str(config.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        # Keep the local footprint small and quiet.
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    logger.info(
        "Local Spark session started (master=%s, driver_memory=%s, shuffle_partitions=%s)",
        config.master,
        config.driver_memory,
        config.shuffle_partitions,
    )
    return session
