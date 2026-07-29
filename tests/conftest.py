"""Shared fixtures: one Spark session, one synthetic environment.

Spark-backed fixtures are session-scoped because starting a JVM per test
would dominate the suite's runtime. Tests marked ``spark`` skip cleanly
when no JDK is available rather than failing with a JVM stack trace.
"""

from __future__ import annotations

import os
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from olist_retention.config import DataConfig, ModellingConfig, SparkConfig

TRAIN_CUTOFF = date(2017, 10, 1)
TEST_CUTOFF = date(2018, 1, 1)
HORIZON_DAYS = 180
OBSERVATION_END = date(2018, 10, 17)


def _java_available() -> bool:
    if shutil.which("java"):
        return True
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        return False
    bin_dir = Path(java_home) / "bin"
    return (bin_dir / "java.exe").exists() or (bin_dir / "java").exists()


def pytest_collection_modifyitems(config, items) -> None:
    """Mark anything that transitively needs the Spark session."""
    for item in items:
        if "spark" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.spark)


@pytest.fixture(scope="session")
def spark():
    if not _java_available():
        pytest.skip("No JDK available for Spark tests")
    from olist_retention.pipeline.session import build_spark_session

    session = build_spark_session(
        SparkConfig(master="local[2]", driver_memory="1g", shuffle_partitions=4, app_name="tests")
    )
    yield session
    session.stop()


def _data_config(base: Path, raw: Path) -> DataConfig:
    return DataConfig(
        raw_dir=raw,
        staging_dir=base / "staging",
        features_dir=base / "features",
        outputs_dir=base / "outputs",
        kaggle_handle="unused/unused",
    )


@pytest.fixture(scope="session")
def synthetic_env(tmp_path_factory, spark) -> DataConfig:
    """Synthetic raw data staged once for the whole suite."""
    from olist_retention.data.synthetic import SyntheticOlist, SyntheticParams
    from olist_retention.pipeline.staging import RawToStaging

    base = tmp_path_factory.mktemp("synthetic")
    raw = base / "raw"
    SyntheticOlist(SyntheticParams(n_customers=250, seed=11)).write_csvs(raw)
    config = _data_config(base, raw)
    RawToStaging(spark, config).run(validate=True)
    return config


@pytest.fixture(scope="session")
def synthetic_snapshots(spark, synthetic_env) -> DataConfig:
    """Feature snapshots at both cutoffs plus the survival export."""
    from olist_retention.pipeline.features import FeatureBuilder
    from olist_retention.pipeline.survival import SurvivalDatasetBuilder

    for cutoff in (TRAIN_CUTOFF, TEST_CUTOFF):
        FeatureBuilder(spark, synthetic_env, cutoff, HORIZON_DAYS).run()
    SurvivalDatasetBuilder(spark, synthetic_env, OBSERVATION_END).run()
    return synthetic_env


@pytest.fixture(scope="session")
def mini_env(tmp_path_factory, spark) -> DataConfig:
    """The handcrafted dataset staged for the precision tests."""
    from olist_retention.pipeline.staging import RawToStaging
    from tests.mini_data import write_mini_dataset

    base = tmp_path_factory.mktemp("mini")
    raw = write_mini_dataset(base / "raw")
    config = _data_config(base, raw)
    RawToStaging(spark, config).run(validate=True)
    return config


@pytest.fixture(scope="session")
def mini_features(spark, mini_env) -> pd.DataFrame:
    from olist_retention.pipeline.features import FeatureBuilder

    builder = FeatureBuilder(spark, mini_env, TEST_CUTOFF, HORIZON_DAYS)
    frame = builder.build().toPandas()
    return frame.set_index("customer_unique_id").sort_index()


@pytest.fixture(scope="session")
def mini_survival(spark, mini_env):
    from olist_retention.pipeline.survival import SurvivalDatasetBuilder

    builder = SurvivalDatasetBuilder(spark, mini_env, OBSERVATION_END)
    csv_path, metadata = builder.run()
    frame = pd.read_csv(csv_path).set_index("customer_unique_id").sort_index()
    return frame, metadata


@pytest.fixture(scope="session")
def trained_model(synthetic_snapshots, tmp_path_factory):
    """A real training run on the synthetic snapshots."""
    from olist_retention.models.train import ChurnModelTrainer

    base = tmp_path_factory.mktemp("model")
    modelling = ModellingConfig(seed=42, cv_folds=3, model_dir=base / "models")
    trainer = ChurnModelTrainer(
        modelling=modelling,
        features_dir=synthetic_snapshots.features_dir,
        outputs_dir=base / "outputs",
        horizon_days=HORIZON_DAYS,
    )
    result = trainer.run(TRAIN_CUTOFF, TEST_CUTOFF)
    return result
