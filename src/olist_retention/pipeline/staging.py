"""Raw CSVs to typed parquet: the bronze to silver step.

Raw files are read with explicit schemas, validated, given corrected column
names (the source's ``lenght`` typo stops here) and written as parquet.
Everything downstream reads the staged tables, never the CSVs, so the
messy details of the source format are dealt with exactly once.

Paths are plain strings through ``pyspark``, so the same job writes to a
local directory on a laptop and to an ``s3://`` prefix on Databricks
without code changes; only the configuration differs.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from olist_retention.config import DataConfig
from olist_retention.data.schemas import TABLES, TableSpec
from olist_retention.data.validate import RawDataValidator, ValidationReport
from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)

# Small dimension tables worth compacting to a single parquet file.
_SINGLE_FILE_TABLES = {"products", "sellers", "category_translation"}

_PRODUCT_RENAMES = {
    "product_name_lenght": "product_name_length",
    "product_description_lenght": "product_description_length",
}


class RawToStaging:
    """Read, validate and persist the raw tables as typed parquet."""

    def __init__(self, spark: SparkSession, data_config: DataConfig) -> None:
        self.spark = spark
        self.config = data_config

    def read_raw(self) -> dict[str, DataFrame]:
        frames: dict[str, DataFrame] = {}
        for name, spec in TABLES.items():
            frames[name] = self._read_table(spec)
        return frames

    def run(self, validate: bool = True) -> ValidationReport | None:
        """Stage every table; returns the validation report when enabled."""
        frames = self.read_raw()

        report: ValidationReport | None = None
        if validate:
            report = RawDataValidator(frames).run()
            report.write(self.config.outputs_dir / "data_validation.json")

        for name, frame in frames.items():
            staged = self._transform(name, frame)
            target = self.config.staging_dir / name
            writer = staged.coalesce(1) if name in _SINGLE_FILE_TABLES else staged
            writer.write.mode("overwrite").parquet(str(target))
            logger.info("Staged %s to %s", name, target)
        return report

    # ------------------------------------------------------------------

    def _read_table(self, spec: TableSpec) -> DataFrame:
        path = self.config.raw_dir / spec.filename
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Raw file {path} is missing. Run 'olist-retention download' first."
            )
        reader = self.spark.read.schema(spec.schema)
        for key, value in spec.options().items():
            reader = reader.option(key, value)
        return reader.csv(str(path))

    @staticmethod
    def _transform(name: str, frame: DataFrame) -> DataFrame:
        if name == "products":
            for source, target in _PRODUCT_RENAMES.items():
                frame = frame.withColumnRenamed(source, target)
        return frame


def read_staged(spark: SparkSession, staging_dir: Path, name: str) -> DataFrame:
    """Read one staged table back; fails clearly if staging has not run."""
    path = staging_dir / name
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Staged table {name!r} not found at {path}. Run 'olist-retention pipeline' first."
        )
    return spark.read.parquet(str(path))
