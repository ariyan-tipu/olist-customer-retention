"""Structural validation of the raw tables, run inside the Spark job.

The checks are split into two severities. Contract breaches (missing
columns, empty tables) raise immediately because nothing downstream can be
trusted after them. Known blemishes of the source data (duplicate reviews,
delivered orders missing a delivery timestamp) are counted and reported
rather than failed, because they are facts about the data the pipeline is
expected to survive, and the numbers belong in the validation report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from olist_retention.data.schemas import TABLES
from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(RuntimeError):
    """Raised when the raw data breaks its structural contract."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class TableReport:
    table: str
    row_count: int
    checks: list[CheckResult] = field(default_factory=list)


@dataclass
class ValidationReport:
    tables: list[TableReport] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"tables": [asdict(t) for t in self.tables]}, indent=2)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        logger.info("Wrote validation report to %s", path)


# Uniqueness contracts that must hold for the pipeline to be correct.
_UNIQUE_KEYS: dict[str, tuple[str, ...]] = {
    "customers": ("customer_id",),
    "orders": ("order_id",),
    "order_items": ("order_id", "order_item_id"),
    "order_payments": ("order_id", "payment_sequential"),
    "products": ("product_id",),
    "sellers": ("seller_id",),
}

# Foreign keys that must be fully covered by the referenced table.
_FOREIGN_KEYS: tuple[tuple[str, str, str, str], ...] = (
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_payments", "order_id", "orders", "order_id"),
    ("order_reviews", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("order_items", "seller_id", "sellers", "seller_id"),
)


class RawDataValidator:
    """Validate a dict of raw DataFrames keyed by TableSpec name."""

    def __init__(self, frames: dict[str, DataFrame]) -> None:
        self.frames = frames

    def run(self) -> ValidationReport:
        report = ValidationReport()
        counts: dict[str, int] = {}

        for name, spec in TABLES.items():
            frame = self.frames.get(name)
            if frame is None:
                raise DataValidationError(f"Table {name!r} was not loaded")

            expected = [f.name for f in spec.schema.fields]
            if list(frame.columns) != expected:
                raise DataValidationError(
                    f"{name}: columns {list(frame.columns)} do not match schema {expected}"
                )

            row_count = frame.count()
            counts[name] = row_count
            if row_count == 0:
                raise DataValidationError(f"{name}: table is empty")

            table_report = TableReport(table=name, row_count=row_count)
            if name in _UNIQUE_KEYS:
                table_report.checks.append(self._check_unique(frame, name, _UNIQUE_KEYS[name]))
            report.tables.append(table_report)

        self._append_cross_table_checks(report)
        self._append_known_blemish_counts(report)

        for table in report.tables:
            for check in table.checks:
                level = logger.info if check.passed else logger.warning
                level("%s / %s: %s", table.table, check.name, check.detail)
        return report

    # ------------------------------------------------------------------

    def _check_unique(self, frame: DataFrame, name: str, keys: tuple[str, ...]) -> CheckResult:
        total = frame.count()
        distinct = frame.select(*keys).distinct().count()
        passed = total == distinct
        return CheckResult(
            name=f"unique({', '.join(keys)})",
            passed=passed,
            detail=f"{total} rows, {distinct} distinct keys",
        )

    def _find_report(self, report: ValidationReport, table: str) -> TableReport:
        return next(t for t in report.tables if t.table == table)

    def _append_cross_table_checks(self, report: ValidationReport) -> None:
        for child, child_key, parent, parent_key in _FOREIGN_KEYS:
            child_frame = self.frames[child]
            parent_frame = self.frames[parent].select(F.col(parent_key).alias("__parent_key"))
            orphans = (
                child_frame.select(F.col(child_key).alias("__child_key"))
                .where(F.col("__child_key").isNotNull())
                .distinct()
                .join(parent_frame, F.col("__child_key") == F.col("__parent_key"), "left_anti")
                .count()
            )
            self._find_report(report, child).checks.append(
                CheckResult(
                    name=f"fk {child}.{child_key} -> {parent}.{parent_key}",
                    passed=orphans == 0,
                    detail=f"{orphans} orphaned keys",
                )
            )

    def _append_known_blemish_counts(self, report: ValidationReport) -> None:
        orders = self.frames["orders"]
        reviews = self.frames["order_reviews"]

        delivered_without_date = orders.where(
            (F.col("order_status") == "delivered") & F.col("order_delivered_customer_date").isNull()
        ).count()
        self._find_report(report, "orders").checks.append(
            CheckResult(
                name="delivered orders missing delivery date",
                passed=True,
                detail=f"{delivered_without_date} rows (kept, flagged downstream)",
            )
        )

        multi_reviewed = reviews.groupBy("order_id").count().where(F.col("count") > 1).count()
        self._find_report(report, "order_reviews").checks.append(
            CheckResult(
                name="orders with multiple reviews",
                passed=True,
                detail=f"{multi_reviewed} orders (latest review wins downstream)",
            )
        )

        bad_scores = reviews.where(
            F.col("review_score").isNull() | ~F.col("review_score").between(1, 5)
        ).count()
        self._find_report(report, "order_reviews").checks.append(
            CheckResult(
                name="review_score in 1..5",
                passed=bad_scores == 0,
                detail=f"{bad_scores} rows outside range",
            )
        )
