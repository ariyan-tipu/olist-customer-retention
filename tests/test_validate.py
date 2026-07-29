"""Raw data validation behaviour against staged synthetic data."""

from __future__ import annotations

import pytest

from olist_retention.data.validate import DataValidationError, RawDataValidator
from olist_retention.pipeline.staging import RawToStaging


def test_validator_passes_on_synthetic(spark, synthetic_env):
    frames = RawToStaging(spark, synthetic_env).read_raw()
    report = RawDataValidator(frames).run()

    assert {t.table for t in report.tables} == {
        "customers",
        "orders",
        "order_items",
        "order_payments",
        "order_reviews",
        "products",
        "sellers",
        "geolocation",
        "category_translation",
    }
    orders_report = next(t for t in report.tables if t.table == "orders")
    assert orders_report.row_count > 0
    assert all(c.passed for t in report.tables for c in t.checks)


def test_missing_column_is_a_contract_breach(spark, synthetic_env):
    frames = RawToStaging(spark, synthetic_env).read_raw()
    frames["customers"] = frames["customers"].drop("customer_state")
    with pytest.raises(DataValidationError, match="customers"):
        RawDataValidator(frames).run()
