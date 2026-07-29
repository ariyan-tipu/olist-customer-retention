"""The synthetic generator: deterministic, schema-faithful, warty."""

from __future__ import annotations

import pandas as pd

from olist_retention.data.schemas import TABLES
from olist_retention.data.synthetic import SyntheticOlist, SyntheticParams

PARAMS = SyntheticParams(n_customers=60, seed=3)


def test_same_seed_gives_identical_tables():
    first = SyntheticOlist(PARAMS).tables()
    second = SyntheticOlist(PARAMS).tables()
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_columns_match_declared_schemas():
    tables = SyntheticOlist(PARAMS).tables()
    for name, frame in tables.items():
        expected = [field.name for field in TABLES[name].schema.fields]
        assert list(frame.columns) == expected, name


def test_customer_id_is_per_order():
    tables = SyntheticOlist(PARAMS).tables()
    assert len(tables["customers"]) == len(tables["orders"])
    assert tables["customers"]["customer_id"].is_unique
    assert tables["customers"]["customer_unique_id"].duplicated().any()


def test_known_data_warts_are_planted():
    tables = SyntheticOlist(PARAMS).tables()
    orders = tables["orders"]
    reviews = tables["order_reviews"]

    delivered_missing_date = orders[
        (orders["order_status"] == "delivered") & orders["order_delivered_customer_date"].isna()
    ]
    assert len(delivered_missing_date) == 2

    duplicated_reviews = reviews["order_id"].duplicated().sum()
    assert duplicated_reviews >= 3

    comments = reviews["review_comment_message"].dropna()
    assert comments.str.contains("\n").any(), "expected a multi-line review comment"
    assert comments.str.contains('"').any(), "expected a quoted review comment"
