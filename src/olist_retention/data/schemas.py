"""Explicit Spark schemas for the raw Olist tables.

Schema inference on CSVs is convenient and wrong just often enough to hurt:
zip code prefixes lose their leading zeros, review scores arrive as strings,
and a silent type change upstream goes unnoticed. Declaring the schema makes
the contract with the source data explicit and lets validation fail loudly.

Two quirks of the source data are preserved deliberately:

* ``customer_id`` is unique per order; ``customer_unique_id`` is the real
  person. Every join that matters happens at ``customer_unique_id`` grain.
* The products table really does spell it ``product_name_lenght``. The typo
  belongs to the raw layer; staging renames it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

_TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"


def _string(name: str) -> StructField:
    return StructField(name, StringType(), nullable=True)


def _int(name: str) -> StructField:
    return StructField(name, IntegerType(), nullable=True)


def _double(name: str) -> StructField:
    return StructField(name, DoubleType(), nullable=True)


def _timestamp(name: str) -> StructField:
    return StructField(name, TimestampType(), nullable=True)


@dataclass(frozen=True)
class TableSpec:
    """How to read one raw CSV: filename, schema and reader options."""

    name: str
    filename: str
    schema: StructType
    read_options: dict[str, str] = field(default_factory=dict)

    def options(self) -> dict[str, str]:
        base = {
            "header": "true",
            "encoding": "UTF-8",
            "timestampFormat": _TIMESTAMP_FORMAT,
            "mode": "PERMISSIVE",
        }
        base.update(self.read_options)
        return base


CUSTOMERS = TableSpec(
    name="customers",
    filename="olist_customers_dataset.csv",
    schema=StructType(
        [
            _string("customer_id"),
            _string("customer_unique_id"),
            # Kept as a string: Brazilian zip prefixes have leading zeros.
            _string("customer_zip_code_prefix"),
            _string("customer_city"),
            _string("customer_state"),
        ]
    ),
)

ORDERS = TableSpec(
    name="orders",
    filename="olist_orders_dataset.csv",
    schema=StructType(
        [
            _string("order_id"),
            _string("customer_id"),
            _string("order_status"),
            _timestamp("order_purchase_timestamp"),
            _timestamp("order_approved_at"),
            _timestamp("order_delivered_carrier_date"),
            _timestamp("order_delivered_customer_date"),
            _timestamp("order_estimated_delivery_date"),
        ]
    ),
)

ORDER_ITEMS = TableSpec(
    name="order_items",
    filename="olist_order_items_dataset.csv",
    schema=StructType(
        [
            _string("order_id"),
            _int("order_item_id"),
            _string("product_id"),
            _string("seller_id"),
            _timestamp("shipping_limit_date"),
            _double("price"),
            _double("freight_value"),
        ]
    ),
)

ORDER_PAYMENTS = TableSpec(
    name="order_payments",
    filename="olist_order_payments_dataset.csv",
    schema=StructType(
        [
            _string("order_id"),
            _int("payment_sequential"),
            _string("payment_type"),
            _int("payment_installments"),
            _double("payment_value"),
        ]
    ),
)

ORDER_REVIEWS = TableSpec(
    name="order_reviews",
    filename="olist_order_reviews_dataset.csv",
    schema=StructType(
        [
            _string("review_id"),
            _string("order_id"),
            _int("review_score"),
            _string("review_comment_title"),
            _string("review_comment_message"),
            _timestamp("review_creation_date"),
            _timestamp("review_answer_timestamp"),
        ]
    ),
    # Free-text reviews contain quoted newlines, so this table needs the
    # multi-line CSV reader (RFC 4180 doubles quotes, hence escape = quote).
    read_options={"multiLine": "true", "quote": '"', "escape": '"'},
)

PRODUCTS = TableSpec(
    name="products",
    filename="olist_products_dataset.csv",
    schema=StructType(
        [
            _string("product_id"),
            _string("product_category_name"),
            _int("product_name_lenght"),
            _int("product_description_lenght"),
            _int("product_photos_qty"),
            _int("product_weight_g"),
            _int("product_length_cm"),
            _int("product_height_cm"),
            _int("product_width_cm"),
        ]
    ),
)

SELLERS = TableSpec(
    name="sellers",
    filename="olist_sellers_dataset.csv",
    schema=StructType(
        [
            _string("seller_id"),
            _string("seller_zip_code_prefix"),
            _string("seller_city"),
            _string("seller_state"),
        ]
    ),
)

GEOLOCATION = TableSpec(
    name="geolocation",
    filename="olist_geolocation_dataset.csv",
    schema=StructType(
        [
            _string("geolocation_zip_code_prefix"),
            _double("geolocation_lat"),
            _double("geolocation_lng"),
            _string("geolocation_city"),
            _string("geolocation_state"),
        ]
    ),
)

CATEGORY_TRANSLATION = TableSpec(
    name="category_translation",
    filename="product_category_name_translation.csv",
    schema=StructType(
        [
            _string("product_category_name"),
            _string("product_category_name_english"),
        ]
    ),
)

TABLES: dict[str, TableSpec] = {
    spec.name: spec
    for spec in (
        CUSTOMERS,
        ORDERS,
        ORDER_ITEMS,
        ORDER_PAYMENTS,
        ORDER_REVIEWS,
        PRODUCTS,
        SELLERS,
        GEOLOCATION,
        CATEGORY_TRANSLATION,
    )
}

EXPECTED_FILES: tuple[str, ...] = tuple(spec.filename for spec in TABLES.values())
