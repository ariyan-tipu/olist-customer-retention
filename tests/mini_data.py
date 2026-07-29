"""A handcrafted nine-table dataset with exactly known timelines.

Every row here exists to pin one behaviour of the pipeline. The cutoff
used by the tests is 2018-01-01 with a 180 day horizon (window closing
2018-06-30) and the survival observation end is 2018-10-17.

Customers:

* CU_A repeats inside the window: label 1; survival event at ~86.83 days.
* CU_B one late, badly reviewed order and never returns: label 0;
  censored at ~301.42 days with late_delivery = 1.
* CU_C ordered before the cutoff but delivery and review happened after
  it: frequency is known at the cutoff, delivery and review features are
  not, so they must be null.
* CU_D only ever cancelled: must not appear in the feature table at all.
* CU_E repeats after the horizon: label 0 for the classifier, yet a
  genuine survival event at ~324.83 days. The pair of assertions is the
  point of running both analyses.
* CU_F has two reviews on one order (latest must win) and two categories
  in the basket (alphabetical tie-break on top_category).
* CU_G placed a second order before the first arrived: counted as a
  pre-delivery repurchaser and censored, with mixed delivery stats.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from olist_retention.data.schemas import TABLES

TS_FORMAT = "%Y-%m-%d %H:%M:%S"

CUTOFF = pd.Timestamp("2018-01-01")
HORIZON_DAYS = 180
OBSERVATION_END = pd.Timestamp("2018-10-17")


def _ts(value: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(value) if value else None


def _order(
    order_id: str,
    customer_id: str,
    status: str,
    purchase: str,
    delivered: str | None,
    estimated: str,
) -> dict:
    purchase_ts = _ts(purchase)
    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "order_status": status,
        "order_purchase_timestamp": purchase_ts,
        "order_approved_at": None if status == "canceled" else purchase_ts + pd.Timedelta(hours=1),
        "order_delivered_carrier_date": (
            None if status == "canceled" else purchase_ts + pd.Timedelta(days=1)
        ),
        "order_delivered_customer_date": _ts(delivered),
        "order_estimated_delivery_date": _ts(estimated),
    }


def build_tables() -> dict[str, pd.DataFrame]:
    orders = [
        _order(
            "A1",
            "cid_A1",
            "delivered",
            "2017-11-10 10:00:00",
            "2017-11-20 14:00:00",
            "2017-11-25 00:00:00",
        ),
        _order(
            "A2",
            "cid_A2",
            "delivered",
            "2018-02-15 10:00:00",
            "2018-02-20 14:00:00",
            "2018-03-01 00:00:00",
        ),
        _order(
            "B1",
            "cid_B1",
            "delivered",
            "2017-12-01 10:00:00",
            "2017-12-20 14:00:00",
            "2017-12-10 00:00:00",
        ),
        _order(
            "C1",
            "cid_C1",
            "delivered",
            "2017-12-28 10:00:00",
            "2018-01-05 14:00:00",
            "2018-01-10 00:00:00",
        ),
        _order("D1", "cid_D1", "canceled", "2017-10-01 10:00:00", None, "2017-10-20 00:00:00"),
        _order(
            "E1",
            "cid_E1",
            "delivered",
            "2017-09-01 10:00:00",
            "2017-09-10 14:00:00",
            "2017-09-15 00:00:00",
        ),
        _order(
            "E2",
            "cid_E2",
            "delivered",
            "2018-08-01 10:00:00",
            "2018-08-10 14:00:00",
            "2018-08-20 00:00:00",
        ),
        _order(
            "F1",
            "cid_F1",
            "delivered",
            "2017-11-01 10:00:00",
            "2017-11-08 14:00:00",
            "2017-11-12 00:00:00",
        ),
        _order(
            "F2",
            "cid_F2",
            "delivered",
            "2018-03-01 10:00:00",
            "2018-03-08 14:00:00",
            "2018-03-15 00:00:00",
        ),
        _order(
            "G1",
            "cid_G1",
            "delivered",
            "2017-11-01 10:00:00",
            "2017-11-30 14:00:00",
            "2017-11-25 00:00:00",
        ),
        _order(
            "G2",
            "cid_G2",
            "delivered",
            "2017-11-15 10:00:00",
            "2017-12-05 14:00:00",
            "2017-12-20 00:00:00",
        ),
    ]

    owner = {
        "cid_A1": ("CU_A", "SP"),
        "cid_A2": ("CU_A", "SP"),
        "cid_B1": ("CU_B", "BA"),
        "cid_C1": ("CU_C", "RJ"),
        "cid_D1": ("CU_D", "PR"),
        "cid_E1": ("CU_E", "MG"),
        "cid_E2": ("CU_E", "MG"),
        "cid_F1": ("CU_F", "RS"),
        "cid_F2": ("CU_F", "RS"),
        "cid_G1": ("CU_G", "SP"),
        "cid_G2": ("CU_G", "SP"),
    }
    customers = [
        {
            "customer_id": cid,
            "customer_unique_id": unique_id,
            "customer_zip_code_prefix": "01310",
            "customer_city": "cidade",
            "customer_state": state,
        }
        for cid, (unique_id, state) in owner.items()
    ]

    def item(order_id: str, item_id: int, product_id: str, price: float, freight: float) -> dict:
        return {
            "order_id": order_id,
            "order_item_id": item_id,
            "product_id": product_id,
            "seller_id": "S1",
            "shipping_limit_date": _ts("2017-01-01 10:00:00"),
            "price": price,
            "freight_value": freight,
        }

    items = [
        item("A1", 1, "P1", 100.0, 20.0),
        item("A2", 1, "P1", 60.0, 20.0),
        item("B1", 1, "P1", 100.0, 20.0),
        item("C1", 1, "P2", 100.0, 20.0),
        item("D1", 1, "P1", 100.0, 20.0),
        item("E1", 1, "P1", 100.0, 20.0),
        item("E2", 1, "P2", 100.0, 20.0),
        item("F1", 1, "P1", 60.0, 10.0),
        item("F1", 2, "P2", 60.0, 10.0),
        item("F2", 1, "P1", 100.0, 20.0),
        item("G1", 1, "P1", 100.0, 20.0),
        item("G2", 1, "P2", 100.0, 20.0),
    ]

    payments = [
        {
            "order_id": row["order_id"],
            "payment_sequential": 1,
            "payment_type": "credit_card",
            "payment_installments": 2,
            "payment_value": 120.0 if row["order_id"] != "A2" else 80.0,
        }
        for row in orders
    ]

    def review(review_id: str, order_id: str, score: int, creation: str) -> dict:
        creation_ts = _ts(creation)
        return {
            "review_id": review_id,
            "order_id": order_id,
            "review_score": score,
            "review_comment_title": None,
            "review_comment_message": None,
            "review_creation_date": creation_ts,
            "review_answer_timestamp": creation_ts + pd.Timedelta(days=1),
        }

    reviews = [
        review("RA1", "A1", 5, "2017-11-22 10:00:00"),
        review("RB1", "B1", 1, "2017-12-21 10:00:00"),
        review("RC1", "C1", 4, "2018-01-06 10:00:00"),
        review("RF1", "F1", 2, "2017-11-09 10:00:00"),
        review("RF2", "F1", 4, "2017-11-15 10:00:00"),
        review("RG1", "G1", 3, "2017-12-01 10:00:00"),
    ]

    products = [
        {
            "product_id": "P1",
            "product_category_name": "brinquedos",
            "product_name_lenght": 30,
            "product_description_lenght": 300,
            "product_photos_qty": 2,
            "product_weight_g": 500,
            "product_length_cm": 20,
            "product_height_cm": 10,
            "product_width_cm": 15,
        },
        {
            "product_id": "P2",
            "product_category_name": "telefonia",
            "product_name_lenght": 25,
            "product_description_lenght": 200,
            "product_photos_qty": 1,
            "product_weight_g": 200,
            "product_length_cm": 15,
            "product_height_cm": 5,
            "product_width_cm": 8,
        },
    ]

    sellers = [
        {
            "seller_id": "S1",
            "seller_zip_code_prefix": "04000",
            "seller_city": "sao paulo",
            "seller_state": "SP",
        }
    ]

    geolocation = [
        {
            "geolocation_zip_code_prefix": "01310",
            "geolocation_lat": -23.56,
            "geolocation_lng": -46.65,
            "geolocation_city": "sao paulo",
            "geolocation_state": "SP",
        }
    ]

    translation = [
        {"product_category_name": "brinquedos", "product_category_name_english": "toys"},
        {"product_category_name": "telefonia", "product_category_name_english": "telephony"},
    ]

    return {
        "customers": pd.DataFrame(customers),
        "orders": pd.DataFrame(orders),
        "order_items": pd.DataFrame(items),
        "order_payments": pd.DataFrame(payments),
        "order_reviews": pd.DataFrame(reviews),
        "products": pd.DataFrame(products),
        "sellers": pd.DataFrame(sellers),
        "geolocation": pd.DataFrame(geolocation),
        "category_translation": pd.DataFrame(translation),
    }


def write_mini_dataset(dest: Path) -> Path:
    """Write the handcrafted tables as raw CSVs with the Olist filenames."""
    dest.mkdir(parents=True, exist_ok=True)
    frames = build_tables()
    for name, frame in frames.items():
        expected = [field.name for field in TABLES[name].schema.fields]
        frame = frame[expected]
        frame.to_csv(dest / TABLES[name].filename, index=False, date_format=TS_FORMAT)
    return dest
