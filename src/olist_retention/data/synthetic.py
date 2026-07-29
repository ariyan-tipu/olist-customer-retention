"""Deterministic synthetic Olist tables for tests and CI.

This is not a statistical replica of the real marketplace. Rates are chosen
so that a few hundred customers produce enough repeat purchases for the
survival models in CI to recover the planted effects; the real dataset's
repeat rate is far lower. What it mirrors faithfully is structure: the
nine-table layout, identifier semantics (``customer_id`` per order,
``customer_unique_id`` per person), timestamp formats, and the awkward
corners of the raw data (cancelled orders with null delivery dates, orders
with duplicate reviews, free text containing quoted newlines), so the
pipeline is exercised against the same hazards it meets on real data.

Planted ground truth: a late first delivery and a bad review both stretch
the Weibull-distributed time to the next purchase (multipliers live in
``SyntheticParams``). CI asserts that the fitted survival models recover
the direction of both effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from olist_retention.data.schemas import (
    CATEGORY_TRANSLATION,
    CUSTOMERS,
    GEOLOCATION,
    ORDER_ITEMS,
    ORDER_PAYMENTS,
    ORDER_REVIEWS,
    ORDERS,
    PRODUCTS,
    SELLERS,
)

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

_CATEGORIES: dict[str, str] = {
    "beleza_saude": "health_beauty",
    "informatica_acessorios": "computers_accessories",
    "moveis_decoracao": "furniture_decor",
    "esporte_lazer": "sports_leisure",
    "utilidades_domesticas": "housewares",
    "brinquedos": "toys",
    "relogios_presentes": "watches_gifts",
    "telefonia": "telephony",
}

_STATES: tuple[str, ...] = ("SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO")
_STATE_WEIGHTS: tuple[float, ...] = (0.42, 0.13, 0.12, 0.06, 0.06, 0.04, 0.09, 0.08)
_CITIES: dict[str, str] = {
    "SP": "sao paulo",
    "RJ": "rio de janeiro",
    "MG": "belo horizonte",
    "RS": "porto alegre",
    "PR": "curitiba",
    "SC": "florianopolis",
    "BA": "salvador",
    "GO": "goiania",
}

# Review score distributions conditioned on whether the delivery was late.
_SCORES = np.array([1, 2, 3, 4, 5])
_SCORE_PROBS_ON_TIME = np.array([0.03, 0.04, 0.08, 0.20, 0.65])
_SCORE_PROBS_LATE = np.array([0.35, 0.20, 0.15, 0.15, 0.15])

_REVIEW_COMMENTS: tuple[str, ...] = (
    "Chegou antes do prazo, recomendo.",
    "Produto de boa qualidade.",
    "Entrega demorou muito.",
    "Veio com defeito, decepcionado.",
    "Muito bom, compraria novamente.",
    'Excelente, "superou" as expectativas.\nEmbalagem caprichada.',
)


@dataclass(frozen=True)
class SyntheticParams:
    """Knobs for the generator, including the planted causal effects."""

    n_customers: int = 800
    seed: int = 7
    window_start: datetime = datetime(2017, 1, 1)
    window_end: datetime = datetime(2018, 10, 17)
    weibull_shape: float = 1.3
    base_scale_days: float = 110.0
    # Multipliers > 1 stretch the time to the next purchase, so in hazard
    # terms both planted effects push the repurchase hazard below 1.
    late_delivery_scale_multiplier: float = 1.9
    low_review_scale_multiplier: float = 1.6
    max_orders_per_customer: int = 4
    review_rate: float = 0.92
    comment_rate: float = 0.30


class SyntheticOlist:
    """Generate the nine Olist tables as pandas frames or CSV files."""

    def __init__(self, params: SyntheticParams | None = None) -> None:
        self.params = params or SyntheticParams()
        self._rng = np.random.default_rng(self.params.seed)
        self._tables: dict[str, pd.DataFrame] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return all tables keyed by TableSpec name, generating once."""
        if self._tables is None:
            self._tables = self._generate()
        return self._tables

    def write_csvs(self, dest: Path) -> Path:
        """Write every table to ``dest`` using the real Olist filenames."""
        dest.mkdir(parents=True, exist_ok=True)
        filenames = {
            CUSTOMERS.name: CUSTOMERS.filename,
            ORDERS.name: ORDERS.filename,
            ORDER_ITEMS.name: ORDER_ITEMS.filename,
            ORDER_PAYMENTS.name: ORDER_PAYMENTS.filename,
            ORDER_REVIEWS.name: ORDER_REVIEWS.filename,
            PRODUCTS.name: PRODUCTS.filename,
            SELLERS.name: SELLERS.filename,
            GEOLOCATION.name: GEOLOCATION.filename,
            CATEGORY_TRANSLATION.name: CATEGORY_TRANSLATION.filename,
        }
        for name, frame in self.tables().items():
            frame.to_csv(
                dest / filenames[name],
                index=False,
                date_format=_TIMESTAMP_FORMAT,
            )
        return dest

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _hex_id(self) -> str:
        return self._rng.bytes(16).hex()

    def _generate(self) -> dict[str, pd.DataFrame]:
        products, translation = self._make_catalogue()
        sellers = self._make_sellers()

        customers: list[dict] = []
        orders: list[dict] = []
        items: list[dict] = []
        payments: list[dict] = []
        reviews: list[dict] = []

        product_ids = products["product_id"].tolist()
        seller_ids = sellers["seller_id"].tolist()

        for _ in range(self.params.n_customers):
            self._simulate_customer(
                product_ids, seller_ids, products, customers, orders, items, payments, reviews
            )

        self._plant_data_warts(orders, reviews)

        frames = {
            CUSTOMERS.name: pd.DataFrame(customers),
            ORDERS.name: self._with_timestamps(
                pd.DataFrame(orders),
                [f.name for f in ORDERS.schema.fields if f.dataType.typeName() == "timestamp"],
            ),
            ORDER_ITEMS.name: self._with_timestamps(pd.DataFrame(items), ["shipping_limit_date"]),
            ORDER_PAYMENTS.name: pd.DataFrame(payments),
            ORDER_REVIEWS.name: self._with_timestamps(
                pd.DataFrame(reviews), ["review_creation_date", "review_answer_timestamp"]
            ),
            PRODUCTS.name: products,
            SELLERS.name: sellers,
            GEOLOCATION.name: self._make_geolocation(),
            CATEGORY_TRANSLATION.name: translation,
        }
        return frames

    @staticmethod
    def _with_timestamps(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            frame[column] = pd.to_datetime(frame[column])
        return frame

    def _make_catalogue(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        categories = list(_CATEGORIES)
        rows = []
        for _ in range(40):
            rows.append(
                {
                    "product_id": self._hex_id(),
                    "product_category_name": categories[
                        int(self._rng.integers(0, len(categories)))
                    ],
                    "product_name_lenght": int(self._rng.integers(20, 60)),
                    "product_description_lenght": int(self._rng.integers(100, 2000)),
                    "product_photos_qty": int(self._rng.integers(1, 6)),
                    "product_weight_g": int(self._rng.integers(100, 8000)),
                    "product_length_cm": int(self._rng.integers(10, 60)),
                    "product_height_cm": int(self._rng.integers(5, 40)),
                    "product_width_cm": int(self._rng.integers(10, 50)),
                }
            )
        translation = pd.DataFrame(
            {
                "product_category_name": list(_CATEGORIES),
                "product_category_name_english": list(_CATEGORIES.values()),
            }
        )
        return pd.DataFrame(rows), translation

    def _make_sellers(self) -> pd.DataFrame:
        rows = []
        for _ in range(12):
            state = self._pick_state()
            rows.append(
                {
                    "seller_id": self._hex_id(),
                    "seller_zip_code_prefix": self._zip_prefix(),
                    "seller_city": _CITIES[state],
                    "seller_state": state,
                }
            )
        return pd.DataFrame(rows)

    def _make_geolocation(self) -> pd.DataFrame:
        rows = []
        for _ in range(50):
            state = self._pick_state()
            rows.append(
                {
                    "geolocation_zip_code_prefix": self._zip_prefix(),
                    "geolocation_lat": float(self._rng.uniform(-33.0, 2.0)),
                    "geolocation_lng": float(self._rng.uniform(-73.0, -34.0)),
                    "geolocation_city": _CITIES[state],
                    "geolocation_state": state,
                }
            )
        return pd.DataFrame(rows)

    def _pick_state(self) -> str:
        return str(self._rng.choice(_STATES, p=_STATE_WEIGHTS))

    def _zip_prefix(self) -> str:
        return f"{int(self._rng.integers(1000, 99999)):05d}"

    # ------------------------------------------------------------------
    # Customer level simulation
    # ------------------------------------------------------------------

    def _simulate_customer(
        self,
        product_ids: list[str],
        seller_ids: list[str],
        products: pd.DataFrame,
        customers: list[dict],
        orders: list[dict],
        items: list[dict],
        payments: list[dict],
        reviews: list[dict],
    ) -> None:
        params = self.params
        unique_id = self._hex_id()
        state = self._pick_state()
        zip_prefix = self._zip_prefix()

        window_days = (params.window_end - params.window_start).days
        purchase_ts = params.window_start + timedelta(
            days=float(self._rng.uniform(0, window_days - 30)),
        )

        for _ in range(params.max_orders_per_customer):
            if purchase_ts >= params.window_end:
                break

            customer_id = self._hex_id()
            customers.append(
                {
                    "customer_id": customer_id,
                    "customer_unique_id": unique_id,
                    "customer_zip_code_prefix": zip_prefix,
                    "customer_city": _CITIES[state],
                    "customer_state": state,
                }
            )

            order = self._simulate_order(customer_id, purchase_ts)
            orders.append(order)
            order_total = self._simulate_items(
                order["order_id"], purchase_ts, product_ids, seller_ids, items
            )
            self._simulate_payments(order["order_id"], order_total, payments)

            late, low_review = self._simulate_review(order, reviews)

            # Time to the next purchase: Weibull, stretched by a bad first
            # experience. This is the ground truth the survival models
            # are expected to recover.
            scale = params.base_scale_days
            if late:
                scale *= params.late_delivery_scale_multiplier
            if low_review:
                scale *= params.low_review_scale_multiplier
            gap_days = float(scale * self._rng.weibull(params.weibull_shape))
            purchase_ts = purchase_ts + timedelta(days=gap_days)

    def _simulate_order(self, customer_id: str, purchase_ts: datetime) -> dict:
        status_draw = float(self._rng.uniform())
        if status_draw < 0.97:
            status = "delivered"
        elif status_draw < 0.99:
            status = "shipped"
        else:
            status = "canceled"

        approved = purchase_ts + timedelta(hours=float(self._rng.uniform(1, 48)))
        carrier = approved + timedelta(days=float(self._rng.uniform(1, 5)))
        estimated = (purchase_ts + timedelta(days=int(self._rng.integers(10, 28)))).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        delay_days = float(self._rng.normal(-5.0, 7.0))
        delivered = estimated + timedelta(days=delay_days)
        earliest = carrier + timedelta(days=1)
        delivered = max(delivered, earliest)

        return {
            "order_id": self._hex_id(),
            "customer_id": customer_id,
            "order_status": status,
            "order_purchase_timestamp": purchase_ts,
            "order_approved_at": approved if status != "canceled" else None,
            "order_delivered_carrier_date": carrier if status != "canceled" else None,
            "order_delivered_customer_date": delivered if status == "delivered" else None,
            "order_estimated_delivery_date": estimated,
        }

    def _simulate_items(
        self,
        order_id: str,
        purchase_ts: datetime,
        product_ids: list[str],
        seller_ids: list[str],
        items: list[dict],
    ) -> float:
        n_items = int(self._rng.integers(1, 4))
        total = 0.0
        for item_number in range(1, n_items + 1):
            price = round(float(self._rng.lognormal(4.2, 0.6)), 2)
            freight = round(float(self._rng.uniform(8, 35)), 2)
            total += price + freight
            items.append(
                {
                    "order_id": order_id,
                    "order_item_id": item_number,
                    "product_id": product_ids[int(self._rng.integers(0, len(product_ids)))],
                    "seller_id": seller_ids[int(self._rng.integers(0, len(seller_ids)))],
                    "shipping_limit_date": purchase_ts + timedelta(days=5),
                    "price": price,
                    "freight_value": freight,
                }
            )
        return round(total, 2)

    def _simulate_payments(self, order_id: str, total: float, payments: list[dict]) -> None:
        split = float(self._rng.uniform()) < 0.06
        payment_type = str(
            self._rng.choice(
                ["credit_card", "boleto", "voucher", "debit_card"],
                p=[0.74, 0.19, 0.04, 0.03],
            )
        )
        installments = int(self._rng.integers(1, 11)) if payment_type == "credit_card" else 1
        if split and total > 20:
            voucher_value = round(min(20.0, total / 4), 2)
            payments.append(
                {
                    "order_id": order_id,
                    "payment_sequential": 1,
                    "payment_type": "voucher",
                    "payment_installments": 1,
                    "payment_value": voucher_value,
                }
            )
            payments.append(
                {
                    "order_id": order_id,
                    "payment_sequential": 2,
                    "payment_type": payment_type,
                    "payment_installments": installments,
                    "payment_value": round(total - voucher_value, 2),
                }
            )
        else:
            payments.append(
                {
                    "order_id": order_id,
                    "payment_sequential": 1,
                    "payment_type": payment_type,
                    "payment_installments": installments,
                    "payment_value": total,
                }
            )

    def _simulate_review(self, order: dict, reviews: list[dict]) -> tuple[bool, bool]:
        """Append a review when applicable; return (late, low_review)."""
        delivered = order["order_delivered_customer_date"]
        estimated = order["order_estimated_delivery_date"]
        late = delivered is not None and delivered > estimated

        if order["order_status"] != "delivered" or delivered is None:
            return late, False
        if float(self._rng.uniform()) > self.params.review_rate:
            return late, False

        probs = _SCORE_PROBS_LATE if late else _SCORE_PROBS_ON_TIME
        score = int(self._rng.choice(_SCORES, p=probs))
        creation = delivered + timedelta(days=float(self._rng.uniform(0, 3)))
        has_comment = float(self._rng.uniform()) < self.params.comment_rate
        comment = None
        if has_comment:
            pool_start = 0 if score >= 4 else 2
            comment = _REVIEW_COMMENTS[int(self._rng.integers(pool_start, len(_REVIEW_COMMENTS)))]
        reviews.append(
            {
                "review_id": self._hex_id(),
                "order_id": order["order_id"],
                "review_score": score,
                "review_comment_title": "recomendo" if has_comment and score >= 4 else None,
                "review_comment_message": comment,
                "review_creation_date": creation,
                "review_answer_timestamp": creation
                + timedelta(days=float(self._rng.uniform(0, 5))),
            }
        )
        return late, score <= 2

    def _plant_data_warts(self, orders: list[dict], reviews: list[dict]) -> None:
        """Reproduce known blemishes of the real dataset deterministically."""
        delivered_orders = [o for o in orders if o["order_status"] == "delivered"]

        # A couple of delivered orders genuinely missing the delivery date.
        for order in delivered_orders[:2]:
            order["order_delivered_customer_date"] = None

        # A few orders carry two review rows.
        for review in list(reviews[:3]):
            duplicate = dict(review)
            duplicate["review_id"] = self._hex_id()
            duplicate["review_creation_date"] = review["review_creation_date"] + timedelta(days=1)
            duplicate["review_score"] = max(1, int(review["review_score"]) - 1)
            reviews.append(duplicate)
