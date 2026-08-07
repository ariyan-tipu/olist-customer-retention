"""Population stability index: what it should and should not report.

None of this needs Spark or a trained model, so the drift machinery is
exercised on every machine that can run the suite at all.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from olist_retention.monitoring.drift import (
    MATERIAL_AT,
    STABLE_BELOW,
    DriftMonitor,
    band_for,
    build_categorical_reference,
    build_numeric_reference,
    build_reference,
)

NUMERIC = ("recency_days", "frequency")
CATEGORICAL = ("customer_region",)


def _frame(rng: np.random.Generator, n: int, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recency_days": rng.normal(100 + shift, 25, n),
            "frequency": rng.poisson(2, n).astype(float),
            "customer_region": rng.choice(["sudeste", "nordeste", "sul"], n),
        }
    )


def test_identical_populations_report_almost_no_drift():
    rng = np.random.default_rng(0)
    reference = build_reference(_frame(rng, 4000), NUMERIC, CATEGORICAL)
    monitor = DriftMonitor(reference, window=4000)
    monitor.observe(_frame(rng, 4000).to_dict("records"))

    snapshot = monitor.snapshot()
    assert snapshot["status"] == "ok"
    assert snapshot["max_psi"] < STABLE_BELOW
    assert snapshot["material_features"] == []


def test_a_shifted_feature_is_caught_and_named():
    rng = np.random.default_rng(1)
    reference = build_reference(_frame(rng, 4000), NUMERIC, CATEGORICAL)
    monitor = DriftMonitor(reference, window=4000)
    # Two standard deviations of movement on one feature only.
    monitor.observe(_frame(rng, 4000, shift=50.0).to_dict("records"))

    snapshot = monitor.snapshot()
    assert snapshot["max_psi_feature"] == "recency_days"
    assert snapshot["max_psi"] >= MATERIAL_AT
    assert "recency_days" in snapshot["material_features"]
    assert "frequency" not in snapshot["material_features"]


def test_a_categorical_shift_is_caught():
    rng = np.random.default_rng(2)
    reference = build_reference(_frame(rng, 3000), NUMERIC, CATEGORICAL)
    monitor = DriftMonitor(reference, window=3000)

    moved = _frame(rng, 3000)
    moved["customer_region"] = "sul"
    monitor.observe(moved.to_dict("records"))

    snapshot = monitor.snapshot()
    assert "customer_region" in snapshot["material_features"]


def test_a_constant_column_is_skipped_rather_than_reported():
    """Quantile edges collapse on a constant column; that is not drift."""
    frame = pd.DataFrame(
        {
            "recency_days": np.zeros(500),
            "frequency": np.arange(500.0),
            "customer_region": ["a"] * 500,
        }
    )
    reference = build_reference(frame, NUMERIC, CATEGORICAL)
    assert "recency_days" in reference["skipped_numeric"]
    assert "recency_days" not in reference["numeric"]
    assert "frequency" in reference["numeric"]


def test_values_outside_the_training_range_land_in_the_end_bins():
    reference = build_numeric_reference(np.arange(1000.0))
    assert reference is not None
    assert reference.edges[0] == -np.inf
    assert reference.edges[-1] == np.inf


def test_missing_values_are_counted_not_dropped_silently():
    values = np.concatenate([np.arange(900.0), np.full(100, np.nan)])
    reference = build_numeric_reference(values)
    assert reference is not None
    assert reference.missing_rate == pytest.approx(0.1)


def test_an_unseen_category_falls_into_the_tail_bucket():
    reference = asdict(build_categorical_reference(["a"] * 90 + ["b"] * 10, max_categories=2))
    monitor = DriftMonitor(
        {"numeric": {}, "categorical": {"customer_region": reference}}, window=200
    )
    monitor.observe([{"customer_region": "never_seen"} for _ in range(100)])
    snapshot = monitor.snapshot()
    assert snapshot["max_psi"] > 0.0


def test_the_monitor_says_it_is_warming_up_before_it_has_enough_rows():
    rng = np.random.default_rng(3)
    monitor = DriftMonitor(build_reference(_frame(rng, 500), NUMERIC, CATEGORICAL))
    monitor.observe(_frame(rng, 5).to_dict("records"))
    assert monitor.snapshot()["status"] == "warming up"


def test_the_monitor_switches_off_without_a_reference():
    monitor = DriftMonitor(None)
    monitor.observe([{"recency_days": 1.0}])
    assert monitor.snapshot()["enabled"] is False


def test_the_window_is_bounded():
    rng = np.random.default_rng(4)
    monitor = DriftMonitor(build_reference(_frame(rng, 500), NUMERIC, CATEGORICAL), window=100)
    monitor.observe(_frame(rng, 900).to_dict("records"))
    snapshot = monitor.snapshot()
    assert snapshot["window_rows"] == 100
    assert snapshot["rows_scored_total"] == 900


def test_the_bands_match_the_documented_thresholds():
    assert band_for(0.05) == "stable"
    assert band_for(0.15) == "moderate"
    assert band_for(0.40) == "material"
