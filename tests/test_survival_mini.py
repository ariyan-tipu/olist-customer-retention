"""Censoring and event logic of the survival export, pinned exactly."""

from __future__ import annotations

import pytest


def test_cohort_is_customers_with_a_delivered_first_order(mini_survival):
    frame, _ = mini_survival
    assert set(frame.index) == {"CU_A", "CU_B", "CU_C", "CU_E", "CU_F", "CU_G"}


def test_event_times_are_measured_from_first_delivery(mini_survival):
    frame, _ = mini_survival
    row = frame.loc["CU_A"]
    assert row["event"] == 1
    # 2017-11-20 14:00 delivery to 2018-02-15 10:00 next purchase.
    assert row["time_days"] == pytest.approx(86.833, abs=0.01)


def test_no_repeat_is_censored_not_churned(mini_survival):
    frame, _ = mini_survival
    row = frame.loc["CU_B"]
    assert row["event"] == 0
    # Censored at the observation end, not labelled as never returning.
    assert row["time_days"] == pytest.approx(301.417, abs=0.01)
    assert row["late_delivery"] == 1
    assert row["review_score"] == 1


def test_repeat_after_classifier_horizon_is_still_an_event(mini_survival):
    """CU_E is a label 0 for the classifier yet a real survival event.

    This asymmetry is the argument for running survival analysis at all:
    the classifier's fixed window throws this information away.
    """
    frame, _ = mini_survival
    row = frame.loc["CU_E"]
    assert row["event"] == 1
    assert row["time_days"] == pytest.approx(324.833, abs=0.01)


def test_pre_delivery_repurchase_is_counted_not_hidden(mini_survival):
    frame, metadata = mini_survival
    assert metadata.pre_delivery_repurchasers == 1
    # CU_G's second order predates the first delivery, so the spell has
    # no post-delivery event and is censored.
    assert frame.loc["CU_G", "event"] == 0


def test_first_order_covariates(mini_survival):
    frame, _ = mini_survival
    assert frame.loc["CU_G", "late_delivery"] == 1
    assert frame.loc["CU_G", "review_score"] == 3
    assert frame.loc["CU_F", "review_score"] == 4
    assert frame.loc["CU_A", "delivery_delay_days"] == -5
