"""The power analysis module against theory, simulation and statsmodels."""

from __future__ import annotations

import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from olist_retention.experiments.power import TwoProportionPower

ANALYSIS = TwoProportionPower(alpha=0.05, target_power=0.8)


def test_required_n_shrinks_as_the_effect_grows():
    baseline = 0.03
    sizes = [
        ANALYSIS.required_n_per_arm(baseline, baseline * (1 + uplift))
        for uplift in (0.05, 0.10, 0.15, 0.20)
    ]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] > 1000  # detecting +20% on a 3% base is still expensive


def test_power_at_required_n_meets_the_target():
    p0, p1 = 0.03, 0.0345
    n = ANALYSIS.required_n_per_arm(p0, p1)
    assert ANALYSIS.power_at(n, p0, p1) >= 0.799
    assert ANALYSIS.power_at(n - max(1, n // 20), p0, p1) < 0.8


def test_simulation_agrees_with_the_analytic_power():
    p0, p1 = 0.03, 0.0345
    n = ANALYSIS.required_n_per_arm(p0, p1)
    result = ANALYSIS.simulate(p0, p1, n, replications=20000, seed=7)
    assert result.empirical_power == pytest.approx(ANALYSIS.power_at(n, p0, p1), abs=0.03)
    # Under the null the same test should reject at close to alpha.
    assert result.empirical_alpha == pytest.approx(0.05, abs=0.01)


def test_sample_size_agrees_with_statsmodels():
    p0, p1 = 0.03, 0.036
    ours = ANALYSIS.required_n_per_arm(p0, p1)
    effect = proportion_effectsize(p1, p0)
    theirs = NormalIndPower().solve_power(
        effect_size=effect, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided"
    )
    # statsmodels uses the arcsine-transform effect size, so the two
    # approximations legitimately differ by a few percent.
    assert ours == pytest.approx(theirs, rel=0.10)


def test_degenerate_inputs_are_rejected():
    with pytest.raises(ValueError):
        ANALYSIS.required_n_per_arm(0.03, 0.03)
    with pytest.raises(ValueError):
        ANALYSIS.required_n_per_arm(0.0, 0.1)
    with pytest.raises(ValueError):
        TwoProportionPower(alpha=0.0, target_power=0.8)
