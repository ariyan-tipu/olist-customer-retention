"""The neural candidate: it behaves like a scikit-learn estimator.

Skipped when the optional extra is absent, which is the common case; the
dedicated CI job installs TensorFlow and runs it for real.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

pytest.importorskip("tensorflow", reason="the tf extra is not installed")

from olist_retention.models.keras_mlp import KerasMlpClassifier, keras_available  # noqa: E402
from olist_retention.models.train import build_candidates  # noqa: E402


@pytest.fixture(scope="module")
def imbalanced():
    """A separable signal at roughly the base rate of the real problem."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 6))
    logit = 1.4 * X[:, 0] - 1.1 * X[:, 1] - 4.2
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return X, y


def test_it_is_available_when_tensorflow_is():
    assert keras_available() is True


def test_clone_works_which_is_what_cross_validation_needs():
    original = KerasMlpClassifier(hidden_units=(8,), epochs=2)
    copy = clone(original)
    assert copy.hidden_units == (8,)
    assert copy.epochs == 2
    assert not hasattr(copy, "model_")


def test_probabilities_have_the_shape_sklearn_expects(imbalanced):
    X, y = imbalanced
    model = KerasMlpClassifier(hidden_units=(8,), epochs=3, random_state=0).fit(X, y)
    probabilities = model.predict_proba(X[:50])

    assert probabilities.shape == (50, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError):
        KerasMlpClassifier().predict_proba(np.zeros((2, 3)))


def test_the_same_seed_gives_the_same_scores(imbalanced):
    X, y = imbalanced
    first = KerasMlpClassifier(hidden_units=(8,), epochs=3, random_state=7).fit(X, y)
    second = KerasMlpClassifier(hidden_units=(8,), epochs=3, random_state=7).fit(X, y)
    assert np.allclose(first.predict_proba(X[:100]), second.predict_proba(X[:100]), atol=1e-5)


def test_it_learns_something_above_the_base_rate(imbalanced):
    """A weak check on purpose: the claim is that it runs and ranks."""
    X, y = imbalanced
    model = KerasMlpClassifier(hidden_units=(16, 8), epochs=12, random_state=0).fit(X, y)
    scores = model.predict_proba(X)[:, 1]
    top_decile = scores >= np.quantile(scores, 0.9)
    assert y[top_decile].mean() > y.mean()


def test_it_survives_cross_validation(imbalanced):
    X, y = imbalanced
    scores = cross_val_score(
        KerasMlpClassifier(hidden_units=(8,), epochs=3, random_state=0),
        X,
        y,
        cv=2,
        scoring="average_precision",
    )
    assert len(scores) == 2
    assert np.isfinite(scores).all()


def test_the_comparison_includes_it_when_asked():
    assert "keras_mlp" in build_candidates(seed=1)
    assert "keras_mlp" not in build_candidates(seed=1, include_keras=False)


def test_balanced_weighting_matches_the_other_candidates():
    model = KerasMlpClassifier()
    weights = model._class_weight(np.array([0] * 99 + [1]))
    assert weights is not None
    assert weights[1] > weights[0]
    assert weights[1] / weights[0] == pytest.approx(99.0)
