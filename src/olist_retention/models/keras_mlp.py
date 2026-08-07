"""A small dense network, offered as the third candidate model.

Why it is here. Tabular data at a base rate near one percent is exactly
the setting where a neural network is widely assumed to help and usually
does not. Running one under the same selection rule as the other two
candidates settles that question with a measurement instead of an
assumption, and the answer goes in the results table whichever way it
falls.

Keras runs on its TensorFlow backend here. TensorFlow is an optional
extra, so importing this module without it installed stays quiet and only
fitting raises.

The scikit-learn wrapper is written out by hand instead of pulled from a
compatibility package, because the contract is short and worth seeing:
`clone` needs `get_params` and `set_params`, and `BaseEstimator` supplies
both as long as `__init__` only records its arguments and derives
nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

# Rows per forward pass. A batch this size is a few megabytes at the
# feature width used here, which keeps the peak inside a small instance.
_DIRECT_CALL_LIMIT = 8192

TENSORFLOW_MISSING = (
    "TensorFlow is not installed. Install the optional extra with "
    'pip install -e ".[tf]" to include the neural candidate.'
)


def keras_available() -> bool:
    """Whether the neural candidate can be fitted in this environment."""
    try:
        import keras  # noqa: F401
    except ImportError:
        return False
    return True


def _import_keras() -> Any:
    try:
        import keras
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(TENSORFLOW_MISSING) from error
    return keras


class KerasMlpClassifier(ClassifierMixin, BaseEstimator):
    """Two hidden layers, dropout, early stopping on a held-out slice.

    Kept deliberately small. The training snapshot has roughly sixty
    thousand rows and about six hundred positives, so capacity is not the
    binding constraint; the number of positive examples is.
    """

    def __init__(
        self,
        hidden_units: Sequence[int] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 512,
        patience: int = 6,
        validation_fraction: float = 0.15,
        balance_classes: bool = True,
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.balance_classes = balance_classes
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------

    def _build(self, keras: Any, n_features: int) -> Any:
        layers = [keras.layers.Input(shape=(n_features,))]
        for units in self.hidden_units:
            layers.append(keras.layers.Dense(units, activation="relu"))
            if self.dropout > 0:
                layers.append(keras.layers.Dropout(self.dropout))
        layers.append(keras.layers.Dense(1, activation="sigmoid"))
        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            # Average precision is the selection measure for the whole
            # comparison, so the network watches the same thing while it
            # trains rather than optimising accuracy at a 1% base rate.
            metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
        )
        return model

    def _class_weight(self, y: np.ndarray) -> dict[int, float] | None:
        """Match the balanced weighting the other two candidates use."""
        if not self.balance_classes:
            return None
        counts = np.bincount(y.astype(int), minlength=2)
        if counts.min() == 0:
            return None
        total = float(counts.sum())
        return {index: total / (2.0 * float(count)) for index, count in enumerate(counts)}

    def fit(self, X: Any, y: Any) -> KerasMlpClassifier:
        keras = _import_keras()
        keras.utils.set_random_seed(self.random_state)

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y).astype("int32").ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Keras carves its validation slice off the end of the array
        # before it shuffles, so an ordered input would validate on an
        # unrepresentative tail. Shuffling first removes that.
        order = np.random.default_rng(self.random_state).permutation(len(X))
        X, y = X[order], y[order]

        self.model_ = self._build(keras, self.n_features_in_)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=self.patience,
                restore_best_weights=True,
            )
        ]
        self.history_ = self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_fraction,
            class_weight=self._class_weight(y),
            callbacks=callbacks,
            verbose=self.verbose,
        ).history
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("KerasMlpClassifier has not been fitted")
        X = np.asarray(X, dtype="float32")
        # Calling the model directly rather than through `predict`. The
        # latter compiles a graph function per input shape, and cross
        # validation hands it a different row count on every fold, so it
        # spends its time retracing and says so in the logs. Direct calls
        # run eagerly, which at these sizes is also the faster of the two.
        # Large inputs are chunked to bound peak memory on a small
        # instance, at a fixed width so the shapes repeat.
        if len(X) <= _DIRECT_CALL_LIMIT:
            positive = np.asarray(self.model_(X, training=False)).ravel()
        else:
            blocks = [
                np.asarray(self.model_(X[start : start + _DIRECT_CALL_LIMIT], training=False))
                for start in range(0, len(X), _DIRECT_CALL_LIMIT)
            ]
            positive = np.concatenate(blocks).ravel()
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __sklearn_tags__(self) -> Any:
        # scikit-learn 1.6 replaced the _estimator_type strings with a
        # tags object. ClassifierMixin fills this in; the override exists
        # to record that the estimator cannot accept sparse input, which
        # is true because Keras wants a dense array.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags
