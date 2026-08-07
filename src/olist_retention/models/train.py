"""Train the repeat-purchase candidates and persist the winner.

The comparison holds the inputs fixed and varies only the learner. A
regularised logistic regression sets the bar, since its coefficients are
readable and anything more complicated has to earn its place against it.
A histogram gradient boosting classifier is the standard strong baseline
on tabular data. A small dense network on the TensorFlow backend joins
them to test the common assumption that a neural model helps on data of
this shape; it is optional at install time, and the metrics file records
which candidates a given run actually compared.

Selection uses stratified cross-validation on the training snapshot
alone. The reported numbers come from the later out-of-time snapshot,
which no selection decision ever touched.

Every candidate shares one preprocessor, so the comparison stays about
the learner and never about the plumbing. It imputes each numeric column
with its median while keeping an indicator of what was missing, scales,
then one-hot encodes the two categorical columns. Scaling does nothing
for the trees and matters a great deal for the network, which is the
second reason the preprocessor is shared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from olist_retention.config import ModellingConfig
from olist_retention.feature_names import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from olist_retention.logging_utils import get_logger
from olist_retention.models.datasets import ModelData, load_snapshot
from olist_retention.models.keras_mlp import KerasMlpClassifier, keras_available
from olist_retention.models.metrics import classification_metrics, save_evaluation_plots
from olist_retention.monitoring.drift import build_reference

logger = get_logger(__name__)

# Version 2 adds the reference feature distribution the scoring service
# needs for drift monitoring. A version 1 artefact has no reference, so
# the scorer rejects it outright instead of serving with the monitor
# quietly switched off.
ARTEFACT_VERSION = 2

# Ascending cost to serve, declared here rather than measured, so it
# cannot be adjusted once results are in. A logistic regression is a dot
# product over the preprocessed row. A histogram gradient booster walks a
# few hundred shallow trees. The network needs TensorFlow present in the
# serving image at all, which is the largest step in the list.
SERVING_COST_ORDER: tuple[str, ...] = (
    "logistic_regression",
    "gradient_boosting",
    "keras_mlp",
)


@dataclass(frozen=True)
class Selection:
    """Which model was chosen, and the reasoning that chose it."""

    winner: str
    leader: str
    contenders: list[str]
    tie_break_applied: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "winner": self.winner,
            "cross_validation_leader": self.leader,
            "within_one_deviation_of_leader": self.contenders,
            "tie_break_applied": self.tie_break_applied,
            "reason": self.reason,
        }


def select_winner(cv_summary: dict[str, dict]) -> Selection:
    """Choose the model to persist from cross-validation alone.

    Primary rule, fixed before anything was scored: the highest mean
    average precision under stratified cross-validation on the training
    snapshot.

    Tie-break, added after the first run that included all three
    candidates: when a candidate's mean sits within one standard
    deviation of the leader's mean, cross-validation has not told the two
    apart, and the cheaper model to serve wins. Both inputs to that
    decision come from the training side, and the out-of-time snapshot is
    still untouched when it runs, so it does not amount to picking a
    model on the test set.

    The reason it exists: with a repeat rate near one percent, the
    fold-to-fold deviation of average precision is roughly a quarter of
    the measure itself. An argmax across candidates that close is reading
    noise, and it will happily hand back a model that costs a great deal
    more to run for a difference it cannot support.
    """
    leader = max(cv_summary, key=lambda name: cv_summary[name]["average_precision_mean"])
    leader_mean = cv_summary[leader]["average_precision_mean"]
    leader_deviation = cv_summary[leader]["average_precision_std"]

    contenders = [
        name
        for name, summary in cv_summary.items()
        if leader_mean - summary["average_precision_mean"] <= leader_deviation
    ]

    def cost_rank(name: str) -> int:
        return (
            SERVING_COST_ORDER.index(name)
            if name in SERVING_COST_ORDER
            else len(SERVING_COST_ORDER)
        )

    cheapest = min(contenders, key=cost_rank)
    if cheapest == leader:
        return Selection(
            winner=leader,
            leader=leader,
            contenders=sorted(contenders),
            tie_break_applied=False,
            reason=(
                f"{leader} leads cross-validated average precision at {leader_mean:.4f} "
                "and is also the cheapest of the candidates within one standard deviation."
            ),
        )
    return Selection(
        winner=cheapest,
        leader=leader,
        contenders=sorted(contenders),
        tie_break_applied=True,
        reason=(
            f"{leader} leads at {leader_mean:.4f} with a fold-to-fold deviation of "
            f"{leader_deviation:.4f}, and {len(contenders)} candidates sit inside that "
            f"deviation, so cross-validation does not separate them. {cheapest} is the "
            "cheapest of those to serve and is selected on that basis."
        ),
    )


@dataclass(frozen=True)
class TrainingResult:
    winner: str
    metrics: dict
    model_path: Path
    metrics_path: Path


def build_preprocessor() -> ColumnTransformer:
    """Shared preprocessing: impute with indicators, scale, one-hot encode."""
    numeric = Pipeline(
        [
            # Nulls carry meaning here (no review yet, nothing delivered
            # yet), so imputation keeps an indicator column per feature
            # instead of silently blending the missing into the median.
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        sparse_threshold=0.0,
    )


def build_candidates(seed: int, include_keras: bool = True) -> dict[str, Pipeline]:
    """The model pipelines under comparison.

    The neural candidate is dropped when TensorFlow is absent, and the
    caller is told so through the returned keys, which end up written
    into the metrics file.
    """
    candidates = {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0, max_iter=2000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.1,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    if include_keras and keras_available():
        candidates["keras_mlp"] = Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", KerasMlpClassifier(random_state=seed)),
            ]
        )
    elif include_keras:
        logger.warning(
            "TensorFlow is not installed, so the neural candidate is skipped. "
            'Install it with pip install -e ".[tf]" to include it.'
        )
    return candidates


class ChurnModelTrainer:
    """End-to-end training run over the two feature snapshots."""

    def __init__(
        self,
        modelling: ModellingConfig,
        features_dir: Path,
        outputs_dir: Path,
        horizon_days: int,
        include_keras: bool = True,
    ) -> None:
        self.modelling = modelling
        self.features_dir = features_dir
        self.outputs_dir = outputs_dir
        self.horizon_days = horizon_days
        self.include_keras = include_keras

    def run(self, train_cutoff: date, test_cutoff: date) -> TrainingResult:
        train = load_snapshot(self.features_dir, train_cutoff)
        test = load_snapshot(self.features_dir, test_cutoff)
        logger.info(
            "Training on %d customers (base rate %.3f%%), testing out-of-time on %d (%.3f%%)",
            len(train),
            100 * train.base_rate,
            len(test),
            100 * test.base_rate,
        )

        candidates = build_candidates(self.modelling.seed, include_keras=self.include_keras)
        cv_summary = {
            name: self._cross_validate(pipeline, train) for name, pipeline in candidates.items()
        }
        selection = select_winner(cv_summary)
        winner = selection.winner
        logger.info("Selected %s. %s", winner, selection.reason)

        holdout: dict[str, dict] = {}
        scores_by_model: dict[str, np.ndarray] = {}
        for name, pipeline in candidates.items():
            pipeline.fit(train.X, train.y)
            scores = pipeline.predict_proba(test.X)[:, 1]
            scores_by_model[name] = scores
            holdout[name] = classification_metrics(test.y, scores)

        strict = self._strict_cohort_metrics(train, test, scores_by_model[winner])
        importance = self._permutation_importance(candidates[winner], test)

        model_dir = self.modelling.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = self.outputs_dir / "models"
        save_evaluation_plots(test.y, scores_by_model, plots_dir)
        self._plot_importance(importance, plots_dir / "permutation_importance.png")

        metrics = {
            "train_cutoff": train_cutoff.isoformat(),
            "test_cutoff": test_cutoff.isoformat(),
            "horizon_days": self.horizon_days,
            "n_train": len(train),
            "n_test": len(test),
            "train_base_rate": round(train.base_rate, 6),
            "test_base_rate": round(test.base_rate, 6),
            # Written down so a reader can tell whether a given run had
            # the optional neural candidate available to it.
            "candidates_compared": sorted(candidates),
            "cross_validation": cv_summary,
            "selection": selection.as_dict(),
            "holdout": holdout,
            "holdout_strict_new_customers": strict,
            "winner": winner,
            "permutation_importance_top15": importance[:15],
        }
        metrics_path = self.outputs_dir / "models" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

        model_card = {
            "artefact_version": ARTEFACT_VERSION,
            "model_name": winner,
            "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "train_cutoff": train_cutoff.isoformat(),
            "test_cutoff": test_cutoff.isoformat(),
            "horizon_days": self.horizon_days,
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "candidates_compared": sorted(candidates),
            "selection": selection.as_dict(),
            "holdout_metrics": holdout[winner],
            # The drift monitor's baseline is cut from the same rows the
            # winner was fitted on, and travels with it, so the two can
            # never be paired up wrongly at deploy time.
            "reference_distribution": build_reference(
                train.X, NUMERIC_FEATURES, CATEGORICAL_FEATURES
            ),
        }
        model_path = model_dir / "churn_model.joblib"
        joblib.dump({"pipeline": candidates[winner], "model_card": model_card}, model_path)
        (model_dir / "model_card.json").write_text(
            json.dumps(model_card, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Persisted %s to %s", winner, model_path)

        return TrainingResult(
            winner=winner, metrics=metrics, model_path=model_path, metrics_path=metrics_path
        )

    # ------------------------------------------------------------------

    def _cross_validate(self, pipeline: Pipeline, train: ModelData) -> dict:
        folds = StratifiedKFold(
            n_splits=self.modelling.cv_folds, shuffle=True, random_state=self.modelling.seed
        )
        results = cross_validate(
            pipeline,
            train.X,
            train.y,
            cv=folds,
            scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
            n_jobs=1,
        )
        return {
            "folds": self.modelling.cv_folds,
            "roc_auc_mean": round(float(np.mean(results["test_roc_auc"])), 4),
            "roc_auc_std": round(float(np.std(results["test_roc_auc"])), 4),
            "average_precision_mean": round(float(np.mean(results["test_average_precision"])), 4),
            "average_precision_std": round(float(np.std(results["test_average_precision"])), 4),
        }

    @staticmethod
    def _strict_cohort_metrics(
        train: ModelData, test: ModelData, winner_scores: np.ndarray
    ) -> dict:
        """Metrics on test customers the training snapshot never saw.

        The full holdout legitimately shares customers with training
        (the same people exist at both cutoffs); this slice shows there
        is no reliance on remembering individuals.
        """
        seen = set(train.ids)
        mask = (~test.ids.isin(seen)).to_numpy()
        if int(mask.sum()) == 0 or len(set(test.y[mask])) < 2:
            return {"n": int(mask.sum()), "note": "not enough new customers to score"}
        metrics = classification_metrics(test.y[mask], winner_scores[mask])
        metrics["share_of_holdout"] = round(float(mask.mean()), 4)
        return metrics

    def _permutation_importance(self, pipeline: Pipeline, test: ModelData) -> list[dict]:
        result = permutation_importance(
            pipeline,
            test.X,
            test.y,
            scoring="average_precision",
            n_repeats=5,
            random_state=self.modelling.seed,
            n_jobs=1,
        )
        ranked = sorted(
            (
                {
                    "feature": feature,
                    "importance_mean": round(float(mean), 6),
                    "importance_std": round(float(std), 6),
                }
                for feature, mean, std in zip(
                    test.X.columns,
                    result.importances_mean,
                    result.importances_std,
                    strict=True,
                )
            ),
            key=lambda row: row["importance_mean"],
            reverse=True,
        )
        return ranked

    @staticmethod
    def _plot_importance(importance: list[dict], path: Path) -> None:
        top = importance[:15][::-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(
            [row["feature"] for row in top],
            [row["importance_mean"] for row in top],
            xerr=[row["importance_std"] for row in top],
        )
        ax.set_xlabel("Drop in average precision when permuted")
        ax.set_title("Permutation importance (out-of-time holdout)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
