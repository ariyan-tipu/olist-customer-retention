"""Evaluation metrics and plots, built around the class imbalance.

With a base repeat rate of a few percent, ROC-AUC flatters any model
because the huge negative class makes ranking look easy. PR-AUC (average
precision) is the headline number here, together with what a campaign
would actually see: precision, recall and lift when targeting the top
decile of predicted scores.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Threshold-free metrics plus the targeting view at the top decile."""
    metrics = {
        "n": int(len(y_true)),
        "base_rate": round(float(np.mean(y_true)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "brier": round(float(brier_score_loss(y_true, scores)), 6),
    }
    metrics.update(targeting_at_share(y_true, scores, share=0.10))
    return metrics


def targeting_at_share(y_true: np.ndarray, scores: np.ndarray, share: float) -> dict:
    """Precision, recall and lift when acting on the top ``share`` of scores."""
    if not 0 < share < 1:
        raise ValueError("share must be in (0, 1)")
    n_selected = max(1, int(round(share * len(scores))))
    order = np.argsort(scores)[::-1]
    selected = order[:n_selected]
    positives_selected = float(np.sum(y_true[selected]))
    total_positives = float(np.sum(y_true))
    precision = positives_selected / n_selected
    recall = positives_selected / total_positives if total_positives else 0.0
    base_rate = float(np.mean(y_true))
    return {
        "targeting_share": share,
        "precision_at_share": round(precision, 4),
        "recall_at_share": round(recall, 4),
        "lift_at_share": round(precision / base_rate, 2) if base_rate else None,
    }


def save_evaluation_plots(
    y_true: np.ndarray, scores_by_model: dict[str, np.ndarray], out_dir: Path
) -> list[Path]:
    """Write ROC, precision-recall and calibration plots for the holdout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, scores):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on the out-of-time holdout")
    ax.legend()
    written.append(_save(fig, out_dir / "roc_curve.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    base_rate = float(np.mean(y_true))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{name} (AP {ap:.3f})")
    ax.axhline(base_rate, linestyle="--", color="grey", linewidth=1, label="base rate")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the out-of-time holdout")
    ax.legend()
    written.append(_save(fig, out_dir / "pr_curve.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, scores in scores_by_model.items():
        mean_pred, frac_pos = _reliability(y_true, scores, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    limit = max((max(np.max(s) for s in scores_by_model.values())), float(np.mean(y_true)))
    ax.plot([0, limit], [0, limit], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed repeat rate")
    ax.set_title("Calibration (decile bins)")
    ax.legend()
    written.append(_save(fig, out_dir / "calibration.png"))

    return written


def _reliability(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reliability curve on score-quantile bins, robust to skewed scores."""
    quantiles = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    quantiles[-1] += 1e-9
    bins = np.clip(np.digitize(scores, quantiles) - 1, 0, n_bins - 1)
    mean_pred, frac_pos = [], []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            mean_pred.append(float(np.mean(scores[mask])))
            frac_pos.append(float(np.mean(y_true[mask])))
    return np.array(mean_pred), np.array(frac_pos)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
