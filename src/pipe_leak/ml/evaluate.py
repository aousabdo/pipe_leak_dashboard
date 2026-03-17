"""
Model evaluation with honest metrics.

No artificial caps. Reports real performance including calibration.
Uses precision-recall curves (better for imbalanced data than ROC alone).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1).
        y_prob: Predicted probabilities P(positive class).

    Returns:
        Dict of metric name -> value. All metrics are honest (no caps).
    """
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    metrics = {
        "n_samples": len(y_true),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if n_pos > 0 and n_neg > 0:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    else:
        metrics["note"] = "Only one class present in test data; limited metrics available."

    return metrics


def compute_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Compute ROC curve data for plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)}


def compute_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Compute Precision-Recall curve data for plotting."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auc = average_precision_score(y_true, y_prob)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_auc": float(auc),
    }


def compute_calibration_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Compute calibration curve data (reliability diagram).

    For each bin of predicted probabilities, compute the actual fraction
    of positive outcomes.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_actual = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_actual.append(y_true[mask].mean())
            bin_counts.append(int(mask.sum()))

    return {
        "predicted": bin_centers,
        "actual": bin_actual,
        "counts": bin_counts,
    }
