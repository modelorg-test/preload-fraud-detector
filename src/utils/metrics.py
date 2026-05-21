"""Fraud detection evaluation metrics.

Provides fraud-specific metrics not available in sklearn:
- Precision at a target false-positive rate
- Alert volume estimation at a target recall level
- Value-at-risk: total fraud amount captured at a given threshold
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def precision_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Compute precision at a target false-positive rate.

    This is the primary operational metric for fraud models —
    maximising fraud catch rate while keeping false alerts below
    a tolerance threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth (0=legitimate, 1=fraud).
    y_prob : np.ndarray
        Predicted fraud probabilities.
    target_fpr : float
        Maximum acceptable false-positive rate (default 1%).

    Returns
    -------
    float
        Precision at the threshold corresponding to target_fpr.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Find the threshold closest to the target FPR
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(thresholds) - 1))
    threshold = thresholds[idx]

    preds = (y_prob >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()

    return float(tp / max(tp + fp, 1))


def alert_volume_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_target: float = 0.50,
) -> float:
    """Estimate the alert volume (fraction flagged) to achieve a recall target.

    Used by operations teams to staff fraud investigation queues.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth.
    y_prob : np.ndarray
        Predicted probabilities.
    recall_target : float
        Desired fraud recall (catch rate).

    Returns
    -------
    float
        Fraction of total transactions that would be flagged.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Find threshold achieving the desired recall
    for i in range(len(recall) - 1, -1, -1):
        if recall[i] >= recall_target:
            if i < len(thresholds):
                threshold = thresholds[i]
                flagged = (y_prob >= threshold).sum()
                return float(flagged / len(y_prob))
            break

    return 1.0  # Flag everything if recall target can't be met


def fraud_value_captured(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the monetary value of fraud captured at a threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth.
    y_prob : np.ndarray
        Predicted probabilities.
    amounts : np.ndarray
        Transaction amounts.
    threshold : float
        Decision threshold.

    Returns
    -------
    dict with total_fraud_value, captured_value, capture_rate.
    """
    preds = (y_prob >= threshold).astype(int)
    fraud_mask = y_true == 1

    total_fraud_value = float(amounts[fraud_mask].sum())
    captured_value = float(amounts[fraud_mask & (preds == 1)].sum())
    capture_rate = captured_value / max(total_fraud_value, 1e-6)

    return {
        "total_fraud_value": total_fraud_value,
        "captured_value": captured_value,
        "capture_rate": capture_rate,
    }
