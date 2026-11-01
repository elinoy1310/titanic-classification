"""Reusable classification metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """
    Calculate binary classification metrics from labels and probabilities.

    Args:
        y_true: Ground-truth binary labels.
        probabilities: Predicted probabilities for the positive class.
        threshold: Decision threshold used to convert probabilities to labels.

    Returns:
        Dictionary containing scalar metrics and a 2x2 confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal lengths.")
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, probabilities))

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def calculate_oof_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Backward-compatible name for metrics calculated on OOF predictions."""
    return calculate_classification_metrics(y_true, probabilities, threshold)
