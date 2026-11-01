"""Evaluation functions shared by Holdout and Cross Validation."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.evaluation.metrics import calculate_classification_metrics
from src.config.schemas import TrainingConfig
from src.inference.model_inference import run_model_inference


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    """
    Calculate metrics and class predictions from model probabilities.

    Args:
        y_true: Ground-truth labels.
        probabilities: Positive-class probabilities.
        threshold: Decision threshold.

    Returns:
        A metrics dictionary and predicted class labels.
    """
    metrics = calculate_classification_metrics(
        y_true=y_true,
        probabilities=probabilities,
        threshold=threshold,
    )
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return metrics, predictions


def evaluate_model(
    model: Any,
    preprocessor: Any,
    X_val_raw: pd.DataFrame,
    y_val: pd.Series,
    training_config: TrainingConfig,
    best_epoch: int,
    best_val_loss: float,
    device: Any,
) -> dict:
    """
    Run inference on a Holdout validation set and calculate its metrics.

    Args:
        model: Trained PyTorch model.
        preprocessor: Fitted preprocessing pipeline.
        X_val_raw: Raw validation features.
        y_val: Validation labels.
        training_config: Training configuration, including decision threshold.
        best_epoch: Epoch at which the best checkpoint was obtained.
        best_val_loss: Best validation loss.
        device: PyTorch device.

    Returns:
        Dictionary containing metrics, probabilities and predictions.
    """
    inference_result = run_model_inference(
        X_raw=X_val_raw,
        preprocessor=preprocessor,
        model=model,
        device=device,
    )
    probabilities = inference_result["probabilities"]
    y_true = y_val.to_numpy(dtype=int)
    threshold = float(training_config.decision_threshold)

    metrics, predictions = evaluate_predictions(
        y_true, probabilities, threshold
    )
    metrics.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "decision_threshold": threshold,
            "evaluation_method": "holdout",
        }
    )

    print("\nValidation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Did Not Survive", "Survived"],
            zero_division=0,
        )
    )

    return {
        "metrics": metrics,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def evaluate_fold(
    model: Any,
    preprocessor: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    device: Any,
    threshold: float,
) -> dict:
    """
    Evaluate one cross-validation fold on its held-out validation partition.

    Args:
        model: Model trained on the fold's training partition.
        preprocessor: Preprocessor fitted on the fold's training partition.
        X_val: Raw validation features for this fold.
        y_val: Validation labels for this fold.
        device: PyTorch device.
        threshold: Classification decision threshold.

    Returns:
        Metrics, probabilities and predicted labels.
    """
    inference_result = run_model_inference(
        X_raw=X_val,
        preprocessor=preprocessor,
        model=model,
        device=device,
    )
    probabilities = inference_result["probabilities"]
    y_true = y_val.to_numpy(dtype=int)
    metrics, predictions = evaluate_predictions(
        y_true, probabilities, threshold
    )

    return {
        "metrics": metrics,
        "probabilities": probabilities,
        "predictions": predictions,
    }
