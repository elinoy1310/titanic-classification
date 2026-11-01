"""Stratified K-Fold training, evaluation, and OOF aggregation."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from src.artifacts.model_io import (
    save_fold_artifacts,
    save_ensemble_manifest,
)
from src.artifacts.results_io import save_cv_results
from src.data.preprocessing import prepare_features
from src.evaluation.evaluation import evaluate_fold
from src.evaluation.metrics import calculate_oof_metrics
from src.evaluation.feature_importance import (
    calculate_feature_importance,
    aggregate_feature_importance,
)
from src.training.trainer import train_model
from src.utils.reproducibility import set_seed
from src.config.schemas import CrossValidationConfig, ModelConfig, PreprocessingConfig


def create_cv_splitter(cv_config: CrossValidationConfig) -> StratifiedKFold:
    """Create a reproducible stratified K-Fold splitter."""
    return StratifiedKFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=cv_config.random_state,
    )


def aggregate_cv_results(fold_results: list[dict]) -> pd.DataFrame:
    """Calculate mean and sample standard deviation of fold metrics."""
    if not fold_results:
        raise ValueError("No fold results were provided.")

    results_df = pd.DataFrame(fold_results)
    metric_columns = [
        "best_epoch",
        "best_val_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    missing_columns = [
        column for column in metric_columns
        if column not in results_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing metric columns: {missing_columns}"
        )

    return pd.DataFrame(
        {
            "mean": results_df[metric_columns].mean(),
            "std": results_df[metric_columns].std(ddof=1),
        }
    )


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    config: ModelConfig,
    preprocessing_config: ModelConfig,
    device: torch.device,
    artifact_dir: Path,
) -> dict:
    """
    Train and evaluate one model per fold and build OOF predictions.

    Each fold receives its own fitted preprocessor and model checkpoint.
    """
    artifact_dir = Path(artifact_dir)
    ensemble_dir = artifact_dir / "cv_ensemble"

    cv_config = config.validation.cross_validation
    feature_config = config.feature_importance
    training_config = config.training

    n_repeats = feature_config.n_repeats
    random_state = feature_config.random_state
    n_splits = cv_config.n_splits
    threshold = training_config.decision_threshold
    cv = create_cv_splitter(cv_config)

    fold_results = []
    fold_artifacts = []
    training_histories = []
    fold_importance_results = []

    oof_probabilities = np.full(len(X), np.nan)
    oof_fold = np.full(len(X), -1)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        set_seed(training_config.seed + fold)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        preprocessor, X_train_processed, X_val_processed, _ = (
            prepare_features(
                X_train, X_val, preprocessing_config
            )
        )

        training_result = train_model(
            X_train_processed=X_train_processed,
            X_val_processed=X_val_processed,
            y_train=y_train,
            y_val=y_val,
            config=config,
            device=device,
        )
        model = training_result["model"]

        evaluation = evaluate_fold(
            model=model,
            preprocessor=preprocessor,
            X_val=X_val,
            y_val=y_val,
            device=device,
            threshold=threshold,
        )

        metrics = {
            "fold": fold,
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            **evaluation["metrics"],
        }
        fold_results.append(metrics)
        oof_probabilities[val_idx] = evaluation["probabilities"]
        oof_fold[val_idx] = fold

        fold_dir = ensemble_dir / f"fold_{fold}"
        fold_artifact = save_fold_artifacts(
            fold=fold,
            model=model,
            preprocessor=preprocessor,
            training_result=training_result,
            preprocessing_config=preprocessing_config,
            config=config,
            fold_dir=fold_dir,
        )
        fold_artifacts.append(fold_artifact)
        training_histories.append(training_result["history"])

        fold_importance = calculate_feature_importance(
            model=model,
            preprocessor=preprocessor,
            X_val_raw=X_val,
            y_val=y_val,
            n_repeats=n_repeats,
            random_state=random_state + fold - 1,
            output_path=fold_dir / "feature_importance.csv",
        )
        fold_importance["fold"] = fold
        fold_importance_results.append(fold_importance)

        print(
            f"Best epoch: {metrics['best_epoch']} | "
            f"F1: {metrics['f1']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    if np.isnan(oof_probabilities).any() or np.any(oof_fold == -1):
        raise RuntimeError(
            "Some samples did not receive an out-of-fold prediction."
        )

    summary = aggregate_cv_results(fold_results)
    y_true = y.to_numpy(dtype=int)
    oof_metrics = calculate_oof_metrics(
        y_true=y_true,
        probabilities=oof_probabilities,
        threshold=threshold,
    )

    oof_predictions = X.reset_index(drop=True).copy()
    oof_predictions["Actual"] = y_true
    oof_predictions["Predicted"] = (
        oof_probabilities >= threshold
    ).astype(int)
    oof_predictions["Probability"] = oof_probabilities
    oof_predictions["Fold"] = oof_fold

    save_cv_results(
        fold_results=fold_results,
        summary=summary,
        oof_predictions=oof_predictions,
        oof_metrics=oof_metrics,
        threshold=threshold,
        training_histories=training_histories,
        artifact_dir=artifact_dir,
    )

    aggregate_feature_importance(
        fold_importance_results=fold_importance_results,
        output_path=artifact_dir / "feature_importance.csv",
    )
    save_ensemble_manifest(
        fold_artifacts=fold_artifacts,
        config=config,
        threshold=threshold,
        ensemble_dir=ensemble_dir,
    )

    print("\n=== Cross-Validation Summary ===")
    print(summary.round(4))
    print("\n=== Overall OOF Metrics ===")
    for name, value in oof_metrics.items():
        print(f"{name}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            oof_predictions["Predicted"],
            labels=[0, 1],
            target_names=["Did Not Survive", "Survived"],
            zero_division=0,
        )
    )
    print(f"\nSaved {len(fold_artifacts)} ensemble models.")
    print("Ensemble directory:", ensemble_dir)

    return {
        "fold_results": fold_results,
        "summary": summary,
        "oof_metrics": oof_metrics,
        "oof_predictions": oof_predictions,
        "fold_artifacts": fold_artifacts,
    }
