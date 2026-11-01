"""Permutation-based feature importance for validation data."""


"""Permutation feature importance utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.inference.model_inference import run_model_inference


def calculate_feature_importance(
    model,
    preprocessor,
    X_val_raw,
    y_val,
    n_repeats=20,
    random_state=42,
    output_path=None,
):
    """Calculate permutation importance on a validation set."""

    if len(np.unique(y_val)) != 2:
        raise ValueError("ROC-AUC requires both classes in the validation set.")

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    device = next(model.parameters()).device
    model.eval()

    y_true = np.asarray(y_val).reshape(-1)
    rng = np.random.default_rng(random_state)

    def predict_probabilities(X):
        """Run the fitted preprocessing and model on a raw feature frame."""
        result = run_model_inference(
            X_raw=X,
            preprocessor=preprocessor,
            model=model,
            device=device,
        )
        return result["probabilities"]

    baseline_probabilities = predict_probabilities(X_val_raw)
    baseline_auc = roc_auc_score(y_true, baseline_probabilities)

    results = []

    for feature in X_val_raw.columns:
        importance_scores = []

        for _ in range(n_repeats):
            X_permuted = X_val_raw.copy(deep=True)
            X_permuted[feature] = rng.permutation(
                X_permuted[feature].to_numpy()
            )

            permuted_probabilities = predict_probabilities(X_permuted)
            permuted_auc = roc_auc_score( y_true, permuted_probabilities)

            importance_scores.append(baseline_auc - permuted_auc)

        results.append({
            "feature": feature,
            "importance_mean": float(np.mean(importance_scores)),
            "importance_std": float(
                np.std(importance_scores, ddof=1)
                if n_repeats > 1 else 0.0
            ),
            "baseline_roc_auc": float(baseline_auc),
            "permuted_roc_auc_mean": float(
                baseline_auc - np.mean(importance_scores)
            ),
            "n_repeats": n_repeats,
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Feature importance saved to: {output_path}")

    return results_df


def aggregate_feature_importance(fold_importance_results: list[pd.DataFrame], output_path: str | Path | None = None) -> pd.DataFrame:
    """Aggregate feature importance results across folds."""

    if not fold_importance_results:
        raise ValueError("No feature importance results were provided.")

    combined = pd.concat(fold_importance_results, ignore_index=True)

    aggregated = (
        combined.groupby("feature")
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
            baseline_roc_auc=("baseline_roc_auc", "mean"),
            permuted_roc_auc_mean=("permuted_roc_auc_mean", "mean"),
            n_folds=("feature", "count"),
        )
        .reset_index()
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    aggregated["importance_std"] = aggregated["importance_std"].fillna(0.0)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(output_path, index=False)

    return aggregated
