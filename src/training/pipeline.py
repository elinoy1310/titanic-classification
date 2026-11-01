"""High-level orchestration for Holdout and Cross Validation training."""

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.artifacts.model_io import save_holdout_artifacts
from src.artifacts.results_io import (
    save_holdout_results,
    save_processed_csv,
)
from src.data.preprocessing import prepare_features
from src.evaluation.evaluation import evaluate_model
from src.evaluation.feature_importance import calculate_feature_importance
from src.training.cross_validation import run_cross_validation
from src.training.trainer import train_model
from src.utils.reproducibility import set_seed
from src.config.schemas import ModelConfig
from src.data.preprocessor import load_config as load_preprocessing_config

TARGET_COLUMN = "Survived"


def run_training_pipeline(
    data_path: Path,
    config: ModelConfig,
    preprocessing_config_path: Path,
    artifact_dir: Path,
    processed_dir: Path,
    save_processed_data: bool,
    device: torch.device,
) -> dict:
    """
    Load data and run the configured Holdout or K-Fold training workflow.

    Args:
        data_path: Path to the Kaggle training CSV.
        config: Model, training, validation, and feature-importance settings.
        preprocessing_config_path: Path to preprocessing YAML.
        artifact_dir: Destination for model and evaluation artifacts.
        processed_dir: Destination for optional processed CSV files.
        save_processed_data: Whether to save transformed train/validation CSVs.
        device: PyTorch execution device.

    Returns:
        Results returned by the selected validation workflow.
    """
    data_path = Path(data_path)
    artifact_dir = Path(artifact_dir)
    processed_dir = Path(processed_dir)

    preprocessing_config = load_preprocessing_config(preprocessing_config_path)
    training_config = config.training
    validation_config = config.validation

    set_seed(training_config.seed)

    df = pd.read_csv(data_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Survived must contain only binary labels 0 and 1.")

    validation_method = validation_config.method.lower()
    if validation_method == "k_fold":
        return run_cross_validation(
            X=X,
            y=y,
            config=config,
            preprocessing_config=preprocessing_config,
            device=device,
            artifact_dir=artifact_dir,
        )
    if validation_method != "holdout":
        raise ValueError(f"Unsupported validation method: {validation_method}")

    split_kwargs = {
        "test_size": validation_config.test_size,
        "random_state": training_config.seed,
    }
    if validation_config.stratified:
        split_kwargs["stratify"] = y

    X_train, X_val, y_train, y_val = train_test_split(X, y, **split_kwargs)
    print("\nTrain size:", len(X_train))
    print("Validation size:", len(X_val))

    preprocessor, X_train_processed, X_val_processed, feature_names = prepare_features(X_train, X_val, preprocessing_config)

    if save_processed_data:
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_val_df = pd.DataFrame( X_val_processed, columns=feature_names, index=X_val.index)
        save_processed_csv( X_train_df, y_train, processed_dir / "train_processed.csv")
        save_processed_csv( X_val_df, y_val, processed_dir / "validation_processed.csv")

    print("\nProcessed train shape:", X_train_processed.shape)
    print("Processed validation shape:", X_val_processed.shape)

    training_result = train_model(
        X_train_processed=X_train_processed,
        X_val_processed=X_val_processed,
        y_train=y_train,
        y_val=y_val,
        config=config,
        device=device,
    )

    model = training_result["model"]
    evaluation = evaluate_model(
        model=model,
        preprocessor=preprocessor,
        X_val_raw=X_val,
        y_val=y_val,
        training_config=training_config,
        best_epoch=training_result["best_epoch"],
        best_val_loss=training_result["best_val_loss"],
        device=device,
    )

    validation_predictions = X_val.reset_index(drop=True).copy()
    validation_predictions["Actual"] = y_val.to_numpy(dtype=int)
    validation_predictions["Predicted"] = evaluation["predictions"]
    validation_predictions["Probability"] = evaluation["probabilities"]

    save_holdout_results(
        metrics=evaluation["metrics"],
        history=training_result["history"],
        predictions=validation_predictions,
        artifact_dir=artifact_dir,
    )
    save_holdout_artifacts(
        model=model,
        preprocessor=preprocessor,
        preprocessing_config=preprocessing_config,
        input_dim=training_result["input_dim"],
        config=config,
        training_config=training_config,
        validation_config=validation_config,
        feature_names=feature_names,
        best_epoch=training_result["best_epoch"],
        best_val_loss=training_result["best_val_loss"],
        artifact_dir=artifact_dir,
    )

    feature_config = config.feature_importance
    calculate_feature_importance(
        model=model,
        preprocessor=preprocessor,
        X_val_raw=X_val,
        y_val=y_val,
        n_repeats=feature_config.n_repeats,
        random_state=feature_config.random_state,
        output_path=artifact_dir / "feature_importance.csv",
    )

    return {
        "metrics": evaluation["metrics"],
        "training_result": training_result,
        "predictions": validation_predictions,
    }
