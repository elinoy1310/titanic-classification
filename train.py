from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
import yaml

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from preprocessing import (
    load_config as load_preprocessing_config,
    fit_preprocessor,
    transform_features,
    save_preprocessor,
)

from model_inference import run_model_inference
from model import TitanicClassifier
from utils.config_utils import load_config

from feature_importance import (
    calculate_feature_importance,
    aggregate_feature_importance,
    
)

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import nn
from scipy.sparse import issparse

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from preprocessing import (
    load_config as load_preprocessing_config,
    fit_preprocessor,
    transform_features,
    save_preprocessor,
)

from model_inference import run_model_inference

from model import TitanicClassifier
from utils.config_utils import load_config
from feature_importance import calculate_feature_importance


DATA_PATH = Path("data/raw/train.csv")
MODEL_CONFIG_PATH = Path("config/model.yaml")

ARTIFACT_DIR = Path("artifacts")
ENSEMBLE_DIR = ARTIFACT_DIR / "cv_ensemble"
PROCESSED_DIR = Path("data/processed")

TARGET_COLUMN = "Survived"

def create_cv_splitter(cv_config):
    """Create a stratified cross-validation splitter."""

    return StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"],
    )
    
def save_fold_artifacts(
    fold,
    model,
    preprocessor,
    training_result,
    preprocessing_config,
    config,
    fold_dir,
):
    """Save the model, preprocessor and training history of a fold."""

    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    model_path = fold_dir / "model.pth"
    preprocessor_path = fold_dir / "preprocessor.joblib"
    history_path = fold_dir / "training_history.csv"

    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "input_dim": training_result["input_dim"],
        "model_config": config["model"],
        "fold": fold,
        "best_epoch": training_result["best_epoch"],
        "best_val_loss": training_result["best_val_loss"],
    }

    torch.save(checkpoint, model_path)

    save_preprocessor(
        preprocessor,
        preprocessing_config,
        path=preprocessor_path,
    )

    pd.DataFrame(training_result["history"]).to_csv(history_path, index=False)

    return {
        "fold": fold,
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "history_path": history_path,
        "input_dim": training_result["input_dim"],
        "best_epoch": training_result["best_epoch"],
        "best_val_loss": training_result["best_val_loss"],
    }
    
def evaluate_fold(
    model,
    preprocessor,
    X_val,
    y_val,
    device,
    threshold,
):
    """Evaluate one fold on its held-out validation data."""

    inference_result = run_model_inference(
        X_raw=X_val,
        preprocessor=preprocessor,
        model=model,
        device=device,
    )

    probabilities = inference_result["probabilities"]
    predictions = (probabilities >= threshold).astype(int)

    y_true = y_val.to_numpy(dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }

    return {
        "metrics": metrics,
        "probabilities": probabilities,
        "predictions": predictions,
    }
    
def calculate_oof_metrics(y_true, probabilities, threshold):
    """Calculate metrics using all out-of-fold predictions."""

    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }
    
def save_cv_results(
    fold_results,
    summary,
    oof_predictions,
    oof_metrics,
    threshold,
    training_histories,
):
    """Save cross-validation results and OOF predictions."""

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fold_results).to_csv(ARTIFACT_DIR / "cross_validation_folds.csv", index=False)

    summary.to_csv(ARTIFACT_DIR / "cross_validation_summary.csv", index=False)
    
    oof_predictions.to_csv( ARTIFACT_DIR / "validation_predictions.csv", index=False)

    # Save the OOF metrics for the dashboard.
    validation_metrics = {
        **oof_metrics,
        "decision_threshold": threshold,
        "evaluation_method": "out_of_fold",
        "n_folds": len(fold_results),
        "fold_mean": summary["mean"].to_dict(),
        "fold_std": summary["std"].to_dict(),
    }

    with open(ARTIFACT_DIR / "validation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(validation_metrics, file, indent=2)

    # Combine the training histories while preserving fold information.
    history_frames = []

    for fold, history in enumerate(training_histories, start=1):
        history_df = pd.DataFrame(history)
        history_df.insert(0, "fold", fold)
        history_frames.append(history_df)

    combined_history = pd.concat(
        history_frames,
        ignore_index=True,
    )

    combined_history.to_csv(
        ARTIFACT_DIR / "training_history.csv",
        index=False,
    )

    return validation_metrics

def save_ensemble_manifest( fold_artifacts, config, threshold ):
    """Save metadata describing all models in the ensemble."""

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "ensemble_type": "probability_average",
        "n_models": len(fold_artifacts),
        "decision_threshold": threshold,
        "model_config": config["model"],
        "folds": [],
    }

    for artifact in fold_artifacts:
        manifest["folds"].append({
            "fold": artifact["fold"],
            "model_path": str(
                artifact["model_path"].relative_to(ENSEMBLE_DIR)
            ),
            "preprocessor_path": str(
                artifact["preprocessor_path"].relative_to(ENSEMBLE_DIR)
            ),
            "history_path": str(
                artifact["history_path"].relative_to(ENSEMBLE_DIR)
            ),
            "input_dim": artifact["input_dim"],
            "best_epoch": artifact["best_epoch"],
            "best_val_loss": artifact["best_val_loss"],
        })

    manifest_path = ENSEMBLE_DIR / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return manifest_path


def aggregate_cv_results(fold_results):
    """Calculate mean and standard deviation of metrics across folds."""

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

    summary = pd.DataFrame({
        "mean": results_df[metric_columns].mean(),
        "std": results_df[metric_columns].std(ddof=1),
    })

    return summary

def run_cross_validation(
    X,
    y,
    config,
    preprocessing_config,
    device,
):
    """Train, evaluate and save all cross-validation models."""

    cv_config = config["validation"]["cross_validation"]
    feature_importance_config = config["feature_importance"]

    n_repeats = feature_importance_config["n_repeats"]
    random_state = feature_importance_config["random_state"]
    training_config = config["training"]

    n_splits = cv_config["n_splits"]
    threshold = training_config["decision_threshold"]

    cv = create_cv_splitter(cv_config)

    fold_results = []
    fold_artifacts = []
    training_histories = []
    fold_importance_results = []

    # Store one out-of-fold prediction for every sample.
    oof_probabilities = np.full(len(X), np.nan)
    oof_fold = np.full(len(X), -1)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1, ):
        print(f"\n=== Fold {fold}/{n_splits} ===")

        # Set a reproducible seed for this fold.
        fold_seed = training_config["seed"] + fold
        set_seed(fold_seed)

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # 1. Fit a separate preprocessor for this fold.
        (
            preprocessor,
            X_train_processed,
            X_val_processed,
            feature_names,
        ) = prepare_features(
            X_train,
            X_val,
            preprocessing_config,
        )

        # 2. Train a fresh model.
        training_result = train_model(
            X_train_processed=X_train_processed,
            X_val_processed=X_val_processed,
            y_train=y_train,
            y_val=y_val,
            config=config,
            device=device,
        )

        model = training_result["model"]

        # 3. Evaluate the model on its held-out fold.
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

        # 4. Store OOF predictions at their original positions.
        oof_probabilities[val_idx] = evaluation["probabilities"]
        oof_fold[val_idx] = fold

        # 5. Save this fold's model and preprocessor.
        fold_dir = ENSEMBLE_DIR / f"fold_{fold}"

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

        # 6. Calculate feature importance on this fold's validation data.
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

    # 7. Verify that every sample received exactly one OOF prediction.
    if np.isnan(oof_probabilities).any() or np.any(oof_fold == -1):
        raise RuntimeError(
            "Some samples did not receive an out-of-fold prediction."
        )

    # 8. Aggregate metrics across folds.
    summary = aggregate_cv_results(fold_results)

    # 9. Calculate metrics across all OOF predictions.
    y_true = y.to_numpy(dtype=int)

    oof_metrics = calculate_oof_metrics(
        y_true=y_true,
        probabilities=oof_probabilities,
        threshold=threshold,
    )

    # 10. Create the OOF prediction table.
    oof_predictions = X.reset_index(drop=True).copy()

    oof_predictions["Actual"] = y_true
    oof_predictions["Predicted"] = (
        oof_probabilities >= threshold).astype(int)
    oof_predictions["Probability"] = oof_probabilities
    oof_predictions["Fold"] = oof_fold

    # 11. Save all CV evaluation results.
    save_cv_results(
        fold_results=fold_results,
        summary=summary,
        oof_predictions=oof_predictions,
        oof_metrics=oof_metrics,
        threshold=threshold,
        training_histories=training_histories,
    )

    # 12. Aggregate feature importance across folds.
    aggregate_feature_importance(
        fold_importance_results=fold_importance_results,
        output_path=ARTIFACT_DIR / "feature_importance.csv",
    )

    # 13. Save the ensemble manifest.
    save_ensemble_manifest(
        fold_artifacts=fold_artifacts,
        config=config,
        threshold=threshold,
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
    print("Ensemble directory:", ENSEMBLE_DIR)

    return {
        "fold_results": fold_results,
        "summary": summary,
        "oof_metrics": oof_metrics,
        "oof_predictions": oof_predictions,
        "fold_artifacts": fold_artifacts,
    }

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
import torch
from torch import nn
import numpy as np

from model import TitanicClassifier

def train_model(X_train_processed, X_val_processed, y_train, y_val, config, device):
    """Train a fresh model and return its best checkpoint and history."""

    training_config = config["training"]

    # Convert processed features and labels to tensors.
    X_train_tensor = torch.tensor(X_train_processed,dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32, device=device)

    y_train_tensor = torch.tensor(y_train.to_numpy(dtype=np.float32).reshape(-1, 1))
    y_val_tensor = torch.tensor(y_val.to_numpy(dtype=np.float32).reshape(-1, 1), device=device)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=training_config["batch_size"],
        shuffle=True
        )

    # Initialize a new model for this training run.
    input_dim = X_train_processed.shape[1]
    model = TitanicClassifier(input_dim, config).to(device)

    # Configure loss and optional class weighting.
    if training_config["loss"] != "bce_with_logits":
        raise ValueError(f"Unsupported loss: {training_config['loss']}")

    class_weighting = training_config["class_weighting"]

    if class_weighting["enabled"]:
        if class_weighting["method"] != "balanced":
            raise ValueError(f"Unsupported class weighting method: {class_weighting['method']}")

        num_negative = int((y_train == 0).sum())
        num_positive = int((y_train == 1).sum())

        if num_negative == 0 or num_positive == 0:
            raise ValueError("Training data must contain both target classes.")

        pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Configure optimizer.
    optimizer_name = training_config["optimizer"].lower()

    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"])

    # Training loop with early stopping.
    early_config = training_config["early_stopping"]
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(training_config["epochs"]):
        model.train()
        total_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(batch_X)

        train_loss = total_train_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"Epoch {epoch + 1:03d}/{training_config['epochs']} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        improved = val_loss < best_val_loss - early_config["min_delta"]

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_config["enabled"] and epochs_without_improvement >= early_config["patience"]:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_weights is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_weights)
    model.eval()

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "input_dim": input_dim,
        # "X_val_tensor": X_val_tensor,
    }

def evaluate_model(
    model,
    preprocessor,
    X_val_raw,
    y_val,
    training_config,
    best_epoch,
    best_val_loss,
    device,
):
    # Run shared preprocessing and inference.
    inference_result = run_model_inference(
        X_raw=X_val_raw,
        preprocessor=preprocessor,
        model=model,
        device=device,
    )

    val_probabilities = inference_result["probabilities"]

    threshold = training_config["decision_threshold"]
    val_predictions = (val_probabilities >= threshold).astype(int)

    y_true = y_val.to_numpy()

    # Preserve original passenger information.
    validation_predictions = X_val_raw.reset_index(drop=True).copy()

    # Add actual and predicted results.
    validation_predictions["Actual"] = y_true
    validation_predictions["Predicted"] = val_predictions
    validation_predictions["Probability"] = val_probabilities

    # Save validation predictions for visualization and analysis.
    validation_predictions.to_csv(
        ARTIFACT_DIR / "validation_predictions.csv",
        index=False,
    )

    # Calculate metrics.
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "decision_threshold": threshold,
        "accuracy": accuracy_score(y_true, val_predictions),
        "precision": precision_score(
            y_true, val_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, val_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, val_predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, val_probabilities),
        "confusion_matrix": confusion_matrix(
            y_true, val_predictions
        ).tolist(),
        "evaluation_method": "holdout",
    }

    print("\nBest epoch:", best_epoch)
    print("\nValidation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            val_predictions,
            labels=[0, 1],
            target_names=["Did Not Survive", "Survived"],
            zero_division=0,
        )
    )

    return metrics

def to_dense_array(data):
    """Convert sparse or dense features to a NumPy array."""
    if issparse(data):
        data = data.toarray()

    return np.asarray(data, dtype=np.float32)

def save_processed_csv(features, target, path):
    """Save processed features and their target labels."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(features, pd.DataFrame):
        processed_df = features.copy()
    else:
        processed_df = pd.DataFrame(features)

    processed_df[TARGET_COLUMN] = np.asarray(target)
    processed_df.to_csv(path, index=False)

def save_artifacts(model, history, metrics, input_dim, config, training_config, validation_config, feature_names, best_epoch, best_val_loss):
    torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "model_config": config["model"],
    },
    ARTIFACT_DIR / "model.pth",
    )

    pd.DataFrame(history).to_csv(
        ARTIFACT_DIR / "training_history.csv", index=False
    )

    with open(
        ARTIFACT_DIR / "validation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    with open(
        ARTIFACT_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "model_config": config["model"],
                "training_config": training_config,
                "validation_config": validation_config,
                "input_dim": input_dim,
                "feature_names": feature_names,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            },
            file,
            indent=2,
        )

    print("\nSaved model weights to:", ARTIFACT_DIR / "model.pth")
    print("Saved validation metrics to:", ARTIFACT_DIR / "validation_metrics.json")

def prepare_features(X_train, X_val, preprocessing_config, ):
    """Fit preprocessing on training data and transform both datasets."""

    preprocessor = fit_preprocessor(X_train, preprocessing_config)

    X_train_processed = to_dense_array(transform_features(preprocessor, X_train))
    X_val_processed = to_dense_array(transform_features(preprocessor, X_val))

    if not np.isfinite(X_train_processed).all():
        raise ValueError("Invalid values found in training features.")

    if not np.isfinite(X_val_processed).all():
        raise ValueError( "Invalid values found in validation features." )
    
    feature_names = (
        preprocessor.named_steps["preprocessor"]
        .get_feature_names_out()
        .tolist()
    )

    return preprocessor, X_train_processed, X_val_processed, feature_names

def main():
    # --------------------------------------------------
    # 1. Load configuration and initialize random seeds
    # --------------------------------------------------
    config = load_config(MODEL_CONFIG_PATH)
    preprocessing_config = load_preprocessing_config()

    training_config = config["training"]
    validation_config = config["validation"]

    seed = training_config["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 2. Load data and split into train/validation
    # --------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Survived must contain only binary labels 0 and 1.")
        
    validation_method = validation_config["method"].lower()

    if validation_method == "k_fold":
        run_cross_validation(
            X=X,
            y=y,
            config=config,
            preprocessing_config=preprocessing_config,
            device=device,
        )
        return

    if validation_method != "holdout":
        raise ValueError(f"Unsupported validation method: {validation_method}")


    # split the data into training and validation sets using stratification if specified
    split_kwargs = {
        "test_size": validation_config["test_size"],
        "random_state": seed,
    }

    if validation_config["stratified"]:
        split_kwargs["stratify"] = y

    X_train, X_val, y_train, y_val = train_test_split(X, y, **split_kwargs)

    print("\nTrain size:", len(X_train))
    print("Validation size:", len(X_val))


    # --------------------------------------------------
    # 3. Fit preprocessing on training data only
    # --------------------------------------------------
    (
        preprocessor,
        X_train_processed,
        X_val_processed,
        feature_names,
    ) = prepare_features(
        X_train=X_train,
        X_val=X_val,
        preprocessing_config=preprocessing_config,
    )

    # Save the fitted preprocessor for holdout inference.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_preprocessor(preprocessor, preprocessing_config)

    # --------------------------------------------------
    # 4. Save processed datasets for later analysis
    # --------------------------------------------------
    X_train_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_val_df = pd.DataFrame(
        X_val_processed,
        columns=feature_names,
        index=X_val.index,
    )

    print("\nProcessed train shape:", X_train_processed.shape)
    print("Processed validation shape:", X_val_processed.shape)
    print("Contains NaN in train:", np.isnan(X_train_processed).any())
    print("Contains NaN in validation:", np.isnan(X_val_processed).any())


    save_processed_csv(
        X_train_df, y_train,
        PROCESSED_DIR / "train_processed.csv"
    )
    save_processed_csv(
        X_val_df, y_val,
        PROCESSED_DIR / "validation_processed.csv"
    )

    # --------------------------------------------------
    # 5. Train the model using the shared training function
    # --------------------------------------------------
    training_result = train_model(
        X_train_processed=X_train_processed,
        X_val_processed=X_val_processed,
        y_train=y_train,
        y_val=y_val,
        config=config,
        device=device,
    )

    model = training_result["model"]
    history = training_result["history"]
    best_epoch = training_result["best_epoch"]
    best_val_loss = training_result["best_val_loss"]
    input_dim = training_result["input_dim"]
    # X_val_tensor = training_result["X_val_tensor"]

    print("\nBest epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)

    # --------------------------------------------------
    # 6. Evaluate the best model checkpoint on the validation set
    # --------------------------------------------------
    metrics = evaluate_model(
    model=model,
    preprocessor=preprocessor,
    X_val_raw=X_val,
    y_val=y_val,
    training_config=training_config,
    best_epoch=best_epoch,
    best_val_loss=best_val_loss,
    device=device,
)
    
    # --------------------------------------------------
    # 7. Save model weights, metrics, and training history
    # --------------------------------------------------
    feature_importance_config = config["feature_importance"]

    n_repeats = feature_importance_config["n_repeats"]
    random_state = feature_importance_config["random_state"]
    calculate_feature_importance(
    model=model,
    preprocessor=preprocessor,
    X_val_raw=X_val,
    y_val=y_val,
    n_repeats=n_repeats,
    random_state=random_state,
    output_path=ARTIFACT_DIR / "feature_importance.csv",
)
    save_artifacts(model, history, metrics, input_dim, config, training_config, validation_config, feature_names, best_epoch, best_val_loss)

if __name__ == "__main__":
    main()
