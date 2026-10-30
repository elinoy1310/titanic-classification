"""Model loading and inference utilities for the Titanic dashboard."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from scipy.sparse import issparse
from model_inference import run_model_inference, run_ensemble_inference


def _ensure_project_importable(project_root: Path) -> None:
    """Make the training project importable for model and joblib classes."""
    root_text = str(project_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # Import the module before joblib loads the saved pipeline, because the
    # serialized FeatureEngineeringTransformer is defined in preprocessing.py.
    importlib.import_module("preprocessing")


def load_trained_model(
    project_root: str | Path,
    model_path: str | Path,
    preprocessor_path: str | Path,
    model_config_path: str | Path,
    device_name: str = "auto",
):
    """Load the user's checkpoint, fitted preprocessing pipeline and model class."""
    project_root = Path(project_root).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    preprocessor_path = Path(preprocessor_path).expanduser().resolve()
    model_config_path = Path(model_config_path).expanduser().resolve()

    for path, label in (
        (project_root, "Project root"),
        (model_path, "Model checkpoint"),
        (preprocessor_path, "Saved preprocessor"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} was not found: {path}")

    _ensure_project_importable(project_root)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but CUDA is not available.")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(model_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Unsupported checkpoint format. Expected model_state_dict and input_dim."
        )

    input_dim = int(checkpoint["input_dim"])

    if model_config_path.is_file():
        with model_config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    elif "model_config" in checkpoint:
        config = {"model": checkpoint["model_config"]}
    else:
        raise FileNotFoundError(
            "Model configuration not found. Provide config/model.yaml or use a "
            "checkpoint containing model_config."
        )

    if "model" not in config:
        if "model_config" in checkpoint:
            config["model"] = checkpoint["model_config"]
        else:
            raise ValueError("The model configuration must contain a 'model' section.")

    model_module = importlib.import_module("model")
    if not hasattr(model_module, "TitanicClassifier"):
        raise ImportError("model.py must define TitanicClassifier.")

    model_class = model_module.TitanicClassifier
    model = model_class(input_dim, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load through the project's preprocessing module to preserve its artifact format.
    preprocessing_module = importlib.import_module("preprocessing")
    if hasattr(preprocessing_module, "load_preprocessor"):
        preprocessor, preprocessing_config = preprocessing_module.load_preprocessor(
            preprocessor_path
        )
    else:
        # Fallback for compatible joblib artifacts.
        import joblib

        artifact = joblib.load(preprocessor_path)
        preprocessor = artifact["pipeline"] if isinstance(artifact, dict) else artifact
        preprocessing_config = artifact.get("config", {}) if isinstance(artifact, dict) else {}

    return {
        "model": model,
        "preprocessor": preprocessor,
        "preprocessing_config": preprocessing_config,
        "config": config,
        "device": device,
        "input_dim": input_dim,
    }


def run_inference(
    raw_df: pd.DataFrame,
    device: torch.device,
    decision_threshold: float = 0.5,
    target_column: str = "Survived",
    use_ensemble: bool = False,
    model=None,
    preprocessor=None,
    ensemble=None,
) -> dict[str, Any]:
    """Run inference using either a single model or an ensemble."""

    if raw_df.empty:
        raise ValueError("The input CSV contains no rows.")

    if raw_df.columns.has_duplicates:
        raise ValueError("The input CSV contains duplicate column names.")

    if not 0.0 <= float(decision_threshold) <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")

    df = raw_df.copy()

    has_labels = target_column in df.columns
    y_true = None

    if has_labels:
        y_true = pd.to_numeric(df.pop(target_column),errors="raise",).to_numpy()

        if pd.isna(y_true).any() or not set(np.unique(y_true)).issubset({0, 1}):
            raise ValueError(f"'{target_column}' must contain only 0/1 labels without missing values.")

    # Run the selected inference mode.
    if use_ensemble:
        if not ensemble:
            raise ValueError("An ensemble must be provided when use_ensemble=True.")

        inference_result = run_ensemble_inference(
            X_raw=df,
            ensemble=ensemble,
            device=device,
        )

        probabilities = inference_result["probabilities"]

        model_probabilities = inference_result["model_probabilities"]
        n_models = inference_result["n_models"]

        feature_dimensions = [member["input_dim"] for member in ensemble]

    else:
        if model is None or preprocessor is None:
            raise ValueError(
                "A model and preprocessor must be provided "
                "when use_ensemble=False."
            )

        inference_result = run_model_inference(
            X_raw=df,
            preprocessor=preprocessor,
            model=model,
            device=device,
        )

        probabilities = inference_result["probabilities"]

        model_probabilities = probabilities.reshape(1, -1)
        n_models = 1

        feature_dimensions = [inference_result["X_processed"].shape[1]]

    predictions = (probabilities >= float(decision_threshold)).astype(int)

    output_df = raw_df.copy().reset_index(drop=True)

    output_df["Survival_Probability"] = probabilities
    output_df["Predicted_Survived"] = predictions

    output_df["Predicted_Label"] = np.where(
        predictions == 1,
        "Survived",
        "Did not survive",
    )

    if use_ensemble:
        for index, fold_probabilities in enumerate(
            model_probabilities,
            start=1,
        ):
            output_df[f"Fold_{index}_Probability"] = fold_probabilities

    metrics = None

    if has_labels:
        y_true = np.asarray(y_true, dtype=int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0 )),
            "recall": float( recall_score( y_true, predictions, zero_division=0 )),
            "f1": float(f1_score(y_true, predictions, zero_division=0 )),
            "confusion_matrix": confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1],
            ).tolist(),
            "roc_auc": (
                float(roc_auc_score(y_true, probabilities))
                if len(np.unique(y_true)) == 2
                else None
            ),
        }

        output_df["Actual_Survived"] = y_true
        output_df["Correct_Prediction"] = (
            y_true == predictions
        )

    return {
        "predictions": output_df,
        "metrics": metrics,
        "has_labels": has_labels,
        "y_true": y_true,
        "probabilities": probabilities,
        "predicted_labels": predictions,
        "model_probabilities": model_probabilities,
        "n_models": n_models,
        "use_ensemble": use_ensemble,
        "threshold": float(decision_threshold),
        "n_rows": len(raw_df),
        "n_features_after_preprocessing": feature_dimensions,
    }


def model_input_dim(model, transformed_dim: int) -> int:
    """Infer input dimension from the first Linear layer when possible."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            expected = int(module.in_features)
            if expected != transformed_dim:
                raise ValueError(
                    f"Preprocessing produced {transformed_dim} features, but the model "
                    f"expects {expected}. Check that model.pth and preprocessor.joblib "
                    "were produced by the same training run."
                )
            return expected
    return transformed_dim


def load_saved_threshold(artifact_dir: str | Path, default: float = 0.5) -> float:
    """Read the decision threshold recorded by train.py, if available."""
    metrics_path = Path(artifact_dir).expanduser() / "validation_metrics.json"
    if not metrics_path.is_file():
        return float(default)
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        return float(metrics.get("decision_threshold", default))
    except (OSError, ValueError, TypeError):
        return float(default)
    

def load_trained_ensemble(
    project_root: str | Path,
    ensemble_dir: str | Path,
    device_name: str = "auto",
):
    """Load all models and their corresponding preprocessors."""

    project_root = Path(project_root).expanduser().resolve()
    ensemble_dir = Path(ensemble_dir).expanduser().resolve()

    manifest_path = ensemble_dir / "manifest.json"

    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root was not found: {project_root}")

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Ensemble manifest was not found: {manifest_path}")

    _ensure_project_importable(project_root)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but CUDA is not available." )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("ensemble_type") != "probability_average":
        raise ValueError("Unsupported ensemble type.")

    folds = manifest.get("folds", [])

    if not folds:
        raise ValueError("No models were found in the ensemble manifest.")

    model_module = importlib.import_module("model")
    preprocessing_module = importlib.import_module("preprocessing")

    ensemble = []

    for fold_info in folds:
        fold = fold_info["fold"]

        model_path = ensemble_dir / fold_info["model_path"]
        preprocessor_path = ensemble_dir / fold_info["preprocessor_path"]

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint for fold {fold} was not found: "
                f"{model_path}"
            )

        if not preprocessor_path.is_file():
            raise FileNotFoundError(
                f"Preprocessor for fold {fold} was not found: "
                f"{preprocessor_path}"
            )

        try:
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                model_path,
                map_location=device,
            )

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint for fold {fold}.")

        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Missing model_state_dict in fold {fold}.")

        input_dim = int(checkpoint["input_dim"])

        model_config = checkpoint["model_config"]
        config = {"model": model_config}

        model_class = model_module.TitanicClassifier

        model = model_class(input_dim, config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        preprocessor, preprocessing_config = preprocessing_module.load_preprocessor(preprocessor_path)

        ensemble.append({
            "fold": fold,
            "model": model,
            "preprocessor": preprocessor,
            "preprocessing_config": preprocessing_config,
            "input_dim": input_dim,
        })

    return {
        "ensemble": ensemble,
        "manifest": manifest,
        "device": device,
        "n_models": len(ensemble),
    }

