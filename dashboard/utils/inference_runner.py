"""Model loading and inference utilities used by the Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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
    
from src.config.schemas import ModelArchitectureConfig
from src.utils.config_utils import load_model_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_project_importable(project_root: str | Path) -> None:
    """Add the training project root and import classes used by saved artifacts."""
    root_text = str(Path(project_root).expanduser().resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # Import before joblib deserializes the fitted custom transformer.
    import src.data.preprocessor  # noqa: F401
    import src.models.model  # noqa: F401


def _resolve_device(device_name: str) -> torch.device:
    """Resolve a requested Torch device and validate CUDA availability."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but CUDA is not available.")
    return device


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a model checkpoint while mapping tensors to the selected device."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid model checkpoint: {path}")
    required = {"model_state_dict", "input_dim", "model_config"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            f"Checkpoint {path} is missing required fields: {sorted(missing)}"
        )
    return checkpoint


def load_trained_model(
    project_root: str | Path,
    model_path: str | Path,
    preprocessor_path: str | Path,
    model_config_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Load a Holdout checkpoint and its matching fitted preprocessor."""
    project_root = Path(project_root).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    preprocessor_path = Path(preprocessor_path).expanduser().resolve()

    for path, label in (
        (project_root, "Project root"),
        (model_path, "Model checkpoint"),
        (preprocessor_path, "Saved preprocessor"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} was not found: {path}")

    _ensure_project_importable(project_root)
    device = _resolve_device(device_name)
    checkpoint = _load_checkpoint(model_path, device)

    config_path = (
        Path(model_config_path).expanduser().resolve()
        if model_config_path is not None
        else None
    )
    if config_path is not None and config_path.is_file():
        model_config = load_model_config(config_path)
    else:
        model_config = ModelArchitectureConfig.model_validate(
            checkpoint["model_config"]
        )

    from src.models.model import TitanicClassifier
    from src.data.preprocessor import load_preprocessor

    model = TitanicClassifier(int(checkpoint["input_dim"]), model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preprocessor, preprocessing_config = load_preprocessor(preprocessor_path)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "preprocessing_config": preprocessing_config,
        "config": model_config,
        "device": device,
        "input_dim": int(checkpoint["input_dim"]),
    }


def load_trained_ensemble(
    project_root: str | Path,
    ensemble_dir: str | Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Load every fold model and preprocessor listed in the ensemble manifest."""
    project_root = Path(project_root).expanduser().resolve()
    ensemble_dir = Path(ensemble_dir).expanduser().resolve()
    manifest_path = ensemble_dir / "manifest.json"

    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root was not found: {project_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Ensemble manifest was not found: {manifest_path}")

    _ensure_project_importable(project_root)
    device = _resolve_device(device_name)

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("ensemble_type") != "probability_average":
        raise ValueError("Unsupported ensemble type.")

    folds = manifest.get("folds", [])
    if not folds:
        raise ValueError("No models were found in the ensemble manifest.")

    from src.models.model import TitanicClassifier
    from src.data.preprocessor import load_preprocessor

    ensemble = []
    for fold_info in folds:
        fold_number = fold_info["fold"]
        # Manifests created on Windows may contain backslashes even when
        # inference is later run on Linux/WSL.
        model_relative_path = Path(str(fold_info["model_path"]).replace("\\", "/"))
        preprocessor_relative_path = Path(
            str(fold_info["preprocessor_path"]).replace("\\", "/")
        )
        model_path = ensemble_dir / model_relative_path
        preprocessor_path = ensemble_dir / preprocessor_relative_path

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint for fold {fold_number} was not found: {model_path}"
            )
        if not preprocessor_path.is_file():
            raise FileNotFoundError(
                f"Preprocessor for fold {fold_number} was not found: {preprocessor_path}"
            )

        checkpoint = _load_checkpoint(model_path, device)
        model_config = ModelArchitectureConfig.model_validate(checkpoint["model_config"])
        model = TitanicClassifier(int(checkpoint["input_dim"]), model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        preprocessor, preprocessing_config = load_preprocessor(preprocessor_path)
        ensemble.append(
            {
                "fold": fold_number,
                "model": model,
                "preprocessor": preprocessor,
                "preprocessing_config": preprocessing_config,
                "input_dim": int(checkpoint["input_dim"]),
            }
        )

    return {
        "ensemble": ensemble,
        "manifest": manifest,
        "device": device,
        "n_models": len(ensemble),
    }


def run_inference(
    raw_df: pd.DataFrame,
    device: torch.device,
    decision_threshold: float = 0.5,
    target_column: str = "Survived",
    use_ensemble: bool = False,
    model: torch.nn.Module | None = None,
    preprocessor: Any = None,
    ensemble: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate predictions and, when labels exist, calculate evaluation metrics."""
    from src.inference.model_inference import (
        run_model_inference,
        run_ensemble_inference,
    )

    if raw_df.empty:
        raise ValueError("The input CSV contains no rows.")
    if raw_df.columns.has_duplicates:
        raise ValueError("The input CSV contains duplicate column names.")
    if not 0.0 <= float(decision_threshold) <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")

    feature_df = raw_df.copy()
    has_labels = target_column in feature_df.columns
    y_true = None

    if has_labels:
        labels = pd.to_numeric(feature_df.pop(target_column), errors="raise")
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError(
                f"'{target_column}' must contain only 0/1 labels without missing values."
            )
        y_true = labels.to_numpy(dtype=int)

    if use_ensemble:
        if not ensemble:
            raise ValueError("An ensemble must be provided when use_ensemble=True.")
        inference_result = run_ensemble_inference(
            X_raw=feature_df,
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
                "A model and preprocessor must be provided when use_ensemble=False."
            )
        inference_result = run_model_inference(
            X_raw=feature_df,
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
        predictions == 1, "Survived", "Did not survive"
    )

    if use_ensemble:
        for index, fold_probabilities in enumerate(model_probabilities, start=1):
            output_df[f"Fold_{index}_Probability"] = fold_probabilities

    metrics = None
    if has_labels:
        metrics = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(
                y_true, predictions, labels=[0, 1]
            ).tolist(),
            "roc_auc": (
                float(roc_auc_score(y_true, probabilities))
                if len(np.unique(y_true)) == 2
                else None
            ),
        }
        output_df["Actual_Survived"] = y_true
        output_df["Correct_Prediction"] = y_true == predictions

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


def model_input_dim(model: torch.nn.Module, transformed_dim: int) -> int:
    """Validate and return the model's expected input feature dimension."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            expected = int(module.in_features)
            if expected != transformed_dim:
                raise ValueError(
                    f"Preprocessing produced {transformed_dim} features, but "
                    f"the model expects {expected}. Ensure the model and "
                    "preprocessor came from the same training run."
                )
            return expected
    return transformed_dim


def load_saved_threshold(
    artifact_dir: str | Path,
    default: float = 0.5,
) -> float:
    """Load the decision threshold from validation_metrics.json when available."""
    metrics_path = Path(artifact_dir).expanduser() / "validation_metrics.json"
    if not metrics_path.is_file():
        return float(default)
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        threshold = float(metrics.get("decision_threshold", default))
        return threshold if 0.0 <= threshold <= 1.0 else float(default)
    except (OSError, ValueError, TypeError):
        return float(default)
