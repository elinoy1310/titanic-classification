"""Shared preprocessing and model inference utilities."""


"""Shared preprocessing and model inference utilities."""

import numpy as np
import pandas as pd
import torch
from typing import Any

from scipy.sparse import issparse

from src.data.preprocessor import transform_features



def run_model_inference(
    X_raw: pd.DataFrame,
    preprocessor,
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """
    Transform raw features and run model inference.

    Returns:
        A dictionary containing:
            - X_processed: Transformed features as a NumPy array.
            - logits: Raw model outputs.
            - probabilities: Sigmoid probabilities.
    """

    if X_raw.empty:
        raise ValueError("Input data contains no rows.")

    # 1. Transform raw features using the fitted preprocessor.
    transformed = transform_features(preprocessor, X_raw)

    # 2. Convert sparse/dense output to a NumPy array.
    if issparse(transformed):
        transformed = transformed.toarray()

    X_processed = np.asarray(transformed, dtype=np.float32)
    
    if X_processed.ndim != 2:
        raise ValueError("Preprocessing must produce a 2D feature matrix.")

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if module.in_features != X_processed.shape[1]:
                raise ValueError(
                    f"Preprocessing produced {X_processed.shape[1]} "
                    f"features, but the model expects {module.in_features}."
                )
            break

    if not np.isfinite(X_processed).all():
        raise ValueError("Preprocessed features contain NaN or infinite values.")

    # 3. Convert features to a PyTorch tensor.
    X_tensor = torch.as_tensor(
        X_processed,
        dtype=torch.float32,
        device=device,
    )

    # 4. Run model inference.
    model.eval()

    with torch.no_grad():
        logits = model(X_tensor).reshape(-1)
        probabilities = torch.sigmoid(logits)

    logits = logits.detach().cpu().numpy()
    probabilities = probabilities.detach().cpu().numpy()

    if len(probabilities) != len(X_raw):
        raise ValueError(
            "Model output size does not match the number of input samples."
        )

    return {
        "X_processed": X_processed,
        "logits": logits,
        "probabilities": probabilities,
    }
    

def run_ensemble_inference(
    X_raw: pd.DataFrame,
    ensemble: list[dict],
    device: torch.device,
) -> dict[str, Any]:
    """Run inference with every model and average their probabilities."""

    if not ensemble:
        raise ValueError("The ensemble contains no models.")

    model_probabilities = []

    for index, member in enumerate(ensemble, start=1):
        if "model" not in member or "preprocessor" not in member:
            raise ValueError(f"Ensemble member {index} must contain model and preprocessor.")

        result = run_model_inference(
            X_raw=X_raw,
            preprocessor=member["preprocessor"],
            model=member["model"],
            device=device,
        )

        model_probabilities.append(result["probabilities"])

    model_probabilities = np.stack(model_probabilities, axis=0)

    probabilities = model_probabilities.mean(axis=0)

    return {
        "probabilities": probabilities,
        "model_probabilities": model_probabilities,
        "n_models": len(ensemble),
    }

