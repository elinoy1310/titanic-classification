"""Feature preparation helpers shared by Holdout and Cross Validation."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from src.config.schemas import PreprocessingConfig

from src.data.preprocessor import fit_preprocessor, transform_features


def to_dense_array(data: Any) -> np.ndarray:
    """Convert sparse or dense transformed features to a float32 NumPy array."""
    if issparse(data):
        data = data.toarray()
    return np.asarray(data, dtype=np.float32)


def prepare_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    preprocessing_config: PreprocessingConfig,
) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
    """Fit preprocessing on training data only, then transform both splits."""
    preprocessor = fit_preprocessor(X_train, preprocessing_config)
    X_train_processed = to_dense_array(transform_features(preprocessor, X_train))
    X_val_processed = to_dense_array(transform_features(preprocessor, X_val))

    if not np.isfinite(X_train_processed).all():
        raise ValueError("Invalid values found in training features.")
    if not np.isfinite(X_val_processed).all():
        raise ValueError("Invalid values found in validation features.")

    feature_names = (
        preprocessor.named_steps["preprocessor"].get_feature_names_out().tolist()
    )
    return preprocessor, X_train_processed, X_val_processed, feature_names
