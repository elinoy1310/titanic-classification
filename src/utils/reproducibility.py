"""Utilities for reproducible experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Seed value used by the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
