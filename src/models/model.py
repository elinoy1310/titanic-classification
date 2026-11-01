"""Configurable PyTorch multilayer perceptron for Titanic survival."""

import torch
from torch import nn

from src.config.schemas import ModelArchitectureConfig, ModelConfig


class TitanicClassifier(nn.Module):
    """Multilayer perceptron that returns a single raw logit per passenger."""

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig | ModelArchitectureConfig,
    ) -> None:
        """Build the network from the typed model configuration."""
        super().__init__()
        model_config = config.model if isinstance(config, ModelConfig) else config

        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }
        activation_name = model_config.activation.lower()
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}")

        layers = []
        previous_dim = input_dim
        for hidden_dim in model_config.hidden_layers:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    activations[activation_name](),
                ]
            )
            if model_config.dropout > 0:
                layers.append(nn.Dropout(model_config.dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, model_config.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for the supplied batch of feature vectors."""
        return self.network(x)
