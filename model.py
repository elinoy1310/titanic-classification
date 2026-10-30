
import torch
from torch import nn
import numpy as np

class TitanicClassifier(nn.Module):
    """Configurable MLP for binary classification."""

    def __init__(self, input_dim, config):
        super().__init__()

        model_config = config["model"]
        hidden_layers = model_config["hidden_layers"]
        dropout_rate = model_config["dropout"]
        activation_name = model_config["activation"].lower()

        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }

        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}")

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(activations[activation_name]())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, model_config["output_dim"]))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


