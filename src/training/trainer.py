"""PyTorch model training implementation."""

import numpy as np
import torch
from torch import nn

from src.models.model import TitanicClassifier
from src.config.schemas import ModelConfig


def train_model(
    X_train_processed: np.ndarray,
    X_val_processed: np.ndarray,
    y_train,
    y_val,
    config: ModelConfig,
    device: torch.device,
) -> dict:
    """
    Train a fresh model and restore its best validation-loss checkpoint.

    Args:
        X_train_processed: Preprocessed training features.
        X_val_processed: Preprocessed validation features.
        y_train: Training labels.
        y_val: Validation labels.
        config: Complete model and training configuration.
        device: Device on which the model is trained.

    Returns:
        Dictionary containing the trained model, history, best epoch,
        best validation loss, and input dimension.
    """
    training_config = config.training

    X_train_tensor = torch.as_tensor(
        X_train_processed, dtype=torch.float32
    )
    X_val_tensor = torch.as_tensor(
        X_val_processed, dtype=torch.float32, device=device
    )
    y_train_tensor = torch.as_tensor(
        y_train.to_numpy(dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32,
    )
    y_val_tensor = torch.as_tensor(
        y_val.to_numpy(dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32,
        device=device,
    )

    train_dataset = torch.utils.data.TensorDataset(
        X_train_tensor, y_train_tensor
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
    )

    input_dim = int(X_train_processed.shape[1])
    model = TitanicClassifier(input_dim, config).to(device)

    if training_config.loss != "bce_with_logits":
        raise ValueError(
            f"Unsupported loss: {training_config.loss}"
        )

    class_weighting = training_config.class_weighting
    if class_weighting.enabled:
        if class_weighting.method != "balanced":
            raise ValueError(
                "Unsupported class weighting method: "
                f"{class_weighting['method']}"
            )

        num_negative = int((y_train == 0).sum())
        num_positive = int((y_train == 1).sum())
        if num_negative == 0 or num_positive == 0:
            raise ValueError(
                "Training data must contain both target classes."
            )

        pos_weight = torch.tensor(
            [num_negative / num_positive],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer_name = training_config.optimizer.lower()
    if optimizer_name != "adam":
        raise ValueError(
            f"Unsupported optimizer: {training_config.optimizer}"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    early_config = training_config.early_stopping
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(training_config.epochs):
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

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"Epoch {epoch + 1:03d}/{training_config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        improved = (
            val_loss < best_val_loss - early_config.min_delta
        )
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

        if (
            early_config.enabled
            and epochs_without_improvement >= early_config.patience
        ):
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_weights is None:
        raise RuntimeError(
            "Training did not produce a valid model checkpoint."
        )

    model.load_state_dict(best_weights)
    model.eval()

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "input_dim": input_dim,
    }
