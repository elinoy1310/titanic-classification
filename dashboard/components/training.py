"""Training-history and loss-curve visualizations."""

import matplotlib.pyplot as plt
import streamlit as st

from .common import find_column, styled_figure




def render_training(data):
    """Render training and validation loss history."""
    st.subheader("Training progress")

    history = data["frames"]["training_history.csv"]

    if history is None or history.empty:
        st.warning("Training history is empty.")
        return

    is_oof = data["metrics"].get("evaluation_method") == "out_of_fold"
    fold_col = find_column(history, ["fold"]) if is_oof else None
    selected_fold = None
    if is_oof and fold_col:
        folds = sorted(history[fold_col].unique())

        selected_fold = st.selectbox(
            "Select fold",
            folds,
            format_func=lambda value: f"Fold {value}",
        )

        history = history[history[fold_col] == selected_fold].copy()

    epoch_col = find_column(history, ["epoch", "epochs", "step"])
    train_col = find_column(
        history,
        ["train_loss", "training_loss", "loss"],
    )
    val_col = find_column(
        history,
        ["val_loss", "validation_loss", "valid_loss"],
    )

    if not train_col and not val_col:
        st.error("Could not find train_loss or val_loss columns.")
        st.write("Available columns:", list(history.columns))
        return

    x = history[epoch_col] if epoch_col else range(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(10, 4.6))

    if train_col:
        ax.plot(
            x,
            history[train_col],
            label="Train loss",
            linewidth=2.2,
            marker="o",
            markersize=3,
        )

    if val_col:
        ax.plot(
            x,
            history[val_col],
            label="Validation loss",
            linewidth=2.2,
            marker="o",
            markersize=3,
        )

        idx = history[val_col].astype(float).idxmin()

        best_x = history.loc[idx, epoch_col] if epoch_col else idx + 1

        best_y = history.loc[idx, val_col]

        ax.scatter(
            [best_x],
            [best_y],
            marker="*",
            s=150,
            zorder=5,
            label="Lowest validation loss",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Fold {selected_fold} - Loss by epoch" if fold_col else "Loss by epoch"
    )

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    styled_figure(fig)

    if train_col and val_col:
        st.info(
            "A widening gap where training loss keeps falling "
            "while validation loss rises can indicate overfitting."
        )

    with st.expander("View training history data"):
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )
