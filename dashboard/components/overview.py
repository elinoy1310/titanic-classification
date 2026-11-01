"""Summary metrics and high-level validation results."""

import streamlit as st


def render_overview(data):
    """Render validation metrics and explain how to interpret them."""
    metrics = data["metrics"]
    st.subheader("Validation performance")
    is_oof = metrics.get("evaluation_method") == "out_of_fold"

    if is_oof:
        st.caption(
            "Metrics are calculated using out-of-fold predictions. "
            "Each sample was evaluated by a model that did not train on it."
        )
    else:
        st.caption("Metrics are calculated on the held-out validation split.")

    specs = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
        ("ROC-AUC", "roc_auc"),
    ]

    cols = st.columns(5)

    fold_mean = metrics.get("fold_mean", {})
    fold_std = metrics.get("fold_std", {})

    for col, (label, key) in zip(cols, specs):
        value = metrics.get(key)

        if isinstance(value, (int, float)):
            col.metric(label, f"{value:.3f}")
        else:
            col.metric(label, "—")

        if is_oof and key in fold_mean:
            mean = fold_mean[key]
            std = fold_std.get(key)

            if mean is not None and std is not None:
                col.caption(f"Fold mean: {mean:.3f} ± {std:.3f}")

    st.divider()

    if is_oof:
        a, b, c = st.columns(3)

        a.metric( "Number of folds", str(metrics.get("n_folds", "—")),)
        predictions_df = data["frames"].get( "validation_predictions.csv" )

        n_samples = len(predictions_df) if predictions_df is not None else 0

        b.metric("Number of evaluated samples", f"{n_samples:,}",)

        c.metric("Decision threshold", f"{metrics.get('decision_threshold', 0.5):.2f}",)

    else:
        a, b = st.columns(2)

        a.metric("Best epoch", str(metrics.get("best_epoch", "—")) )

        b.metric("Decision threshold",f"{metrics.get('decision_threshold', 0.5):.2f}")

    st.subheader("How to read these results")

    if is_oof:
        st.markdown(
            "- **OOF metrics** summarize predictions across all samples.\n"
            "- **Fold mean ± std** describes the average performance and "
            "variation across the validation folds.\n"
            "- OOF results are not an evaluation of the final Ensemble "
            "on an independent test set."
        )
    else:
        st.markdown(
            "- **Accuracy** measures the share of correct predictions.\n"
            "- **Precision vs. recall** exposes the trade-off between "
            "false positives and false negatives.\n"
            "- **ROC-AUC** measures ranking quality across thresholds."
        )

    with st.expander("Show all classification metrics"):
        st.json(metrics)

    folds = data["frames"].get("cross_validation_folds.csv")

    if is_oof and folds is not None and not folds.empty:
        st.subheader("Cross-validation fold results")
        st.dataframe(
            folds,
            use_container_width=True,
            hide_index=True,
        )

