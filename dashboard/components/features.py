import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from .common import find_column, styled_figure


def render_features(data):
    st.subheader("Feature importance")
    st.write(
        "This view ranks original input features by how much they contribute "
        "to validation performance."
    )
    importance = data["frames"].get("feature_importance.csv")
    if importance is None:
        st.warning(
            "Feature importance file not found. Add "
            "`artifacts/feature_importance.csv` with at least `feature` and "
            "`importance_mean` columns."
        )
        st.markdown(
            "**Recommended:** permutation importance on the held-out validation set. "
            "It measures how much a validation metric drops when one original "
            "feature is shuffled and the complete preprocessing + model pipeline "
            "is run again."
        )
        return
    if importance.empty:
        st.info("feature_importance.csv exists but contains no rows.")
        return

    feature_col = find_column(importance, ["feature", "feature_name", "column"])
    value_col = find_column(
        importance,
        ["importance", "mean_importance", "importance_mean", "mean_abs_shap", "permutation_importance"],
    )
    if not feature_col or not value_col:
        st.error("Could not identify feature and importance columns.")
        st.write("Available columns:", list(importance.columns))
        return

    plot_data = importance.copy()
    plot_data[value_col] = pd.to_numeric(plot_data[value_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[value_col]).sort_values(value_col, ascending=False)
    if plot_data.empty:
        st.warning("No numeric importance values to plot.")
        return

    feature_count = len(plot_data)
    options = list(range(1, min(30, feature_count) + 1))
    default_count = min(12, feature_count)
    max_features = st.select_slider(
        "Number of features to display",
        options=options,
        value=default_count,
        key="feature_importance_display_count",
    )
    shown = plot_data.head(max_features).sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4.2, .36 * len(shown))))
    ax.barh(shown[feature_col].astype(str), shown[value_col], alpha=.9)
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_ylabel("")
    ax.set_title("Permutation importance by original feature")
    ax.grid(axis="x", alpha=.25)
    styled_figure(fig)

    st.markdown("#### Importance table")
    sorted_table = plot_data.sort_values(value_col, ascending=False)
    st.dataframe(sorted_table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download feature importance",
        sorted_table.to_csv(index=False).encode("utf-8"),
        "feature_importance_sorted.csv",
        "text/csv",
    )

    st.markdown("#### How this was calculated")
    # st.write(
    #     "Permutation importance is calculated on the held-out validation set. "
    #     "For each original input feature, its values are shuffled while the other "
    #     "columns remain unchanged. The complete preprocessing pipeline and model "
    #     "are then run again, and the decrease in ROC-AUC is recorded. This is "
    #     "repeated several times and summarized in the table."
    # )
    is_oof = data["metrics"].get("evaluation_method") == "out_of_fold"

    if is_oof:
        st.caption(
            "Feature importance is aggregated across validation folds. "
            "The standard deviation represents variation between folds."
        )
    else:
        st.caption(
            "Feature importance is calculated using permutation repeats "
            "on the held-out validation set. The standard deviation "
            "represents variation between permutation repeats."
        )
    
    # st.markdown(
    #     "- **feature** — original input column before preprocessing.\n"
    #     "- **importance_mean** — average decrease in ROC-AUC after shuffling this feature. "
    #     "A larger positive value means the model relied more on this feature for validation predictions.\n"
    #     "- **importance_std** — standard deviation of the decrease across repetitions; "
    #     "a larger value means the estimate varied more between repetitions.\n"
    #     "- **baseline_roc_auc** — ROC-AUC on the original, unmodified validation data.\n"
    #     "- **permuted_roc_auc_mean** — average ROC-AUC after shuffling this feature.\n"
    #     "- **n_repeats** — number of shuffling repetitions used."
    # )
    
#     st.markdown(
#     "- **feature** — original input column before preprocessing.\n"
#     "- **importance_mean** — average decrease in ROC-AUC across folds.\n"
#     "- **importance_std** — standard deviation of the mean importance "
#     "between folds.\n"
#     "- **baseline_roc_auc** — average baseline ROC-AUC across folds.\n"
#     "- **permuted_roc_auc_mean** — average ROC-AUC after permutation.\n"
#     "- **n_folds** — number of folds included in the aggregation."
# )
    st.info(
        "Importance is a global measure of predictive usefulness, not causality or "
        "the direction of a feature's effect. Correlated features can share or mask importance."
    )
