"""Streamlit UI for running inference with the saved Titanic model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import json
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# from utils.inference_runner import load_saved_threshold, load_trained_model, run_inference
from .common import styled_figure
from utils.inference_runner import (
    load_saved_threshold,
    load_trained_model,
    load_trained_ensemble,
    run_inference,
)

def render_inference(default_project_root: Path, default_artifact_dir: Path):
    st.subheader("Inference workspace")
    st.write(
        "Run the saved PyTorch model on a CSV file. If the CSV contains `Survived`, "
        "the app also calculates evaluation metrics; otherwise it shows predictions only."
    )

    with st.container(border=True):
        st.markdown("#### 1 · Model selection")
        metrics_path = default_artifact_dir / "validation_metrics.json"

        with metrics_path.open("r", encoding="utf-8") as file:
            saved_metrics = json.load(file)

        evaluation_method = saved_metrics.get("evaluation_method", "holdout")

        if evaluation_method == "out_of_fold":
            inference_mode = "Ensemble (K-Fold)"
            st.caption(
                "The selected training run used cross-validation. "
                "Inference will use the ensemble of fold models."
            )
        elif evaluation_method == "holdout":
            inference_mode = "Single Model (Holdout)"
            st.caption(
                "The selected training run used a holdout split. "
                "Inference will use the single model trained on the training split."
            )
        else:
            raise ValueError(
                f"Unsupported evaluation method: {evaluation_method}"
            )

        st.info(f"Selected inference mode: **{inference_mode}**")

        project_root_text = st.text_input(
            "Training project root",
            value=str(default_project_root),
            help="Folder containing model.py and preprocessing.py.",
            key="inference_project_root",
        )

        if inference_mode == "Ensemble (K-Fold)":

            ensemble_dir_text = st.text_input(
                "Ensemble directory",
                value=str(default_artifact_dir / "cv_ensemble"),
                key="inference_ensemble_dir",
            )

        else:
            c1, c2 = st.columns(2)

            with c1:
                model_path_text = st.text_input(
                    "Model checkpoint (.pth)",
                    value=str(default_artifact_dir / "model.pth"),
                    key="inference_model_path",
                )

                preprocessor_path_text = st.text_input(
                    "Fitted preprocessor (.joblib)",
                    value=str(default_artifact_dir / "preprocessor.joblib"),
                    key="inference_preprocessor_path",
                )

            with c2:
                config_path_text = st.text_input(
                    "Model config (.yaml)",
                    value=str(default_project_root / "config" / "model.yaml"),
                    key="inference_config_path",
                )

        artifact_dir_text = st.text_input(
            "Artifacts folder (for saved threshold)",
            value=str(default_artifact_dir),
            key="inference_artifact_dir",
        )

    with st.container(border=True):
        st.markdown("#### 2 · Input dataset")
        uploaded_csv = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            key="inference_csv_upload",
        )
        st.caption(
            "Expected columns are the original, unprocessed Titanic input columns. "
            "Do not upload data that has already been one-hot encoded or scaled."
        )

    saved_threshold = load_saved_threshold(
        st.session_state.get("inference_artifact_dir", str(default_artifact_dir))
    )
    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(saved_threshold),
        step=0.01,
        help="Loaded from validation_metrics.json by default; adjust it to explore the precision/recall trade-off.",
        key="inference_threshold",
    )

    run_clicked = st.button("Run inference", type="primary", use_container_width=True)
    
    if run_clicked:
        try:
            if uploaded_csv is None:
                raise ValueError(
                    "Choose a CSV file before running inference."
                )

            raw_df = pd.read_csv(uploaded_csv)
            source_label = uploaded_csv.name

            with st.spinner("Loading model and running predictions..."):

                if inference_mode == "Ensemble (K-Fold)":

                    loaded = load_trained_ensemble(
                        project_root=project_root_text,
                        ensemble_dir=ensemble_dir_text,
                    )

                    result = run_inference(
                        raw_df=raw_df,
                        device=loaded["device"],
                        decision_threshold=threshold,
                        use_ensemble=True,
                        ensemble=loaded["ensemble"],
                    )

                    result["model_type"] = "Ensemble"
                    result["n_models"] = loaded["n_models"]

                else:

                    loaded = load_trained_model(
                        project_root=project_root_text,
                        model_path=model_path_text,
                        preprocessor_path=preprocessor_path_text,
                        model_config_path=config_path_text,
                    )

                    result = run_inference(
                        raw_df=raw_df,
                        model=loaded["model"],
                        preprocessor=loaded["preprocessor"],
                        device=loaded["device"],
                        decision_threshold=threshold,
                    )

                    result["model_type"] = "Single Model"
                    result["n_models"] = 1

            result["source_label"] = source_label
            result["device"] = str(loaded["device"])

            st.session_state["inference_result"] = result
            st.session_state["inference_error"] = None

        except Exception as exc:
            st.session_state["inference_result"] = None
            st.session_state["inference_error"] = str(exc)

    error = st.session_state.get("inference_error")
    if error:
        st.error("Inference could not be completed.")
        st.code(error)
        st.info(
            "Check that model.pth and preprocessor.joblib came from the same training run, "
            "that the project root contains model.py and preprocessing.py, and that the CSV "
            "contains the original input columns expected by the saved preprocessor."
        )

    result = st.session_state.get("inference_result")
    if not result:
        st.info("Choose the model files and browse for a CSV file, then select **Run inference**.")
        return

    st.divider()
    st.markdown("### Results")
    feature_dimensions = ", ".join(
        map(str, result["n_features_after_preprocessing"])
    )

    st.caption(
        f"Dataset: {result['source_label']} · "
        f"{result['n_rows']:,} rows · "
        f"Model: {result['model_type']} · "
        f"Models used: {result['n_models']} · "
        f"Feature dimensions: {feature_dimensions} · "
        f"Device: {result['device']}"
    )

    if result["has_labels"]:
        metrics = result["metrics"]
        metric_items = [
            ("Accuracy", metrics["accuracy"]),
            ("Precision", metrics["precision"]),
            ("Recall", metrics["recall"]),
            ("F1-score", metrics["f1"]),
            ("ROC-AUC", metrics["roc_auc"]),
        ]
        columns = st.columns(len(metric_items))
        for column, (label, value) in zip(columns, metric_items):
            with column:
                column.metric(label, "N/A" if value is None else f"{value:.3f}")
        st.caption("Metrics are calculated against the `Survived` labels present in this CSV.")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Confusion matrix")
            matrix = confusion_matrix(
                result["y_true"], result["predicted_labels"], labels=[0, 1]
            )
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            image = ax.imshow(matrix, cmap="Blues")
            ax.set_xticks([0, 1], labels=["Did not survive", "Survived"])
            ax.set_yticks([0, 1], labels=["Did not survive", "Survived"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion matrix")
            cutoff = matrix.max() / 2 if matrix.size else 0
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, str(int(matrix[i, j])), ha="center", va="center",
                        color="white" if matrix[i, j] > cutoff else "#152536",
                        fontsize=13, fontweight="bold",
                    )
            fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
            styled_figure(fig)

        with right:
            st.markdown("#### ROC curve")
            if len(set(result["y_true"])) == 2:
                fpr, tpr, _ = roc_curve(result["y_true"], result["probabilities"])
                auc = roc_auc_score(result["y_true"], result["probabilities"])
                fig, ax = plt.subplots(figsize=(5.2, 4.2))
                ax.plot(fpr, tpr, linewidth=2.4, label=f"Model (AUC = {auc:.3f})")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
                ax.set(
                    xlabel="False Positive Rate", ylabel="True Positive Rate",
                    title="ROC curve", xlim=(0, 1), ylim=(0, 1.02),
                )
                ax.grid(True, alpha=.25)
                ax.legend(frameon=False, loc="lower right")
                styled_figure(fig)
            else:
                st.info("ROC-AUC and the ROC curve require both target classes in the CSV.")
    else:
        st.info(
            "This CSV does not contain `Survived`, so evaluation metrics and ROC/confusion-matrix "
            "plots are unavailable. The model predictions below are still valid inference outputs."
        )

    predictions = result["predictions"]
    st.markdown("#### Prediction table")
    if result["has_labels"]:
        filter_option = st.selectbox(
            "Rows to display",
            ["All rows", "Incorrect predictions only", "Correct predictions only"],
            key="inference_result_filter",
        )
        if filter_option == "Incorrect predictions only":
            predictions = predictions[predictions["Correct_Prediction"] == False]  # noqa: E712
        elif filter_option == "Correct predictions only":
            predictions = predictions[predictions["Correct_Prediction"] == True]  # noqa: E712

    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions as CSV",
        data=result["predictions"].to_csv(index=False).encode("utf-8"),
        file_name="titanic_inference_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
