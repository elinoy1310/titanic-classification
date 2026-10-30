<!-- omit from toc -->
# Titanic Classification
based on  **[Titanic dataset from Kaggle](https://www.kaggle.com/competitions/titanic/data)**.
<!-- omit from toc -->
## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. (Optional) Create a virtual environment](#2-optional-create-a-virtual-environment)
  - [3. Install dependencies](#3-install-dependencies)
- [Kaggle API Setup](#kaggle-api-setup)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Download the Dataset](#download-the-dataset)
- [Model Training](#model-training)
- [Streamlit Dashboard](#streamlit-dashboard)
  - [1. Evaluation](#1-evaluation)
  - [2. Inference](#2-inference)
- [Architecture and Design Choices](#architecture-and-design-choices)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Evaluation Strategy](#evaluation-strategy)
  - [Application Design](#application-design)

## Project Structure
CHANGE!
```text
titanic-classification/
│
├── config/
│   ├── model.yaml
│   └── preprocessing.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── scripts/
│   └── download_data.py
│
├── artifacts/
│   ├── model.pth
│   ├── preprocessor.joblib
│   ├── model_metadata.json
│   ├── validation_metrics.json
│   ├── validation_predictions.csv
│   ├── training_history.csv
│   └── feature_importance.csv
│
├── titanic_dashboard/
│   ├── ds_app.py
│   ├── components/
│   └── utils/
│
├── train.py
├── model.py
├── preprocessing.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/elinoy1310/titanic-classification
cd titanic-classification
```

### 2. (Optional) Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Kaggle API Setup

The dataset is downloaded directly from Kaggle using the Kaggle API.

1. Create a Kaggle account if you do not already have one.
2. Generate an API token from your Kaggle account settings.
3. Configure your Kaggle API credentials locally according to the [official Kaggle API documentation](https://www.kaggle.com/docs/api).
4. Accept the Titanic competition rules on Kaggle before downloading the data.

## Exploratory Data Analysis (EDA)

The exploratory data analysis is performed in a Jupyter Notebook:

```text
notebooks/eda.ipynb
```

To run the notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

Alternatively, open the notebook directly in Visual Studio Code or JupyterLab.

## Download the Dataset
MAYBE CHANGE
Run the following command from the project root:

```bash
python scripts/download_data.py
```

The script downloads only `train.csv` into:

```text
data/raw/train.csv
```

The project uses only the training dataset provided by Kaggle. A separate validation set is created during model training.

## Model Training
CHANGE: ADD A note that it base on config file + the paths to config files

To train the classification model and generate the evaluation artifacts, run:

```bash
python train.py
```
MAYBE CHANGE : not add to much details
The training pipeline performs the following steps:

1. Loads the Titanic dataset from `data/raw/train.csv`.
2. Splits the data into training and validation sets.
3. Fits the preprocessing pipeline using training data only.
4. Transforms the training and validation features.
5. Trains a PyTorch classification model.
6. Uses early stopping to select the best model checkpoint.
7. Evaluates the best model on the held-out validation set.
8. Calculates permutation-based feature importance.
9. Saves the trained model, preprocessor, and evaluation results.

Generated artifacts are saved in the `artifacts/` directory.

## Streamlit Dashboard

The project includes a Streamlit application with two main workspaces: **Evaluation** and **Inference**.

Run the dashboard from the project root:
maybe CHANGE: move the file
```bash
streamlit run dashboard/ds_app.py
```

### 1. Evaluation

CHANGE: ADD A SCREENSHOT OF EACH SECTION
The Evaluation workspace presents the results of the model training and validation process.

It includes the following sections:

- **Overview:** Key validation metrics, including Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- **Training Progress:** Training and validation loss curves across epochs.
- **Validation Analysis:** Confusion matrix, ROC curve, and prediction explorer.
- **Feature Importance:** Permutation Importance analysis to understand the contribution of the original input features to model performance.
- **Artifacts:** Information about the generated model artifacts and saved evaluation results.

### 2. Inference
CHANGE: ADD A SCREENSHOT
The Inference workspace allows users to run the trained model on an external CSV dataset.

The interface allows users to:

1. Select a CSV file.
2. Load the trained model and fitted preprocessing pipeline.
3. Generate survival predictions and probabilities.
4. View the predictions in a table.

If the input dataset contains the `Survived` target column, additional evaluation metrics and visualizations are displayed.

If the target column is not available, the application displays predictions without calculating evaluation metrics.

## Architecture and Design Choices

CHANGE: think what i should  explain gere
### Data Preprocessing

A Scikit-learn preprocessing pipeline is used to handle missing values, encode categorical variables, scale numerical features, and perform feature engineering.

Additional features are extracted from existing columns, including passenger titles, cabin decks, ticket prefixes, and family size.

The preprocessing pipeline is fitted exclusively on the training data to prevent data leakage. The fitted pipeline is saved and reused during inference to ensure consistent transformations.

### Model Architecture
A binary classification neural network is implemented using PyTorch.

The model is trained using Binary Cross Entropy with Logits Loss and the Adam optimizer. Early stopping is used to select the best checkpoint based on validation loss.

### Evaluation Strategy

The dataset is divided into training and validation sets using a reproducible split.

Model performance is evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

Additional evaluation tools include a confusion matrix, ROC curve, training history visualization, and permutation-based feature importance.

Permutation Importance measures how much model performance decreases when an individual input feature is randomly shuffled.

### Application Design

The Streamlit application separates model evaluation from inference.

The Evaluation workspace visualizes the training and validation results saved during training.

The Inference workspace loads the saved model and preprocessing pipeline to generate predictions on external datasets.

The application is organized into separate components and utility modules to improve maintainability and make future extensions easier.
