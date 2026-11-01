"""Configurable Titanic feature engineering and preprocessing pipeline."""

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler
)
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.schemas import PreprocessingConfig
from src.utils.config_utils import load_preprocessing_config


TARGET_COLUMN = "Survived"

NUMERIC_CANDIDATES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "Cabin_Recorded",
]

CATEGORICAL_CANDIDATES = [
    "Sex",
    "Pclass",
    "Embarked",
    "Title",
    "Deck",
    "Ticket_Prefix",
]


def load_config(config_path: str | Path) -> PreprocessingConfig:
    """Load and validate preprocessing settings from a YAML file."""
    return load_preprocessing_config(config_path)


def extract_title(name):
    """Extract a passenger title from a name."""
    match = re.search(r",\s*([^.]*)\.", str(name))

    if match:
        return match.group(1).strip()

    return "Unknown"

def group_rare_titles(titles, min_frequency):
    """Replace title categories below the frequency threshold with 'Rare'."""
    counts = titles.value_counts()
    rare_titles = counts[counts < min_frequency].index

    return titles.where(~titles.isin(rare_titles), "Rare")

def extract_deck(cabin):
    """Extract the first cabin deck letter."""
    if pd.isna(cabin) or not str(cabin).strip():
        return "Unknown"

    return str(cabin).strip()[0].upper()


def extract_ticket_prefix(ticket):
    """Extract the normalized prefix of a ticket."""
    if pd.isna(ticket):
        return "Unknown"

    prefix = re.sub(r"[\d./ ]", "", str(ticket)).upper()

    return prefix if prefix else "Numeric Only"


def add_engineered_features(data: pd.DataFrame, config: PreprocessingConfig, rare_titles=None):
    """
    Create optional features before dropping their source columns.

    This function does not learn statistics from the data.
    """
    data = data.copy()
    options = config.feature_engineering

    if options.family_size.enabled:
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    if options.title.enabled:
        data["Title"] = data["Name"].map(extract_title)

        title_config = options.title

        if title_config.group_rare:
            if rare_titles is None:
                counts = data["Title"].value_counts()

                rare_titles = set( counts[ counts < title_config.min_frequency ].index )

            data["Title"] = data["Title"].where(
                ~data["Title"].isin(rare_titles),
                "Rare",
            )


    if options.deck.enabled:
        data["Deck"] = data["Cabin"].map(extract_deck)

    if options.ticket_prefix.enabled:
        data["Ticket_Prefix"] = data["Ticket"].map(extract_ticket_prefix)

    if config.transformations.fare.log1p:
        data["Fare"] = np.log1p(data["Fare"])

    if options.cabin_recorded.enabled:
        data["Cabin_Recorded"] = (
            data["Cabin"].notna().astype(int)
        )

    columns_to_drop = config.columns.drop
    existing_columns = [col for col in columns_to_drop if col in data.columns]

    data = data.drop(columns=existing_columns)

    return data

def build_categorical_pipeline(config: PreprocessingConfig, encoder):
    """Build categorical imputation and encoding steps."""
    steps = []

    if config.missing_values.enabled:
        steps.append(("imputer", SimpleImputer(strategy=config.missing_values.categorical_strategy)))

    steps.append(("encoder", encoder))

    return Pipeline(steps)

def build_preprocessor(config: PreprocessingConfig, training_features: pd.DataFrame):
    """
    Build a scikit-learn preprocessing pipeline.

    Column types are determined from the training schema and
    the known semantics of the Titanic features.
    """
    numeric_columns = [
        col for col in NUMERIC_CANDIDATES
        if col in training_features.columns
    ]

    categorical_columns = [
        col for col in CATEGORICAL_CANDIDATES
        if col in training_features.columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No supported features remain after preprocessing.")



    # Missing-value handling
    if config.missing_values.enabled:
        numeric_steps = [
            ("imputer", SimpleImputer(
                strategy=config.missing_values.numeric_strategy
            ))
        ]
    else:
        numeric_steps = []

    # Scaling
    if config.scaling.enabled:
        scaler_name = config.scaling.method

        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "maxabs": MaxAbsScaler,
        }

        if scaler_name not in scalers:
            raise ValueError(f"Unsupported scaling method: {scaler_name}")

        numeric_steps.append(("scaler", scalers[scaler_name]()))

    if numeric_columns and not numeric_steps:
       numeric_steps.append(("identity", FunctionTransformer()))

    # Categorical encoding
    transformers = []

    if config.categorical_encoding.enabled:
        encoding_method = config.categorical_encoding.method

        if encoding_method == "onehot":
            if categorical_columns:
                categorical_pipeline = build_categorical_pipeline(config, OneHotEncoder(handle_unknown="ignore"))

                transformers.append(("categorical", categorical_pipeline, categorical_columns))

        elif encoding_method == "ordinal":
            if "Pclass" in categorical_columns:
                ordinal_categories = config.categorical_encoding.ordinal_features.get("Pclass")

                if ordinal_categories != [1, 2, 3]:
                    raise ValueError("For ordinal encoding, configure Pclass as [1, 2, 3].")

                pclass_pipeline = build_categorical_pipeline(
                                        config,
                                        OrdinalEncoder(
                                            categories=[[1, 2, 3]],
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1,
                                        ),
                                    )

                transformers.append(
                    ("pclass_ordinal", pclass_pipeline, ["Pclass"])
                )

            other_categorical_columns = [ col for col in categorical_columns if col != "Pclass" ]

            if other_categorical_columns:
                other_categorical_pipeline =  build_categorical_pipeline(
                                                config,
                                                OneHotEncoder(handle_unknown="ignore"),
                                            )

                transformers.append(
                    (
                        "other_categorical",
                        other_categorical_pipeline,
                        other_categorical_columns,
                    )
                )

        else:
            raise ValueError(
                f"Unsupported encoding method: {encoding_method}"
            )

    else:
        if categorical_columns:
            raise ValueError(
                "Categorical encoding is disabled, but categorical "
                "features remain. Enable encoding or define another "
                "explicit handling strategy."
            )

    # Numeric features
    if numeric_columns:
        numeric_pipeline = Pipeline(numeric_steps)
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    # Do not silently discard unsupported columns.
    supported_columns = set(numeric_columns + categorical_columns)
    unsupported_columns = [
        col for col in training_features.columns
        if col not in supported_columns
    ]

    if unsupported_columns:
        raise ValueError(
            "Unsupported columns remain after feature engineering: "
            f"{unsupported_columns}. Add a handling rule or drop them "
            "in the configuration."
        )

    return Pipeline(
        steps=[
            ("features", FeatureEngineeringTransformer(config)),
            ("preprocessor", ColumnTransformer(transformers=transformers)),
        ]
    )


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Apply deterministic feature engineering using training statistics."""

    def __init__(self, config: PreprocessingConfig):
        """Store the feature-engineering configuration without fitting data."""
        self.config = config

    def fit(self, X, y=None):
        """Learn rare-title categories from the training partition only."""
        options = self.config.feature_engineering
        title_config = options.title

        self.rare_titles_ = set()

        if title_config.enabled and title_config.group_rare:
            titles = X["Name"].map(extract_title)
            counts = titles.value_counts()

            self.rare_titles_ = set( counts[counts < title_config.min_frequency].index )

        return self

    def transform(self, X):
        """Apply configured feature engineering using fitted statistics."""
        if not hasattr(self, "rare_titles_"):
            raise RuntimeError("FeatureEngineeringTransformer must be fitted first.")

        return add_engineered_features(
            X,
            self.config,
            rare_titles=self.rare_titles_,
        )

def split_features_target(data):
    """Separate model inputs from the target column."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = data.drop(columns=[TARGET_COLUMN]).copy()
    y = data[TARGET_COLUMN].copy()

    return X, y

def fit_preprocessor(X_train: pd.DataFrame, config: PreprocessingConfig):
    """Fit the preprocessing pipeline on training data only."""
    training_features = add_engineered_features(X_train, config)
    pipeline = build_preprocessor(config, training_features)
    pipeline.fit(X_train)

    return pipeline

def transform_features(pipeline, data):
    """Transform features using an already-fitted pipeline."""
    return pipeline.transform(data)


def save_preprocessor(pipeline, config: PreprocessingConfig, path: str | Path):
    """Save the fitted pipeline and its configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipeline,
            "config": config.model_dump(mode="json"),
        },
        path,
    )


def load_preprocessor(path: str | Path):
    """Load the saved fitted pipeline and configuration."""
    artifact = joblib.load(path)

    pipeline = artifact["pipeline"]
    saved_config = artifact["config"]

    # Older joblib artifacts stored the preprocessing config as a plain dict.
    if isinstance(saved_config, PreprocessingConfig):
        preprocessing_config = saved_config
    else:
        preprocessing_config = PreprocessingConfig.model_validate(saved_config)

    # Older fitted transformers also retain the original dict on the object.
    feature_transformer = pipeline.named_steps.get("features")
    if feature_transformer is not None and hasattr(feature_transformer, "config"):
        feature_transformer.config = preprocessing_config

    return pipeline, preprocessing_config
