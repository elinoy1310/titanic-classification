"""Typed Pydantic schemas matching the project's YAML configuration files."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class ConfigModel(BaseModel):
    """Base class for validated configuration sections."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelArchitectureConfig(ConfigModel):
    """Neural-network architecture settings."""

    input_dim: int | Literal["auto"] = "auto"
    hidden_layers: list[PositiveInt] = Field(min_length=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    activation: Literal["relu", "tanh", "gelu"]
    output_dim: int = 1


class EarlyStoppingConfig(ConfigModel):
    """Early-stopping settings."""

    enabled: bool
    monitor: Literal["val_loss"]
    patience: int = Field(ge=1)
    min_delta: float = Field(ge=0.0)


class ClassWeightingConfig(ConfigModel):
    """Class-weighting settings for imbalanced targets."""

    enabled: bool
    method: Literal["balanced"]


class TrainingConfig(ConfigModel):
    """Optimization and training-loop settings."""

    seed: int
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    optimizer: Literal["adam"]
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss: Literal["bce_with_logits"]
    early_stopping: EarlyStoppingConfig
    decision_threshold: float = Field(ge=0.0, le=1.0)
    class_weighting: ClassWeightingConfig


class CrossValidationConfig(ConfigModel):
    """Stratified K-Fold settings."""

    n_splits: int = Field(ge=2)
    shuffle: bool
    random_state: int


class ValidationConfig(ConfigModel):
    """Validation strategy settings."""

    method: Literal["holdout", "k_fold"]
    test_size: float = Field(gt=0.0, lt=1.0)
    stratified: bool
    cross_validation: CrossValidationConfig


class FeatureImportanceConfig(ConfigModel):
    """Permutation feature-importance settings."""

    n_repeats: int = Field(ge=1)
    random_state: int


class ModelConfig(ConfigModel):
    """Complete model-training configuration."""

    model: ModelArchitectureConfig
    training: TrainingConfig
    validation: ValidationConfig
    feature_importance: FeatureImportanceConfig


class MissingValuesConfig(ConfigModel):
    """Missing-value imputation settings."""

    enabled: bool
    numeric_strategy: Literal["median", "mean"]
    categorical_strategy: Literal["most_frequent", "constant"]


class ColumnsConfig(ConfigModel):
    """Input columns to drop before model fitting."""

    drop: list[str] = Field(default_factory=list)


class FamilySizeConfig(ConfigModel):
    """Family-size feature toggle."""

    enabled: bool


class TitleConfig(ConfigModel):
    """Passenger-title feature settings."""

    enabled: bool
    group_rare: bool = False
    min_frequency: int = Field(default=10, ge=1)


class FeatureToggleConfig(ConfigModel):
    """Simple feature-engineering toggle."""

    enabled: bool


class FeatureEngineeringConfig(ConfigModel):
    """Feature-engineering settings."""

    family_size: FamilySizeConfig
    title: TitleConfig
    cabin_recorded: FeatureToggleConfig
    deck: FeatureToggleConfig
    ticket_prefix: FeatureToggleConfig


class ScalingConfig(ConfigModel):
    """Numerical feature scaling settings."""

    enabled: bool
    method: Literal["standard", "minmax", "robust", "maxabs"]


class CategoricalEncodingConfig(ConfigModel):
    """Categorical feature encoding settings."""

    enabled: bool
    method: Literal["onehot", "ordinal"]
    ordinal_features: dict[str, list[Any]] = Field(default_factory=dict)


class FareTransformationConfig(ConfigModel):
    """Fare transformation settings."""

    log1p: bool = False


class TransformationsConfig(ConfigModel):
    """Feature transformation settings."""

    fare: FareTransformationConfig = Field(default_factory=FareTransformationConfig)


class OutliersConfig(ConfigModel):
    """Outlier-handling settings."""

    enabled: bool = False


class PreprocessingConfig(ConfigModel):
    """Complete preprocessing configuration."""

    missing_values: MissingValuesConfig
    columns: ColumnsConfig
    feature_engineering: FeatureEngineeringConfig
    scaling: ScalingConfig
    categorical_encoding: CategoricalEncodingConfig
    transformations: TransformationsConfig = Field(default_factory=TransformationsConfig)
    outliers: OutliersConfig = Field(default_factory=OutliersConfig)
