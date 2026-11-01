"""Load YAML files and validate them as typed Pydantic configuration models."""

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from src.config.schemas import ModelConfig, PreprocessingConfig

PydanticConfig = TypeVar("PydanticConfig", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return its root mapping."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    return raw_config


def load_typed_config(
    path: str | Path,
    schema: type[PydanticConfig],
) -> PydanticConfig:
    """Load a YAML mapping and validate it against the supplied Pydantic schema."""
    raw_config = load_yaml(path)
    try:
        return schema.model_validate(raw_config)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid configuration in '{path}':\n{exc}"
        ) from exc


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate model, training, validation, and feature-importance settings."""
    return load_typed_config(path, ModelConfig)


def load_preprocessing_config(path: str | Path) -> PreprocessingConfig:
    """Load and validate preprocessing settings."""
    return load_typed_config(path, PreprocessingConfig)


def load_config(path: str | Path) -> ModelConfig:
    """Backward-compatible alias for loading the main model configuration."""
    return load_model_config(path)
