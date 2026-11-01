"""Command-line entry point for training the Titanic classifier."""

import argparse
from pathlib import Path

import torch

from src.training.pipeline import run_training_pipeline
from src.utils.config_utils import load_model_config


DEFAULT_DATA_PATH = Path("data/raw/train.csv")
DEFAULT_MODEL_CONFIG_PATH = Path("config/model.yaml")
DEFAULT_PREPROCESSING_CONFIG_PATH = Path("config/preprocessing.yaml")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(description="Train and evaluate the Titanic survival classifier.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--preprocessing-config-path", type=Path, default=DEFAULT_PREPROCESSING_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--save-processed-data", action="store_true", help="Save transformed train and validation CSV files for inspection.")
    parser.add_argument( "--device", choices=("auto", "cpu", "cuda"), default="auto", help="Torch device; auto selects CUDA when available.")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed from the YAML configuration.")
    return parser.parse_args()


def main() -> None:
    """Load configuration and run the configured training workflow."""
    args = parse_args()
    config = load_model_config(args.config_path)

    if args.seed is not None:
        config.training.seed = args.seed

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)

    run_training_pipeline(
        data_path=args.data_path,
        config=config,
        preprocessing_config_path=args.preprocessing_config_path,
        artifact_dir=args.artifact_dir,
        processed_dir=args.processed_dir,
        save_processed_data=args.save_processed_data,
        device=device,
    )


if __name__ == "__main__":
    main()
