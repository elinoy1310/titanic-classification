# import json
# from pathlib import Path
# from typing import Any
# import pandas as pd


# def _read_json(path: Path) -> dict[str, Any]:
#     with path.open("r", encoding="utf-8") as file:
#         value = json.load(file)
#     if not isinstance(value, dict):
#         raise ValueError(f"Expected a JSON object in {path.name}")
#     return value


# def load_artifacts(artifact_dir: Path) -> dict[str, Any]:
#     if not artifact_dir.exists() or not artifact_dir.is_dir():
#         raise FileNotFoundError(f"Artifacts directory not found: {artifact_dir}")

#     metrics_path = artifact_dir / "validation_metrics.json"
#     history_path = artifact_dir / "training_history.csv"
#     predictions_path = artifact_dir / "validation_predictions.csv"
#     importance_path = artifact_dir / "feature_importance.csv"

#     missing = [p.name for p in (metrics_path, history_path, predictions_path) if not p.is_file()]
#     if missing:
#         raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing))

#     metrics = _read_json(metrics_path)
#     history = pd.read_csv(history_path)
#     predictions = pd.read_csv(predictions_path)
#     importance = pd.read_csv(importance_path) if importance_path.is_file() else None

#     return {
#         "metrics": metrics,
#         "frames": {
#             "training_history.csv": history,
#             "validation_predictions.csv": predictions,
#             "feature_importance.csv": importance,
#         },
#         "paths": {
#             "metrics": metrics_path,
#             "history": history_path,
#             "predictions": predictions_path,
#             "importance": importance_path if importance_path.is_file() else None,
#         },
#     }



import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")

    return value


def _read_csv_if_exists(path: Path):
    """Read a CSV file if it exists."""

    if not path.is_file():
        return None

    return pd.read_csv(path)


def load_artifacts(artifact_dir: Path) -> dict[str, Any]:
    """Load training, validation and feature importance artifacts."""

    if not artifact_dir.exists() or not artifact_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {artifact_dir}")

    metrics_path = artifact_dir / "validation_metrics.json"
    history_path = artifact_dir / "training_history.csv"
    predictions_path = artifact_dir / "validation_predictions.csv"

    required = [
        metrics_path,
        history_path,
        predictions_path,
    ]

    missing = [ path.name for path in required if not path.is_file() ]

    if missing:
        raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing))

    metrics = _read_json(metrics_path)

    frames = {
        "training_history.csv": _read_csv_if_exists(history_path),
        "validation_predictions.csv": _read_csv_if_exists(predictions_path),
        "feature_importance.csv": _read_csv_if_exists(artifact_dir / "feature_importance.csv"),
        "cross_validation_folds.csv": _read_csv_if_exists(artifact_dir / "cross_validation_folds.csv"),
        # "cross_validation_summary.csv": _read_csv_if_exists(artifact_dir / "cross_validation_summary.csv"),
    }

    paths = {
        name: (
            artifact_dir / name
            if (artifact_dir / name).is_file()
            else None
        )
        for name in frames
    }

    paths["metrics"] = metrics_path

    return {
        "metrics": metrics,
        "frames": frames,
        "paths": paths,
    }

