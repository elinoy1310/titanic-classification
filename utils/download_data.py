
from pathlib import Path
import subprocess

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [
        "kaggle",
        "competitions",
        "download",
        "-c", "titanic",
        "-f", "train.csv",
        "-p", str(DATA_DIR),
    ],
    check=True,
)

print(f"Download completed. Files are in: {DATA_DIR}")
