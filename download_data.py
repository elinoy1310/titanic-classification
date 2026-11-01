"""Download the Titanic competition training CSV using the Kaggle CLI."""

import argparse
import subprocess
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_COMPETITION = "titanic"
DEFAULT_FILE_NAME = "train.csv"
DEFAULT_KAGGLE_EXECUTABLE = "kaggle"


def parse_args() -> argparse.Namespace:
    """Parse Kaggle download options."""
    parser = argparse.ArgumentParser(
        description="Download a file from a Kaggle competition."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--file-name", default=DEFAULT_FILE_NAME)
    parser.add_argument(
        "--kaggle-executable",
        default=DEFAULT_KAGGLE_EXECUTABLE,
        help="Kaggle CLI executable name or path.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded ZIP archive after extracting the requested file.",
    )
    return parser.parse_args()


def download_data(
    output_dir: Path,
    competition: str,
    file_name: str,
    kaggle_executable: str = DEFAULT_KAGGLE_EXECUTABLE,
    keep_archive: bool = False,
) -> Path:
    """Download and, if necessary, extract the requested competition file."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        kaggle_executable,
        "competitions",
        "download",
        "-c",
        competition,
        "-f",
        file_name,
        "-p",
        str(output_dir),
    ]
    subprocess.run(command, check=True)

    downloaded_file = output_dir / file_name
    archive_path = output_dir / f"{file_name}.zip"

    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            archive_members = archive.namelist()
            matching_member = next(
                (member for member in archive_members
                 if Path(member).name == Path(file_name).name),
                None,
            )
            if matching_member is None:
                raise FileNotFoundError(
                    f"{file_name!r} was not found inside {archive_path}."
                )
            extracted = archive.read(matching_member)
            downloaded_file.write_bytes(extracted)

        if not keep_archive:
            archive_path.unlink()

    if not downloaded_file.is_file():
        raise FileNotFoundError(
            f"Kaggle download completed, but {downloaded_file} was not created."
        )

    print(f"Download completed: {downloaded_file}")
    return downloaded_file


def main() -> None:
    """Run the Kaggle download command."""
    args = parse_args()
    download_data(
        output_dir=args.output_dir,
        competition=args.competition,
        file_name=args.file_name,
        kaggle_executable=args.kaggle_executable,
        keep_archive=args.keep_archive,
    )


if __name__ == "__main__":
    main()
