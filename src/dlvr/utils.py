from pathlib import Path
from zipfile import ZipFile


def find_deepcrack_data() -> Path:
    """Find the DeepCrack dataset zip file in the current directory."""
    current_dir = Path(__file__).parent.parent.parent
    for file in current_dir.glob("data/*"):
        if file.suffix == ".zip" and "deepcrack" in file.stem.lower():
            return file
    raise FileNotFoundError(f"DeepCrack dataset zip file not found. looking in {current_dir}")


def extract_deepcrack_data() -> None:
    """Extract the DeepCrack dataset from the zip file."""
    zip_file = find_deepcrack_data()
    with ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(
            zip_file.with_suffix("")
        )  # Extract to the same directory as the zip file
