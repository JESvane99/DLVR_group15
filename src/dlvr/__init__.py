from dlvr.utils import (
    check_train_val_duplicates,
    extract_deepcrack_data,
    find_deepcrack_data,
    move_duplicates_to_review_dir,
)


def main() -> None:
    print("Hello from dlvr!")


def prepare_data(data: dict | None = None) -> None:
    """Prepare the DeepCrack dataset by extracting it if necessary."""
    try:
        find_deepcrack_data()
    except FileNotFoundError:
        print(
            "DeepCrack dataset zip file not found. Please download it and place it in the 'data' directory."
        )
        return
    if not (find_deepcrack_data().with_suffix("").exists()):
        extract_deepcrack_data()
    print("DeepCrack dataset is ready.")
    if data is not None:
        check_train_val_duplicates(data.get("train_dataset"), data.get("val_dataset"))
