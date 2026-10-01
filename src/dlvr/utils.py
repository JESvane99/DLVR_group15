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


def find_train_val_duplicates(train_dataset, val_dataset, threshold=0.95, batch_size=32):
    """
    Find likely duplicate images across training and validation datasets.

    Why? A validation image that also appears in the training set makes
    validation accuracy misleading.

    How? A pretrained ResNet-18 extracts one feature vector per image. Pairs
    with cosine similarity at or above `threshold` are returned for manual
    review. A high threshold is intentionally used because similar is not
    necessarily identical: augmentation, compression, or resizing can change
    pixel values.

    Note that this is just one type of train/val data leakage — there are many
    other ways to leak information. For example, if the validation set contains
    images of the same objects (e.g., patients) as the training set, the model
    may perform well on validation but fail to generalize to new objects.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet18_Weights, resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def dataset_paths(dataset):
        return [path for path in dataset.image_paths]

    train_paths = dataset_paths(train_dataset)
    val_paths = dataset_paths(val_dataset)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    def extract_features(dataset):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        features = []
        with torch.inference_mode():
            for images, _ in loader:
                batch_features = model(images.to(device))
                features.append(F.normalize(batch_features, p=2, dim=1).cpu())
        return torch.cat(features)

    train_features = extract_features(train_dataset)
    val_features = extract_features(val_dataset)
    similarity = val_features @ train_features.T
    val_indices, train_indices = torch.where(similarity >= threshold)

    pairs = []
    for val_index, train_index in zip(val_indices.tolist(), train_indices.tolist()):
        pairs.append(
            {
                "train_path": train_paths[train_index],
                "val_path": val_paths[val_index],
                "similarity": float(similarity[val_index, train_index]),
            }
        )

    pairs.sort(key=lambda pair: pair["similarity"], reverse=True)
    print(f"Found {len(pairs)} possible train/validation duplicate pair(s).")
    for pair in pairs[:10]:
        print(f"{pair['similarity']:.3f}  train: {pair['train_path']}  val: {pair['val_path']}")
    return pairs


def move_duplicates_to_review_dir(duplicates):
    """Move duplicate images to a review directory for manual inspection."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    review_dir = data_dir / "train_val_duplicates"
    review_dir.mkdir(exist_ok=True)
    for i, pair in enumerate(duplicates):
        train_path = Path(pair["train_path"])
        val_path = Path(pair["val_path"])
        train_review_path = review_dir / f"train_{train_path.name}_{i}"
        val_review_path = review_dir / f"val_{val_path.name}_{i}"
        train_path.move(train_review_path)
        val_path.move(val_review_path)
    print(f"Moved {len(duplicates)} duplicate pairs to {review_dir.resolve()} for review.")


def check_train_val_duplicates(train_dataset, val_dataset, threshold=0.95, batch_size=32):
    """Check for likely duplicate images across training and validation datasets."""
    duplicates = find_train_val_duplicates(train_dataset, val_dataset, threshold, batch_size)
    if duplicates:
        move_duplicates_to_review_dir(duplicates)
        print(
            "Duplicate images found between training and validation datasets. "
            "They have been moved to the 'train_val_duplicates' directory for review."
        )


def check_tensor_stats(loader, n_batches=5):
    """
    Check the mean and std of pixel values after normalisation.
    After ImageNet normalisation, mean should be near 0 and std near 1.
    Large deviations suggest the wrong normalisation was applied.
    """
    all_means = []
    all_stds = []

    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        all_means.append(images.mean(dim=[0, 2, 3]).numpy())  # per channel
        all_stds.append(images.std(dim=[0, 2, 3]).numpy())

    mean = np.array(all_means).mean(axis=0)
    std = np.array(all_stds).mean(axis=0)

    print("After normalisation (should be near mean≈0, std≈1 per channel):")
    for c, name in enumerate(["R", "G", "B"]):
        print(f"  {name}: mean={mean[c]:.3f}, std={std[c]:.3f}")

    if abs(mean).max() > 0.5:
        print("WARNING: Mean far from 0 — check normalisation values.")
    if abs(std - 1).max() > 0.5:
        print("WARNING: Std far from 1 — check normalisation values.")
