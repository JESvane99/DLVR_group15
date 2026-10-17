from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SimpleCrackDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose,
        target_transform: transforms.Compose,
        mode: str = "train",  # "train" or "test"
    ):
        root = Path(root) if isinstance(root, str) else root
        self.transform = transform
        self.target_transform = target_transform
        if mode == "train":
            self.img_dir = root / "train_img"
            self.lab_dir = root / "train_lab"
        elif mode == "test":
            self.img_dir = root / "test_img"
            self.lab_dir = root / "test_lab"
        else:
            raise ValueError("mode must be 'train' or 'test'")

        if not self.img_dir.exists() or not self.lab_dir.exists():
            raise FileNotFoundError(
                f"Check dataset folders: {self.img_dir} or {self.lab_dir} not found."
            )

        # List .jpg images
        self.image_paths = sorted(self.img_dir.glob("*.jpg"))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .jpg images found in {self.img_dir}")

        # Pair by stem -> mask is stem + .png in lab_dir
        self.pairs = []
        missing_masks = 0
        for p in self.image_paths:
            mask_path = self.lab_dir / (p.stem + ".png")
            if mask_path.exists():
                self.pairs.append((p, mask_path))
            else:
                missing_masks += 1
        if missing_masks:
            print(f"Warning: {missing_masks} images had no matching .png mask and were skipped.")
        if len(self.pairs) == 0:
            raise RuntimeError("No image-mask pairs found.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        # Load
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        img = self.transform(img)

        mask = self.target_transform(mask)

        return img, mask
