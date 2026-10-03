from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split


class SimpleCrackDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        mode: str = "train",  # "train" or "test"
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        super().__init__()
        data_root = Path(data_root) if isinstance(data_root, str) else data_root
        if mode == "train":
            self.img_dir = data_root / "train_img"
            self.lab_dir = data_root / "train_lab"
        elif mode == "test":
            self.img_dir = data_root / "test_img"
            self.lab_dir = data_root / "test_lab"
        else:
            raise ValueError("mode must be 'train' or 'test'")

        # List .jpg images
        self.image_paths = sorted(self.img_dir.glob("*.jpg"))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .jpg images found in {self.img_dir}")

        # Pair by stem -> mask is stem + .png in lab_dir
        self.pairs = []
        missing_masks = []
        for p in self.image_paths:
            mask_path = self.lab_dir / (p.stem + ".png")
            if mask_path.exists():
                self.pairs.append((p, mask_path))
            else:
                missing_masks.append(p)
        if missing_masks:
            print(
                f"Warning: {len(missing_masks)} images had no matching .png mask and were skipped."
            )
            for p in missing_masks:
                print(f"  Missing mask for image: {p}")
        if len(self.pairs) == 0:
            raise RuntimeError("No image-mask pairs found.")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        # Load
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Mask to binary tensor (0/1), shape (1,H,W)
            mask_arr = np.array(mask)
            mask_bin = (mask_arr > 128).astype(np.float32)  # threshold at 128
            mask = torch.from_numpy(mask_bin).unsqueeze(0)

        return img, mask

    def get_train_val_split(self, val_ratio=0.2, seed=42):
        """Return subsets for train and validation"""
        generator = torch.Generator().manual_seed(seed)  # for reproducibility
        return random_split(self, [1 - val_ratio, val_ratio], generator=generator)
