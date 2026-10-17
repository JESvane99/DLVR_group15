from dlvr.utils.setup import set_seed
from segmentation_models_pytorch.encoders import get_preprocessing_params
from torch import Generator
from torch.utils.data import random_split

from dlvr.simple_crack_dataset import SimpleCrackDataset


def test_preprocessing_params():
    preprocess_params = get_preprocessing_params("resnet18", pretrained="imagenet")
    assert isinstance(preprocess_params, dict)
    assert "mean" in preprocess_params
    assert "std" in preprocess_params


# test to see if set_seed works


def test_set_seed():
    set_seed(42)

    train_dataset = SimpleCrackDataset(
        root="data/deepcrack_dataset",
        transform=None,
        target_transform=None,
        mode="train",
    )

    G1 = Generator().manual_seed(42)

    train1, val1 = random_split(train_dataset, [0.8, 0.2], generator=G1)
    train2, val2 = random_split(train_dataset, [0.8, 0.2])

    assert train1.indices == train2.indices
    assert val1.indices == val2.indices
