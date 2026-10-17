import copy

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_params
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dlvr.config import (
    DatasetConfig,
    EarlyStoppingConfig,
    LossFunctionConfig,
    ModelConfig,
    OptimizerConfig,
)
from dlvr.simple_crack_dataset import SimpleCrackDataset


def choose_optimizer(model: torch.nn.Module, cfg: OptimizerConfig) -> optim.Optimizer:
    """Choose optimizer based on configuration."""
    match cfg.name.lower():
        case "adam":
            return optim.Adam(model.parameters(), lr=cfg.learning_rate)
        case "sgd":
            return optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
        case _:
            raise ValueError(f"Unsupported optimizer: {cfg.name}")


def get_transforms(cfg: ModelConfig) -> tuple[transforms.Compose, transforms.Compose, list, list]:
    """Define transformations for images and masks.

    returns: data_transform, target_transform, mean, std"""
    preprocess_params = get_preprocessing_params(
        encoder_name=cfg.backbone, pretrained=cfg.encoder_weights
    )

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=preprocess_params["mean"], std=preprocess_params["std"]),
        ]
    )

    target_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return (data_transform, target_transform, preprocess_params["mean"], preprocess_params["std"])


def get_train_val_dataloaders(
    data_cfg: DatasetConfig, model_cfg: ModelConfig
) -> tuple[DataLoader, DataLoader, list, list]:
    """build train, validation, and test dataloaders."""
    data_transform, target_transform, mean, std = get_transforms(model_cfg)

    full_train_dataset = SimpleCrackDataset(
        data_cfg.dataset_path,
        mode="train",
        transform=data_transform,
        target_transform=target_transform,
    )
    val_ratio = data_cfg.validation_train_split_ratio
    train_ratio = 1 - val_ratio
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_ratio, val_ratio], generator=torch.Generator()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=data_cfg.shuffle_train_data,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )

    return train_loader, val_loader, mean, std


def get_test_dataloader(data_cfg: DatasetConfig, model_cfg: ModelConfig):
    """build test dataloader."""
    data_transform, target_transform, _, _ = get_transforms(model_cfg)

    test_dataset = SimpleCrackDataset(
        data_cfg.dataset_path,
        mode="test",
        transform=data_transform,
        target_transform=target_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )

    return test_loader


class EarlyStopping:
    def __init__(self, cfg: EarlyStoppingConfig):
        assert cfg.mode in ("max", "min")
        self.patience = cfg.patience
        self.mode = cfg.mode
        self.best = None  # best monitored value so far
        self.best_epoch = None  # epoch (0-based) at which it occurred
        self.best_state = None  # deep copy of the model's state_dict at that epoch
        self.bad_epochs = 0  # consecutive epochs without improvement

    def step(self, value, model, epoch):
        """Call once per epoch. Returns True if training should stop.
        value: the value of the monitored metric (e.g., validation IoU)
        model: the model being trained (to save its state_dict)
        epoch: the current epoch number (0-based)
        """
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best
        else:
            improved = value < self.best

        if improved:
            self.bad_epochs = 0
            self.best = value
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())

        else:
            self.bad_epochs += 1
        # True if training runs out of patience
        return self.patience <= self.bad_epochs


# 2. Define helper to download the pretrained U-Net
def get_pretrained_unet(cfg: ModelConfig) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name=cfg.backbone,  # Use a stable ResNet34 backbone
        encoder_weights=cfg.encoder_weights,  # Download weights pretrained on ImageNet
        in_channels=cfg.in_channels,  # RGB input channels
        classes=1,  # Binary segmentation output
        decoder_use_norm=cfg.decoder_use_normalization,  # Use config normalization in the decoder
        decoder_interpolation=cfg.decoder_interpolation,  # Use config interpolation in the decoder
    )
    return model


def precision_recall(logits, targets, cfg: LossFunctionConfig):
    """Compute precision and recall for binary segmentation tasks.
    threshold determines the binary conversion of the predicted probabilities.
    eps is a small value to avoid division by zero.
    """
    threshold = cfg.binary_threshold
    eps = cfg.epsilon
    preds = (torch.sigmoid(logits) > threshold).float().view(-1)
    targets = targets.view(-1)
    true_positive_count = (preds * targets).sum()
    false_positive_count = (preds * (1 - targets)).sum()
    false_negative_count = ((1 - preds) * targets).sum()
    precision = (true_positive_count + eps) / (true_positive_count + false_positive_count + eps)
    recall = (true_positive_count + eps) / (true_positive_count + false_negative_count + eps)
    return precision.item(), recall.item()


def IoU(
    logits,
    targets,
    cfg: LossFunctionConfig,
):
    """Compute Intersection over Union (IoU) for binary segmentation tasks.
    threshold determines the binary conversion of the predicted probabilities.
    """
    threshold = cfg.binary_threshold

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()
