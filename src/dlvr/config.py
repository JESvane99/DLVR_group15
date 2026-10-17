import functools
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger


@dataclass
class WandBConfig:
    name: str = "baseline_unet"
    project: str = "dlvr"
    entity: str = "group15"
    job_type: str = "train"
    log_model: str = "all"
    save_code: bool = True


@dataclass
class DatasetConfig:
    dataset_path: Path = Path("data/deepcrack_dataset")
    image_size: tuple[int, int] = (384, 544)
    validation_train_split_ratio: float = 0.2
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train_data: bool = True


@dataclass
class ModelConfig:
    in_channels: int = 3
    n_classes: int = 1
    unet_depth: int = 5
    backbone: str = "resnet34"  # also refered to as encoder_name
    encoder_weights: str = "imagenet"  # place weights come from
    decoder_use_normalization: str = "batchnorm"  # see segmentation_models_pytorch for options
    decoder_interpolation: str = "nearest"  # see segmentation_models_pytorch for options


@dataclass
class EarlyStoppingConfig:
    patience: int = 5
    mode: str = "max"  # 'min' or 'max'


@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 0.001
    momentum: float = 0.0
    weight_decay: float = 0.0
    epochs: int = 15


@dataclass
class PlotConfig:
    n_samples: int = 4
    stats_batches: int = 5
    visualise_samples_amount: int = 2
    test_visualisation_samples: int = 4
    binary_image_threshold: float = 1.0


@dataclass
class LossFunctionConfig:
    binary_threshold: float = 0.5  # threshold for binary classification
    epsilon: float = 1e-7  # small value to avoid division by zero in metrics calculations


class ConfigBuilder:
    def __init__(self):
        self._wandb = None
        self._dataset = None
        self._model = None
        self._early_stopping = None
        self._lossfn = None
        self._plot = None
        self._optimizer = None

    def wandb(
        self,
        name: str = "baseline_unet",
        project: str = "dlvr",
        entity: str = "group15",
        job_type: str = "train",
        log_model: str = "all",
        save_code: bool = True,
    ):
        self._wandb = WandBConfig(
            name=name,
            project=project,
            entity=entity,
            job_type=job_type,
            log_model=log_model,
            save_code=save_code,
        )
        return self._wandb

    def dataset(
        self,
        dataset_path: Path = Path("data/deepcrack_dataset"),
        image_size: tuple[int, int] = (384, 544),
        validation_train_split_ratio: float = 0.2,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self._dataset = DatasetConfig(
            dataset_path=dataset_path,
            image_size=image_size,
            validation_train_split_ratio=validation_train_split_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return self._dataset

    def model(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        unet_depth: int = 5,
        backbone: str = "resnet34",
        encoder_weights: str = "imagenet",
        decoder_use_normalization: str = "batchnorm",
        decoder_interpolation: str = "nearest",
    ):
        self._model = ModelConfig(
            in_channels=in_channels,
            n_classes=n_classes,
            unet_depth=unet_depth,
            backbone=backbone,
            encoder_weights=encoder_weights,
            decoder_use_normalization=decoder_use_normalization,
            decoder_interpolation=decoder_interpolation,
        )
        return self._model

    def early_stopping(self, patience: int = 5, mode: str = "max"):
        self._early_stopping = EarlyStoppingConfig(patience=patience, mode=mode)
        return self._early_stopping

    def optimizer(
        self,
        name: str = "adam",
        learning_rate: float = 0.001,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epochs: int = 15,
    ):
        self._optimizer = OptimizerConfig(
            name=name,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            epochs=epochs,
        )
        return self._optimizer

    def loss_function(
        self,
        binary_threshold: float = 0.5,
        epsilon: float = 1e-7,
    ):
        self._lossfn = LossFunctionConfig(
            binary_threshold=binary_threshold,
            epsilon=epsilon,
        )
        return self._lossfn

    def plot(
        self,
        n_samples: int = 4,
        stats_batches: int = 5,
        visualise_samples_amount: int = 2,
        test_visualisation_samples: int = 4,
        binary_image_threshold: float = 1.0,
    ):
        self._plot = PlotConfig(
            n_samples=n_samples,
            stats_batches=stats_batches,
            visualise_samples_amount=visualise_samples_amount,
            test_visualisation_samples=test_visualisation_samples,
            binary_image_threshold=binary_image_threshold,
        )
        return self._plot


def get_optimizer_class(optimizer_name: str):
    optimizer_classes = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    try:
        return optimizer_classes[optimizer_name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}") from error


def get_loss_function(loss_name: str):
    loss_functions = {
        "bce": torch.nn.BCELoss,
        "bcewithlogits": torch.nn.BCEWithLogitsLoss,
    }
    try:
        return loss_functions[loss_name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unsupported loss function: {loss_name}") from error


def logger_wraps(*, entry=True, exit=True, level="DEBUG"):

    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry:
                logger_.log(
                    level,
                    "Entering '{}' with {} positional and {} keyword arguments",
                    name,
                    len(args),
                    len(kwargs),
                )
            result = func(*args, **kwargs)
            if exit:
                logger_.log(level, "Exiting '{}' (result={})", name, result)
            return result

        return wrapped

    return wrapper
