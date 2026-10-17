import random
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dlvr.config import (
    DatasetConfig,
    EarlyStoppingConfig,
    LossFunctionConfig,
    ModelConfig,
    OptimizerConfig,
    PlotConfig,
    WandBConfig,
    get_optimizer_class,
    logger_wraps,
)
from dlvr.model import (
    IoU,
    choose_optimizer,
    get_pretrained_unet,
)


@logger_wraps()
def set_seed(seed):
    """Makes weight initialisation and batch order reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@logger_wraps()
def run_experiment_old(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: Callable,
    run_config: RunConfig,
) -> tuple[nn.Module, dict]:
    """Trains one model from scratch and logs the result. Returns (model, history)."""
    output_dir = run_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(run_config.seed)

    # define optimizer
    optimizer_class = get_optimizer_class(run_config.optimizer.lower())
    optimizer_kwargs = run_config.get_optimizer_kwargs()
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    history = {
        "iter_loss": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "avg_train_ious": [],
        "val_ious": [],
    }
    best_val_iou = 0.0
    start = time.time()
    for epoch in range(run_config.epochs):
        model.train()

        loss, running, seen, diverged, train_iou = train_step(
            history, train_loader, model, loss_function, optimizer, run_config
        )
        if diverged:
            if run_config.verbose:
                print(
                    f"  [{run_config.name} | seed {run_config.seed}] diverged "
                    f"(loss = {loss.item()}) in epoch {epoch + 1}"
                )
            break
        val_loss, val_acc, val_iou = evaluate(model, val_loader, loss_function, run_config)
        history["train_loss"].append(running / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ious"].append(val_iou)
        history["avg_train_ious"].append(train_iou / len(train_loader))
    runtime = time.time() - start

    result = dict(
        asdict(run_config),
        diverged=diverged,
        train_loss=history["train_loss"][-1],
        val_loss=history["val_loss"][-1],
        val_acc=history["val_acc"][-1],
        val_ious=history["val_ious"][-1],
        avg_train_ious=history["avg_train_ious"][-1],
        runtime_s=round(runtime, 1),
    )
    write_log(result, run_config.output_dir)

    print(
        f"  [{run_config.name} | seed {run_config.seed}] train loss {result['train_loss']:.3f} | "
        f"val loss {result['val_loss']:.3f} | val acc {result['val_acc']:.3f} | {runtime:.0f}s"
    )

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        checkpoint_path = run_config.checkpoint_path.absolute()
        torch.save(model.state_dict(), checkpoint_path)
        print(f"--> Saved new best model with Val IoU : {best_val_iou:.4f}")

    return model, history


@logger_wraps()
def train_step(
    history: dict,
    train_loader: DataLoader,
    model: nn.Module,
    loss_function: Callable,
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
) -> tuple[Tensor, float, int, bool, float]:
    diverged = False
    running, seen, train_iou = 0.0, 0, 0.0
    loss = torch.tensor(float("nan"))
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_function(logits, yb)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            diverged = True
            break
        history["iter_loss"].append(loss.item())
        running += loss.item() * len(xb)
        seen += len(xb)
        train_iou += IoU(logits, yb, threshold=config.metric_threshold)

    return loss, running, seen, diverged, train_iou


def get_log(output_dir: Path = Path("out")):
    out_path = output_dir.absolute()
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path / "experiment_log.csv"


@logger_wraps()
def write_log(result: dict, output_dir: Path = Path("out")) -> None:
    experiment_log = get_log(output_dir)
    pd.DataFrame([result]).to_csv(experiment_log, index=False, mode="a")


def log_wandb(log_dict: dict, run: wandb.Run | None):
    """Log metrics to Weights & Biases. handles the case when run is None (i.e., W&B logging is disabled)."""
    if run is None:
        return

    run.log(log_dict)


def init_wandb(cfg: WandBConfig):
    """Initialize Weights & Biases session using the config dict."""
    return wandb.init(
        project="crack-segmentation-unet",
        name=f"lr:{cfg.learning_rate}-e:{cfg.epochs}-batch_size:{cfg.batch_size}-seed:{cfg.seed}",
        config=asdict(cfg),
    )


def run_experiment(
    wandbcfg: WandBConfig,
    datasetcfg: DatasetConfig,
    modelcfg: ModelConfig,
    earlystoppingcfg: EarlyStoppingConfig,
    optimizercfg: OptimizerConfig,
    lossfncfg: LossFunctionConfig,
    plotcfg: PlotConfig,
    wandb_log: bool = False,
    seeds: tuple[int, ...] = (42, 67, 69, 420),
):
    """Run a single experiment with the given configuration."""
    # 1. Initialize Weights & Biases session using the config dict
    for seed in seeds:
        # Set manual seed for reproducibility
        set_seed(seed)

        # Set device
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(DEVICE)

        # Defining model dynamically based on configuration
        model = get_pretrained_unet(modelcfg)

        # Moderated learning rate populated dynamically from config
        optimizer = choose_optimizer(model, optimizercfg)

        # Calling loss function
        loss_function = nn.BCEWithLogitsLoss()  # only one strategy as of now.

        # Initialize Weights & Biases session
        run = init_wandb(wandbcfg) if wandb_log else None

        if run is not None:
            run.finish()  # Ensure the W&B run is properly closed after the experiment
