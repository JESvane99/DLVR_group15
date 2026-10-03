import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn, seed
from torch.utils.data import DataLoader


def set_seed(seed):
    """Makes weight initialisation and batch order reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: Callable):
    model.eval()
    total_loss, correct = 0.0, 0
    for xb, yb in dataloader:
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader)


def run_experiment(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: Callable,
    run_config: dict,
    verbose: bool = True,
):
    """Trains one model from scratch and logs the result. Returns (model, history)."""

    set_seed(run_config["seed"])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=run_config["lr"], momentum=run_config["optimizer_momentum"]
    )

    history = {"iter_loss": [], "train_loss": [], "val_loss": [], "val_acc": []}
    diverged = False
    start = time.time()
    for epoch in range(run_config["epochs"]):
        model.train()
        loss, running, seen, diverged = train_step(
            history, train_loader, model, loss_function, optimizer
        )
        if diverged:
            if verbose:
                print(
                    f"  [{name} | seed {seed}] diverged (loss = {loss.item()}) in epoch {epoch + 1}"
                )
            break
        val_loss, val_acc = evaluate(model, val_loader, loss_function)
        history["train_loss"].append(running / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    runtime = time.time() - start

    result = dict(
        run_config,
        diverged=diverged,
        train_loss=history["train_loss"][-1] if history["train_loss"] else float("nan"),
        val_loss=history["val_loss"][-1] if history["val_loss"] else float("nan"),
        val_acc=history["val_acc"][-1] if history["val_acc"] else float("nan"),
        runtime_s=round(runtime, 1),
    )
    write_log(result)

    print(
        f"  [{name} | seed {seed}] train loss {result['train_loss']:.3f} | "
        f"val loss {result['val_loss']:.3f} | val acc {result['val_acc']:.3f} | {runtime:.0f}s"
    )
    return model, history


def train_step(
    history: dict,
    train_loader: DataLoader,
    model: nn.Module,
    loss_function: Callable,
    optimizer: torch.optim.Optimizer,
) -> tuple[Tensor, float, int, bool]:
    running, seen = 0.0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_function(model(xb), yb)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            diverged = True
            break
        history["iter_loss"].append(loss.item())
        running += loss.item() * len(xb)
        seen += len(xb)
    return loss, running, seen, diverged


def get_log():
    out_path = Path.cwd() / "out"
    out_path.mkdir(exist_ok=True)
    return out_path / "experiment_log.csv"


def write_log(result: dict) -> None:
    experiment_log = get_log()
    pd.DataFrame(result).to_csv(experiment_log, index=False, mode="a")
