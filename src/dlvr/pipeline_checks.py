import math

import matplotlib.pyplot as plt
import torch


def run_all_pipeline_checks(
    model, x_tr, y_tr, x_va, tr_idx, va_idx, train_step, loss_fn
):
    """
    Run all pipeline checks.
    """
    check_data(x_tr, y_tr, x_va)
    check_split(tr_idx, va_idx)
    check_initial_loss(model, x_tr, y_tr, loss_fn, num_classes=len(torch.unique(y_tr)))
    overfit_one_batch(model, x_tr, y_tr, train_step)


def check_data(x_tr, y_tr, x_va, mean=None, std=None, n_show=8):
    """Check A: per-channel statistics of both splits + a look at some training images."""
    for split, x in [("train", x_tr), ("val", x_va)]:
        m = x.mean(dim=(0, 2, 3)).tolist()
        s = x.std(dim=(0, 2, 3)).tolist()
        print(
            f"{split:>5}: mean per channel = {[round(v, 3) for v in m]}, "
            f"std per channel = {[round(v, 3) for v in s]}, "
            f"min = {x.min().item():.2f}, max = {x.max().item():.2f}"
        )
    imgs = x_tr[:n_show].detach().cpu()
    if mean is not None:  # undo normalisation for display
        imgs = imgs * std + mean
    imgs = (imgs - imgs.amin(dim=(1, 2, 3), keepdim=True)) / (
        imgs.amax(dim=(1, 2, 3), keepdim=True)
        - imgs.amin(dim=(1, 2, 3), keepdim=True)
        + 1e-8
    )
    _, axes = plt.subplots(1, n_show, figsize=(1.6 * n_show, 2))
    for ax, img, label in zip(axes, imgs, y_tr[:n_show].tolist()):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(str(label))
        ax.axis("off")
    plt.suptitle("Training images with their labels (display rescaled per image)")
    plt.tight_layout()
    plt.show()


def check_split(tr_idx, va_idx):
    """Check B: are any examples in both splits?"""
    overlap = set(tr_idx.tolist()) & set(va_idx.tolist())
    print(
        f"Train size {len(tr_idx)}, val size {len(va_idx)}, overlap: {len(overlap)} examples"
    )
    if overlap:
        print(
            f"WARNING: {len(overlap) / len(va_idx):.0%} of the validation set is also in the training set!"
        )


@torch.no_grad()
def check_initial_loss(model, x, y, loss_fn, num_classes=10):
    """Check C: loss of the untrained model vs. the expected ln(C)."""
    model.eval()
    loss = loss_fn(model(x[:512]), y[:512]).item()
    print(
        f"Initial loss: {loss:.3f}   (expected ≈ ln({num_classes}) = {math.log(num_classes):.3f})"
    )


def overfit_one_batch(model, x, y, train_step, steps=300, lr=0.01, batch_size=32):
    """Check D: can the model memorise a single small batch?"""
    xb, yb = x[:batch_size], y[:batch_size]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    losses, accs = [], []
    for _ in range(steps):
        loss, acc = train_step(model, optimizer, xb, yb)
        losses.append(loss)
        accs.append(acc)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2))
    ax1.plot(losses)
    ax1.set_yscale("log")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Training loss (log scale)")
    ax1.set_title("Loss on one batch")
    ax1.grid(alpha=0.3)
    ax2.plot(accs)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy on the same batch")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Final loss: {losses[-1]:.4f}, final accuracy: {accs[-1]:.2f}")
