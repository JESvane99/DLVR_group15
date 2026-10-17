import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from dlvr.model import IoU


def visualise_samples(dataset, n=4, mean=[0, 0, 0], std=[1, 1, 1]):
    """
    Show n examples with proper denormalization.
    """
    _fig, axes = plt.subplots(n, 2, figsize=(12, n * 5))
    for i in range(n):
        indx = np.random.randint(0, len(dataset))
        img_tensor, mask_tensor = dataset[indx]

        # Denormalize image: class-wise (img * std) + mean
        img = img_tensor.clone()
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]

        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
        mask = mask_tensor.permute(1, 2, 0).squeeze().numpy()

        ax_img = axes[i][0] if n > 1 else axes[0]
        ax_mask = axes[i][1] if n > 1 else axes[1]

        ax_img.imshow(img)
        ax_img.set_title("Denormalized Input Image")
        ax_img.axis("off")

        ax_mask.imshow(mask, cmap="gray")
        ax_mask.set_title("Ground Truth Mask")
        ax_mask.axis("off")

    plt.suptitle("Images and corresponding ground truth masks", fontsize=14)
    plt.tight_layout()
    plt.show()


def vizualize_test_predictions(
    model, test_loader, device, n_samples=4, mean=[0, 0, 0], std=[1, 1, 1]
):
    model.eval()
    images, masks = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    _fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    for i in range(n_samples):
        # Denormalize image
        img = images[i].clone()
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Denormalized Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i].squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i].squeeze(), cmap="gray")
        axes[i, 2].set_title("Model prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def check_data(x_tr: torch.Tensor, x_va: torch.Tensor) -> None:
    """Check A: per-channel statistics of both splits."""
    result = ["split", "mean per channel", "std per channel", "min, max"]
    lines = ["Data statistics:"]
    lines.append(f"Train size: {len(x_tr)}, Val size: {len(x_va)}")
    lines.append(f"Train shape: {x_tr.shape}, Val shape: {x_va.shape}")
    lines.append(f"{result[0]:<10} | {result[1]:<30} | {result[2]:<30} | {result[3]:<20}")
    lines.append("-" * 100)
    for split, x in [("train", x_tr), ("val", x_va)]:
        m = x.mean(dim=(0, 2, 3)).tolist()
        s = x.std(dim=(0, 2, 3)).tolist()
        result[0] = f"{split}"
        result[1] = f"{[round(v, 3) for v in m]}"
        result[2] = f"{[round(v, 3) for v in s]}"
        result[3] = f"{x.min().item():.2f}, {x.max().item():.2f}"
        lines.append(f"{result[0]:<10} | {result[1]:<30} | {result[2]:<30} | {result[3]:<20}")
    print("\n".join(lines))


def check_tensor_stats(loader: DataLoader, n_batches=5):
    """Check mean and std of pixel values after normalization."""
    all_means = []
    all_stds = []
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        all_means.append(images.mean(dim=[0, 2, 3]).numpy())
        all_stds.append(images.std(dim=[0, 2, 3]).numpy())

    mean = np.array(all_means).mean(axis=0)
    std = np.array(all_stds).mean(axis=0)
    print("After normalisation (should be near mean≈0, std≈1 per channel):")
    print("### Ask about meaning of this check when using pretrained weights ###")
    for c, name in enumerate(["R", "G", "B"]):
        print(f"  {name}: mean={mean[c]:.3f}, std={std[c]:.3f}")


def check_split(tr_idx: list[int], va_idx: list[int]) -> None:
    """Check B: check for overlap between splits."""
    overlap = set(tr_idx) & set(va_idx)
    print(f"Train size {len(tr_idx)}, val size {len(va_idx)}, overlap: {len(overlap)} examples")
    if overlap:
        print(f"WARNING: {len(overlap) / len(va_idx):.0%} of validation set overlaps with train!")


@torch.no_grad()
def check_initial_loss(
    model,
    x,
    y,
    loss_fn,
) -> None:
    """Check C: loss of untrained model."""
    model.eval()
    loss = loss_fn(model(x[:12]), y[:12]).item()
    # Using math.log(2) since we evaluate binary outcomes (probability ~0.5 for random initial weights)
    print(f"Initial loss: {loss:.3f}   (expected ≈ ln(2) = {math.log(2):.3f})")


def overfit_one_batch(model, images, masks, loss_fn, steps=300, lr=0.01, output_dir=Path("out")):
    """Check D: overfit a single small batch."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    losses, accs = [], []

    for step in tqdm(range(steps), desc="Overfitting Single Batch"):
        acc = 0.0
        total_loss = 0.0
        for xb, yb in zip(images, masks):
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            acc += IoU(logits, yb)
            total_loss += loss.item()

        losses.append(total_loss / len(images))
        accs.append(acc / len(images))

    print(f"Overfit one batch: final loss {losses[-1]:.4f}, final accuracy {accs[-1]:.2f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2))
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
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pipeline_check_overfit.png", bbox_inches="tight")
    plt.show()


def run_checks(train_loader, val_loader, loss_function, model, mean, std):
    # Move input tensors to the same device as the model
    images, masks = next(iter(train_loader))
    device_images = images.to(model.device)
    device_masks = masks.to(model.device)

    # data checks
    check_tensor_stats(train_loader)
    visualise_samples(train_loader.dataset, n=2, mean=mean, std=std)
    check_data(next(iter(train_loader))[0], next(iter(val_loader))[0])

    # model checks
    # train and val loader datasets are subset which hold indices of the original dataset
    check_split(train_loader.dataset.indices, val_loader.dataset.indices)

    check_initial_loss(model, device_images, device_masks, loss_function)
    # Ensure we pass 4D batch slices (e.g., [1, C, H, W]) instead of 3D squeezed tensors
    batch_images_list = [device_images[i : i + 1] for i in range(device_images.size(0))]
    batch_masks_list = [device_masks[i : i + 1] for i in range(device_masks.size(0))]

    overfit_one_batch(copy.deepcopy(model), batch_images_list, batch_masks_list, loss_function)
    # Get one batch from train and val loaders
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    # Check data statistics
    check_data(train_batch[0], val_batch[0])
