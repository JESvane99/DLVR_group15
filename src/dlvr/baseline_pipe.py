# %% [markdown]


# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split  # Added train_test_split
from torch import optim
from torch.utils.data import DataLoader, Subset  # Added Subset
from torchvision import transforms

from dlvr.loss_functions import BCELoss
from dlvr.metrics import IoU
from dlvr.pipeline_checks import run_all_pipeline_checks
from dlvr.simple_crack_dataset import SimpleCrackDataset
from dlvr.unet_model import UNet
from dlvr.utils import check_tensor_stats
from dlvr.visualisation_utils import visualise_samples

# %%
## loading dataset path
cwd = Path.cwd().parents[1].absolute()
print(cwd)
deepcrack_dataset_path = [ds for ds in cwd.glob("data/deepcrack*/**")][1]
print(deepcrack_dataset_path)

# %% [markdown]
# ## Pytorch dataset object
#
# Define dataset object for correct handling of test and train data

# %%


# %%
# Define the target path in Google Drive for persistent storage
# The dataset is expected to be here, making it accessible across sessions and shareable.

if deepcrack_dataset_path.exists():
    print(*deepcrack_dataset_path.glob("*"), sep="\n")

# %% [markdown]
# ### Define dataset transformations

# %%
data_transform = transforms.Compose(
    [
        transforms.Resize((384, 544)),  # Resize to a consistent size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # \\TODO: Check if this normalization is appropriate for the dataset
    ]
)

target_transform = transforms.Compose(
    [
        transforms.Resize((384, 544)),  # Resize masks to a consistent size
        transforms.ToTensor(),
    ]
)

# %% [markdown]
# ### Load training data

# %%
full_train_dataset = SimpleCrackDataset(
    deepcrack_dataset_path,
    mode="train",
    transform=data_transform,
    target_transform=target_transform,
)


# %% [markdown]
# ## Load Test data

# %%
test_dataset = SimpleCrackDataset(
    deepcrack_dataset_path,
    mode="test",
    transform=data_transform,
    target_transform=target_transform,
)


# %% [markdown]
# ## Spliting Train for Train and Validation dataset

# %%
train_dataset, val_dataset = train_test_split(full_train_dataset, test_size=0.2, random_state=42)


train_indices, val_indices = train_test_split(
    list(range(len(full_train_dataset))), test_size=0.20, random_state=42
)

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

# %%
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# %% [markdown]
# ## check image stats

# %%


# Check one batch
images, masks = next(iter(train_loader))
print(f"Batch image shape: {images.shape}")  # (32, 3, 224, 224)
print(f"Batch masks shape: {masks.shape}")  # (32,)

check_tensor_stats(train_loader)

# %% [markdown]
# ### Visualize random selection of the data

# %%
visualise_samples(dataset=train_dataset, n=2)

# %% [markdown]
# ### Defining
#
# model, loss function, and optimizer
# %%

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining model
model = UNet(in_channels=3, n_class=1).to(device)

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Calling loss function
loss_function = BCELoss()

# %% [markdown]
# ### running a few checks before training
# pipeline checks:

# %%

run_all_pipeline_checks(
    model=model,
    x_train=images,
    y_train=masks,
    x_validation=next(iter(val_loader))[0],
    train_idx=train_indices,
    validation_idx=val_indices,
    train_step=lambda m, x, y: loss_function(m(x), y),
    loss_function=loss_function,
)

# %% [markdown]
# ### Training Loop

# %%
NUM_EPOCHS = 15

best_val_iou = 0
train_losses = []
val_losses = []
train_ious = []
val_ious = []

for epoch in range(NUM_EPOCHS):
    model.train()

    train_loss = 0
    train_iou = 0

    # Clear CUDA cache before each epoch to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)
        loss = loss_function(outputs, masks)

        # we need also backward
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iou += IoU(outputs, masks)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_iou = train_iou / len(train_loader)
    train_losses.append(avg_train_loss)
    train_ious.append(avg_train_iou)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}"
    )

    # Validation Phase
    model.eval()
    val_loss = 0
    val_iou = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_function(outputs, masks)

            val_loss += loss.item()
            val_iou += IoU(outputs, masks)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    val_losses.append(avg_val_loss)
    val_ious.append(avg_val_iou)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}"
    )

    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        torch.save(model.state_dict(), "best_unet_baseline.pth")
        print(f"--> Saved new best model with Val IoU : {best_val_iou:.4f}")

# %% [markdown]
# Plotting metrics

# %%
plt.figure(figsize=(12, 5))

# Graph for Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()

# Graph for IoU
plt.subplot(1, 2, 2)
plt.plot(train_ious, label="Train IoU")
plt.plot(val_ious, label="Val IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("IoU")
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Test

# %%
model.load_state_dict(torch.load("best_unet_baseline.pth"))
model.eval()

test_loss = 0.0
test_iou = 0.0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        l = loss_function(outputs, masks)

        test_loss += l.item()
        test_iou += IoU(outputs, masks)

final_test_loss = test_loss / len(test_loader)
final_test_iou = test_iou / len(test_loader)

print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test IoU: {final_test_iou:.4f}")


# %%
def vizualize_test_predictions(model, test_loader, n_samples=4):
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

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))

    for i in range(n_samples):
        # Image(re-permutation from [C, H, W] on [H, W, C])
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalization
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i].squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i].squeeze(), cmap="gray")
        axes[i, 2].set_title("Model prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


vizualize_test_predictions(model, test_loader, n_samples=4)
