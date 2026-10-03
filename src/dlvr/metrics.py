import torch


def IoU(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
):  ## think about the threshold !!!
    """Compute Intersection over Union (IoU) for binary segmentation tasks.
    Args:
        logits (torch.Tensor): The raw output from the model (before activation).
        targets (torch.Tensor): The ground truth binary masks.
        threshold (float): Threshold to convert probabilities to binary predictions.
    Returns:
        float: The IoU score.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)  # adding value for smoothing and avoiding zero
    return iou.item()
