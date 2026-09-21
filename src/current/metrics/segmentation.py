import torch


def dice_score(predictions, targets, smooth=1e-6):
    """Dice coefficient - works on both probabilities and binary"""
    # If predictions are logits (raw model output), apply sigmoid
    # If predictions are already binary, skip sigmoid
    if predictions.max() > 1.0 or predictions.min() < 0:
        # Logits detected, apply sigmoid
        predictions = torch.sigmoid(predictions)

    predictions_flat = predictions.flatten(1)
    targets_flat = targets.flatten(1)

    intersection = (predictions_flat * targets_flat).sum(dim=1)
    dice = (2 * intersection + smooth) / (
            predictions_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth
    )
    return dice.mean()


def iou_score(predictions, targets, smooth=1.0):
    """IoU coefficient"""
    if predictions.max() > 1.0 or predictions.min() < 0:
        predictions = torch.sigmoid(predictions)

    predictions_flat = predictions.flatten(1)
    targets_flat = targets.flatten(1)

    intersection = (predictions_flat * targets_flat).sum(dim=1)
    union = (predictions_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection)
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()
