import torch


def dice_score(predictions, targets, smooth=1e-6):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    intersection = (predictions * targets).sum()
    dice = (
                   2 * intersection + smooth
           ) / (
                   predictions.sum() + targets.sum() + smooth
           )

    return dice


def iou_score(predictions, target, threshold=0.5, smooth=1.0):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    predictions = predictions.view(-1)
    target = target.view(-1)
    intersection = (target * predictions).sum()
    union = (predictions.sum() + target.sum() - intersection)
    iou = (intersection + smooth) / (union + smooth)
    return iou
