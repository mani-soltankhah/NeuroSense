import torch


def dice_score(predictions, target, threshold=0.5, smooth=1.0):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    predictions = predictions.view(-1)
    target = target.view(-1)

    intersection = (target * predictions).sum()
    dice = ((2.0 * intersection + smooth) /
            (target.sum() + predictions.sum() + smooth))

    return dice


def iou_score(predictions, target, threshold=0.5, smooth=1.0):
    predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    predictions = predictions.view(-1)
    target = target.view(-1)

    intersection = (predictions * target).sum()

    union = (predictions.sum() + target.sum() - intersection)

    iou = ((intersection + smooth) /
           (union + smooth))

    return iou
