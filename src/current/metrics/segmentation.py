import torch


def dice_score(predictions, target, threshold=0.5, smooth=1.0):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    dice_scores = []
    for pred, mask in zip(predictions, target):
        pred = pred.view(-1)
        mask = mask.view(-1)
        intersection = (pred * mask).sum()
        dice = (2 * intersection + smooth) / (pred.sum() + mask.sum() + smooth)
        dice_scores.append(dice)

    return torch.mean(torch.stack(dice_scores))


def iou_score(predictions, target, threshold=0.5, smooth=1.0):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    iou_scores = []
    for pred, mask in zip(predictions, target):
        pred = pred.view(-1)
        mask = mask.view(-1)
        intersection = (pred * mask).sum()
        union = (mask.sum() + pred.sum() - intersection)
        iou = ((intersection + smooth) / (union + smooth))
        iou_scores.append(iou)
    return torch.mean(torch.stack(iou_scores))
