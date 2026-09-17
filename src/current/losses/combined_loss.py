import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)

        predictions = predictions.flatten(1)
        targets = targets.flatten(1)

        intersection = (predictions * targets).sum(dim=1)

        dice = (
                (2.0 * intersection + self.smooth) /
                (predictions.sum() + targets.sum() + self.smooth)
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, target):
        bce_loss = self.bce(predictions, target)
        dice_loss = self.dice(predictions, target)

        loss = (
                self.bce_weight * bce_loss
                + self.dice_weight * dice_loss
        )
        return loss
