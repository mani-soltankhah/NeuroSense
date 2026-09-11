import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, target):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(predictions.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (predictions * target).sum(dim=1)
        dice = (
                (2. * intersection + self.smooth) /
                (predictions.sum(dim=1) + target.sum(dim=1) + self.smooth)
        )
        return 1 - dice.mean()
