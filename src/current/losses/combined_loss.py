import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, target):
        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        target = target.view(-1)

        intersection = (predictions * target).sum()

        dice = (
                (2.0 * intersection + self.smooth) /
                (predictions.sum() + target.sum() + self.smooth)
        )
        return 1.0 - dice


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


import torch

from src.current.models.unet import UNet

model = UNet(
    in_channels=1,
    out_channels=1
)

criterion = BCEDiceLoss()

x = torch.randn(8, 1, 224, 224)
target = torch.randint(
    0,
    2,
    (8, 1, 224, 224)
).float()

output = model(x)

loss = criterion(output, target)

print("Output shape:", output.shape)
print("Target shape:", target.shape)
print("Loss:", loss)
