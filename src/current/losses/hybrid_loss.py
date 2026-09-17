import torch.nn as nn
from src.current.losses.focal_tversky import FocalTverskyLoss
from src.current.losses.dice_loss import DiceLoss


class DiceFocalTverskyLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()

        self.dice = DiceLoss()

        self.focal_tversky = FocalTverskyLoss(
            alpha=0.2,
            beta=0.8,
            gamma=0.75
        )

        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        dice_loss = self.dice(outputs, targets)
        ft_loss = self.focal_tversky(outputs, targets)

        return (
                self.dice_weight * dice_loss +
                (1 - self.dice_weight) * ft_loss
        )
