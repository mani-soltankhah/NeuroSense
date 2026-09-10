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


loss_fn = DiceLoss()

mask1 = torch.randint(
    0,
    2,
    (8, 1, 224, 224)
).float()

loss = loss_fn(mask1, mask1)

print(loss)
pred = torch.randn(8, 1, 224, 224)
mask2 = torch.randint(
    0,
    2,
    (8, 1, 224, 224)
).float()

wrong_prediction = 1 - mask2

loss = loss_fn(wrong_prediction, mask2)

print(loss)

from src.current.models.unet import UNet

model = UNet(
    in_channels=1,
    out_channels=1
)

image = torch.randn(
    8, 1, 224, 224
)

mask = torch.randint(
    0,
    2,
    (8, 1, 224, 224)
).float()

prediction = model(image)

loss = loss_fn(
    prediction,
    mask
)

print(prediction.shape)
print(loss)
