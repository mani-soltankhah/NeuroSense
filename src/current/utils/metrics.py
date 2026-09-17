import torch


def dice_score(predictions, targets, smooth=1e-6):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()

    predictions = predictions.flatten(1)
    targets = targets.flatten(1)

    intersection = (predictions * targets).sum(dim=1)

    dice = (
            (2 * intersection + smooth)
            /
            (
                    predictions.sum(dim=1)
                    +
                    targets.sum(dim=1)
                    +
                    smooth
            )
    )

    return dice.mean()


def iou_score(predictions, targets, threshold=0.5, smooth=1e-6):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    predictions = predictions.flatten(1)
    targets = targets.flatten(1)

    intersection = (predictions * targets).sum(dim=1)

    union = (
            predictions.sum(dim=1)
            +
            targets.sum(dim=1)
            -
            intersection
    )

    iou = (
                  intersection + smooth
          ) / (
                  union + smooth
          )

    return iou.mean()
