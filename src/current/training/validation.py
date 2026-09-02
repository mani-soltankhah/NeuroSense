import torch
from src.current.metrics.segmentation import dice_score, iou_score


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            predictions = model(images)
            loss = criterion(predictions, masks)
            dice = dice_score(predictions, masks)

            iou = iou_score(predictions, masks)

            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()

    return (
        total_loss / len(loader),
        total_dice / len(loader),
        total_iou / len(loader)
    )
