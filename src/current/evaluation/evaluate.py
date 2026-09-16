import torch
from src.current.utils.metrics import dice_score, iou_score


def evaluate(model, device, test_loader):
    model.eval()
    total_dice = 0
    total_iou = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            total_dice += dice_score(
                predictions,
                masks
            ).item()

            total_iou += iou_score(
                predictions,
                masks
            ).item()
    print(f"Test Dice: {total_dice / len(test_loader):.4f}")
    print(f"Test IoU: {total_iou / len(test_loader):.4f}")
