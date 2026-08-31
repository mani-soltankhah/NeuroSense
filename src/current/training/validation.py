import torch


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            loss = criterion(predictions, masks)

            total_loss += loss.item()

    average_loss = total_loss / len(loader)

    return average_loss
