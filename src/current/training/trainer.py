import torch
from src.current.utils.metrics import iou_score, dice_score


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        for images, masks in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            predictions = self.model(images)
            loss = self.criterion(predictions, masks)
            dice = dice_score(predictions, masks)
            iou = iou_score(predictions, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()

        last_weight = next(self.model.parameters())

        return (
            total_loss / len(self.val_loader),
            total_dice / len(self.val_loader),
            total_iou / len(self.val_loader)
        )

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)
                dice = dice_score(predictions, masks)
                iou = iou_score(predictions, masks)

                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()

        return (
            total_loss / len(self.val_loader),
            total_dice / len(self.val_loader),
            total_iou / len(self.val_loader)
        )

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_dice, val_iou = self.validate()
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Validation Loss: {val_loss}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Dice: {val_dice:.4f}")
            print(f"Validation IoU: {val_iou:.4f}")
