import torch
from src.current.utils.metrics import iou_score, dice_score
from pathlib import Path


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device,
                 checkpoint_path="models/best_model.pth"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.history = {
            "train_loss": [],
            "train_dice": [],
            "train_iou": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": []
        }

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        for images, masks in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            predictions = self.model(images)

            loss = self.criterion(predictions, masks)
            loss.backward()

            self.optimizer.step()

            dice = dice_score(predictions, masks)
            iou = iou_score(predictions, masks)

            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()

        # last_weight = next(self.model.parameters())

        return (
            total_loss / len(self.train_loader),
            total_dice / len(self.train_loader),
            total_iou / len(self.train_loader)
        )

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

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
        best_epoch = 0
        best_dice = 0

        for epoch in range(epochs):
            train_loss, train_dice, train_iou = self.train_one_epoch()
            val_loss, val_dice, val_iou = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_dice"].append(train_dice)
            self.history["train_iou"].append(train_iou)

            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_dice)
            self.history["val_iou"].append(val_iou)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Dice: {train_dice:.4f}")
            print(f"Train IoU: {train_iou:.4f}")

            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Dice: {val_dice:.4f}")
            print(f"Validation IoU: {val_iou:.4f}")
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1

                Path(self.checkpoint_path).parent.mkdir(
                    parents=True,
                    exist_ok=True
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_dice": val_dice
                    },
                    self.checkpoint_path
                )
                print(
                    f"Best model saved at epoch {best_epoch} "
                    f"with Dice {best_dice:.4f}"
                )
        return self.history
