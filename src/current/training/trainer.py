import torch


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        first_weight = next(self.model.parameters()).clone()
        for images, masks in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            predictions = self.model(images)
            loss = self.criterion(predictions, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        last_weight = next(self.model.parameters())

        print(
            torch.equal(
                first_weight,
                last_weight
            )
        )
        return total_loss / len(self.train_loader)
