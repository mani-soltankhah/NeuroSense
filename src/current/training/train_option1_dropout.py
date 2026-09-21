"""
OPTION 1: Dropout (Best for overfitting)
Run with: python train_option1_dropout.py
"""

import torch
from torch.optim import AdamW
from src.current.models.unet_dropout import AttentionUnetDropout
from src.current.training.trainer import Trainer
from src.current.losses.focal_tversky_2 import FocalTverskyLoss
from src.current.datasets.dataloader import train_loader, val_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = AttentionUnetDropout(in_channels=1, out_channels=1, dropout_rate=0.3)
    model = model.to(device)
    print(f"Model: AttentionUnetDropout with dropout=0.3")

    criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device,
                      scheduler=scheduler,
                      checkpoint_path="models/option1_dropout.pth")
    trainer.fit(epochs=50)


if __name__ == '__main__':
    main()
