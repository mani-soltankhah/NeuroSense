"""
OPTION 2: Aggressive Hyperparameters (Higher beta=0.8, gamma=1.0)
Run with: python train_option2_aggressive.py

Why more aggressive:
- beta=0.8 instead of 0.7 → penalize false negatives 80% more (vs false positives)
- gamma=1.0 instead of 0.75 → focus even more on hard examples
"""

import torch
from torch.optim import AdamW
from src.current.models.unet import AttentionUnet
from src.current.training.trainer import Trainer
from src.current.losses.focal_tversky_2 import FocalTverskyLoss
from src.current.datasets.dataloader import train_loader, val_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = AttentionUnet(in_channels=1, out_channels=1)
    model = model.to(device)
    print(f"Model: AttentionUnet (standard)")

    # AGGRESSIVE PARAMETERS
    criterion = FocalTverskyLoss(alpha=0.2, beta=0.8, gamma=1.0)
    print(f"Loss: FocalTverskyLoss(alpha=0.2, beta=0.8, gamma=1.0) - AGGRESSIVE")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device,
                      scheduler=scheduler,
                      checkpoint_path="models/option2_aggressive.pth")
    trainer.fit(epochs=50)


if __name__ == '__main__':
    main()
