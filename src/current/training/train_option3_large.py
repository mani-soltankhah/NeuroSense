"""
OPTION 3: Larger Model (2x channels everywhere)
Run with: python train_option3_large.py

Why larger:
- 2x more parameters → can learn more complex patterns
- 128 channels instead of 64 at first layer
- 2048 channels in bottleneck instead of 1024
- Better capacity for tumor features
"""

import torch
from torch.optim import AdamW
from src.current.models.unet_dropout import AttentionUnetLarge
from src.current.training.trainer import Trainer
from src.current.losses.focal_tversky_2 import FocalTverskyLoss
from src.current.datasets.dataloader import train_loader, val_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = AttentionUnetLarge(in_channels=1, out_channels=1, dropout_rate=0.3)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: AttentionUnetLarge (2x channels)")
    print(f"Total parameters: {total_params:,}")

    criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device,
                      scheduler=scheduler,
                      checkpoint_path="models/option3_large.pth")
    trainer.fit(epochs=50)


if __name__ == '__main__':
    main()
