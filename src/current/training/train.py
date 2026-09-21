import torch
from torch.optim import AdamW
from src.current.models.unet import UNet, AttentionUnet
from src.current.training.trainer import Trainer
from src.current.losses.dice_loss import DiceLoss
from src.current.losses.combined_loss import BCEDiceLoss
from src.current.datasets.dataloader import train_loader, val_loader
from src.current.losses.focal_tversky import FocalTverskyLoss
from src.current.losses.hybrid_loss import DiceFocalTverskyLoss


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    model = AttentionUnet(in_channels=1, out_channels=1)
    model = model.to(device)

    criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=5)

    trainer = Trainer(model, train_loader, val_loader,
                      criterion, optimizer, device, scheduler=scheduler,
                      checkpoint_path="models/AttentionsUnet_model.pth")
    trainer.fit(epochs=50)


if __name__ == "__main__":
    main()
