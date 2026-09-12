import torch
from src.current.models.unet import UNet
from src.current.training.trainer import Trainer
from src.current.losses.combined_loss import BCEDiceLoss
from src.current.datasets.dataloader import train_loader, val_loader


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    model = UNet(in_channels=1, out_channels=1)
    model = model.to(device)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=5)
    trainer = Trainer(model, train_loader, val_loader,
                      criterion, optimizer, device, scheduler=scheduler)
    trainer.fit(epochs=50)


if __name__ == "__main__":
    main()
