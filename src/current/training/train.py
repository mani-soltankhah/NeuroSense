import torch
from src.current.models.unet import UNet
from src.current.training.trainer import Trainer
from src.current.losses.dice_loss import DiceLoss
from src.current.datasets.test_dataloader import train_loader, val_loader

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)

model = UNet(in_channels=1, out_channels=1)
model = model.to(device)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

trainer = Trainer(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device
)

trainer.fit(epochs=10)
