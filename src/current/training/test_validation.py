import torch
from src.current.models.unet import UNet
from src.current.losses.combined_loss import BCEDiceLoss
from src.current.training.optimizer import create_optimizer
from src.current.training.train import train_one_epoch
from src.current.datasets.test_dataloader import train_loader, val_loader
from src.current.training.validation import validate

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet(
    in_channels=1,
    out_channels=1
).to(device)

criterion = BCEDiceLoss()

optimizer = create_optimizer(model)

train_loss = train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
)

val_loss = validate(
    model,
    val_loader,
    criterion,
    device
)

print("Train Loss:", train_loss)
print("Validation Loss:", val_loss)
