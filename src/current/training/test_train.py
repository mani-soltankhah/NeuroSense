import torch
from src.current.models.unet import UNet
from src.current.losses.combined_loss import BCEDiceLoss
from src.current.training.optimizer import create_optimizer
from src.current.training.train import train_one_epoch
from src.current.datasets.test_dataloader import train_loader

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet(
    in_channels=1,
    out_channels=1
).to(device)

criterion = BCEDiceLoss()

optimizer = create_optimizer(model)

loss = train_one_epoch(
    model=model,
    loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device
)

print("Train Loss:", loss)
