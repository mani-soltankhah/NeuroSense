import torch

from src.current.models.unet import UNet
from src.current.training.optimizer import create_optimizer
from src.current.losses.combined_loss import BCEDiceLoss

model = UNet(in_channels=1, out_channels=1)
criterion = BCEDiceLoss()
optimizer = create_optimizer(model)

x = torch.randn(8, 1, 224, 224)
target = torch.randint(
    0,
    2,
    (8, 1, 224, 224)
).float()

output = model(x)
loss = criterion(output, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"Loss: {loss.item()}")
