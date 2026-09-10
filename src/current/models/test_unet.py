import torch

from src.current.models.unet import UNet

model = UNet(in_channels=1, out_channels=1)
x = torch.randn(8, 1, 224, 224)

output = model(x)

print("Input:")
print(x.shape)

print("Output:")
print(output.shape)
