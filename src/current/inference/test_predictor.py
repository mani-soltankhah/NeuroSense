from pathlib import Path
import torch
from src.current.datasets.test_dataloader import test_loader, val_loader
from src.current.inference.predictor import Predictor
from src.current.models.unet import UNet
from src.current.utils.checkpoint import load_checkpoint
from src.current.visualization.visualize import (
    show_segmentation,
    create_segmentation_figure,
    visualize_dataset,
    save_segmentation
)
from src.current.utils.metrics import dice_score, iou_score

BASE_DIR = Path(r"D:\Portfolio\NeuroSense")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=1, out_channels=1)
model = load_checkpoint(
    model,
    BASE_DIR / "src" / "current" / "training" / "models" / "best_model.pth",
    device
)

output_path_single = BASE_DIR / "src" / "current" / "visualization" / "plot_images" / "single"
output_path_dataset = BASE_DIR / "src" / "current" / "visualization" / "plot_images" / "dataset"

output_path_single.mkdir(parents=True, exist_ok=True)
output_path_dataset.mkdir(parents=True, exist_ok=True)

model.to(device)

predictor = Predictor(model, device)

images, masks = next(iter(test_loader))
image = images[0]
mask = masks[0]

input_image = image.unsqueeze(0)
prediction = predictor.predict(input_image)

print("Prediction shape:", prediction.shape)
print("Unique values:", torch.unique(prediction))
print("Prediction sum (foreground pixels):", prediction.sum().item())
prediction = prediction.to("cpu")
show_segmentation(image, mask, prediction)
dice = dice_score(prediction, mask)
iou = iou_score(prediction, mask)
fig = create_segmentation_figure(image, mask, prediction, dice, iou)
save_segmentation(fig, output_path_single / "single_prediction.png")

visualize_dataset(
    loader=val_loader,
    predictor=predictor,
    output_dir=output_path_dataset,
    num_last_images=None
)

print(f"\n✅ Check this exact folder for your images:\n{output_path_dataset.absolute()}")
