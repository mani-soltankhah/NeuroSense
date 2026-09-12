import torch
from src.current.models.unet import UNet
from src.current.datasets.dataloader import test_loader
from src.current.utils.metrics import dice_score, iou_score
from src.current.inference.predictor import Predictor
from src.current.visualization.visualize import visualize_dataset

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)


def main():
    model = UNet(in_channels=1, out_channels=1)
    model = model.to(device)

    checkpoint = torch.load(r"D:\Portfolio\NeuroSense\src\current\training\models\best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    predictor = Predictor(model, device)
    visualize_dataset(test_loader, predictor,
                      "D:/Portfolio/NeuroSense/src/current/evaluation/results",
                      num_last_images=None
                      )


if __name__ == '__main__':
    main()
