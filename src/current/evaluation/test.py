import torch

from src.current.evaluation.find_best_threshold import test_multiple_thresholds
from src.current.evaluation.visualize import SegmentationVisualizer
from src.current.models.unet import UNet, AttentionUnet
from src.current.datasets.dataloader import test_loader
from src.current.utils.metrics import dice_score, iou_score
from src.current.inference.predictor import Predictor
from src.current.visualization.visualize import visualize_dataset

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)


def main():
    # model = UNet(in_channels=1, out_channels=1)
    model = AttentionUnet(in_channels=1, out_channels=1)
    model = model.to(device)

    checkpoint = torch.load(
        r"D:\Portfolio\NeuroSense\src\current\training\models\AttentionUnet_nonAugment_FocalTversky.pth",
        map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    best_threshold, all_results = test_multiple_thresholds(model, test_loader, device)
    predictor = Predictor(model, device, threshold=best_threshold)
    visualize_dataset(test_loader, predictor,
                      "D:/Portfolio/NeuroSense/src/current/evaluation/results_2",
                      num_last_images=50
                      )
    # visualizer = SegmentationVisualizer(
    #     model,
    #     test_loader,
    #     device,
    #     threshold=0.1,
    #     save_dir="results/visualization"
    # )
    # visualizer.run()


if __name__ == '__main__':
    main()
