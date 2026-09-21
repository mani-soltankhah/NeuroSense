"""
ENSEMBLE: Combine all 3 models for final prediction
Run with: python ensemble_eval.py

Why ensemble works:
- Option 1 (dropout) handles overfitting
- Option 2 (aggressive) catches harder cases
- Option 3 (large) gets fine details
- Voting/averaging their outputs = best overall Dice
"""

import torch
import torch.nn.functional as F
from src.current.models.unet_dropout import AttentionUnetDropout, AttentionUnetLarge
from src.current.models.unet import AttentionUnet
from src.current.datasets.dataloader import test_loader
from src.current.utils.metrics import dice_score, iou_score


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading 3 models...")

    # Load Model 1: Dropout
    model1 = AttentionUnetDropout(in_channels=1, out_channels=1, dropout_rate=0.3)
    checkpoint1 = torch.load("models/option1_dropout.pth", map_location=device)
    model1.load_state_dict(checkpoint1['model_state_dict'])
    model1 = model1.to(device)
    model1.eval()
    print("✅ Model 1 (Dropout) loaded")

    # Load Model 2: Aggressive
    model2 = AttentionUnet(in_channels=1, out_channels=1)
    checkpoint2 = torch.load("models/option2_aggressive.pth", map_location=device)
    model2.load_state_dict(checkpoint2['model_state_dict'])
    model2 = model2.to(device)
    model2.eval()
    print("✅ Model 2 (Aggressive) loaded")

    # Load Model 3: Large
    model3 = AttentionUnetLarge(in_channels=1, out_channels=1, dropout_rate=0.3)
    checkpoint3 = torch.load("models/option3_large.pth", map_location=device)
    model3.load_state_dict(checkpoint3['model_state_dict'])
    model3 = model3.to(device)
    model3.eval()
    print("✅ Model 3 (Large) loaded")

    print("\nEvaluating ensemble on test set...")
    print("=" * 70)

    total_dice_ensemble = 0
    total_iou_ensemble = 0
    total_dice_m1 = 0
    total_dice_m2 = 0
    total_dice_m3 = 0
    count = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Get predictions from all 3 models
            pred1 = torch.sigmoid(model1(images))
            pred2 = torch.sigmoid(model2(images))
            pred3 = torch.sigmoid(model3(images))

            # Ensemble: Average probabilities
            ensemble_pred = (pred1 + pred2 + pred3) / 3.0

            # Compute Dice for each
            dice1 = dice_score(pred1, masks)
            dice2 = dice_score(pred2, masks)
            dice3 = dice_score(pred3, masks)
            dice_ens = dice_score(ensemble_pred, masks)
            iou_ens = iou_score(ensemble_pred, masks)

            total_dice_m1 += dice1.item()
            total_dice_m2 += dice2.item()
            total_dice_m3 += dice3.item()
            total_dice_ensemble += dice_ens.item()
            total_iou_ensemble += iou_ens.item()

            count += 1

    print(f"Model 1 (Dropout):   Dice {total_dice_m1 / count:.4f}")
    print(f"Model 2 (Aggressive): Dice {total_dice_m2 / count:.4f}")
    print(f"Model 3 (Large):     Dice {total_dice_m3 / count:.4f}")
    print("-" * 70)
    print(f"ENSEMBLE (Average):  Dice {total_dice_ensemble / count:.4f}")
    print(f"ENSEMBLE (Average):  IoU  {total_iou_ensemble / count:.4f}")
    print("=" * 70)

    # Find best model
    best_dice = max(total_dice_m1, total_dice_m2, total_dice_m3) / count
    if best_dice == total_dice_m1 / count:
        print(f"Best single model: Option 1 (Dropout)")
    elif best_dice == total_dice_m2 / count:
        print(f"Best single model: Option 2 (Aggressive)")
    else:
        print(f"Best single model: Option 3 (Large)")

    if total_dice_ensemble / count > best_dice:
        print(f"✅ ENSEMBLE is BETTER than best single model!")
    else:
        print(f"Best single model is slightly better, but ensemble is more robust")


if __name__ == "__main__":
    main()
