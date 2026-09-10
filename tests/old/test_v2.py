import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from archive_older_codes.old.datasets.dataset import (
    BrainTumorDataset,
    test_df,
    image_transform,
    mask_transform
)
from archive_older_codes.old.models.segmentation.segmentation_v2 import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"/Models/segmentation_only/best_v2.pth"

SAVE_DIR = r"/Models/segmentation_only/test_predictions"

os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
THRESHOLD = 0.35

test_dataset = BrainTumorDataset(
    test_df,
    image_transform=image_transform,
    mask_transform=mask_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("=" * 60)
print("TEST ONLY SEGMENTATION")
print("=" * 60)

print("Device:", DEVICE)
print("Test samples:", len(test_dataset))

model = UNet()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully")


def dice_score(pred, target, eps=1e-7):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    intersection = (pred * target).sum()

    return (
            2 * intersection + eps
    ) / (
            pred.sum() + target.sum() + eps
    )


def iou_score(pred, target, eps=1e-7):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    intersection = (pred * target).sum()

    union = (
            pred.sum()
            +
            target.sum()
            -
            intersection
    )

    return (
            intersection + eps
    ) / (
            union + eps
    )


def make_overlay(image, mask):
    image = image.astype(np.uint8)

    overlay = np.stack(
        [image, image, image],
        axis=-1
    )

    overlay[mask > 0] = [
        255,
        0,
        0
    ]

    return overlay


dice_scores = []
iou_scores = []

counter = 0

with torch.no_grad():
    for images, labels, masks in tqdm(test_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (
                probs > THRESHOLD
        ).float()

        for i in range(len(preds)):
            dice = dice_score(
                preds[i],
                masks[i]
            )

            iou = iou_score(
                preds[i],
                masks[i]
            )

            dice_scores.append(
                dice.item()
            )

            iou_scores.append(
                iou.item()
            )

            image = (
                images[i]
                .squeeze()
                .cpu()
                .numpy()
            )

            image = (
                    (image * 0.5 + 0.5)
                    * 255
            )

            image = np.clip(
                image,
                0,
                255
            ).astype(
                np.uint8
            )

            pred_mask = (
                    preds[i]
                    .squeeze()
                    .cpu()
                    .numpy()
                    *
                    255
            ).astype(
                np.uint8
            )

            gt_mask = (
                    masks[i]
                    .squeeze()
                    .cpu()
                    .numpy()
                    *
                    255
            ).astype(
                np.uint8
            )

            overlay = make_overlay(
                image,
                pred_mask
            )

            comparison = np.concatenate(
                [
                    np.stack(
                        [
                            image,
                            image,
                            image
                        ],
                        axis=-1
                    ),
                    np.stack(
                        [
                            gt_mask,
                            gt_mask,
                            gt_mask
                        ],
                        axis=-1
                    ),
                    overlay
                ],
                axis=1
            )

            Image.fromarray(pred_mask).save(
                os.path.join(
                    SAVE_DIR,
                    f"prediction_{counter}.png"
                )
            )

            Image.fromarray(overlay).save(
                os.path.join(
                    SAVE_DIR,
                    f"overlay_{counter}.png"
                )
            )

            Image.fromarray(comparison).save(
                os.path.join(
                    SAVE_DIR,
                    f"comparison_{counter}.png"
                )
            )

            counter += 1

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    "Mean Dice:",
    np.mean(dice_scores)
)

print(
    "Median Dice:",
    np.median(dice_scores)
)

print(
    "Mean IoU:",
    np.mean(iou_scores)
)

print(
    "Median IoU:",
    np.median(iou_scores)
)

print(
    "\nSaved:",
    SAVE_DIR
)
