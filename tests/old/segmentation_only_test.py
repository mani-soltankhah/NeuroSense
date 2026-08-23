# D:\Portfolio\NeuroSense\Data\segmentation_only_test.py

import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"/")

METADATA_PATH = PROJECT_ROOT / "Data" / "metadata.csv"

MODEL_DIR = PROJECT_ROOT / "Models" / "segmentation_only"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best.pth"

PREDICTION_DIR = MODEL_DIR / "predictions"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-3

NUM_WORKERS = 0

SEED = 42

THRESHOLD = 0.5

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class BrainTumorSegmentationDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = row["path"]

        with h5py.File(path, "r") as f:

            image = np.asarray(
                f["cjdata"]["image"][()]
            ).astype(np.float32)

            mask = np.asarray(
                f["cjdata"]["tumorMask"][()]
            ).astype(np.float32)

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = torch.from_numpy(image)

        if image.ndim == 2:

            image = image.unsqueeze(0)

        elif image.ndim == 3:

            if image.shape[0] != 1:
                image = image[:1]

        image = F.interpolate(
            image.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # ----------------------------------------------------
        # NORMALIZE IMAGE
        # ----------------------------------------------------

        image_min = image.min()

        image_max = image.max()

        if image_max > image_min:

            image = (
                            image - image_min
                    ) / (
                            image_max - image_min
                    )

        else:

            image = torch.zeros_like(image)

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask = torch.from_numpy(mask)

        if mask.ndim == 2:

            mask = mask.unsqueeze(0)

        elif mask.ndim == 3:

            if mask.shape[0] != 1:
                mask = mask[:1]

        # IMPORTANT:
        # NEAREST preserves binary mask labels.

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest"
        ).squeeze(0)

        # ----------------------------------------------------
        # FORCE BINARY
        # ----------------------------------------------------

        mask = (mask > 0).float()

        return image, mask


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SEGMENTATION ONLY TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

print()

df = pd.read_csv(METADATA_PATH)

# ------------------------------------------------------------
# Normalize split column
# ------------------------------------------------------------

df["split"] = (
    df["split"]
    .astype(str)
    .str.lower()
    .str.strip()
)

train_df = df[
    df["split"].isin(
        ["train", "training"]
    )
].copy()

val_df = df[
    df["split"].isin(
        ["val", "validation"]
    )
].copy()

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# ============================================================
# DATASETS
# ============================================================

train_dataset = BrainTumorSegmentationDataset(
    train_df
)

val_dataset = BrainTumorSegmentationDataset(
    val_df
)

# ============================================================
# DATASET SANITY CHECK
# ============================================================

print()
print("=" * 70)
print("DATASET SANITY CHECK")
print("=" * 70)

for name, dataset in [
    ("TRAIN", train_dataset),
    ("VALIDATION", val_dataset)
]:
    image, mask = dataset[0]

    print()
    print(f"{name} SAMPLE")

    print(
        f"Image shape      : {image.shape}"
    )

    print(
        f"Mask shape       : {mask.shape}"
    )

    print(
        f"Image dtype      : {image.dtype}"
    )

    print(
        f"Mask dtype       : {mask.dtype}"
    )

    print(
        f"Mask unique      : {torch.unique(mask)}"
    )

    print(
        f"Tumor pixels     : {mask.sum().item()}"
    )

    print(
        f"Image min/max    : "
        f"{image.min().item():.4f} / "
        f"{image.max().item():.4f}"
    )

# ============================================================
# VALIDATION MASK SANITY CHECK
# ============================================================

print()
print("=" * 70)
print("VALIDATION MASK SANITY CHECK")
print("=" * 70)

val_tumor_pixels = []

for i in range(
        min(10, len(val_dataset))
):
    _, mask = val_dataset[i]

    pixels = int(mask.sum().item())

    val_tumor_pixels.append(pixels)

    print(
        f"Sample {i + 1:03d} | "
        f"Tumor pixels after resize: {pixels}"
    )

print()

print(
    f"First 10 total tumor pixels: "
    f"{sum(val_tumor_pixels)}"
)

if sum(val_tumor_pixels) == 0:
    raise RuntimeError(
        "\n"
        "FATAL ERROR:\n"
        "Validation masks became empty after preprocessing.\n"
        "STOPPING BEFORE TRAINING."
    )

print(
    "PASS: Validation masks survive preprocessing."
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# DATALOADER SANITY CHECK
# ============================================================

print()
print("=" * 70)
print("DATALOADER SANITY CHECK")
print("=" * 70)

images, masks = next(iter(val_loader))

print(
    f"Batch images shape : {images.shape}"
)

print(
    f"Batch masks shape  : {masks.shape}"
)

print(
    f"Batch tumor pixels : "
    f"{masks.sum().item():.0f}"
)

print(
    f"Batch mask unique  : "
    f"{torch.unique(masks)}"
)

if masks.sum().item() == 0:
    raise RuntimeError(
        "\n"
        "FATAL ERROR:\n"
        "Validation DataLoader produced empty masks.\n"
        "STOPPING."
    )

print(
    "PASS: Validation DataLoader contains tumor pixels."
)


# ============================================================
# MODEL
# ============================================================

class SegmentationCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)

        return x


# ============================================================
# DICE
# ============================================================

def dice_score(
        logits,
        targets,
        threshold=0.5,
        smooth=1e-6
):
    probabilities = torch.sigmoid(logits)

    predictions = (
            probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    intersection = (
            predictions * targets
    ).sum(dim=1)

    denominator = (
            predictions.sum(dim=1)
            +
            targets.sum(dim=1)
    )

    dice = (
                   2.0 * intersection + smooth
           ) / (
                   denominator + smooth
           )

    return dice.mean()


# ============================================================
# SOFT DICE LOSS
# ============================================================

def soft_dice_loss(
        logits,
        targets,
        smooth=1.0
):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.view(
        probabilities.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    intersection = (
            probabilities * targets
    ).sum(dim=1)

    denominator = (
            probabilities.sum(dim=1)
            +
            targets.sum(dim=1)
    )

    dice = (
                   2.0 * intersection + smooth
           ) / (
                   denominator + smooth
           )

    return 1.0 - dice.mean()


# ============================================================
# IOU
# ============================================================

def iou_score(
        logits,
        targets,
        threshold=0.5,
        smooth=1e-6
):
    probabilities = torch.sigmoid(logits)

    predictions = (
            probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    intersection = (
            predictions * targets
    ).sum(dim=1)

    union = (
            predictions.sum(dim=1)
            +
            targets.sum(dim=1)
            -
            intersection
    )

    # If both are empty, IoU = 1.
    # But our dataset should contain tumors,
    # so this should not dominate the metric.

    iou = (
                  intersection + smooth
          ) / (
                  union + smooth
          )

    return iou.mean()


# ============================================================
# LOSS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def total_loss(
        logits,
        targets
):
    bce = bce_loss(
        logits,
        targets
    )

    dice = soft_dice_loss(
        logits,
        targets
    )

    return bce + dice


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING SEGMENTATION MODEL")
print("=" * 70)

model = SegmentationCNN().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Model created successfully.")


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
        model,
        loader
):
    model.eval()

    total_loss_value = 0.0

    all_dice = []

    all_iou = []

    total_gt_pixels = 0

    total_pred_pixels = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(images)

            loss = total_loss(
                logits,
                masks
            )

            total_loss_value += (
                    loss.item()
                    *
                    images.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                    probabilities >= THRESHOLD
            ).float()

            # ------------------------------------------------
            # Per sample metrics
            # ------------------------------------------------

            for i in range(
                    images.size(0)
            ):
                pred = predictions[i].view(-1)

                target = masks[i].view(-1)

                intersection = (
                        pred * target
                ).sum()

                pred_pixels = pred.sum()

                gt_pixels = target.sum()

                union = (
                        pred_pixels
                        +
                        gt_pixels
                        -
                        intersection
                )

                # ------------------------------------------------
                # IMPORTANT:
                # Every validation sample has a tumor.
                # ------------------------------------------------

                dice = (
                               2.0 * intersection
                       ) / (
                               pred_pixels
                               +
                               gt_pixels
                               +
                               1e-6
                       )

                iou = (
                          intersection
                      ) / (
                              union
                              +
                              1e-6
                      )

                all_dice.append(
                    dice.item()
                )

                all_iou.append(
                    iou.item()
                )

                total_gt_pixels += (
                    gt_pixels.item()
                )

                total_pred_pixels += (
                    pred_pixels.item()
                )

    mean_loss = (
            total_loss_value
            /
            len(loader.dataset)
    )

    mean_dice = float(
        np.mean(all_dice)
    )

    median_dice = float(
        np.median(all_dice)
    )

    mean_iou = float(
        np.mean(all_iou)
    )

    median_iou = float(
        np.median(all_iou)
    )

    return {
        "loss": mean_loss,
        "dice": mean_dice,
        "median_dice": median_dice,
        "iou": mean_iou,
        "median_iou": median_iou,
        "gt_pixels": total_gt_pixels,
        "pred_pixels": total_pred_pixels,
        "sample_dice": all_dice,
    }


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING SEGMENTATION-ONLY TRAINING")
print("=" * 70)

best_dice = -1.0

best_epoch = 0

for epoch in range(
        1,
        EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch:02d}/{EPOCHS}"
    )

    for images, masks in progress:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = total_loss(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        running_loss += (
                loss.item()
                *
                images.size(0)
        )

        # ----------------------------------------------------
        # Progress Dice
        # ----------------------------------------------------

        with torch.no_grad():
            batch_dice = dice_score(
                logits,
                masks
            )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_dice.item():.4f}"
        )

    train_loss = (
            running_loss
            /
            len(train_loader.dataset)
    )

    # ========================================================
    # TRAIN METRICS
    # ========================================================

    train_eval = evaluate(
        model,
        train_loader
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_eval = evaluate(
        model,
        val_loader
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()

    print(
        f"Epoch {epoch:03d}/{EPOCHS}"
    )

    print(
        f"Train Loss : {train_loss:.6f}"
    )

    print(
        f"Train Dice : {train_eval['dice']:.6f}"
    )

    print(
        f"Train IoU  : {train_eval['iou']:.6f}"
    )

    print(
        f"Val Loss   : {val_eval['loss']:.6f}"
    )

    print(
        f"Val Dice   : {val_eval['dice']:.6f}"
    )

    print(
        f"Val IoU    : {val_eval['iou']:.6f}"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_eval["dice"] > best_dice:
        best_dice = val_eval["dice"]

        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_loss":
                    val_eval["loss"],

                "val_dice":
                    val_eval["dice"],

                "val_iou":
                    val_eval["iou"],
            },
            BEST_MODEL_PATH
        )

        print()
        print(
            ">>> BEST MODEL SAVED"
        )

        print(
            f">>> Epoch: {epoch}"
        )

        print(
            f">>> Val Dice: "
            f"{val_eval['dice']:.6f}"
        )

        print(
            f">>> Path: "
            f"{BEST_MODEL_PATH}"
        )

# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation Dice: "
    f"{checkpoint['val_dice']:.6f}"
)

print(
    f"Best validation IoU: "
    f"{checkpoint['val_iou']:.6f}"
)

# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

final_results = evaluate(
    model,
    val_loader
)

print(
    f"Mean Dice   : "
    f"{final_results['dice']:.6f}"
)

print(
    f"Median Dice : "
    f"{final_results['median_dice']:.6f}"
)

print(
    f"Mean IoU    : "
    f"{final_results['iou']:.6f}"
)

print(
    f"Median IoU  : "
    f"{final_results['median_iou']:.6f}"
)

print()

print(
    f"Ground Truth tumor pixels : "
    f"{final_results['gt_pixels']:,.0f}"
)

print(
    f"Predicted tumor pixels    : "
    f"{final_results['pred_pixels']:,.0f}"
)

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

model.eval()

all_probabilities = []

all_masks = []

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(
            logits
        )

        all_probabilities.append(
            probabilities.cpu()
        )

        all_masks.append(
            masks.cpu()
        )

all_probabilities = torch.cat(
    all_probabilities,
    dim=0
)

all_masks = torch.cat(
    all_masks,
    dim=0
)

for threshold in [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]:

    predictions = (
            all_probabilities >= threshold
    ).float()

    dices = []

    ious = []

    for i in range(
            predictions.size(0)
    ):
        pred = predictions[i].view(-1)

        target = all_masks[i].view(-1)

        intersection = (
                pred * target
        ).sum()

        pred_pixels = pred.sum()

        gt_pixels = target.sum()

        union = (
                pred_pixels
                +
                gt_pixels
                -
                intersection
        )

        dice = (
                       2.0 * intersection
               ) / (
                       pred_pixels
                       +
                       gt_pixels
                       +
                       1e-6
               )

        iou = (
                  intersection
              ) / (
                      union
                      +
                      1e-6
              )

        dices.append(
            dice.item()
        )

        ious.append(
            iou.item()
        )

    predicted_pixels = (
        predictions.sum().item()
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Dice: {np.mean(dices):.4f} | "
        f"IoU: {np.mean(ious):.4f} | "
        f"Predicted pixels: "
        f"{predicted_pixels:,.0f}"
    )

# ============================================================
# PROBABILITY ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PROBABILITY ANALYSIS")
print("=" * 70)

probabilities_flat = (
    all_probabilities
    .numpy()
    .reshape(-1)
)

print(
    f"Probability Min    : "
    f"{probabilities_flat.min():.6f}"
)

print(
    f"Probability Max    : "
    f"{probabilities_flat.max():.6f}"
)

print(
    f"Probability Mean   : "
    f"{probabilities_flat.mean():.6f}"
)

print(
    f"Probability Median : "
    f"{np.median(probabilities_flat):.6f}"
)

# ============================================================
# PIXEL ANALYSIS
# ============================================================

predictions_05 = (
        all_probabilities >= 0.5
).float()

gt_pixels = (
    all_masks.sum().item()
)

pred_pixels = (
    predictions_05.sum().item()
)

total_pixels = (
    all_masks.numel()
)

print()
print("=" * 70)
print("PIXEL ANALYSIS")
print("=" * 70)

print(
    f"Ground Truth tumor pixels : "
    f"{gt_pixels:,.0f}"
)

print(
    f"Predicted tumor pixels    : "
    f"{pred_pixels:,.0f}"
)

print(
    f"Ground Truth tumor ratio  : "
    f"{gt_pixels / total_pixels:.8f}"
)

print(
    f"Predicted tumor ratio     : "
    f"{pred_pixels / total_pixels:.8f}"
)

# ============================================================
# PER SAMPLE DICE
# ============================================================

print()
print("=" * 70)
print("PER-SAMPLE DICE")
print("=" * 70)

for i in range(
        min(20, len(val_dataset))
):
    pred = (
        predictions_05[i]
        .view(-1)
    )

    target = (
        all_masks[i]
        .view(-1)
    )

    intersection = (
            pred * target
    ).sum()

    pred_pixels_i = pred.sum()

    gt_pixels_i = target.sum()

    dice = (
                   2.0 * intersection
           ) / (
                   pred_pixels_i
                   +
                   gt_pixels_i
                   +
                   1e-6
           )

    print(
        f"Sample {i + 1:03d} | "
        f"GT pixels: {gt_pixels_i.item():6.0f} | "
        f"Pred pixels: {pred_pixels_i.item():6.0f} | "
        f"Dice: {dice.item():.4f}"
    )

# ============================================================
# SAVE VISUAL PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("SAVING SAMPLE PREDICTIONS")
print("=" * 70)

try:

    import matplotlib.pyplot as plt

    model.eval()

    sample_indices = [
        0,
        1,
        2,
        3,
        4
    ]

    with torch.no_grad():

        for sample_index in sample_indices:
            image, mask = val_dataset[
                sample_index
            ]

            image_input = image.unsqueeze(0).to(
                DEVICE
            )

            logits = model(
                image_input
            )

            probability = torch.sigmoid(
                logits
            )[0, 0].cpu().numpy()

            prediction = (
                    probability >= 0.5
            ).astype(np.float32)

            image_np = (
                image[0].numpy()
            )

            mask_np = (
                mask[0].numpy()
            )

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(16, 4)
            )

            axes[0].imshow(
                image_np,
                cmap="gray"
            )

            axes[0].set_title(
                "MRI"
            )

            axes[0].axis("off")

            axes[1].imshow(
                mask_np,
                cmap="gray"
            )

            axes[1].set_title(
                "Ground Truth"
            )

            axes[1].axis("off")

            axes[2].imshow(
                probability,
                cmap="hot",
                vmin=0,
                vmax=1
            )

            axes[2].set_title(
                "Probability"
            )

            axes[2].axis("off")

            axes[3].imshow(
                prediction,
                cmap="gray"
            )

            axes[3].set_title(
                "Prediction >= 0.5"
            )

            axes[3].axis("off")

            plt.tight_layout()

            output_path = (
                    PREDICTION_DIR
                    /
                    f"sample_{sample_index + 1:03d}.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"Saved: {output_path}"
            )


except ImportError:

    print(
        "matplotlib is not installed."
    )

    print(
        "Skipping visual predictions."
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("SEGMENTATION TEST COMPLETED")
print("=" * 70)

print(
    f"Best Epoch      : {best_epoch}"
)

print(
    f"Best Val Dice   : {best_dice:.6f}"
)

print(
    f"Final Val Dice  : "
    f"{final_results['dice']:.6f}"
)

print(
    f"Final Val IoU   : "
    f"{final_results['iou']:.6f}"
)

print()

print(
    f"Best model saved at:"
)

print(
    BEST_MODEL_PATH
)

print()

print(
    f"Prediction images saved at:"
)

print(
    PREDICTION_DIR
)

print("=" * 70)