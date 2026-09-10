import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from archive_older_codes.old.datasets.dataset import BrainTumorDataset

# ------------------------------------------------------------
# IMPORT DATASET
# ------------------------------------------------------------

# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 40

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

IMAGE_SIZE = 224

NUM_WORKERS = 0

PATIENCE = 8

DICE_WEIGHT = 0.6
BCE_WEIGHT = 0.4

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"/"
)

METADATA_PATH = BASE_DIR / "Data" / "metadata.csv"

MODEL_DIR = (
        BASE_DIR
        / "Models"
        / "segmentation_only"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = MODEL_DIR / "best_v2.pth"

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

# ============================================================
# TRANSFORMS
# ============================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=InterpolationMode.BILINEAR
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    ),
])

mask_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=InterpolationMode.NEAREST
    ),

    transforms.ToTensor(),
])

# ============================================================
# DATASET
# ============================================================

import pandas as pd

df = pd.read_csv(
    METADATA_PATH
)

train_df = df[
    df["split"] == "train"
    ].copy()

val_df = df[
    df["split"] == "val"
    ].copy()

test_df = df[
    df["split"] == "test"
    ].copy()

print("=" * 70)
print("SEGMENTATION ONLY V2")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print()

print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    f"Train samples: {len(train_df)}"
)

print(
    f"Validation samples: {len(val_df)}"
)

print(
    f"Test samples: {len(test_df)}"
)

# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = BrainTumorDataset(
    train_df,
    image_transform=image_transform,
    mask_transform=mask_transform
)

val_dataset = BrainTumorDataset(
    val_df,
    image_transform=image_transform,
    mask_transform=mask_transform
)

test_dataset = BrainTumorDataset(
    test_df,
    image_transform=image_transform,
    mask_transform=mask_transform
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ============================================================
# DATASET SANITY CHECK
# ============================================================

print()
print("=" * 70)
print("DATASET SANITY CHECK")
print("=" * 70)

train_image, train_label, train_mask = train_dataset[0]

val_image, val_label, val_mask = val_dataset[0]

print()
print("TRAIN SAMPLE")

print(
    "Image shape      :",
    train_image.shape
)

print(
    "Mask shape       :",
    train_mask.shape
)

print(
    "Image dtype      :",
    train_image.dtype
)

print(
    "Mask dtype       :",
    train_mask.dtype
)

print(
    "Mask unique      :",
    torch.unique(train_mask)
)

print(
    "Tumor pixels     :",
    train_mask.sum().item()
)

print()
print("VALIDATION SAMPLE")

print(
    "Image shape      :",
    val_image.shape
)

print(
    "Mask shape       :",
    val_mask.shape
)

print(
    "Image dtype      :",
    val_image.dtype
)

print(
    "Mask dtype       :",
    val_mask.dtype
)

print(
    "Mask unique      :",
    torch.unique(val_mask)
)

print(
    "Tumor pixels     :",
    val_mask.sum().item()
)

# ============================================================
# BATCH CHECK
# ============================================================

images, labels, masks = next(
    iter(train_loader)
)

print()
print("TRAIN BATCH")

print(
    "Images shape     :",
    images.shape
)

print(
    "Labels shape     :",
    labels.shape
)

print(
    "Masks shape      :",
    masks.shape
)

print(
    "Total tumor px   :",
    masks.sum().item()
)

print(
    "Mask unique      :",
    torch.unique(masks)
)

if masks.sum().item() <= 0:
    raise RuntimeError(
        "ERROR: Training batch contains no tumor pixels."
    )

print(
    "PASS: Dataset contains valid tumor masks."
)


# ============================================================
# U-NET BUILDING BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.enc1 = DoubleConv(
            1,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        # ----------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # ----------------------------------------------------
        # POOLING
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):
        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        p1 = self.pool(e1)

        e2 = self.enc2(p1)

        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        p3 = self.pool(e3)

        e4 = self.enc4(p3)

        p4 = self.pool(e4)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(p4)

        # ====================================================
        # DECODER
        # ====================================================

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # ====================================================
        # OUTPUT LOGITS
        # ====================================================

        return self.output(d1)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)

model = UNet()

model = model.to(DEVICE)

print(
    "Model created successfully."
)


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
        logits,
        targets,
        smooth=1.0
):
    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.contiguous()

    targets = targets.contiguous()

    intersection = (
            probabilities * targets
    ).sum(
        dim=(1, 2, 3)
    )

    denominator = (
            probabilities.sum(
                dim=(1, 2, 3)
            )
            +
            targets.sum(
                dim=(1, 2, 3)
            )
    )

    dice = (
            (2.0 * intersection + smooth)
            /
            (denominator + smooth)
    )

    return (
            1.0 - dice.mean()
    )


# ============================================================
# COMBINED LOSS
# ============================================================

bce_criterion = nn.BCEWithLogitsLoss()


def combined_loss(
        logits,
        targets
):
    bce = bce_criterion(
        logits,
        targets
    )

    dice = dice_loss(
        logits,
        targets
    )

    loss = (
            BCE_WEIGHT * bce
            +
            DICE_WEIGHT * dice
    )

    return loss, bce, dice


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def calculate_metrics(
        logits,
        targets,
        threshold=0.5,
        smooth=1e-6
):
    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
            probabilities >= threshold
    ).float()

    intersection = (
            predictions * targets
    ).sum(
        dim=(1, 2, 3)
    )

    prediction_area = predictions.sum(
        dim=(1, 2, 3)
    )

    target_area = targets.sum(
        dim=(1, 2, 3)
    )

    union = (
            prediction_area
            +
            target_area
            -
            intersection
    )

    dice = (
            (2.0 * intersection + smooth)
            /
            (
                    prediction_area
                    +
                    target_area
                    +
                    smooth
            )
    )

    iou = (
            (intersection + smooth)
            /
            (union + smooth)
    )

    return (
        dice.mean().item(),
        iou.mean().item()
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch():
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0

    total_dice = 0.0
    total_iou = 0.0

    batches = 0

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, _, masks in progress:
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

        logits = model(
            images
        )

        loss, bce, dloss = combined_loss(
            logits,
            masks
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        dice, iou = calculate_metrics(
            logits.detach(),
            masks
        )

        total_loss += loss.item()

        total_bce += bce.item()

        total_dice_loss += dloss.item()

        total_dice += dice

        total_iou += iou

        batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{dice:.4f}"
        )

    return (
        total_loss / batches,
        total_bce / batches,
        total_dice_loss / batches,
        total_dice / batches,
        total_iou / batches
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate():
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0

    total_dice = 0.0
    total_iou = 0.0

    batches = 0

    for images, _, masks in val_loader:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            images
        )

        loss, bce, dloss = combined_loss(
            logits,
            masks
        )

        dice, iou = calculate_metrics(
            logits,
            masks
        )

        total_loss += loss.item()

        total_bce += bce.item()

        total_dice_loss += dloss.item()

        total_dice += dice

        total_iou += iou

        batches += 1

    return (
        total_loss / batches,
        total_bce / batches,
        total_dice_loss / batches,
        total_dice / batches,
        total_iou / batches
    )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

best_val_dice = -1.0
best_epoch = 0

epochs_without_improvement = 0

history = []

for epoch in range(
        1,
        EPOCHS + 1
):

    print()
    print(
        f"Epoch {epoch:02d}/{EPOCHS}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    (
        train_loss,
        train_bce,
        train_dloss,
        train_dice,
        train_iou
    ) = train_one_epoch()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        val_loss,
        val_bce,
        val_dloss,
        val_dice,
        val_iou
    ) = validate()

    # --------------------------------------------------------
    # CURRENT LR
    # --------------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        val_dice
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        f"Train Loss : {train_loss:.6f}"
    )

    print(
        f"Train BCE  : {train_bce:.6f}"
    )

    print(
        f"Train DiceL: {train_dloss:.6f}"
    )

    print(
        f"Train Dice : {train_dice:.6f}"
    )

    print(
        f"Train IoU  : {train_iou:.6f}"
    )

    print()

    print(
        f"Val Loss   : {val_loss:.6f}"
    )

    print(
        f"Val BCE    : {val_bce:.6f}"
    )

    print(
        f"Val DiceL  : {val_dloss:.6f}"
    )

    print(
        f"Val Dice   : {val_dice:.6f}"
    )

    print(
        f"Val IoU    : {val_iou:.6f}"
    )

    print(
        f"LR         : {current_lr:.8f}"
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "train_dice": train_dice,

        "train_iou": train_iou,

        "val_loss": val_loss,

        "val_dice": val_dice,

        "val_iou": val_iou,

        "lr": current_lr

    })

    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        best_epoch = epoch

        epochs_without_improvement = 0

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_val_dice":
                best_val_dice,

            "val_iou":
                val_iou,

            "config": {

                "image_size":
                    IMAGE_SIZE,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "dice_weight":
                    DICE_WEIGHT,

                "bce_weight":
                    BCE_WEIGHT

            }

        }

        torch.save(
            checkpoint,
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
            f">>> Val Dice: {val_dice:.6f}"
        )

        print(
            f">>> Path: {BEST_MODEL_PATH}"
        )


    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:
        print()
        print(
            "EARLY STOPPING"
        )

        print(
            f"No improvement for "
            f"{PATIENCE} epochs."
        )

        break

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
    f"Best epoch       : "
    f"{checkpoint['epoch']}"
)

print(
    f"Best Val Dice    : "
    f"{checkpoint['best_val_dice']:.6f}"
)

print(
    f"Model path       : "
    f"{BEST_MODEL_PATH}"
)

# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


@torch.no_grad()
def evaluate_dataset(
        loader,
        threshold=0.5
):
    model.eval()

    sample_dices = []
    sample_ious = []

    total_gt_pixels = 0
    total_pred_pixels = 0

    for images, _, masks in loader:
        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        logits = model(
            images
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
                probabilities >= threshold
        ).float()

        intersection = (
                predictions * masks
        ).sum(
            dim=(1, 2, 3)
        )

        pred_pixels = predictions.sum(
            dim=(1, 2, 3)
        )

        gt_pixels = masks.sum(
            dim=(1, 2, 3)
        )

        union = (
                pred_pixels
                +
                gt_pixels
                -
                intersection
        )

        dice = (
                (2.0 * intersection + 1e-6)
                /
                (
                        pred_pixels
                        +
                        gt_pixels
                        +
                        1e-6
                )
        )

        iou = (
                (intersection + 1e-6)
                /
                (union + 1e-6)
        )

        sample_dices.extend(
            dice.cpu().numpy().tolist()
        )

        sample_ious.extend(
            iou.cpu().numpy().tolist()
        )

        total_gt_pixels += (
            gt_pixels.sum().item()
        )

        total_pred_pixels += (
            pred_pixels.sum().item()
        )

    return (
        float(np.mean(sample_dices)),
        float(np.median(sample_dices)),
        float(np.mean(sample_ious)),
        float(np.median(sample_ious)),
        total_gt_pixels,
        total_pred_pixels
    )


(
    mean_dice,
    median_dice,
    mean_iou,
    median_iou,
    gt_pixels,
    pred_pixels
) = evaluate_dataset(
    val_loader,
    threshold=0.5
)

print()
print(
    f"Mean Dice   : {mean_dice:.6f}"
)

print(
    f"Median Dice : {median_dice:.6f}"
)

print(
    f"Mean IoU    : {mean_iou:.6f}"
)

print(
    f"Median IoU  : {median_iou:.6f}"
)

print()
print(
    f"Ground Truth tumor pixels : "
    f"{gt_pixels:.0f}"
)

print(
    f"Predicted tumor pixels    : "
    f"{pred_pixels:.0f}"
)

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


@torch.no_grad()
def threshold_analysis():
    model.eval()

    # Store probabilities and masks
    all_probabilities = []
    all_masks = []

    for images, _, masks in val_loader:
        images = images.to(
            DEVICE
        )

        logits = model(
            images
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_probabilities.append(
            probabilities.cpu()
        )

        all_masks.append(
            masks.cpu()
        )

    probabilities = torch.cat(
        all_probabilities,
        dim=0
    )

    masks = torch.cat(
        all_masks,
        dim=0
    )

    for threshold in thresholds:
        predictions = (
                probabilities >= threshold
        ).float()

        intersection = (
                predictions * masks
        ).sum(
            dim=(1, 2, 3)
        )

        pred_area = predictions.sum(
            dim=(1, 2, 3)
        )

        gt_area = masks.sum(
            dim=(1, 2, 3)
        )

        union = (
                pred_area
                +
                gt_area
                -
                intersection
        )

        dice = (
                (2.0 * intersection + 1e-6)
                /
                (
                        pred_area
                        +
                        gt_area
                        +
                        1e-6
                )
        )

        iou = (
                (intersection + 1e-6)
                /
                (union + 1e-6)
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Dice: {dice.mean():.4f} | "
            f"IoU: {iou.mean():.4f} | "
            f"Predicted pixels: "
            f"{predictions.sum().item():.0f}"
        )


threshold_analysis()

# ============================================================
# PROBABILITY ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PROBABILITY ANALYSIS")
print("=" * 70)


@torch.no_grad()
def probability_analysis():
    model.eval()

    probabilities_list = []

    for images, _, _ in val_loader:
        images = images.to(
            DEVICE
        )

        logits = model(
            images
        )

        probabilities = torch.sigmoid(
            logits
        )

        probabilities_list.append(
            probabilities.cpu()
        )

    probabilities = torch.cat(
        probabilities_list,
        dim=0
    )

    print(
        f"Probability Min    : "
        f"{probabilities.min().item():.6f}"
    )

    print(
        f"Probability Max    : "
        f"{probabilities.max().item():.6f}"
    )

    print(
        f"Probability Mean   : "
        f"{probabilities.mean().item():.6f}"
    )

    print(
        f"Probability Median : "
        f"{probabilities.median().item():.6f}"
    )


probability_analysis()

# ============================================================
# SAVE SAMPLE PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("SAVING SAMPLE PREDICTIONS")
print("=" * 70)

PREDICTION_DIR = (
        MODEL_DIR
        / "predictions_v2"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

from PIL import Image


@torch.no_grad()
def save_predictions(
        num_samples=10,
        threshold=0.5
):
    model.eval()

    saved = 0

    for images, _, masks in val_loader:

        images = images.to(
            DEVICE
        )

        logits = model(
            images
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
                probabilities >= threshold
        ).float()

        for i in range(
                images.size(0)
        ):

            if saved >= num_samples:
                return

            image = images[i].cpu()

            mask = masks[i].cpu()

            prediction = predictions[i].cpu()

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            image = image.squeeze(0)

            image = (
                    image * 0.5
                    + 0.5
            )

            image = (
                    image.clamp(0, 1)
                    * 255
            ).byte().numpy()

            # ------------------------------------------------
            # GT
            # ------------------------------------------------

            gt = (
                    mask.squeeze(0)
                    .numpy()
                    * 255
            ).astype(
                np.uint8
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            pred = (
                    prediction.squeeze(0)
                    .numpy()
                    * 255
            ).astype(
                np.uint8
            )

            # ------------------------------------------------
            # CREATE RGB VISUALIZATION
            #
            # R = prediction
            # G = ground truth
            # B = image
            # ------------------------------------------------

            visualization = np.zeros(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    3
                ),
                dtype=np.uint8
            )

            visualization[:, :, 0] = pred

            visualization[:, :, 1] = gt

            visualization[:, :, 2] = image

            output_path = (
                    PREDICTION_DIR
                    / f"sample_{saved + 1:03d}.png"
            )

            Image.fromarray(
                visualization
            ).save(
                output_path
            )

            print(
                f"Saved: {output_path}"
            )

            saved += 1


save_predictions(
    num_samples=10,
    threshold=0.5
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("SEGMENTATION V2 COMPLETED")
print("=" * 70)

print(
    f"Best Epoch      : "
    f"{checkpoint['epoch']}"
)

print(
    f"Best Val Dice   : "
    f"{checkpoint['best_val_dice']:.6f}"
)

print(
    f"Final Mean Dice : "
    f"{mean_dice:.6f}"
)

print(
    f"Final Mean IoU  : "
    f"{mean_iou:.6f}"
)

print()
print(
    "Best model saved at:"
)

print(
    BEST_MODEL_PATH
)

print()
print(
    "Prediction samples saved at:"
)

print(
    PREDICTION_DIR
)

print()
print(
    "Process finished."
)
