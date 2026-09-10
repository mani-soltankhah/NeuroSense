from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================================
# CONFIG
# ============================================================

METADATA_PATH = Path(
    r"/Data/metadata.csv"
)

NUM_SAMPLES = 10

NUM_EPOCHS = 200

BATCH_SIZE = NUM_SAMPLES

LEARNING_RATE = 1e-3

DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.5

CLASSIFICATION_WEIGHT = 1.0
SEGMENTATION_WEIGHT = 1.0

THRESHOLD = 0.5

MODEL_SAVE_PATH = Path(
    r"/Models/brain_tumor_multitask_overfit_best.pth"
)

SEED = 42

# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MULTITASK MODEL OVERFIT TEST")
print("=" * 70)

print("Device:", device)
print("Number of samples:", NUM_SAMPLES)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)

# ============================================================
# TRANSFORMS
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# DATASET
# ============================================================

class BrainTumorDataset(Dataset):

    def __init__(
            self,
            dataframe,
            transform=None
    ):
        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        path = row["path"]

        # ----------------------------------------------------
        # READ HDF5
        # ----------------------------------------------------

        with h5py.File(path, "r") as f:
            image = f["cjdata"]["image"][()]

            mask = f["cjdata"]["tumorMask"][()]

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = image.astype(
            "float32"
        )

        image_min = image.min()
        image_max = image.max()

        image = (
                255.0
                * (image - image_min)
                / (
                        image_max
                        - image_min
                        + 1e-8
                )
        )

        image = image.astype(
            "uint8"
        )

        image = Image.fromarray(
            image
        )

        # ----------------------------------------------------
        # MASK
        #
        # IMPORTANT:
        # NEAREST interpolation must be used.
        # Otherwise 0/1 mask values become gray values.
        # ----------------------------------------------------

        mask = Image.fromarray(
            mask.astype("uint8")
        )

        mask = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.NEAREST
        )(mask)

        mask = np.array(
            mask,
            dtype=np.float32
        )

        mask = (
                mask > 0
        ).astype(
            np.float32
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        # ----------------------------------------------------
        # IMAGE TRANSFORM
        # ----------------------------------------------------

        if self.transform:
            image = self.transform(
                image
            )

        # ----------------------------------------------------
        # LABEL
        #
        # Original:
        # 1 = Meningioma
        # 2 = Glioma
        # 3 = Pituitary
        #
        # PyTorch:
        # 0 = Meningioma
        # 1 = Glioma
        # 2 = Pituitary
        # ----------------------------------------------------

        label = int(
            row["label"]
        ) - 1

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return (
            image,
            label,
            mask
        )


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
            self,
            smooth=1.0
    ):
        super().__init__()

        self.smooth = smooth

    def forward(
            self,
            logits,
            targets
    ):
        probs = torch.sigmoid(
            logits
        )

        probs = probs.view(
            probs.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
                probs * targets
        ).sum(
            dim=1
        )

        dice = (
                       2.0 * intersection
                       + self.smooth
               ) / (
                       probs.sum(dim=1)
                       + targets.sum(dim=1)
                       + self.smooth
               )

        return (
                1.0 - dice.mean()
        )


# ============================================================
# DICE METRIC
# ============================================================

def dice_score(
        logits,
        targets,
        threshold=0.5
):
    probs = torch.sigmoid(
        logits
    )

    predictions = (
            probs >= threshold
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
    ).sum(
        dim=1
    )

    dice = (
                   2.0 * intersection
           ) / (
                   predictions.sum(dim=1)
                   + targets.sum(dim=1)
                   + 1e-8
           )

    return dice.mean().item()


# ============================================================
# IOU METRIC
# ============================================================

def iou_score(
        logits,
        targets,
        threshold=0.5
):
    probs = torch.sigmoid(
        logits
    )

    predictions = (
            probs >= threshold
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
    ).sum(
        dim=1
    )

    union = (
            predictions
            + targets
            - predictions * targets
    ).sum(
        dim=1
    )

    iou = (
            intersection
            / (
                    union
                    + 1e-8
            )
    )

    return iou.mean().item()


# ============================================================
# MODEL
# ============================================================

class BrainTumorMultiTaskCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder = nn.Sequential(

            # 224 -> 112
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(2),

            # 112 -> 56
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(2),

            # 56 -> 28
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(2),

            # 28 -> 28
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(
                inplace=True
            )
        )

        # ====================================================
        # CLASSIFICATION HEAD
        # ====================================================

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),

            nn.Flatten(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                128,
                3
            )
        )

        # ====================================================
        # SEGMENTATION HEAD
        # ====================================================

        self.segmentation = nn.Sequential(

            # 28 -> 56
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),

            # 56 -> 112
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),

            # 112 -> 224
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(
                inplace=True
            ),

            # 224 -> 224
            nn.Conv2d(
                16,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):
        features = self.encoder(
            x
        )

        class_logits = self.classifier(
            features
        )

        mask_logits = self.segmentation(
            features
        )

        return (
            class_logits,
            mask_logits
        )


# ============================================================
# LOAD METADATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(
    METADATA_PATH
)

train_df = df[
    df["split"] == "train"
    ].copy()

print(
    "Total train samples:",
    len(train_df)
)

# ============================================================
# SELECT FIXED 10 SAMPLES
# ============================================================

# We intentionally select fixed samples.
# No random sampling here.

overfit_df = train_df.head(
    NUM_SAMPLES
).copy()

print(
    "Selected samples:",
    len(overfit_df)
)

print()
print(
    "Labels:"
)

print(
    overfit_df["label"].tolist()
)

# ============================================================
# CREATE DATASET
# ============================================================

dataset = BrainTumorDataset(
    overfit_df,
    transform=image_transform
)

# ============================================================
# CREATE DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# DATASET SANITY CHECK
# ============================================================

print()
print("=" * 70)
print("DATASET SANITY CHECK")
print("=" * 70)

images, labels, masks = next(
    iter(loader)
)

print(
    "Images shape:",
    images.shape
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Masks shape:",
    masks.shape
)

print(
    "Image dtype:",
    images.dtype
)

print(
    "Mask dtype:",
    masks.dtype
)

print(
    "Mask unique values:",
    torch.unique(masks)
)

print(
    "Total GT tumor pixels:",
    masks.sum().item()
)

if masks.sum().item() == 0:
    raise RuntimeError(
        "ERROR: Selected 10 samples contain zero tumor pixels!"
    )

# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)

model = BrainTumorMultiTaskCNN().to(
    device
)

print(
    "Model created successfully."
)

# ============================================================
# LOSSES
# ============================================================

classification_loss_fn = (
    nn.CrossEntropyLoss()
)

bce_loss_fn = (
    nn.BCEWithLogitsLoss()
)

dice_loss_fn = DiceLoss()

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING OVERFIT TEST")
print("=" * 70)

best_dice = -1.0

for epoch in range(
        1,
        NUM_EPOCHS + 1
):

    model.train()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    images = images.to(
        device
    )

    labels = labels.to(
        device
    )

    masks = masks.to(
        device
    )

    class_logits, mask_logits = model(
        images
    )

    # --------------------------------------------------------
    # Classification loss
    # --------------------------------------------------------

    classification_loss = (
        classification_loss_fn(
            class_logits,
            labels
        )
    )

    # --------------------------------------------------------
    # Segmentation losses
    # --------------------------------------------------------

    bce_loss = (
        bce_loss_fn(
            mask_logits,
            masks
        )
    )

    dice_loss = (
        dice_loss_fn(
            mask_logits,
            masks
        )
    )

    segmentation_loss = (
            BCE_WEIGHT * bce_loss
            +
            DICE_WEIGHT * dice_loss
    )

    # --------------------------------------------------------
    # Total loss
    # --------------------------------------------------------

    total_loss = (
            CLASSIFICATION_WEIGHT
            * classification_loss
            +
            SEGMENTATION_WEIGHT
            * segmentation_loss
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    optimizer.zero_grad()

    total_loss.backward()

    optimizer.step()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    with torch.no_grad():

        predictions = torch.argmax(
            class_logits,
            dim=1
        )

        classification_accuracy = (
            (
                    predictions
                    == labels
            )
            .float()
            .mean()
            .item()
        )

        dice = dice_score(
            mask_logits,
            masks,
            threshold=THRESHOLD
        )

        iou = iou_score(
            mask_logits,
            masks,
            threshold=THRESHOLD
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    if (
            epoch == 1
            or epoch % 10 == 0
    ):
        print(
            f"\nEpoch {epoch:03d}/{NUM_EPOCHS}"
        )

        print(
            f"Total Loss          : {total_loss.item():.6f}"
        )

        print(
            f"Classification Loss  : {classification_loss.item():.6f}"
        )

        print(
            f"BCE Loss             : {bce_loss.item():.6f}"
        )

        print(
            f"Dice Loss            : {dice_loss.item():.6f}"
        )

        print(
            f"Classification Acc   : "
            f"{classification_accuracy:.4f}"
        )

        print(
            f"Segmentation Dice    : "
            f"{dice:.4f}"
        )

        print(
            f"Segmentation IoU     : "
            f"{iou:.4f}"
        )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if dice > best_dice:
        best_dice = dice

        MODEL_SAVE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dice": dice,
                "iou": iou,
                "classification_accuracy":
                    classification_accuracy
            },
            MODEL_SAVE_PATH
        )

# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST OVERFIT MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_SAVE_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Best Dice:",
    checkpoint["dice"]
)

print(
    "Best IoU:",
    checkpoint["iou"]
)

print(
    "Best Classification Accuracy:",
    checkpoint[
        "classification_accuracy"
    ]
)

# ============================================================
# FINAL EVALUATION
# ============================================================

with torch.no_grad():
    class_logits, mask_logits = model(
        images
    )

    class_predictions = torch.argmax(
        class_logits,
        dim=1
    )

    class_accuracy = (
        (
                class_predictions
                == labels
        )
        .float()
        .mean()
        .item()
    )

print()
print("=" * 70)
print("FINAL OVERFIT RESULTS")
print("=" * 70)

print(
    f"Classification Accuracy : "
    f"{class_accuracy * 100:.2f}%"
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

for threshold in thresholds:
    dice = dice_score(
        mask_logits,
        masks,
        threshold=threshold
    )

    iou = iou_score(
        mask_logits,
        masks,
        threshold=threshold
    )

    with torch.no_grad():
        probabilities = torch.sigmoid(
            mask_logits
        )

        predicted_pixels = (
                probabilities
                >= threshold
        ).sum().item()

    print(
        f"Threshold {threshold:.2f} | "
        f"Dice: {dice:.4f} | "
        f"IoU: {iou:.4f} | "
        f"Predicted pixels: {predicted_pixels}"
    )

# ============================================================
# PROBABILITY ANALYSIS
# ============================================================

with torch.no_grad():
    probabilities = torch.sigmoid(
        mask_logits
    )

print()
print("=" * 70)
print("PROBABILITY ANALYSIS")
print("=" * 70)

print(
    "Probability Min    :",
    probabilities.min().item()
)

print(
    "Probability Max    :",
    probabilities.max().item()
)

print(
    "Probability Mean   :",
    probabilities.mean().item()
)

print(
    "Probability Median :",
    probabilities.median().item()
)

# ============================================================
# PIXEL ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PIXEL ANALYSIS")
print("=" * 70)

ground_truth_pixels = (
    masks.sum().item()
)

predicted_pixels_05 = (
        probabilities >= 0.5
).sum().item()

print(
    "Ground Truth tumor pixels:",
    ground_truth_pixels
)

print(
    "Predicted tumor pixels @ 0.5:",
    predicted_pixels_05
)

print(
    "Ground Truth tumor ratio:",
    ground_truth_pixels
    / masks.numel()
)

print(
    "Predicted tumor ratio @ 0.5:",
    predicted_pixels_05
    / masks.numel()
)

# ============================================================
# PER-SAMPLE DICE
# ============================================================

print()
print("=" * 70)
print("PER-SAMPLE DICE")
print("=" * 70)

with torch.no_grad():
    probabilities = torch.sigmoid(
        mask_logits
    )

    predictions = (
            probabilities >= 0.5
    ).float()

for i in range(
        NUM_SAMPLES
):
    prediction = predictions[
                 i:i + 1
                 ]

    target = masks[
             i:i + 1
             ]

    prediction_flat = (
        prediction.view(-1)
    )

    target_flat = (
        target.view(-1)
    )

    intersection = (
            prediction_flat
            * target_flat
    ).sum()

    dice = (
                   2.0 * intersection
           ) / (
                   prediction_flat.sum()
                   + target_flat.sum()
                   + 1e-8
           )

    print(
        f"Sample {i + 1:02d} | "
        f"GT pixels: {int(target.sum().item()):6d} | "
        f"Pred pixels: {int(prediction.sum().item()):6d} | "
        f"Dice: {dice.item():.4f}"
    )

# ============================================================
# FINAL INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("OVERFIT TEST INTERPRETATION")
print("=" * 70)

if best_dice >= 0.80:

    print(
        "PASS:"
    )

    print(
        "The model CAN memorize the segmentation masks."
    )

    print(
        "The architecture/loss pipeline is capable "
        "of learning segmentation."
    )

    print(
        "The next problem is likely generalization."
    )


elif best_dice >= 0.50:

    print(
        "PARTIAL:"
    )

    print(
        "The model is learning segmentation,"
    )

    print(
        "but not strongly enough."
    )

    print(
        "Architecture/loss/optimization likely "
        "needs improvement."
    )


else:

    print(
        "FAIL:"
    )

    print(
        "The model cannot properly overfit "
        "10 training samples."
    )

    print(
        "This strongly suggests a problem with "
        "architecture, loss, optimization, "
        "or the training pipeline."
    )

print()
print(
    "Best model saved at:"
)

print(
    MODEL_SAVE_PATH
)

print()
print("=" * 70)
print("OVERFIT TEST COMPLETED")
print("=" * 70)
