from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================================
# CONFIG
# ============================================================

METADATA_PATH = Path(
    r"/Data/metadata.csv"
)

CHECKPOINT_PATH = Path(
    r"/Models/brain_tumor_multitask_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32

NUM_CLASSES = 3

CLASS_NAMES = [
    "Meningioma",
    "Glioma",
    "Pituitary"
]

# Thresholds for segmentation analysis
SEGMENTATION_THRESHOLDS = [
    0.50,
    0.30,
    0.20,
    0.15,
    0.10,
    0.05
]

# Number of samples to visualize
NUM_VISUALIZATION_SAMPLES = 10

OUTPUT_DIR = Path(
    r"D:\Portfolio\NeuroSense\Models\brain_tumor_multitask\evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("DEVICE")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([

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

    def __getitem__(
            self,
            index
    ):
        row = self.df.iloc[index]

        path = row["path"]

        # ----------------------------------------------------
        # Read HDF5 / MATLAB v7.3
        # ----------------------------------------------------

        with h5py.File(
                path,
                "r"
        ) as f:
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
        # IMAGE TRANSFORM
        # ----------------------------------------------------

        if self.transform:
            image = self.transform(
                image
            )

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask = mask.astype(
            "float32"
        )

        mask = Image.fromarray(
            mask
        )

        mask = mask.resize(
            (224, 224),
            resample=Image.Resampling.NEAREST
        )

        mask = np.array(
            mask
        ).astype(
            "float32"
        )

        mask = torch.from_numpy(
            mask
        )

        mask = mask.unsqueeze(
            0
        )

        mask = (
                mask > 0.5
        ).float()

        # ----------------------------------------------------
        # LABEL
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
# LOAD METADATA
# ============================================================

print()
print("=" * 70)
print("LOADING METADATA")
print("=" * 70)

df = pd.read_csv(
    METADATA_PATH
)

train_df = df[
    df["split"] == "train"
    ]

val_df = df[
    df["split"] == "val"
    ]

test_df = df[
    df["split"] == "test"
    ]

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)

# ============================================================
# DATASET
# ============================================================

test_dataset = BrainTumorDataset(
    test_df,
    transform=transform
)

# ============================================================
# DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

class BrainTumorMultiTaskCNN(
    nn.Module
):

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

            nn.BatchNorm2d(
                16
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            # 112 -> 56

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            # 56 -> 28

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            # 28 -> 28

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

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

            nn.BatchNorm2d(
                64
            ),

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

            nn.BatchNorm2d(
                32
            ),

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

            nn.BatchNorm2d(
                16
            ),

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

    def forward(
            self,
            x
    ):
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
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model = BrainTumorMultiTaskCNN()

model = model.to(
    DEVICE
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

# ------------------------------------------------------------
# Checkpoint format
# ------------------------------------------------------------

if (
        isinstance(
            checkpoint,
            dict
        )
        and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "N/A"
        )
    )

    print(
        "Checkpoint validation loss:",
        checkpoint.get(
            "val_loss",
            "N/A"
        )
    )

    print(
        "Checkpoint validation accuracy:",
        checkpoint.get(
            "val_accuracy",
            "N/A"
        )
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print()
print(
    "Model loaded successfully."
)


# ============================================================
# DICE
# ============================================================

def dice_score(
        prediction,
        target,
        smooth=1e-6
):
    prediction = prediction.float()
    target = target.float()

    intersection = (
            prediction * target
    ).sum()

    denominator = (
            prediction.sum()
            + target.sum()
    )

    dice = (
                   2.0 * intersection
                   + smooth
           ) / (
                   denominator
                   + smooth
           )

    return dice.item()


# ============================================================
# IOU
# ============================================================

def iou_score(
        prediction,
        target,
        smooth=1e-6
):
    prediction = prediction.float()
    target = target.float()

    intersection = (
            prediction * target
    ).sum()

    union = (
            prediction
            + target
            - prediction * target
    ).sum()

    iou = (
                  intersection
                  + smooth
          ) / (
                  union
                  + smooth
          )

    return iou.item()


# ============================================================
# EVALUATION STORAGE
# ============================================================

all_predictions = []
all_targets = []

all_dice = {
    threshold: []
    for threshold in SEGMENTATION_THRESHOLDS
}

all_iou = {
    threshold: []
    for threshold in SEGMENTATION_THRESHOLDS
}

# ============================================================
# GLOBAL SEGMENTATION STATISTICS
# ============================================================

all_probability_values = []

total_predicted_pixels = {
    threshold: 0
    for threshold in SEGMENTATION_THRESHOLDS
}

total_ground_truth_pixels = 0

# ============================================================
# VISUALIZATION COUNTER
# ============================================================

visualization_count = 0

# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 70)
print("EVALUATING TEST SET")
print("=" * 70)

with torch.no_grad():
    for batch_index, (
            images,
            labels,
            masks
    ) in enumerate(test_loader):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        class_logits, mask_logits = model(
            images
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        class_predictions = torch.argmax(
            class_logits,
            dim=1
        )

        all_predictions.extend(
            class_predictions.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

        # ----------------------------------------------------
        # Segmentation probabilities
        # ----------------------------------------------------

        mask_probabilities = torch.sigmoid(
            mask_logits
        )

        # ----------------------------------------------------
        # Probability statistics
        # ----------------------------------------------------

        all_probability_values.append(
            mask_probabilities
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        # ----------------------------------------------------
        # Ground truth pixels
        # ----------------------------------------------------

        total_ground_truth_pixels += (
            masks.sum()
            .item()
        )

        # ----------------------------------------------------
        # Segmentation metrics
        # ----------------------------------------------------

        for threshold in SEGMENTATION_THRESHOLDS:

            predicted_masks = (
                    mask_probabilities
                    >= threshold
            ).float()

            total_predicted_pixels[
                threshold
            ] += (
                predicted_masks
                .sum()
                .item()
            )

            # -----------------------------------------------
            # Per-image Dice / IoU
            # -----------------------------------------------

            for i in range(
                    images.size(0)
            ):
                pred = predicted_masks[
                    i
                ]

                target = masks[
                    i
                ]

                dice = dice_score(
                    pred,
                    target
                )

                iou = iou_score(
                    pred,
                    target
                )

                all_dice[
                    threshold
                ].append(
                    dice
                )

                all_iou[
                    threshold
                ].append(
                    iou
                )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        if (
                visualization_count
                < NUM_VISUALIZATION_SAMPLES
        ):

            for i in range(
                    images.size(0)
            ):

                if (
                        visualization_count
                        >= NUM_VISUALIZATION_SAMPLES
                ):
                    break

                image = images[
                    i
                ].cpu().numpy()[0]

                target_mask = masks[
                    i
                ].cpu().numpy()[0]

                probability = mask_probabilities[
                    i
                ].cpu().numpy()[0]

                prediction_mask = (
                        probability
                        >= 0.5
                ).astype(
                    np.float32
                )

                # Undo normalization
                image_display = (
                        image * 0.5
                        + 0.5
                )

                image_display = np.clip(
                    image_display,
                    0,
                    1
                )

                # ------------------------------------------------
                # Save image
                # ------------------------------------------------

                fig, axes = plt.subplots(
                    1,
                    4,
                    figsize=(16, 4)
                )

                axes[0].imshow(
                    image_display,
                    cmap="gray"
                )

                axes[0].set_title(
                    "MRI"
                )

                axes[0].axis(
                    "off"
                )

                axes[1].imshow(
                    target_mask,
                    cmap="gray"
                )

                axes[1].set_title(
                    "Ground Truth"
                )

                axes[1].axis(
                    "off"
                )

                axes[2].imshow(
                    probability,
                    cmap="hot",
                    vmin=0,
                    vmax=1
                )

                axes[2].set_title(
                    "Prediction Probability"
                )

                axes[2].axis(
                    "off"
                )

                axes[3].imshow(
                    prediction_mask,
                    cmap="gray"
                )

                axes[3].set_title(
                    "Prediction >= 0.5"
                )

                axes[3].axis(
                    "off"
                )

                plt.tight_layout()

                output_path = (
                        OUTPUT_DIR
                        / f"sample_{visualization_count + 1}.png"
                )

                plt.savefig(
                    output_path,
                    dpi=150,
                    bbox_inches="tight"
                )

                plt.close()

                visualization_count += 1

# ============================================================
# PROBABILITY STATISTICS
# ============================================================

all_probability_values = np.concatenate(
    all_probability_values
)

print()
print("=" * 70)
print("SEGMENTATION PROBABILITY ANALYSIS")
print("=" * 70)

print(
    "Min:",
    float(
        np.min(
            all_probability_values
        )
    )
)

print(
    "Max:",
    float(
        np.max(
            all_probability_values
        )
    )
)

print(
    "Mean:",
    float(
        np.mean(
            all_probability_values
        )
    )
)

print(
    "Median:",
    float(
        np.median(
            all_probability_values
        )
    )
)

# ============================================================
# PIXEL STATISTICS
# ============================================================

print()
print("=" * 70)
print("SEGMENTATION PIXEL STATISTICS")
print("=" * 70)

print(
    "Ground truth tumor pixels:",
    total_ground_truth_pixels
)

total_pixels = (
        len(test_dataset)
        * 224
        * 224
)

ground_truth_ratio = (
        total_ground_truth_pixels
        / total_pixels
)

print(
    "Ground truth tumor ratio:",
    ground_truth_ratio
)

for threshold in SEGMENTATION_THRESHOLDS:
    predicted_pixels = (
        total_predicted_pixels[
            threshold
        ]
    )

    predicted_ratio = (
            predicted_pixels
            / total_pixels
    )

    print()

    print(
        f"Threshold >= {threshold:.2f}"
    )

    print(
        "Predicted tumor pixels:",
        predicted_pixels
    )

    print(
        "Predicted tumor ratio:",
        predicted_ratio
    )

# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

precision = precision_score(
    all_targets,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_targets,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_targets,
    all_predictions,
    average="weighted",
    zero_division=0
)

print()
print("=" * 70)
print("CLASSIFICATION RESULTS")
print("=" * 70)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()

print(
    classification_report(
        all_targets,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_targets,
    all_predictions
)

print(
    "Confusion Matrix:"
)

print(
    cm
)

# ============================================================
# SEGMENTATION RESULTS BY THRESHOLD
# ============================================================

print()
print("=" * 70)
print("SEGMENTATION RESULTS BY THRESHOLD")
print("=" * 70)

for threshold in SEGMENTATION_THRESHOLDS:
    dice_values = np.array(
        all_dice[
            threshold
        ]
    )

    iou_values = np.array(
        all_iou[
            threshold
        ]
    )

    mean_dice = np.mean(
        dice_values
    )

    median_dice = np.median(
        dice_values
    )

    mean_iou = np.mean(
        iou_values
    )

    median_iou = np.median(
        iou_values
    )

    print()

    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        f"Mean Dice   : {mean_dice:.4f}"
    )

    print(
        f"Median Dice : {median_dice:.4f}"
    )

    print(
        f"Mean IoU    : {mean_iou:.4f}"
    )

    print(
        f"Median IoU  : {median_iou:.4f}"
    )

# ============================================================
# BEST THRESHOLD
# ============================================================

best_threshold = max(
    SEGMENTATION_THRESHOLDS,
    key=lambda t: np.mean(
        all_dice[t]
    )
)

best_dice = np.mean(
    all_dice[
        best_threshold
    ]
)

best_iou = np.mean(
    all_iou[
        best_threshold
    ]
)

print()
print("=" * 70)
print("BEST SEGMENTATION THRESHOLD")
print("=" * 70)

print(
    "Best threshold:",
    best_threshold
)

print(
    "Best Mean Dice:",
    best_dice
)

print(
    "Best Mean IoU:",
    best_iou
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Classification Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Classification F1       : {f1:.4f}"
)

print(
    f"Best Segmentation Dice : {best_dice:.4f}"
)

print(
    f"Best Segmentation IoU  : {best_iou:.4f}"
)

print()

print(
    "Visualization samples saved to:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Evaluation completed successfully."
)

print("=" * 70)
