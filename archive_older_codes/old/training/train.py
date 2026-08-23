from pathlib import Path

import torch
import torch.nn as nn
from dataset import (
    train_loader,
    val_loader
)
from torch.optim import Adam
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 3

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

SEGMENTATION_WEIGHT = 0.5

PATIENCE = 5

MODEL_DIR = Path(
    r"/Models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
        MODEL_DIR / "brain_tumor_multitask_best.pth"
)


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

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 112 -> 56
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 56 -> 28
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 28 -> 28
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),
        )

        # ====================================================
        # CLASSIFICATION HEAD
        # ====================================================

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),

            nn.Flatten(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                NUM_CLASSES
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

            nn.ReLU(inplace=True),

            # 56 -> 112
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            # 112 -> 224
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True),

            # 224 -> 224
            nn.Conv2d(
                16,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):
        features = self.encoder(x)

        class_logits = self.classifier(
            features
        )

        mask_logits = self.segmentation(
            features
        )

        return class_logits, mask_logits


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

        probs = probs.view(-1)

        targets = targets.view(-1)

        intersection = (
                probs * targets
        ).sum()

        dice = (
                       2.0 * intersection
                       + self.smooth
               ) / (
                       probs.sum()
                       + targets.sum()
                       + self.smooth
               )

        return 1.0 - dice


# ============================================================
# MODEL
# ============================================================

model = BrainTumorMultiTaskCNN()

model = model.to(DEVICE)

print("=" * 60)
print("DEVICE")
print("=" * 60)

print(DEVICE)

print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

print(model)

# ============================================================
# LOSSES
# ============================================================

classification_criterion = nn.CrossEntropyLoss()

bce_criterion = nn.BCEWithLogitsLoss()

dice_criterion = DiceLoss()

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# EARLY STOPPING VARIABLES
# ============================================================

best_val_loss = float("inf")

epochs_without_improvement = 0

best_epoch = 0

# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_total_loss = 0.0

    train_classification_loss = 0.0

    train_segmentation_loss = 0.0

    train_correct = 0

    train_total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1:02d}/{EPOCHS}"
    )

    for images, labels, masks in progress:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # ZERO GRADIENT
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        class_logits, mask_logits = model(
            images
        )

        # ----------------------------------------------------
        # CLASSIFICATION LOSS
        # ----------------------------------------------------

        classification_loss = (
            classification_criterion(
                class_logits,
                labels
            )
        )

        # ----------------------------------------------------
        # SEGMENTATION LOSS
        # ----------------------------------------------------

        bce_loss = (
            bce_criterion(
                mask_logits,
                masks
            )
        )

        dice_loss = (
            dice_criterion(
                mask_logits,
                masks
            )
        )

        segmentation_loss = (
                bce_loss + dice_loss
        )

        # ----------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------

        loss = (
                classification_loss
                +
                SEGMENTATION_WEIGHT
                * segmentation_loss
        )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        batch_size = images.size(0)

        train_total_loss += (
                loss.item() * batch_size
        )

        train_classification_loss += (
                classification_loss.item()
                * batch_size
        )

        train_segmentation_loss += (
                segmentation_loss.item()
                * batch_size
        )

        predictions = (
            class_logits.argmax(
                dim=1
            )
        )

        train_correct += (
                predictions == labels
        ).sum().item()

        train_total += batch_size

    # ========================================================
    # TRAIN METRICS
    # ========================================================

    train_loss = (
            train_total_loss
            /
            train_total
    )

    train_cls_loss = (
            train_classification_loss
            /
            train_total
    )

    train_seg_loss = (
            train_segmentation_loss
            /
            train_total
    )

    train_accuracy = (
            train_correct
            /
            train_total
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_total_loss = 0.0

    val_classification_loss = 0.0

    val_segmentation_loss = 0.0

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        for images, labels, masks in val_loader:
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            class_logits, mask_logits = model(
                images
            )

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            classification_loss = (
                classification_criterion(
                    class_logits,
                    labels
                )
            )

            # ------------------------------------------------
            # SEGMENTATION
            # ------------------------------------------------

            bce_loss = (
                bce_criterion(
                    mask_logits,
                    masks
                )
            )

            dice_loss = (
                dice_criterion(
                    mask_logits,
                    masks
                )
            )

            segmentation_loss = (
                    bce_loss + dice_loss
            )

            # ------------------------------------------------
            # TOTAL
            # ------------------------------------------------

            loss = (
                    classification_loss
                    +
                    SEGMENTATION_WEIGHT
                    * segmentation_loss
            )

            batch_size = images.size(0)

            val_total_loss += (
                    loss.item()
                    * batch_size
            )

            val_classification_loss += (
                    classification_loss.item()
                    * batch_size
            )

            val_segmentation_loss += (
                    segmentation_loss.item()
                    * batch_size
            )

            predictions = (
                class_logits.argmax(
                    dim=1
                )
            )

            val_correct += (
                    predictions == labels
            ).sum().item()

            val_total += batch_size

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    val_loss = (
            val_total_loss
            /
            val_total
    )

    val_cls_loss = (
            val_classification_loss
            /
            val_total
    )

    val_seg_loss = (
            val_segmentation_loss
            /
            val_total
    )

    val_accuracy = (
            val_correct
            /
            val_total
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"\nEpoch {epoch + 1:02d}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Classification Loss: "
        f"{train_cls_loss:.4f}"
    )

    print(
        f"Train Segmentation Loss: "
        f"{train_seg_loss:.4f}"
    )

    print(
        f"Train Acc: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Classification Loss: "
        f"{val_cls_loss:.4f}"
    )

    print(
        f"Val Segmentation Loss: "
        f"{val_seg_loss:.4f}"
    )

    print(
        f"Val Acc: {val_accuracy:.4f}"
    )

    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch + 1

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy
            },
            BEST_MODEL_PATH
        )

        print(
            "\n>>> BEST MODEL SAVED"
        )

        print(
            f">>> Epoch: {epoch + 1}"
        )

        print(
            f">>> Val Loss: {val_loss:.4f}"
        )


    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= PATIENCE:
        print(
            "\n" + "=" * 60
        )

        print(
            "EARLY STOPPING"
        )

        print(
            "=" * 60
        )

        print(
            f"Best Epoch: {best_epoch}"
        )

        print(
            f"Best Validation Loss: "
            f"{best_val_loss:.4f}"
        )

        break

# ============================================================
# FINISHED
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRAINING FINISHED"
)

print(
    "=" * 60
)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best model saved at:"
)

print(
    BEST_MODEL_PATH
)
