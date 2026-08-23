import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ============================================================
# CONFIG
# ============================================================

METADATA_PATH = Path(
    r"/Data/metadata.csv"
)

MODEL_PATH = Path(
    r"D:\Portfolio\NeuroSense\Models\best_model.pth"
)

NUM_PATIENTS = 20
MAX_SLICES_PER_PATIENT = 10

IMAGE_SIZE = 224

# Grad-CAM threshold
CAM_THRESHOLD = 0.5

SEED = 42

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Meningioma",
    1: "Glioma",
    2: "Pituitary"
}


# ============================================================
# MODEL
# ============================================================

class BrainTumorCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 224x224
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 112x112
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 56x56
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 28x28
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

model = BrainTumorCNN().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded.")

# ============================================================
# TARGET LAYER
# ============================================================

# Last convolutional layer
target_layer = model.features[12]

print(
    "Grad-CAM target layer:",
    target_layer
)

# ============================================================
# METADATA
# ============================================================

df = pd.read_csv(METADATA_PATH)

test_df = df[
    df["split"] == "test"
    ].copy()

print(
    "Test images:",
    len(test_df)
)

print(
    "Test patients:",
    test_df["patient_id"].nunique()
)

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# IMAGE LOADER
# ============================================================

def load_sample(path):
    with h5py.File(path, "r") as f:
        cjdata = f["cjdata"]

        image = cjdata["image"][()]
        mask = cjdata["tumorMask"][()]

    # --------------------------------------------------------
    # Image normalization
    # --------------------------------------------------------

    image = image.astype(np.float32)

    image_min = image.min()
    image_max = image.max()

    image = (
            255
            * (image - image_min)
            / (image_max - image_min + 1e-8)
    )

    image = image.astype(np.uint8)

    # --------------------------------------------------------
    # PIL image
    # --------------------------------------------------------

    pil_image = Image.fromarray(image)

    image_tensor = transform(
        pil_image
    )

    # --------------------------------------------------------
    # Resize mask
    # --------------------------------------------------------

    mask_image = Image.fromarray(
        mask.astype(np.uint8)
    )

    mask_image = mask_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask = np.array(
        mask_image
    )

    mask = (
            mask > 0
    ).astype(np.uint8)

    return image_tensor, mask


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
        model,
        image,
        target_class
):
    activations = None
    gradients = None

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def forward_hook(
            module,
            input,
            output
    ):
        nonlocal activations

        activations = output

    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def backward_hook(
            module,
            grad_input,
            grad_output
    ):
        nonlocal gradients

        gradients = grad_output[0]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    model.zero_grad()

    output = model(image)

    score = output[
        0,
        target_class
    ]

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    score.backward()

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    forward_handle.remove()
    backward_handle.remove()

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    cam = (
            weights * activations
    ).sum(
        dim=1,
        keepdim=True
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = F.relu(cam)

    # --------------------------------------------------------
    # Resize to image size
    # --------------------------------------------------------

    cam = F.interpolate(
        cam,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        mode="bilinear",
        align_corners=False
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cam = cam[0, 0]

    cam_min = cam.min()
    cam_max = cam.max()

    cam = (
                  cam - cam_min
          ) / (
                  cam_max - cam_min + 1e-8
          )

    return cam.detach().cpu().numpy()


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
        cam,
        tumor_mask,
        threshold=0.5
):
    cam_mask = (
            cam >= threshold
    ).astype(np.uint8)

    tumor_mask = (
            tumor_mask > 0
    ).astype(np.uint8)

    intersection = np.logical_and(
        cam_mask,
        tumor_mask
    ).sum()

    union = np.logical_or(
        cam_mask,
        tumor_mask
    ).sum()

    cam_pixels = cam_mask.sum()
    tumor_pixels = tumor_mask.sum()

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if union == 0:
        iou = 0.0
    else:
        iou = (
                intersection
                / union
        )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    denominator = (
            cam_pixels
            + tumor_pixels
    )

    if denominator == 0:
        dice = 0.0
    else:
        dice = (
                2 * intersection
                / denominator
        )

    # --------------------------------------------------------
    # Tumor coverage
    #
    # How much of the actual tumor
    # was inside Grad-CAM?
    # --------------------------------------------------------

    if tumor_pixels == 0:
        coverage = 0.0
    else:
        coverage = (
                intersection
                / tumor_pixels
        )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "coverage": float(coverage),
        "cam_pixels": int(cam_pixels),
        "tumor_pixels": int(tumor_pixels),
        "intersection": int(intersection)
    }


# ============================================================
# SELECT PATIENTS
# ============================================================

patients = (
    test_df[
        "patient_id"
    ]
    .drop_duplicates()
    .tolist()
)

random.shuffle(
    patients
)

selected_patients = patients[
                    :NUM_PATIENTS
                    ]

print(
    "\nSelected patients:",
    len(selected_patients)
)

# ============================================================
# EVALUATION
# ============================================================

results = []

correct_samples = 0
total_samples = 0

for patient_id in selected_patients:

    patient_df = test_df[
        test_df["patient_id"]
        == patient_id
        ].copy()

    # --------------------------------------------------------
    # Randomly select slices if patient has too many
    # --------------------------------------------------------

    if len(patient_df) > MAX_SLICES_PER_PATIENT:
        patient_df = patient_df.sample(
            MAX_SLICES_PER_PATIENT,
            random_state=SEED
        )

    for _, row in patient_df.iterrows():

        path = row["path"]

        true_label = (
                int(row["label"]) - 1
        )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        image, tumor_mask = load_sample(
            path
        )

        image = image.unsqueeze(
            0
        ).to(device)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                image
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = (
                output.argmax(
                    dim=1
                ).item()
            )

            confidence = probabilities[
                0,
                prediction
            ].item()

        total_samples += 1

        # ----------------------------------------------------
        # Only evaluate correctly classified images
        # ----------------------------------------------------

        if prediction != true_label:
            continue

        correct_samples += 1

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        cam = generate_gradcam(
            model,
            image,
            prediction
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            cam,
            tumor_mask,
            threshold=CAM_THRESHOLD
        )

        results.append({

            "patient_id": patient_id,

            "path": path,

            "true_label":
                CLASS_NAMES[true_label],

            "prediction":
                CLASS_NAMES[prediction],

            "confidence":
                confidence,

            "iou":
                metrics["iou"],

            "dice":
                metrics["dice"],

            "coverage":
                metrics["coverage"],

            "cam_pixels":
                metrics["cam_pixels"],

            "tumor_pixels":
                metrics["tumor_pixels"],

            "intersection":
                metrics["intersection"]

        })

# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    "\n" + "=" * 60
)

print(
    "GRAD-CAM EVALUATION RESULTS"
)

print(
    "=" * 60
)

print(
    "Total evaluated slices:",
    total_samples
)

print(
    "Correctly classified:",
    correct_samples
)

print(
    "Grad-CAM evaluated:",
    len(results_df)
)

if len(results_df) > 0:

    print(
        "\nMean IoU:",
        f"{results_df['iou'].mean():.4f}"
    )

    print(
        "Median IoU:",
        f"{results_df['iou'].median():.4f}"
    )

    print(
        "Std IoU:",
        f"{results_df['iou'].std():.4f}"
    )

    print(
        "\nMean Dice:",
        f"{results_df['dice'].mean():.4f}"
    )

    print(
        "Median Dice:",
        f"{results_df['dice'].median():.4f}"
    )

    print(
        "Std Dice:",
        f"{results_df['dice'].std():.4f}"
    )

    print(
        "\nMean Tumor Coverage:",
        f"{results_df['coverage'].mean():.4f}"
    )

    print(
        "Median Tumor Coverage:",
        f"{results_df['coverage'].median():.4f}"
    )

    # --------------------------------------------------------
    # By class
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "BY CLASS"
    )

    print(
        "=" * 60
    )

    class_results = (
        results_df
        .groupby("true_label")
        [
            [
                "iou",
                "dice",
                "coverage"
            ]
        ]
        .mean()
    )

    print(
        class_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = Path(
        r"/gradcam/gradcam_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_path
    )

else:

    print(
        "\nNo correctly classified samples."
    )
