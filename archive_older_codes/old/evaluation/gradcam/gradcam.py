import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from dataset import BrainTumorDataset, transform
from model import BrainTumorCNN

matplotlib.use('TkAgg')

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = r"D:\Portfolio\NeuroSense\Models\best_model.pth"

METADATA_PATH = r"/Data/metadata.csv"

CLASS_NAMES = [
    "Meningioma",
    "Glioma",
    "Pituitary"
]

# ============================================================
# LOAD MODEL
# ============================================================

model = BrainTumorCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("=" * 60)
print("GRAD-CAM")
print("=" * 60)

print("Device:", DEVICE)
print("Model loaded.")

# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(
    METADATA_PATH
)

test_df = df[
    df["split"] == "test"
    ].reset_index(drop=True)

print(
    "Test samples:",
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
# LOAD TUMOR MASK
# ============================================================

def load_tumor_mask(path):
    with h5py.File(path, "r") as f:
        mask = f["cjdata"]["tumorMask"][()]

    mask = mask.astype("float32")

    mask = Image.fromarray(mask)

    mask = mask.resize(
        (224, 224),
        Image.Resampling.NEAREST
    )

    mask = np.array(mask)

    return mask > 0


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
        model,
        input_tensor,
        target_layer,
        target_class
):
    activations = None
    gradients = None

    # --------------------------------------------------------
    # Hooks
    # --------------------------------------------------------

    def forward_hook(
            module,
            input,
            output
    ):
        nonlocal activations

        activations = output

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

    output = model(
        input_tensor
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    model.zero_grad()

    score = output[
        0,
        target_class
    ]

    score.backward()

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    forward_handle.remove()
    backward_handle.remove()

    # --------------------------------------------------------
    # Calculate weights
    # --------------------------------------------------------

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # --------------------------------------------------------
    # Weighted activation maps
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

    cam = F.relu(
        cam
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    # --------------------------------------------------------
    # Remove batch/channel dimensions
    # --------------------------------------------------------

    cam = cam.squeeze()

    cam = (
        cam
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Normalize 0 → 1
    # --------------------------------------------------------

    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    return cam


# ============================================================
# SELECT SAMPLE
# ============================================================

sample_index = 0

image, true_label = test_dataset[
    sample_index
]

row = test_df.iloc[
    sample_index
]

input_tensor = image.unsqueeze(
    0
).to(DEVICE)

print("\n" + "=" * 60)
print("SAMPLE")
print("=" * 60)

print(
    "Patient:",
    row["patient_id"]
)

print(
    "True label:",
    CLASS_NAMES[
        true_label.item()
    ]
)

print(
    "Image:",
    image.shape
)

# ============================================================
# MODEL PREDICTION
# ============================================================

with torch.no_grad():
    output = model(
        input_tensor
    )

    probabilities = F.softmax(
        output,
        dim=1
    )

    predicted_class = output.argmax(
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

print(
    "Prediction:",
    CLASS_NAMES[
        predicted_class
    ]
)

print(
    f"Confidence: {confidence:.4f}"
)

# ============================================================
# TARGET LAYER
# ============================================================

target_layer = model.features[12]

# ============================================================
# GENERATE GRAD-CAM
# ============================================================

cam = generate_gradcam(
    model=model,
    input_tensor=input_tensor,
    target_layer=target_layer,
    target_class=predicted_class
)

# ============================================================
# LOAD TUMOR MASK
# ============================================================

tumor_mask = load_tumor_mask(
    row["path"]
)

# ============================================================
# CREATE GRAD-CAM BINARY MASK
# ============================================================

threshold = 0.5

gradcam_mask = (
        cam >= threshold
)

# ============================================================
# CALCULATE IoU
# ============================================================

intersection = np.logical_and(
    gradcam_mask,
    tumor_mask
).sum()

union = np.logical_or(
    gradcam_mask,
    tumor_mask
).sum()

iou = (
        intersection /
        (union + 1e-8)
)

print("\n" + "=" * 60)
print("XAI ANALYSIS")
print("=" * 60)

print(
    "Tumor pixels:",
    tumor_mask.sum()
)

print(
    "Grad-CAM pixels:",
    gradcam_mask.sum()
)

print(
    f"Intersection: {intersection}"
)

print(
    f"Union: {union}"
)

print(
    f"Grad-CAM IoU: {iou:.4f}"
)

# ============================================================
# ORIGINAL IMAGE
# ============================================================

original = (
    image
    .squeeze()
    .cpu()
    .numpy()
)

# Undo normalization

original = (
                   original * 0.5
           ) + 0.5

original = np.clip(
    original,
    0,
    1
)

# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(
    figsize=(16, 4)
)

# ------------------------------------------------------------
# ORIGINAL
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    1
)

plt.imshow(
    original,
    cmap="gray"
)

plt.title(
    "Original\n"
    f"True: {CLASS_NAMES[true_label.item()]}"
)

plt.axis("off")

# ------------------------------------------------------------
# GRAD-CAM
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    2
)

plt.imshow(
    cam,
    cmap="jet"
)

plt.title(
    "Grad-CAM\n"
    f"Pred: {CLASS_NAMES[predicted_class]}"
)

plt.axis("off")

# ------------------------------------------------------------
# TUMOR MASK
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    3
)

plt.imshow(
    tumor_mask,
    cmap="gray"
)

plt.title(
    "Ground Truth\nTumor Mask"
)

plt.axis("off")

# ------------------------------------------------------------
# OVERLAY
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    4
)

plt.imshow(
    original,
    cmap="gray"
)

plt.imshow(
    cam,
    cmap="jet",
    alpha=0.45
)

plt.contour(
    tumor_mask,
    levels=[0.5],
    linewidths=1.5
)

plt.title(
    "Grad-CAM + Tumor Mask\n"
    f"IoU: {iou:.4f}"
)

plt.axis("off")

plt.tight_layout()

plt.show()
