from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from torch.utils.data import DataLoader

from archive_older_codes.old.datasets.dataset import BrainTumorDataset, transform
from archive_older_codes.old.models.model import BrainTumorCNN

METADATA_PATH = Path(
    r"/Data/old/metadata.csv"
)

MODEL_PATH = Path(
    r"D:\Portfolio\NeuroSense\Models\best_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

df = pd.read_csv(METADATA_PATH)

test_df = df[
    df["split"] == "test"
    ].copy()

print("=" * 60)
print("PATIENT-LEVEL EVALUATION")
print("=" * 60)

print(
    f"Test images: {len(test_df)}"
)

print(
    f"Test patients: {test_df['patient_id'].nunique()}"
)

test_dataset = BrainTumorDataset(
    test_df,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

model = BrainTumorCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("\nDevice:", DEVICE)
print("Model loaded.")

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        ).cpu()

        all_predictions.extend(
            predictions.tolist()
        )

        all_labels.extend(
            labels.tolist()
        )

test_df["prediction"] = all_predictions

image_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("IMAGE-LEVEL RESULTS")
print("=" * 60)

print(
    f"Accuracy: {image_accuracy:.4f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Meningioma",
            "Glioma",
            "Pituitary"
        ],
        digits=4
    )
)

patient_results = []
for patient_id, group in test_df.groupby(
        "patient_id"
):
    true_label = group["label"].iloc[0] - 1
    predictions = group["prediction"]
    patient_prediction = (
        predictions
        .value_counts()
        .idxmax()
    )

    patient_results.append({
        "patient_id": patient_id,
        "true_label": true_label,
        "prediction": patient_prediction,
        "num_slices": len(group)
    })

patient_df = pd.DataFrame(
    patient_results
)

patient_accuracy = accuracy_score(
    patient_df["true_label"],
    patient_df["prediction"]
)

print("\n" + "=" * 60)
print("PATIENT-LEVEL RESULTS")
print("=" * 60)

print(
    f"Patients evaluated: {len(patient_df)}"
)

print(
    f"Patient Accuracy: {patient_accuracy:.4f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        patient_df["true_label"],
        patient_df["prediction"]
    )
)

print("\nClassification Report:")

print(
    classification_report(
        patient_df["true_label"],
        patient_df["prediction"],
        target_names=[
            "Meningioma",
            "Glioma",
            "Pituitary"
        ],
        digits=4
    )
)

print("\n" + "=" * 60)
print("PATIENT DETAILS")
print("=" * 60)

for _, row in patient_df.iterrows():
    true_name = [
        "Meningioma",
        "Glioma",
        "Pituitary"
    ][row["true_label"]]

    pred_name = [
        "Meningioma",
        "Glioma",
        "Pituitary"
    ][row["prediction"]]

    status = (
        "CORRECT"
        if row["true_label"] == row["prediction"]
        else "WRONG"
    )

    print(
        f"{row['patient_id']} | "
        f"Slices: {row['num_slices']:2d} | "
        f"Actual: {true_name:10s} | "
        f"Predicted: {pred_name:10s} | "
        f"{status}"
    )
