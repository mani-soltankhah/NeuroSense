import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from src.current.datasets.processed_brain_tumor import ProcessedBrainTumorDataset
from src.current.utils.preprocessing import BrainMRIProcessor

data_dir = Path(r"D:\Portfolio\NeuroSense\Data\Raw")
mat_files = list(data_dir.glob("*.mat"))
df = pd.DataFrame(
    {
        "path": mat_files,
        "label": [1] * len(mat_files)
    }
)

data = []
for mat_file in mat_files:
    with h5py.File(mat_file, "r") as f:
        cjdata = f["cjdata"]
        label = int(cjdata["label"][()].flatten()[0])
        pid_raw = cjdata['PID'][()]
        if isinstance(pid_raw, np.ndarray):
            pid = "".join(chr(int(x)) for x in pid_raw.flatten())
        elif isinstance(pid_raw, bytes):
            pid = pid_raw.decode('utf-8')
        else:
            pid = str(pid_raw)
        data.append({
            "path": mat_file,
            "patient_id": pid,
            "label": label

        })
df = pd.DataFrame(data)
patient_df = df.groupby("patient_id").first().reset_index()

train_patients, temp_patients = train_test_split(
    patient_df,
    test_size=0.3,
    stratify=patient_df["label"],
    random_state=42
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.5,
    stratify=temp_patients["label"],
    random_state=42
)
train_patient_ids = train_patients["patient_id"]
val_patient_ids = val_patients["patient_id"]
test_patient_ids = test_patients["patient_id"]

train_df = df[df['patient_id'].isin(train_patient_ids)]
val_df = df[df['patient_id'].isin(val_patient_ids)]
test_df = df[df['patient_id'].isin(test_patient_ids)]

train_dataset = ProcessedBrainTumorDataset(
    "D:/Portfolio/NeuroSense/Data/Processed/train",
    augment=True
)
val_dataset = ProcessedBrainTumorDataset(
    "D:/Portfolio/NeuroSense/Data/Processed/val"
)
test_dataset = ProcessedBrainTumorDataset(
    "D:/Portfolio/NeuroSense/Data/Processed/test"
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)

if __name__ == "__main__":
    images, masks = next(iter(train_loader))

    print(images.shape)
    print(masks.shape)
