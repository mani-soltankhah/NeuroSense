import numpy as np
from sklearn.model_selection import train_test_split
import h5py
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from src.current.datasets.brain_tumor import BrainTumorDataset
from src.current.utils.preprocessing import BrainMRIProcessor
from src.current.utils.visualization import show_sample
from numpy import *

matplotlib.use('TkAgg')
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

processor = BrainMRIProcessor(image_size=(384, 384))

train_dataset = BrainTumorDataset(train_df, processor=processor)
val_dataset = BrainTumorDataset(val_df, processor=processor)
test_dataset = BrainTumorDataset(test_df, processor=processor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                         num_workers=0, pin_memory=True)

# images, masks = next(iter(train_loader))
# print(images.shape)
# print(masks.shape)
#
# print(images.dtype)
# print(masks.dtype)
#
# print(images.min())
# print(images.max())
#
# print(torch.unique(masks))

# print(len(train_df))
# print(len(val_df))
# print(len(test_df))
#
# print(train_df["patient_id"].nunique())
# print(val_df["patient_id"].nunique())
# print(test_df["patient_id"].nunique())
#
# print(train_df["label"].value_counts())
# print(val_df["label"].value_counts())
# print(test_df["label"].value_counts())
#
# print(val_df["patient_id"].value_counts().describe())
# print(test_df["patient_id"].value_counts().describe())

# print(len(train_df))
# print(len(val_df))
# print(len(test_df))
#
# print(
#     train_df["patient_id"].nunique(),
#     val_df["patient_id"].nunique(),
#     test_df["patient_id"].nunique()
# )
# print(set(train_patients) & set(val_patients))
# print(set(train_patients) & set(test_patients))
# print(set(val_patients) & set(test_patients))

# print(train_df["label"].value_counts(normalize=True))
# print(val_df["label"].value_counts(normalize=True))
# print(test_df["label"].value_counts(normalize=True))

# data = []
# for path in mat_files:
#     with h5py.File(path, 'r') as f:
#         cjdata = f['cjdata']
#         label = cjdata['label'][()]
#         pid = cjdata['PID'][()]
#
#         label = int(label[0][0])
#         pid = int(pid[0][0])
#
#         data.append({
#             "path": path,
#             "label": label,
#             "patient_id": pid
#         })
# df = pd.DataFrame(data)
#
# print(df.head())
# print(df["label"].value_counts())
# print(df["patient_id"].nunique())
#
# train_patients, temp_patients = train_test_split(df["patient_id"].unique(), test_size=0.3, random_state=42)
# val_patient, test_patient = train_test_split(temp_patients, test_size=0.5, random_state=42)
#
# train_df = df[df['patient_id'].isin(train_patients)]
# val_df = df[df['patient_id'].isin(val_patient)]
# test_df = df[df['patient_id'].isin(test_patient)]
#
# print(f"Train {len(train_df)}")
# print(f"Val {len(val_df)}")
# print(f"Test {len(test_df)}")
#
# print(
#     "Patient split:",
#     train_df.patient_id.nunique(),
#     val_df.patient_id.nunique(),
#     test_df.patient_id.nunique()
# )
#
# # train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
# # val_df, test_df = train_test_split(df, test_size=0.5, random_state=42)
# # print("Train:", len(train_df))
# # print("Validation:", len(val_df))
# # print("Test:", len(test_df))
# #
# processor = BrainMRIProcessor(image_size=(384, 384))
# train_dataset = BrainTumorDataset(train_df, processor=processor)
# val_dataset = BrainTumorDataset(val_df, processor=processor)
# test_dataset = BrainTumorDataset(test_df, processor=processor)
#
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#
# images, masks = next(iter(train_loader))
#
# print(images.shape)
# print(masks.shape)
#
# print(torch.unique(masks))
#
# images, masks = next(iter(train_loader))
#
# show_sample(
#     images[0],
#     masks[0]
# )
