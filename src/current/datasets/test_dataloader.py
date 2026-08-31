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

matplotlib.use('TkAgg')

data_dir = Path(r"D:\Portfolio\NeuroSense\Data\Raw")
mat_files = list(data_dir.glob("*.mat"))
df = pd.DataFrame(
    {
        "path": mat_files,
        "label": [1] * len(mat_files)
    }
)

train_df, temp_df = train_test_split(df, test_size=0.3)
val_df, test_df = train_test_split(df, test_size=0.5)
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

processor = BrainMRIProcessor(image_size=(224, 224))
train_dataset = BrainTumorDataset(train_df, processor=processor)
val_dataset = BrainTumorDataset(val_df, processor=processor)
test_dataset = BrainTumorDataset(test_df, processor=processor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

images, masks = next(iter(train_loader))

print(images.shape)
print(masks.shape)

print(torch.unique(masks))

images, masks = next(iter(train_loader))

show_sample(
    images[0],
    masks[0]
)
