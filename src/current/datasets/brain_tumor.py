import h5py
from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path

from src.current.utils.preprocessing import BrainMRIProcessor


class BrainTumorDataset(Dataset):
    def __init__(self, dataframe, processor=None, image_transform=None, mask_transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['path']
        with h5py.File(path, 'r') as f:
            cjdata = f['cjdata']
            image = cjdata['image'][()]
            tumor_mask = cjdata['tumorMask'][()]
        if self.processor:
            image, tumor_mask = self.processor.preprocess(image, tumor_mask)
        return image, tumor_mask


data_dir = Path(r"D:\Portfolio\NeuroSense\Data\Raw")
mat_files = list(data_dir.glob("*.mat"))
df = pd.DataFrame(
    {
        "path": [
            mat_files[0]
        ],

        "label": [
            1,
        ]
    }
)

dataset = BrainTumorDataset(df)
image, mask = dataset[0]

processor = BrainMRIProcessor(
    image_size=(224, 224)
)

dataset = BrainTumorDataset(
    df,
    processor=processor
)

image, mask = dataset[0]

print(image.shape)
print(mask.shape)
print(image.dtype)
print(mask.dtype)
print(torch.unique(mask))
