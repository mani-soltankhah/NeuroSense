from torch.utils.data import Dataset
from pathlib import Path
import torch
import random


class ProcessedBrainTumorDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'masks'

        self.images = sorted(self.image_dir.glob("*.pt"), key=lambda x: int(x.stem))
        self.masks = sorted(self.mask_dir.glob("*.pt"), key=lambda x: int(x.stem))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = torch.load(image_path)
        mask = torch.load(mask_path)
        if self.augment:
            if random.random() > 0.5:
                image = torch.flip(image, [2])
                mask = torch.flip(mask, [2])

        return image, mask
