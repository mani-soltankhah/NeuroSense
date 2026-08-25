import random

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from src.current.datasets.brain_tumor import BrainTumorDataset


def normalize_image(image):
    image = image.astype('float32')
    brain_mask = image != 0
    brain_pixels = image[brain_mask]
    mean = brain_pixels.mean()
    std = brain_pixels.std()
    normalized_image = np.zeros_like(image)
    normalized_image[brain_mask] = (image[brain_mask] - mean) / std
    return normalized_image


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
random_files = random.sample(mat_files, 10)
normalized = normalize_image(image)

for path in random_files:
    with h5py.File(path, "r") as f:
        cjdata = f["cjdata"]
        image = cjdata["image"][()]
        tumor_mask = cjdata['tumorMask'][()]
        normalized_random = normalize_image(image)
        normalized_brain = normalized_random[normalized_random != 0]

        print(
            f"""
            Whole image:
            mean: {normalized_random.mean():.2f}
            std: {normalized_random.std():.2f}

            Brain only:
            mean: {normalized_brain.mean():.2f}
            std: {normalized_brain.std():.2f}
            """
        )
