import random
import torch
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


def image_to_tensor(image):
    tensor_image = torch.tensor(image)
    return tensor_image


def add_channel_dimension(image):
    return image.unsqueeze(0)


def mask_to_tensor(mask):
    tensor_mask = torch.tensor(mask)
    tensor_mask = tensor_mask.type(torch.float32)
    return tensor_mask.unsqueeze(0)


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
normalized_image = normalize_image(image)
normalized_tumor = normalized_image[mask == 1]
normalized_background = normalized_image[mask == 0]
tensor_image = image_to_tensor(normalized_image)
tensor_image = add_channel_dimension(tensor_image)

tensor_mask = mask_to_tensor(mask)

print("Type:", type(tensor_mask))
print("Dtype:", tensor_mask.dtype)
print("Shape:", tensor_mask.shape)
print("Unique:", torch.unique(tensor_mask))

# for path in random_files:
#     with h5py.File(path, "r") as f:
#         cjdata = f["cjdata"]
#         image = cjdata["image"][()]
#         tumor_mask = cjdata['tumorMask'][()]
#         normalized_random = normalize_image(image)
#
#         healthy_brain_mask = (image != 0) & (tumor_mask == 0)
#         normalized_healthy_brain = normalized_random[healthy_brain_mask]
#
#         normalized_image = normalize_image(image)
#         normalized_tumor = normalized_image[tumor_mask == 1]
#         normalized_background = normalized_image[tumor_mask == 0]
#         normalized_brain = normalized_random[normalized_random != 0]
#         print(
#             f"""
#             Whole image:
#             min: {normalized_random.min():.2f}
#             max: {normalized_random.max():.2f}
#             """
#         )
#         # print(f"""
#         # name: {path.name}
#         #
#         # Tumor:
#         # mean: {normalized_tumor.mean():.3f}
#         # std: {normalized_tumor.std():.3f}
#         #
#         # Healthy brain:
#         # mean: {normalized_healthy_brain.mean():.3f}
#         # std: {normalized_healthy_brain.std():.3f}
#         # """)
