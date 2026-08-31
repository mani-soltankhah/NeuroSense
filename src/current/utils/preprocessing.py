import random
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


class BrainMRIProcessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def normalize_image(self, image):
        image = image.astype('float32')
        brain_mask = image != 0
        brain_pixels = image[brain_mask]
        mean = brain_pixels.mean()
        std = brain_pixels.std()
        normalized_image = np.zeros_like(image)
        normalized_image[brain_mask] = (image[brain_mask] - mean) / std
        return normalized_image

    def image_to_tensor(self, image):
        return torch.from_numpy(image)

    def add_channel_dimension(self, image):
        return image.unsqueeze(0)

    def mask_to_tensor(self, mask):
        tensor_mask = torch.tensor(mask)
        tensor_mask = tensor_mask.type(torch.float32)
        return tensor_mask.unsqueeze(0)

    def resize_image(self, image, size):
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
        return image.squeeze(0)
    
    def resize_mask(self, image, size):
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=size, mode='nearest')
        return image.squeeze(0)

    def preprocess(self, image, mask):
        image = self.normalize_image(image)
        image = self.image_to_tensor(image)
        image = self.add_channel_dimension(image)
        image = self.resize_image(image, self.image_size)

        mask = self.mask_to_tensor(mask)
        mask = self.resize_mask(mask, self.image_size)

        return image, mask
