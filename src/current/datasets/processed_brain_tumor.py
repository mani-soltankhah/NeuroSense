from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import albumentations as A
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

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = torch.load(image_path)
        mask = torch.load(mask_path)

        # بررسی ابعاد
        # print(f"Original image shape: {image.shape}")
        # print(f"Original mask shape: {mask.shape}")

        if self.augment:
            if image.dim() == 3:
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = image.unsqueeze(-1).numpy()

            if mask.dim() == 3:
                mask_np = mask.permute(1, 2, 0).numpy()
            else:
                mask_np = mask.unsqueeze(-1).numpy()

            # print(f"Image numpy shape: {image_np.shape}")
            # print(f"Mask numpy shape: {mask_np.shape}")

            transformed = self.transform(image=image_np, mask=mask_np)

            image_aug = transformed['image']
            mask_aug = transformed['mask']

            if image_aug.ndim == 3 and image_aug.shape[2] == 1:
                image = torch.from_numpy(image_aug.squeeze(-1)).unsqueeze(0).float()
            else:
                image = torch.from_numpy(image_aug).permute(2, 0, 1)

            if mask_aug.ndim == 3 and mask_aug.shape[2] == 1:
                mask = torch.from_numpy(mask_aug.squeeze(-1)).unsqueeze(0).float()
            else:
                mask = torch.from_numpy(mask_aug).permute(2, 0, 1)


        else:
            if image.dim() == 2:
                image = image.unsqueeze(0)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)

            # if random.random() > 0.5:
            #     image = torch.flip(image, [2])
            #     mask = torch.flip(mask, [2])
            # if random.random() > 0.5:
            #     angle = random.choice([-15, -10, 10, 15])
            #     image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            #     mask = TF.rotate(image, angle, interpolation=TF.InterpolationMode.NEAREST)

        return image, mask
