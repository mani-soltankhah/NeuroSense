from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path
import torch


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
            ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = torch.load(image_path)
        mask = torch.load(mask_path)

        if self.augment:
            if image.dim() == 3:
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = image.unsqueeze(-1).numpy()

            if mask.dim() == 3:
                mask_np = mask.permute(1, 2, 0).numpy()
            else:
                mask_np = mask.unsqueeze(-1).numpy()

            transformed = self.transform(image=image_np, mask=mask_np)
            image_aug = transformed['image']
            mask_aug = transformed['mask']

            if image_aug.ndim == 3 and image_aug.shape[2] == 1:
                image = torch.from_numpy(image_aug.squeeze(-1)).unsqueeze(0).float()
            else:
                image = torch.from_numpy(image_aug).permute(2, 0, 1).float()

            if mask_aug.ndim == 3 and mask_aug.shape[2] == 1:
                mask = torch.from_numpy(mask_aug.squeeze(-1)).unsqueeze(0).float()
            else:
                mask = torch.from_numpy(mask_aug).permute(2, 0, 1).float()

        else:
            if image.dim() == 2:
                image = image.unsqueeze(0).float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).float()

        return image, mask
