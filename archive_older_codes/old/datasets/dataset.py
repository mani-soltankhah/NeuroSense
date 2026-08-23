from pathlib import Path

import h5py
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

METADATA_PATH = Path(
    r"/Data/old/metadata.csv"
)

image_transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BILINEAR
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    ),
])

mask_transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.NEAREST
    ),

    transforms.ToTensor(),
])


class BrainTumorDataset(Dataset):
    def __init__(
            self,
            dataframe,
            image_transform=None,
            mask_transform=None
    ):

        self.df = dataframe.reset_index(drop=True)

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row["path"]
        with h5py.File(path, "r") as f:

            cjdata = f["cjdata"]

            image = cjdata["image"][()]
            tumor_mask = cjdata["tumorMask"][()]

        image = image.astype("float32")
        image_min = image.min()
        image_max = image.max()

        image = (
                255.0
                * (image - image_min)
                / (image_max - image_min + 1e-8)
        )
        image = image.astype("uint8")
        image = Image.fromarray(
            image,
            mode="L"
        )
        tumor_mask = tumor_mask.astype("uint8")
        tumor_mask = tumor_mask * 255
        tumor_mask = Image.fromarray(
            tumor_mask,
            mode="L"
        )

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:

            tumor_mask = self.mask_transform(tumor_mask)

        else:

            tumor_mask = transforms.ToTensor()(tumor_mask)

        tumor_mask = (
                tumor_mask > 0.5
        ).float()

        label = int(row["label"]) - 1
        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label, tumor_mask


df = pd.read_csv(
    METADATA_PATH
)

train_df = df[
    df["split"] == "train"
    ].copy()

val_df = df[
    df["split"] == "val"
    ].copy()

test_df = df[
    df["split"] == "test"
    ].copy()

train_dataset = BrainTumorDataset(
    train_df,
    image_transform=image_transform,
    mask_transform=mask_transform
)

val_dataset = BrainTumorDataset(
    val_df,
    image_transform=image_transform,
    mask_transform=mask_transform
)

test_dataset = BrainTumorDataset(
    test_df,
    image_transform=image_transform,
    mask_transform=mask_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

if __name__ == "__main__":

    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    image, label, mask = train_dataset[0]

    print("\nSingle sample:")

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Image dtype:",
        image.dtype
    )

    print(
        "Label:",
        label
    )

    print(
        "Label dtype:",
        label.dtype
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Mask dtype:",
        mask.dtype
    )

    print(
        "Mask unique values:",
        torch.unique(mask)
    )

    print(
        "Tumor pixels:",
        int(mask.sum().item())
    )

    images, labels, masks = next(iter(train_loader))
    print("\nBatch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Masks shape:", masks.shape)
    print("Images dtype:", images.dtype)
    print("Labels dtype:", labels.dtype)
    print("Masks dtype:", masks.dtype)
    print("Labels:", labels)
    print("Mask unique values:", torch.unique(masks))
    print("Total tumor pixels:", int(masks.sum().item()))
    if masks.sum() == 0:
        print("\nWARNING: Batch contains no tumor pixels.")
    else:
        print("\nTumor masks detected successfully.")
    print("\nDataset test: PASSED")
