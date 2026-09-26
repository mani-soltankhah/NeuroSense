import json
from pathlib import Path
import cv2
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class BraTSDataset(Dataset):
    def __init__(self, dataset_root, split):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.metadata_path = self.dataset_root / "train.jsonl"
        self.splits_path = self.dataset_root / "splits.json"

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Invalid split: {split}. "
                "Expected 'train', 'val', or 'test'."
            )
        self.rows = self._load_metadata()
        self.subjects = self._load_split_subjects()
        self.scans = self._build_scan_index()
        self.samples = self._build_slice_index()

    def _resolve_path(self, relative_path):
        prefix = "data/nii/BraTS2023_GLI/"
        if not relative_path.startswith(prefix):
            raise RuntimeError(
                f"Unexpected dataset path: {relative_path}"
            )
        relative_path = Path(relative_path[len(prefix):])
        path = self.dataset_root / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )
        return path

    def _load_metadata(self):
        rows = []
        with self.metadata_path.open("r", encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def _load_slice(self, path, slice_idx):
        volume = nib.load(path)
        array = volume.get_fdata()
        slice_data = array[:, :, slice_idx]
        return slice_data

    def _load_split_subjects(self):
        with self.splits_path.open("r", encoding='utf-8') as f:
            splits = json.load(f)
        return set(splits['subjects'][self.split])

    def _build_scan_index(self):
        scans = []

        for row in self.rows:
            patient_id = row["patient_id"]
            subject_id = patient_id.rsplit("-", 1)[0]
            if subject_id not in self.subjects:
                continue
            scans.append(
                {
                    "subject_id": subject_id,
                    "patient_id": patient_id,
                    "modalities": row["modalities"],
                    "mask": row["mask"],
                    "t1c_path": self._resolve_path(
                        row["modalities"]["t1c"]
                    ),

                    "mask_path": self._resolve_path(
                        row["mask"]
                    ),
                }
            )

        return scans

    def _build_slice_index(self):
        samples = []
        for scan in self.scans:
            volume = nib.load(scan["t1c_path"])
            num_slices = volume.shape[2]
            for slice_idx in range(num_slices):
                samples.append(
                    {
                        "t1c_path": scan["t1c_path"],
                        "mask_path": scan["mask_path"],
                        "subject_id": scan["subject_id"],
                        "patient_id": scan["patient_id"],
                        "slice_idx": slice_idx,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self._load_slice(
            sample["t1c_path"],
            sample["slice_idx"],
        )
        mask = self._load_slice(
            sample["mask_path"],
            sample["slice_idx"],
        )
        mask = (mask > 0).astype(np.float32)

        brain_mask = image > 0
        nonzero = image[brain_mask]
        if nonzero.size > 0:
            mean = nonzero.mean()
            std = nonzero.std()
            image = (image - mean) / (std + 1e-8)
            image[~brain_mask] = 0
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [224, 224], interpolation=InterpolationMode.NEAREST)

        mask = (mask > 0.5).float()
        return image, mask


if __name__ == "__main__":
    dataset = BraTSDataset(
        dataset_root=r"D:\Datasets\BraTS-GLI",
        split="train",
    )

    print("Dataset:")
    print("  subjects:", len(dataset.subjects))
    print("  scans:", len(dataset.scans))
    print("  samples:", len(dataset))

    for i in range(len(dataset)):
        image, mask = dataset[i]

        if mask.sum() > 0:
            print("Found positive slice:", i)

            print("Image:")
            print(" shape:", image.shape)
            print(" min:", image.min().item())
            print(" max:", image.max().item())

            print("\nMask:")
            print(" shape:", mask.shape)
            print(" unique:", torch.unique(mask))
            print(" tumor pixels:", mask.sum().item())

            break

    print("\nSample:")
    print("  image shape:", image.shape)
    print("  image dtype:", image.dtype)
    print("  image min:", image.min().item())
    print("  image max:", image.max().item())

    print("  mask shape:", mask.shape)
    print("  mask dtype:", mask.dtype)
    print("  mask unique:", torch.unique(mask))
