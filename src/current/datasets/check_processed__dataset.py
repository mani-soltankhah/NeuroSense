from src.current.datasets.processed_brain_tumor import ProcessedBrainTumorDataset
import torch

dataset = ProcessedBrainTumorDataset(
    "D:/Portfolio/NeuroSense/Data/Processed/test"
)

print("length:", len(dataset))

for i in [0, 100, 200, 300]:
    image, mask = dataset[i]

    print("\nindex:", i)
    print("image shape:", image.shape)
    print("mask shape:", mask.shape)
    print("image min/max:", image.min(), image.max())
    print("mask unique:", torch.unique(mask))
    print("mask sum:", mask.sum())
