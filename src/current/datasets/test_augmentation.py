from src.current.datasets.processed_brain_tumor import ProcessedBrainTumorDataset

dataset_original = ProcessedBrainTumorDataset("D:/Portfolio/NeuroSense/Data/Processed/train", augment=False)

dataset_augmented = ProcessedBrainTumorDataset("D:/Portfolio/NeuroSense/Data/Processed/train", augment=True)

image_original, mask_original = dataset_original[10]
image_aug, mask_aug = dataset_augmented[10]

print(image_original.shape)
print(mask_original.shape)

print(image_aug.shape)
print(mask_aug.shape)

print(mask_original.unique())
print(mask_aug.unique())

print(image_original.dtype)
print(mask_original.dtype)
print(image_original.shape)
print(mask_original.unique())

print(image_aug.dtype)
print(mask_aug.dtype)
print(image_aug.shape)
print(mask_aug.unique())
