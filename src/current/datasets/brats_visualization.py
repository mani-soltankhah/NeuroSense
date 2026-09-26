from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from brats_dataset import BraTSDataset
import matplotlib

matplotlib.use('TkAgg')

# ۱. لود دیتاست
dataset = BraTSDataset(
    dataset_root=r"D:\Datasets\BraTS-GLI",
    split="train",
)

# ۲. انتخاب یک بیمار (Scan) برای جلوگیری از پیمایش کل دیتاست
# (اگر می‌خواهید بیمار خاصی را ببینید، ایندکس آن را تغییر دهید)
scan_info = dataset.scans[0]
target_patient_id = scan_info['patient_id']
print(f"Visualizing scan: {target_patient_id}")

# پیدا کردن ایندکس تمام اسلایس‌های این بیمار در دیتاست
scan_sample_indices = [
    i for i, sample in enumerate(dataset.samples)
    if sample['patient_id'] == target_patient_id
]

# ۳. پیمایش اسلایس‌های این بیمار برای پیدا کردن بهترین نما
max_tumor_area = 0
best_tumor_idx = scan_sample_indices[0]

max_brain_area = 0
best_brain_idx = scan_sample_indices[0]

for idx in scan_sample_indices:
    image, mask = dataset[idx]

    # مساحت مغز (پیکسل‌های غیرصفر)
    brain_area = (image != 0).sum().item()
    # مساحت تومور
    tumor_area = mask.sum().item()

    if tumor_area > max_tumor_area:
        max_tumor_area = tumor_area
        best_tumor_idx = idx

    if brain_area > max_brain_area:
        max_brain_area = brain_area
        best_brain_idx = idx

print(f"Max Tumor Area: {max_tumor_area} at Slice {dataset.samples[best_tumor_idx]['slice_idx']}")
print(f"Max Brain Area: {max_brain_area} at Slice {dataset.samples[best_brain_idx]['slice_idx']}")

# ۴. رسم تصاویر
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# --- ردیف اول: اسلایس با بیشترین مساحت تومور ---
image_t, mask_t = dataset[best_tumor_idx]
image_t = image_t.squeeze().numpy()
mask_t = mask_t.squeeze().numpy()
slice_num_t = dataset.samples[best_tumor_idx]['slice_idx']

axes[0, 0].imshow(image_t, cmap="gray")
axes[0, 0].set_title(f"T1c MRI (Max Tumor - Slice {slice_num_t})")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask_t, cmap="gray")
axes[0, 1].set_title("Tumor Mask")
axes[0, 1].axis("off")

axes[0, 2].imshow(image_t, cmap="gray")
axes[0, 2].contour(mask_t, colors='red', linewidths=1.5)  # استفاده از کانتور به جای overlay شفاف
axes[0, 2].set_title("Overlay (Contour)")
axes[0, 2].axis("off")

# --- ردیف دوم: اسلایس با بیشترین مساحت مغز (نمای کامل) ---
image_b, mask_b = dataset[best_brain_idx]
image_b = image_b.squeeze().numpy()
mask_b = mask_b.squeeze().numpy()
slice_num_b = dataset.samples[best_brain_idx]['slice_idx']

axes[1, 0].imshow(image_b, cmap="gray")
axes[1, 0].set_title(f"T1c MRI (Max Brain - Slice {slice_num_b})")
axes[1, 0].axis("off")

axes[1, 1].imshow(mask_b, cmap="gray")
axes[1, 1].set_title("Tumor Mask")
axes[1, 1].axis("off")

axes[1, 2].imshow(image_b, cmap="gray")
axes[1, 2].contour(mask_b, colors='red', linewidths=1.5)
axes[1, 2].set_title("Overlay (Contour)")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()
