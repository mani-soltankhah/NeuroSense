from src.current.datasets.processed_brain_tumor import ProcessedBrainTumorDataset
import matplotlib.pyplot as plt
import torch
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

data_path = "D:/Portfolio/NeuroSense/Data/Processed/train"
original_dataset = ProcessedBrainTumorDataset(data_path, augment=False)
augmented_dataset = ProcessedBrainTumorDataset(data_path, augment=True)

index = 10
original_image, original_mask = original_dataset[index]
augmented_image, augmented_mask = augmented_dataset[index]

print("Original mask:", original_mask.unique())
print("Augmented mask:", augmented_mask.unique())


def prepare_image(image):
    image = image.squeeze(0).numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image


def prepare_mask(mask):
    return mask.squeeze(0).numpy()


original_image = prepare_image(original_image)
augmented_image = prepare_image(augmented_image)
original_mask = prepare_mask(original_mask)
augmented_mask = prepare_mask(augmented_mask)


# تابع کمکی برای overlay کردن ماسک روی تصویر
def overlay_mask_on_image(image, mask, alpha=0.4):
    """
    ماسک را به رنگ قرمز روی تصویر overlay می‌کند
    """
    # تبدیل تصویر به RGB
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image

    # ایجاد overlay قرمز برای ماسک
    overlay = image_rgb.copy()
    mask_bool = mask > 0.5
    overlay[mask_bool] = [1, 0, 0]  # رنگ قرمز

    # ترکیب تصویر و overlay
    result = (1 - alpha) * image_rgb + alpha * overlay
    return result


# ایجاد تصاویر overlay
original_overlay = overlay_mask_on_image(original_image, original_mask)
augmented_overlay = overlay_mask_on_image(augmented_image, augmented_mask)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# ردیف اول: تصاویر خام
axes[0, 0].imshow(original_image, cmap="gray")
axes[0, 0].set_title("Original Image", fontsize=12)
axes[0, 0].axis("off")

axes[0, 1].imshow(augmented_image, cmap="gray")
axes[0, 1].set_title("Augmented Image", fontsize=12)
axes[0, 1].axis("off")

axes[0, 2].axis("off")

# ردیف دوم: ماسک‌ها
axes[1, 0].imshow(original_mask, cmap="gray")
axes[1, 0].set_title("Original Mask", fontsize=12)
axes[1, 0].axis("off")

axes[1, 1].imshow(augmented_mask, cmap="gray")
axes[1, 1].set_title("Augmented Mask", fontsize=12)
axes[1, 1].axis("off")

# ستون سوم: Overlay
axes[0, 2].imshow(original_overlay)
axes[0, 2].set_title("Original: Image + Mask Overlay", fontsize=12, color='green')
axes[0, 2].axis("off")

axes[1, 2].imshow(augmented_overlay)
axes[1, 2].set_title("Augmented: Image + Mask Overlay", fontsize=12, color='red')
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()

# بررسی هم‌پوشانی ماسک با ناحیه تومور
print("\n=== تحلیل هم‌پوشانی ===")
print(f"Original mask area: {original_mask.sum():.0f} pixels")
print(f"Augmented mask area: {augmented_mask.sum():.0f} pixels")
print(f"Area difference: {abs(original_mask.sum() - augmented_mask.sum()):.0f} pixels")
