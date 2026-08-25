import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
PATH = r"D:\Portfolio\NeuroSense\Data\Raw\2944.mat"

print("=" * 60)
print("MASK DEBUG")
print("=" * 60)

with h5py.File(PATH, "r") as f:
    print("\nTop-level keys:")
    print(list(f.keys()))

    print("\nCJDATA keys:")
    print(list(f["cjdata"].keys()))

    print("\nImage:")
    image = np.squeeze(f["cjdata"]["image"][()])
    print("Shape:", image.shape)
    print("Dtype:", image.dtype)
    print("Min:", image.min())
    print("Max:", image.max())

    print("\nTumor mask:")
    mask = np.squeeze(f["cjdata"]["tumorMask"][()])
    print("Shape:", mask.shape)
    print("Dtype:", mask.dtype)
    print("Min:", mask.min())
    print("Max:", mask.max())

    print("\nUnique values:")
    unique = np.unique(mask)
    print("First 50 unique values:", unique[:50])
    print("Number of unique values:", len(unique))
    print("Non-zero pixels:", np.count_nonzero(mask))
    print("Total pixels:", mask.size)
    print("Tumor ratio:", np.count_nonzero(mask) / mask.size)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original MRI Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="hot")
axes[1].set_title("Tumor Mask")
axes[1].axis("off")

axes[2].imshow(image, cmap="gray")
axes[2].imshow(mask, cmap="Reds", alpha=0.6)
axes[2].set_title("Image with Tumor Mask Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()
