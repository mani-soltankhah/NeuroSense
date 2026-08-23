import h5py
import numpy as np

PATH = r"/Data/Raw/457.mat"

print("=" * 60)
print("MASK DEBUG")
print("=" * 60)

with h5py.File(PATH, "r") as f:
    print("\nTop-level keys:")
    print(list(f.keys()))

    print("\nCJDATA keys:")
    print(list(f["cjdata"].keys()))

    print("\nImage:")
    image = f["cjdata"]["image"][()]

    print("Shape:", image.shape)
    print("Dtype:", image.dtype)
    print("Min:", image.min())
    print("Max:", image.max())

    print("\nTumor mask:")

    mask = f["cjdata"]["tumorMask"][()]

    print("Shape:", mask.shape)
    print("Dtype:", mask.dtype)
    print("Min:", mask.min())
    print("Max:", mask.max())

    print("\nUnique values:")

    unique = np.unique(mask)

    print(unique[:50])

    print(
        "Number of unique values:",
        len(unique)
    )

    print(
        "Non-zero pixels:",
        np.count_nonzero(mask)
    )

    print(
        "Total pixels:",
        mask.size
    )

    print(
        "Tumor ratio:",
        np.count_nonzero(mask) / mask.size
    )
