from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

METADATA_PATH = Path(
    r"/Data/metadata.csv"
)

# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("VALIDATION MASK DEBUG")
print("=" * 70)

df = pd.read_csv(METADATA_PATH)

val_df = df[df["split"] == "val"].reset_index(drop=True)

print(f"Total validation samples: {len(val_df)}")

# ============================================================
# STORAGE
# ============================================================

tumor_pixels = []

samples_with_tumor = 0
samples_without_tumor = 0

invalid_masks = 0

# ============================================================
# CHECK ALL VALIDATION MASKS
# ============================================================

print()
print("=" * 70)
print("CHECKING VALIDATION MASKS")
print("=" * 70)

for index, row in val_df.iterrows():

    path = row["path"]

    try:

        with h5py.File(path, "r") as f:

            mask = f["cjdata"]["tumorMask"][()]

        # ----------------------------------------------------
        # Validate mask
        # ----------------------------------------------------

        mask = np.asarray(mask)

        if mask.ndim != 2:
            print(
                f"[WARNING] Sample {index}: "
                f"unexpected shape {mask.shape}"
            )

            invalid_masks += 1
            continue

        # ----------------------------------------------------
        # Convert to binary
        # ----------------------------------------------------

        mask_binary = (mask > 0).astype(np.uint8)

        pixels = int(mask_binary.sum())

        tumor_pixels.append(pixels)

        if pixels > 0:
            samples_with_tumor += 1
        else:
            samples_without_tumor += 1

        # ----------------------------------------------------
        # Print first 10 samples
        # ----------------------------------------------------

        if index < 10:
            print(
                f"Sample {index + 1:03d} | "
                f"Shape: {mask.shape} | "
                f"Dtype: {mask.dtype} | "
                f"Tumor pixels: {pixels:6d} | "
                f"Path: {path}"
            )

    except Exception as e:

        print(
            f"[ERROR] Sample {index}: "
            f"{path}"
        )

        print(f"        {e}")

        invalid_masks += 1

# ============================================================
# STATISTICS
# ============================================================

tumor_pixels = np.asarray(tumor_pixels)

print()
print("=" * 70)
print("VALIDATION MASK STATISTICS")
print("=" * 70)

print(
    f"Validation samples checked : {len(tumor_pixels)}"
)

print(
    f"Samples with tumor          : {samples_with_tumor}"
)

print(
    f"Samples without tumor       : {samples_without_tumor}"
)

print(
    f"Invalid masks               : {invalid_masks}"
)

if len(tumor_pixels) > 0:
    print(
        f"Total tumor pixels         : "
        f"{tumor_pixels.sum():,.0f}"
    )

    print(
        f"Minimum tumor pixels       : "
        f"{tumor_pixels.min():,.0f}"
    )

    print(
        f"Maximum tumor pixels       : "
        f"{tumor_pixels.max():,.0f}"
    )

    print(
        f"Mean tumor pixels          : "
        f"{tumor_pixels.mean():,.2f}"
    )

    print(
        f"Median tumor pixels        : "
        f"{np.median(tumor_pixels):,.2f}"
    )

# ============================================================
# MASK VALUE CHECK
# ============================================================

print()
print("=" * 70)
print("MASK VALUE CHECK")
print("=" * 70)

unique_values_global = set()

for index, row in val_df.head(10).iterrows():

    path = row["path"]

    try:

        with h5py.File(path, "r") as f:

            mask = f["cjdata"]["tumorMask"][()]

        unique_values = np.unique(mask)

        unique_values_global.update(
            unique_values.tolist()
        )

        print(
            f"Sample {index + 1:03d} | "
            f"Unique values: {unique_values}"
        )

    except Exception as e:

        print(
            f"Sample {index + 1:03d} | ERROR: {e}"
        )

print()
print(
    "Global unique values in first 10 masks:",
    sorted(unique_values_global)
)

# ============================================================
# FINAL VERDICT
# ============================================================

print()
print("=" * 70)
print("FINAL VERDICT")
print("=" * 70)

if len(tumor_pixels) == 0:

    print("FAIL: No validation masks were successfully loaded.")

elif samples_with_tumor == 0:

    print(
        "FAIL: All validation masks are empty."
    )

    print(
        "The validation mask pipeline is broken."
    )

elif tumor_pixels.sum() == 0:

    print(
        "FAIL: Total tumor pixels = 0."
    )

else:

    print(
        "PASS: Validation masks contain tumor pixels."
    )

    print(
        "Segmentation training can proceed."
    )

print("=" * 70)
