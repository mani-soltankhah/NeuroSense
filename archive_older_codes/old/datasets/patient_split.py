from pathlib import Path

import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(r"/Data/Raw")

OUTPUT_METADATA = DATA_DIR.parent / "metadata.csv"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# LABELS
# ============================================================

LABEL_NAMES = {
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary",
}


# ============================================================
# READ PID
# ============================================================

def decode_pid(pid_array):
    """
    PID is stored as ASCII codes inside the MATLAB/HDF5 file.
    Example:
        [49, 48, 48, 51, 54, 48]
        -> "100360"
    """

    pid_array = pid_array.flatten()

    return "".join(chr(int(x)) for x in pid_array)


# ============================================================
# BUILD METADATA
# ============================================================

records = []

mat_files = sorted(DATA_DIR.glob("*.mat"))

print("=" * 60)
print("BUILDING METADATA")
print("=" * 60)

print(f"MAT files found: {len(mat_files)}")

for i, mat_path in enumerate(mat_files, start=1):

    try:
        with h5py.File(mat_path, "r") as f:

            cjdata = f["cjdata"]

            label = int(cjdata["label"][()][0][0])

            pid = decode_pid(cjdata["PID"][()])

            image_shape = tuple(cjdata["image"].shape)

            records.append({
                "file": mat_path.name,
                "path": str(mat_path),
                "patient_id": pid,
                "label": label,
                "class_name": LABEL_NAMES[label],
                "image_height": image_shape[0],
                "image_width": image_shape[1],
            })

    except Exception as e:
        print(f"ERROR: {mat_path.name}")
        print(e)

    if i % 500 == 0:
        print(f"Processed: {i}/{len(mat_files)}")

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

print("\n" + "=" * 60)
print("METADATA SUMMARY")
print("=" * 60)

print(f"Total images: {len(df)}")
print(f"Unique patients: {df['patient_id'].nunique()}")

print("\nImages per class:")
print(df["class_name"].value_counts())

print("\nPatients per class:")
print(
    df.groupby("class_name")["patient_id"]
    .nunique()
)

# ============================================================
# PATIENT TABLE
# ============================================================

patients = (
    df[["patient_id", "label", "class_name"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print("\nPatient table:")
print(f"Total patients: {len(patients)}")

# ============================================================
# CHECK THAT EACH PATIENT HAS ONLY ONE LABEL
# ============================================================

patient_label_counts = (
    df.groupby("patient_id")["label"]
    .nunique()
)

multi_label_patients = patient_label_counts[
    patient_label_counts > 1
    ]

if len(multi_label_patients) > 0:
    raise ValueError(
        "Some patients have multiple tumor labels!"
    )

print("Patient label consistency: OK")

# ============================================================
# SPLIT PATIENTS
# ============================================================

# First:
# 70% train
# 30% temporary

train_patients, temp_patients = train_test_split(
    patients,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=patients["label"],
    random_state=RANDOM_STATE,
)

# Then split temporary:
# 15% validation
# 15% test

relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=relative_test_size,
    stratify=temp_patients["label"],
    random_state=RANDOM_STATE,
)

# ============================================================
# ADD SPLIT COLUMN
# ============================================================

train_ids = set(train_patients["patient_id"])
val_ids = set(val_patients["patient_id"])
test_ids = set(test_patients["patient_id"])


def assign_split(patient_id):
    if patient_id in train_ids:
        return "train"

    if patient_id in val_ids:
        return "val"

    if patient_id in test_ids:
        return "test"

    raise ValueError(f"Unknown patient: {patient_id}")


df["split"] = df["patient_id"].apply(assign_split)

# ============================================================
# SAVE METADATA
# ============================================================

df.to_csv(
    OUTPUT_METADATA,
    index=False
)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("PATIENT-LEVEL SPLIT")
print("=" * 60)

print(
    f"Train patients: {len(train_patients)}"
)

print(
    f"Validation patients: {len(val_patients)}"
)

print(
    f"Test patients: {len(test_patients)}"
)

print("\nImage distribution:")
print(
    df["split"].value_counts()
)

print("\nPatients by split and class:")

patient_split_summary = (
    patients.assign(
        split=patients["patient_id"].map(
            lambda x:
            "train" if x in train_ids
            else "val" if x in val_ids
            else "test"
        )
    )
    .groupby(["split", "class_name"])
    .size()
)

print(patient_split_summary)

# ============================================================
# LEAKAGE CHECK
# ============================================================

train_set = set(train_patients["patient_id"])
val_set = set(val_patients["patient_id"])
test_set = set(test_patients["patient_id"])

print("\n" + "=" * 60)
print("LEAKAGE CHECK")
print("=" * 60)

print(
    "Train ∩ Validation:",
    len(train_set & val_set)
)

print(
    "Train ∩ Test:",
    len(train_set & test_set)
)

print(
    "Validation ∩ Test:",
    len(val_set & test_set)
)

if (
        len(train_set & val_set) == 0
        and len(train_set & test_set) == 0
        and len(val_set & test_set) == 0
):
    print("\nPatient-level split: PASSED")
else:
    raise RuntimeError(
        "Patient leakage detected!"
    )

print("\nMetadata saved to:")
print(OUTPUT_METADATA)
