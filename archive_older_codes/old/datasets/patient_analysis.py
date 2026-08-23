from pathlib import Path
from collections import defaultdict, Counter
import h5py


DATA_DIR = Path(r"/Data/Raw")


CLASS_NAMES = {
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary"
}


def decode_pid(pid_array):
    return "".join(chr(int(x)) for x in pid_array.flatten())


patients = defaultdict(list)

files = sorted(DATA_DIR.glob("*.mat"))

for file_path in files:

    with h5py.File(file_path, "r") as f:

        cjdata = f["cjdata"]

        pid = decode_pid(cjdata["PID"][:])
        label = int(cjdata["label"][()][0][0])

        patients[pid].append({
            "file": file_path.name,
            "label": label
        })


print("=" * 60)
print("PATIENT ANALYSIS")
print("=" * 60)

print(f"Total files: {len(files)}")
print(f"Unique patients: {len(patients)}")


# --------------------------------------------------
# Images per patient
# --------------------------------------------------

images_per_patient = [
    len(samples)
    for samples in patients.values()
]

print("\nImages per patient:")
print(f"Min: {min(images_per_patient)}")
print(f"Max: {max(images_per_patient)}")
print(f"Mean: {sum(images_per_patient) / len(images_per_patient):.2f}")


# --------------------------------------------------
# Patient class distribution
# --------------------------------------------------

patient_classes = Counter()

mixed_patients = []

for pid, samples in patients.items():

    labels = set(sample["label"] for sample in samples)

    for label in labels:
        patient_classes[label] += 1

    if len(labels) > 1:
        mixed_patients.append(
            (pid, labels)
        )


print("\nPatients per class:")

for label, count in sorted(patient_classes.items()):
    print(
        f"{label} ({CLASS_NAMES[label]}): {count}"
    )


# --------------------------------------------------
# Mixed-label patients
# --------------------------------------------------

print("\nPatients with multiple tumor labels:")

print(f"Count: {len(mixed_patients)}")

for pid, labels in mixed_patients[:20]:

    label_names = [
        CLASS_NAMES[label]
        for label in sorted(labels)
    ]

    print(
        f"{pid}: {label_names}"
    )


# --------------------------------------------------
# Largest patients
# --------------------------------------------------

print("\nPatients with most images:")

largest = sorted(
    patients.items(),
    key=lambda x: len(x[1]),
    reverse=True
)

for pid, samples in largest[:10]:

    label = samples[0]["label"]

    print(
        f"PID {pid} | "
        f"{len(samples)} images | "
        f"{CLASS_NAMES[label]}"
    )