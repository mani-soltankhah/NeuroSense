from pathlib import Path

path = Path(
    "D:/Portfolio/NeuroSense/Data/Processed/test/images"
)

for name in ["960.pt", "961.pt", "962.pt"]:
    print(name, len(list(path.glob(name))))
