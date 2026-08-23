import pandas as pd

from brain_tumor import BrainTumorDataset

df = pd.DataFrame(
    {
        "path": [
            "sample1.mat",
            "sample2.mat"
        ],

        "label": [
            1,
            2
        ]
    }
)

dataset = BrainTumorDataset(df)
print(f"length of dataset: {len(dataset)}")
print(f"First sample: {dataset[0]}")
