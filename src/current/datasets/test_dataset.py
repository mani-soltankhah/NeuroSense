import h5py
import random
import numpy as np
import pandas as pd
from pathlib import Path
from brain_tumor import BrainTumorDataset

data_dir = Path(r"D:\Portfolio\NeuroSense\Data\Raw")
mat_files = list(data_dir.glob("*.mat"))
df = pd.DataFrame(
    {
        "path": [
            mat_files[0]
        ],

        "label": [
            1,
        ]
    }
)

dataset = BrainTumorDataset(df)
image, mask = dataset[0]
random_files = random.sample(mat_files, 10)

for path in random_files:
    with h5py.File(path, "r") as f:
        cjdata = f["cjdata"]
        image = cjdata["image"][()]
        image = image.astype('float32')
        tumor_mask = cjdata['tumorMask'][()]
        brain_mask = image != 0

        brain_pixels = image[image > 0]
        mean = brain_pixels.mean()
        std = brain_pixels.std()
        size = brain_pixels.size
        tumor_pixels = image[tumor_mask == 1]
        background_pixels = image[tumor_mask == 0]
        print(f"""
        name: {path}
        brain pixel count: {brain_pixels.size}
        mean: {mean}
        std: {std}
        size: {size}
        """)

        print(f'''
        background pixels:
                   name: {path}
                   type: {image.dtype}
                   min: {background_pixels.min()}
                   max: {background_pixels.max()}
                   mean: {background_pixels.mean()}''')
        print(f'''
        tumor pixels:
            name: {path}
            type: {image.dtype}
            min: {tumor_pixels.min()}
            max: {tumor_pixels.max()}
            mean: {tumor_pixels.mean()}''')

        print(f'''
            name: {path}
            min: {brain_pixels.min()}
            max: {brain_pixels.max()}
            mean: {brain_pixels.mean()}
            median: {np.median(brain_pixels)}
            std: {brain_pixels.std()}
            percentile 1% : {np.percentile(brain_pixels, 1)}
            percentile 50% : {np.percentile(brain_pixels, 50)}
            percentile 99% : {np.percentile(brain_pixels, 99)} ''')
        print(path.name)
        print(f'''
        min():
         {image.min()}
         max():
         {image.max()}
         mean:
         {image.mean()}
         percentile 1%:
         {np.percentile(image, 1)}
         percentile 50%:
         {np.percentile(image, 50)}
         percentile 99%:
         {np.percentile(image, 99)}
         median:
         {np.median(image)}
         ''')
        print(np.sum(image == 0))
        print((np.sum(image == 0) / image.size) * 100)

stds = []
for path in mat_files:
    with h5py.File(path, "r") as f:
        cjdata = f["cjdata"]
        image = cjdata["image"][()]
        image = image.astype('float32')
        tumor_mask = cjdata['tumorMask'][()]
        brain_mask = image != 0
        brain_pixels = image[image > 0]
        mean = brain_pixels.mean()
        std = brain_pixels.std()
        stds.append(std)
        if brain_pixels.size == 0:
            print("Empty image:", path)
            print(brain_pixels.mean())
            print(brain_pixels.std())

print("min std:", np.min(stds))
print("max std:", np.max(stds))
print("mean std:", np.mean(stds))
print("zero std images:", np.sum(np.array(stds) == 0))
