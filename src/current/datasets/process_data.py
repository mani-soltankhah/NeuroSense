import torch
import h5py
from pathlib import Path
from tqdm import tqdm
from src.current.utils.preprocessing import BrainMRIProcessor
from src.current.datasets.test_dataloader import train_df, val_df, test_df


def preprocess_and_save(dataframe, output_dir, processor):
    output_dir = Path(output_dir)
    image_dir = output_dir / 'images'
    mask_dir = output_dir / 'masks'

    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        mat_path = row['path']
        with h5py.File(mat_path, 'r') as f:
            cjdata = f['cjdata']
            image = cjdata['image'][()]
            mask = cjdata['tumorMask'][()]
        image, mask = processor.preprocess(image, mask)
        torch.save(image, image_dir / f'{index:04d}.pt')
        torch.save(mask, mask_dir / f'{index:04d}.pt')


if __name__ == "__main__":
    processor = BrainMRIProcessor(
        image_size=(224, 224)
    )

    preprocess_and_save(
        train_df,
        "D:/Portfolio/NeuroSense/Data/Processed/train",
        processor
    )
    preprocess_and_save(
        val_df,
        "D:/Portfolio/NeuroSense/Data/Processed/val",
        processor
    )

    preprocess_and_save(
        test_df,
        "D:/Portfolio/NeuroSense/Data/Processed/test",
        processor
    )
    print('saved')
