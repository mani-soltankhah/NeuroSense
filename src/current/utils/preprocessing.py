import random
import numpy as np
import pandas as pd
from pathlib import Path
from src.current.datasets.brain_tumor import BrainTumorDataset


def normalize_image(image):
    image = image.astype('float32')
    brain_mask = image != 0
