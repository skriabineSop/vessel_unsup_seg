from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from skimage import io

class Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, workdir):
        """
        Args:
            workdir (string): Directory with all the images.
        """
        self.load_dir = workdir
        self.data = os.listdir(workdir)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(self.load_dir, file))

        return image
