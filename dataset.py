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
        data = []
        for path, subdirs, files in os.walk(workdir):
            for name in files:
                if '.npy' in name:
                    print(os.path.join(path, name))
                    data.append(os.path.join(path, name))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.load_dir, self.data[idx])).astype(float)
        image = image.reshape((1,) + image.shape)
        return image
