import numpy as np
import torch
from skimage import io
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

savedir = '/mnt/raid/UnsupSegment/patches'
workdir = '/mnt/raid/UnsupSegment/180509_IgG_10-43-24'
impath = '/mnt/raid/UnsupSegment/180509_IgG_10-43-24/10-43-24_IgG_UltraII[01 x 08]_C00.ome.tif'
patchShape = (40, 40, 40)


def from_tif_to_patch(file):
    image = io.imread(os.path.join(workdir, file))
    print(image.shape)
    cpt = 0
    for i in tqdm(range(0, image.shape[0] - patchShape[0], patchShape[0])):
        for j in range(0, image.shape[1] - patchShape[1], patchShape[1]):
            for k in range(0, image.shape[2] - patchShape[2], patchShape[2]):
                cpt += 1
                np.save(os.path.join(savedir, file[:-8] + '_patch_' + str(i) + '_' + str(j) + '_' + str(k)),
                        image[i:i + patchShape[0],
                        j:j + patchShape[1],
                        k:k + patchShape[2]])
                # print(cpt, 'patch saved over', (image.shape[0] // patchShape[0]) *
                #       (image.shape[1] // patchShape[1]) *
                #       (image.shape[2] // patchShape[2]))


for files in os.listdir(workdir):
    if '.tif' in files:
        print(files)
        from_tif_to_patch(files)




