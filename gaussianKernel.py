import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader  # , Dataset
from dataset import Dataset
from visualize import plot3d, show
import numpy as np
import math


class GaussianKernel():

    def __init__(self, sigma, kernel_size):
        self.kernel_size = kernel_size
        self.sigma = sigma

        kernel = np.ones((kernel_size, kernel_size, kernel_size))
        mean = np.ones(3) * (kernel_size - 1) / 2.
        variance = sigma ** 2

        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                for k in range(kernel.shape[2]):
                    vec = [i, j, k]
                    kernel[i, j, k] = (1. / (2. * math.pi * variance)) * \
                                      np.exp(-(np.linalg.norm(vec - mean) ** 2.) / (2 * variance))

        kernel = kernel / np.max(kernel)
        gaussian_kernel = kernel.reshape((1,) + kernel.shape)
        gaussian_kernel = torch.from_numpy(gaussian_kernel)

        self.kernel = gaussian_kernel


if __name__ == "__main__":

    kernel_size = 11
    sigma = 3

    img = GaussianKernel(sigma, kernel_size).kernel
    print(np.array(img).reshape((kernel_size, kernel_size, kernel_size)))
    plot3d(np.array(img).reshape((kernel_size, kernel_size, kernel_size)))
    show()
