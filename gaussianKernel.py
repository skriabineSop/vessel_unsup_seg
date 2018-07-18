import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader # , Dataset
from dataset import Dataset
from visualize import plot3d, show
import numpy as np
import math

kernel_size = 5
sigma = 3

class GaussianKernel():

    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma
       # X, Y, Z = np.mgrid[0:kernel_size, 0:kernel_size, 0:kernel_size]
        kernel = np.ones((kernel_size, kernel_size, kernel_size))
        mean = np.ones(3)*(kernel_size - 1) / 2.
        variance = sigma ** 2.
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                for k in range(kernel.shape[2]):
                    vec=[i,j,k]
                    kernel[i,j,k] = (1. / (2. * math.pi * variance)) * \
                                      np.exp(
                                          -((np.linalg.norm(vec - mean) ** 2.)) / \
                                          (2 * variance)
                                      )
        gaussian_kernel = kernel / np.sum(kernel)
        gaussian_kernel = gaussian_kernel.reshape((1,) + gaussian_kernel.shape)

        gaussian_kernel = torch.from_numpy(gaussian_kernel)
        self.kernel = gaussian_kernel

    def convolution(self):
        # Create a x, y , z coordinate grid of shape (kernel_size, kernel_size, kernel_size)
        # x_cord = np.arange(kernel_size)
        # x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        # y_grid = x_grid.t()
        # z_grid=x_grid()
        # xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)

        kernel = np.mgrid[0:kernel_size, 0:kernel_size, 0:kernel_size]

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((kernel - mean)**2., dim=-1) /\
                              (2*variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.reshape((1,) + gaussian_kernel.shape)
        gaussian_kernel = Variable(gaussian_kernel)
        # Reshape to 2d depthwise convolutional weight
        #gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        #gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        # gaussian_filter = nn.Conv3d(in_channels=channels, out_channels=channels,
        #                             kernel_size=kernel_size, groups=channels, bias=False)
        #
        # gaussian_filter.weight.data = gaussian_kernel
        # gaussian_filter.weight.requires_grad = False
        return gaussian_kernel


if __name__ == "__main__":
    img=GaussianKernel(kernel_size, sigma).kernel
    plot3d(np.array(img).reshape((kernel_size, kernel_size, kernel_size)))
    show()