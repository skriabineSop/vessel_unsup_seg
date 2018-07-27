import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader  # , Dataset
from dataset import Dataset
from visualize import plot3d, show
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from gaussianKernel import get_gaussian_filter
logdir = 'logs'


def get_total_variation(latent):
    # is this differentiable ?

    x_diff = np.gradient(latent, axis=2)
    y_diff = np.gradient( latent, axis=3)
    z_diff = np.gradient(latent, axis=4)

    grad_norm2 = x_diff ** 2 + y_diff ** 2 + z_diff ** 2
    result = np.sum(np.sqrt(grad_norm2))

    return result


def totalVariationLoss(latent):  #, input, Lambda):
    # criterion = nn.MSELoss()
    # L2 = criterion(latent, input)
    V = get_total_variation(latent)
    # result = L2 + Lambda*V
    return V


class Total_variation_loss(nn.Module):

    def __init__(self):
        super(Total_variation_loss, self).__init__()

    def forward(self, x):
        return totalVariationLoss(x)


if __name__ == "__main__":

    ones = np.random.rand(8, 2, 20, 40, 40)
    zeros = np.random.rand(8, 2, 20, 40, 40)
    latent = np.concatenate((ones*0.5, ones*0.5), axis=2)
    latent = torch.from_numpy(latent).cuda().float()
    input = np.concatenate((ones*0.5, zeros*0.5), axis=2)
    input = torch.from_numpy(input).cuda().float()
    criterion2 = Total_variation_loss()

    loss = criterion2(latent)

    print(loss)