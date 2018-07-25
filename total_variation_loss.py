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
    result=0.
    for i in range(latent.shape[1]):
        #print(latent[0, i].shape)
        x_diff = np.gradient(latent[0, i], -1, axis=0)
        y_diff = np.gradient( latent[0, i], -1, axis=1)
        z_diff = np.gradient(latent[0, i], -1, axis=2)
        grad_norm2 = x_diff ** 2 + y_diff ** 2 + z_diff ** 2
        #print(grad_norm2)
        norm = np.sum(np.sqrt(grad_norm2))
        result+=norm
    return result


def totalVariationLoss(latent, input, Lambda):
    criterion = nn.MSELoss()
    L2 = criterion(latent, input)
    V = get_total_variation(latent)
    result = L2 + Lambda*V
    return result


class Total_variation_loss(nn.Module):

    def __init__(self):
        super(Total_variation_loss, self).__init__()

    def forward(self, x, input, l):
        return totalVariationLoss(x, input, 0.01)


if __name__ == "__main__":

    ones = np.random.rand(1, 2, 20, 40, 40)
    zeros = np.random.rand(1, 2, 20, 40, 40)
    latent = np.concatenate((ones*0.5, ones*0.5), axis=2)
    latent = torch.from_numpy(latent).cuda().float()
    imput= np.concatenate((ones*0.5, zeros*0.5), axis=2)
    imput = torch.from_numpy(imput).cuda().float()
    criterion2 = Total_variation_loss()

    # print(np.unique(latent[0, 1, :, :, 0]))

    loss = criterion2(latent, imput, 0.01)

    print(loss)