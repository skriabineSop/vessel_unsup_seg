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

def normalizedSoftCutLoss(input, kernel):
    ones = torch.from_numpy(np.ones((1, 2, 40, 40, 40))).cuda().float()
    res = torch.from_numpy(np.zeros(1)).cuda().float()
    print("res", res)
    for i in range(input.shape[1]):
        gauss = F.conv3d(input[:, i:i + 1, :, :, :],
                                  kernel[np.newaxis, :], bias=None, stride=1, padding=(5, 5, 5))
        print("gauss", gauss.shape, gauss)
        #np.save(os.path.join(logdir, 'gauss' + 'test', gauss.data[0, 0]))
        # plot3d(np.array(input.data[0,0]).reshape((40, 40, 40)))
        # show()
        # plot3d(np.array(gauss.data).reshape((40, 40, 40)))
        # show()
        print("input", input.shape)
        mul = torch.mul(input[:, i:i + 1, :, :, :], gauss)
        # plot3d(np.array(gauss.data).reshape((40, 40, 40)))
        # show()
        print("mul", mul.shape)
        print("mul", mul)

        numerator = torch.sum(mul)

        print("numerator", numerator)

        # sum_gauss = torch.sum(kernel)
        sum_gauss = F.conv3d(ones[:, i:i + 1, :, :, :],
                         kernel[np.newaxis, :], bias=None, stride=1, padding=(5, 5, 5))

        print("sum gauss", sum_gauss)

        sub_denom = torch.mul(input[:, i:i + 1, :, :, :], sum_gauss)

        print("sub denom", sub_denom)

        denom = torch.sum(sub_denom)

        print("denom", denom)

        res += numerator / denom

        print("res", res)

    result =torch.add(torch.mul(res, -1), input.shape[1])
    return result


class Soft_cut_loss(nn.Module):

    def __init__(self):
        super(Soft_cut_loss, self).__init__()

    def forward(self, x, kernel):
        return normalizedSoftCutLoss(x, kernel)


if __name__ == "__main__":

    ones = np.ones((1, 2, 20, 40, 40))
    zeros = np.zeros((1, 2, 20, 40, 40))
    latent = np.concatenate((ones, ones), axis=2)
    latent = torch.from_numpy(latent).cuda().float()

    criterion2 = Soft_cut_loss()

    # print(np.unique(latent[0, 1, :, :, 0]))

    sigma = 4
    kernel_size = 11
    kernel = get_gaussian_filter(sigma, kernel_size)

    soft_loss = criterion2(latent, kernel)

    print(soft_loss)