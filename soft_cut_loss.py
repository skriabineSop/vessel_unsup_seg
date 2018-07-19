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


def normalizedSoftCutLoss(input, kernel):
    for i in range(input.shape[1]):
        gauss = F.conv3d(input[:, i:i + 1, :, :, :],
                                  kernel[np.newaxis, :], bias=None, stride=1, padding=(5, 5, 5))
        gauss =gauss
        mul = torch.mul(input[:, i:i + 1, :, :, :], gauss)
        numerator = torch.sum(mul)
        sum_gauss = torch.sum(gauss)
        sub_denom = torch.mul(input[:, i:i + 1, :, :, :], sum_gauss)
        denom = torch.sum(sub_denom)
        res = numerator / denom
        # res = (torch.sum(torch.mul(x[:, i:i + 1, :, :, :], F.conv3d(x[:, i:i + 1, :, :, :], kernel[np.newaxis, :], bias=None, stride=1,padding=(5, 5, 5))))) / (torch.sum(torch.mul(x[:, i:i + 1, :, :, :],torch.sum(F.conv3d(x[:, i:i + 1, :, :, :],kernel[np.newaxis, :], bias=None, stride=1,padding=(5, 5, 5))))))
        #print(res)
    result =torch.add(torch.mul(res, -1), input.shape[1])#input.shape[1] - res
    print('result :', res, result.type())
    return result

class Soft_cut_loss(nn.Module):

    def __init__(self):
        super(Soft_cut_loss, self).__init__()

    def forward(self, x, kernel):
        print('compute soft loss')
            # if torch.cuda.is_available():
        #     res = Variable(torch.tensor(0).float().cuda(), requires_grad=True)
        # else:
        #     res = Variable(torch.tensor(0).float(), requires_grad=True)
        return normalizedSoftCutLoss(x, kernel)



