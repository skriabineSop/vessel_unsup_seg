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
from gaussianKernel import GaussianKernel
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

datadir = '/mnt/raid/UnsupSegment/patches/10-43-24_IgG_UltraII[02 x 05]_C00'
logdir = 'logs/training180718_1'
savedmodeldir = 'savedModels'
sigma = 4
kernel_size = 11
Lambda = 1  # Used to weight relative importance of reconstruction loss and min cut loss

writer = SummaryWriter(logdir)


def get_gaussian_filter(sigma, kernel_size):
    if torch.cuda.is_available():
        gauss_filter = GaussianKernel(sigma, kernel_size).kernel.float().cuda()
    else:
        gauss_filter = GaussianKernel(sigma, kernel_size).kernel.float()
    return gauss_filter


def soft_cut_loss(x, kernel):
    if torch.cuda.is_available():
        res = Variable(torch.tensor(0).float().cuda(), requires_grad=True)
    else:
        res = Variable(torch.tensor(0).float(), requires_grad=True)

    for i in range(x.shape[1]):
        gauss = F.conv3d(x[:, i:i+1, :, :, :],
                         kernel[np.newaxis, :], bias=None, stride=1, padding=(5, 5, 5))
        mul = torch.mul(x[:, i:i+1, :, :, :], gauss)
        numerator = torch.sum(mul)
        sum_gauss = torch.sum(gauss)
        sub_denom = torch.mul(x[:, i:i+1, :, :, :], sum_gauss)
        denom = torch.sum(sub_denom)
        res = numerator/denom

    result = x.shape[1] - res

    return result


# autoencoder test
class autoencoder(nn.Module):

    def __init__(self):

        self.soft_cut_kernel = get_gaussian_filter(sigma, kernel_size)

        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=1, padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(32, 16, 3, stride=1, padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 2, 3, stride=1, padding=(1, 1, 1)),
            nn.Softmax(dim=1))

        self.decoder = nn.Sequential(
            nn.Conv3d(2, 16, 3, stride=1, padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=1, padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(32, 1, 3, stride=1, padding=(1, 1, 1)))

        self.soft_cut_loss = 0

    def forward(self, x):
        x = self.encoder(x)
        latent = x
        self.soft_cut_loss = soft_cut_loss(x, self.soft_cut_kernel)
        x = self.decoder(x)
        return x, latent


def preprocess(x):
    x = torch.clamp(x, max=20000.)
    x = torch.clamp(x, min=500.)
    x -= 500.
    x /= 19500
    return x


def main():

    print("load dataset")
    dataset = Dataset(datadir)
    dataloader = DataLoader(dataset, shuffle=True,  batch_size=16)

    print("Initialize model")
    if torch.cuda.is_available():
        model = autoencoder().cuda()
    else:
        model = autoencoder()
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    print('parameters', model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    print("begin training")
    num_iteration = 0

    for epoch in range(num_epochs):
        for data in dataloader:
            if torch.cuda.is_available():
                img = data.float().cuda()
            else:
                img = data.float()
            # ===================preprocess=====================
            img = preprocess(img)
            img = Variable(img)
            # ===================forward=====================
            output, latent = model(img)
            reconstruction_loss = criterion(output, img)

            # loss = reconstruction_loss + Lambda * model.soft_cut_loss
            loss = model.soft_cut_loss

            writer.add_scalar('Train/Loss', loss, num_iteration)
            writer.add_scalar('Train/ReconstructionLoss', reconstruction_loss, num_iteration)
            writer.add_scalar('Train/SoftCutLoss', model.soft_cut_loss, num_iteration)
            writer.add_scalar('Train/Loss', loss, num_iteration)

            if num_iteration % 200 == 0:
                writer.add_image('Train/Input', img.data[0, :, 20], num_iteration)
                writer.add_image('Train/Output', output.data[0, :, 20], num_iteration)
                np.save(os.path.join(logdir, 'kernel_' + str(num_iteration)), model.soft_cut_kernel[0])
                np.save(os.path.join(logdir, 'output_' + str(num_iteration)), output.data[0, 0])
                np.save(os.path.join(logdir, 'input_' + str(num_iteration)), img.data[0, 0])
                np.save(os.path.join(logdir, 'latent1_' + str(num_iteration)), latent.data[0, 0])
                np.save(os.path.join(logdir, 'latent2_' + str(num_iteration)), latent.data[1, 0])

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()

            print('gradient', loss.grad, type(loss))

            optimizer.step()
            num_iteration += 1

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data))

        torch.save(model.state_dict(), os.path.join(savedmodeldir, 'sim_autoencoder.pth'))


def test():
    dataset = Dataset(datadir)
    dataloader = DataLoader(dataset)
    if torch.cuda.is_available():
        model = autoencoder().cuda()
    else:
        model = autoencoder()

    for data in dataloader:
        if torch.cuda.is_available():
            img = data.float().cuda()
        else:
            img = data.float()
        plot3d(np.array(img).reshape((40, 40, 40)))
        show()

        output, latent = model(Variable(img))
        plot3d(np.array(output.data).reshape((40, 40, 40)))
        show()


if __name__ == "__main__":
    main()
