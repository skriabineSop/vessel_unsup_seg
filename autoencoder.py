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
from gaussianKernel import GaussianKernel, get_gaussian_filter
from soft_cut_loss import Soft_cut_loss

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

datadir = '/mnt/raid/UnsupSegment/patches/10-43-24_IgG_UltraII[02 x 05]_C00' # ordi fixe
# datadir = '/home/paul.bertin/PycharmProjects/vessel_unsup_seg/data/toyDataset' # ordi perso
logdir = 'logs/training230718_8'
savedmodeldir = 'savedModels/230718_8/'
sigma = 4
kernel_size = 11
Lambda = 0.6  # Used to weight relative importance of reconstruction loss and min cut loss
l=0.01
writer = SummaryWriter(logdir)


# def soft_cut_loss(x, kernel):
#     if torch.cuda.is_available():
#         res = Variable(torch.tensor(0).float().cuda(), requires_grad=True)
#     else:
#         res = Variable(torch.tensor(0).float(), requires_grad=True)
#
#     for i in range(x.shape[1]):
#         gauss = F.conv3d(x[:, i:i+1, :, :, :],
#                          kernel[np.newaxis, :], bias=None, stride=1, padding=(5, 5, 5))
#         mul = torch.mul(x[:, i:i+1, :, :, :], gauss)
#         numerator = torch.sum(mul)
#         sum_gauss = torch.sum(gauss)
#         sub_denom = torch.mul(x[:, i:i+1, :, :, :], sum_gauss)
#         denom = torch.sum(sub_denom)
#         res = numerator/denom
#
#     result = x.shape[1] - res
#
#     return result


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
            nn.Softmax(dim=1)
            )

        self.decoder = nn.Sequential(
            nn.Conv3d(2, 16, 3, stride=1, padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=1, padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(32, 1, 3, stride=1, padding=(1, 1, 1)))

        self.soft_cut_loss = 0

    def forward(self, x):
        x = self.encoder(x)
        #self.soft_cut_loss = soft_cut_loss(x, self.soft_cut_kernel)
        thresh=Variable(torch.Tensor([1/x.data.shape[1]])).float().cuda()
        x=(x>=thresh).float().cuda()
        self.intermediate = x
        x = self.decoder(x)
        return x, self.intermediate


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
    criterion1 = nn.MSELoss()
    criterion2 = Soft_cut_loss()

    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # print('parameters', model.parameters())
    # optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=1e-6)
    # optimizer_model = torch.optim.SGD(model.parameters(), lr=1e-1)
    print("begin training")
    num_iteration = 0

    cpt = 0

    for epoch in range(num_epochs):
        for data in dataloader:

            cpt += 1

            if torch.cuda.is_available():
                img = data.float().cuda()
            else:
                img = data.float()
            # ===================preprocess=====================
            img = preprocess(img)
            img = Variable(img)
            # ===================forward=====================
            output, latent = model(img)
            reconstruction_loss = criterion1(output, img)
            soft_loss = criterion2(model.intermediate, model.soft_cut_kernel)
            # loss = reconstruction_loss + Lambda * model.soft_cut_loss
            # loss = model.soft_cut_loss
            if cpt < 2000:
                print('cpt<1200')
                loss = reconstruction_loss
            elif cpt < 4000  and cpt > 2000:
                Lambda = l*(cpt-1200.)/1200
                print('lanbda:', Lambda)
                loss = reconstruction_loss + (Lambda*soft_loss)
            else:
                loss = reconstruction_loss + l*soft_loss
            # ===================get infos=====================
            writer.add_scalar('Train/Loss', loss, num_iteration)
            writer.add_scalar('Train/ReconstructionLoss', reconstruction_loss, num_iteration)
            writer.add_scalar('Train/SoftCutLoss', soft_loss, num_iteration)
            writer.add_scalar('Train/Loss', loss, num_iteration)

            if num_iteration % 200 == 0:
                writer.add_image('Train/Input', img.data[0, :, :, :, 20], num_iteration)
                writer.add_image('Train/Output', output.data[0, :, :, :, 20], num_iteration)
                writer.add_image('Train/latent1', latent.data[0, 0, :, :, 20], num_iteration)
                writer.add_image('Train/latent2', latent.data[0, 1, :, :, 20], num_iteration)
                # writer.add_image('Train/latent3', latent.data[0, 2, :, :, 20], num_iteration)
                # writer.add_image('Train/latent4', latent.data[0, 3, :, :, 20], num_iteration)
                # writer.add_image('Train/latent5', latent.data[0, 4, :, :, 20], num_iteration)


                np.save(os.path.join(logdir, 'kernel_' + str(num_iteration)), model.soft_cut_kernel[0])
                np.save(os.path.join(logdir, 'output_' + str(num_iteration)), output.data[0, 0])
                np.save(os.path.join(logdir, 'input_' + str(num_iteration)), img.data[0, 0])
                np.save(os.path.join(logdir, 'latent1_' + str(num_iteration)), latent.data[0, 0])
                np.save(os.path.join(logdir, 'latent2_' + str(num_iteration)), latent.data[0, 1])
                # np.save(os.path.join(logdir, 'latent3_' + str(num_iteration)), latent.data[0, 2])
                # np.save(os.path.join(logdir, 'latent4_' + str(num_iteration)), latent.data[0, 3])
                # np.save(os.path.join(logdir, 'latent5 _' + str(num_iteration)), latent.data[0, 4])

            # ===================backward====================
            optimizer_model.zero_grad()
            loss.backward()
            #
            print('loss', loss)
            print('reconstruction loss:', reconstruction_loss)
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data[0])
            #         print("grad", param.grad[0])
            #         break
            optimizer_model.step()

            # loss2 = reconstruction_loss
            # # print('gradient', loss.grad, type(loss))
            # optimizer_encoder.zero_grad()
            # loss2.backward()
            # optimizer_encoder.step()

            num_iteration += 1

            if num_iteration % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(savedmodeldir, 'sim_autoencoder'+str(num_iteration)+'.pth'))
            # ===================log========================
        print('epoch [{}/{}]'
              .format(epoch + 1, num_epochs))

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
