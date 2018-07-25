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
from total_variation_loss import Total_variation_loss
from skimage import io
from visualize import twoClassesMap, get_two_views
import shutil

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

datadir = '/mnt/raid/UnsupSegment/patches/10-43-24_IgG_UltraII[02 x 05]_C00' # ordi fixe
# datadir = '/home/paul.bertin/PycharmProjects/vessel_unsup_seg/data/toyDataset' # ordi perso

date = "250718"
trainingNumber = "3"

logdir = 'logs/training' + date + "_" + trainingNumber
savedmodeldir = 'savedModels/' + date + "_" + trainingNumber
sigma = 4
kernel_size = 11
Lambda = 0.6  # Used to weight relative importance of reconstruction loss and min cut loss
l = 0 # 1e-11
l1=0.01

# autoencoder test
class autoencoder(nn.Module):

    def __init__(self):

        self.soft_cut_kernel = get_gaussian_filter(sigma, kernel_size)

        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(2, 64, 3, stride=1, padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(64, 32, 3, stride=1, padding=(1, 1, 1)),
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
            nn.Conv3d(32, 2, 3, stride=1, padding=(1, 1, 1)))

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
    x = torch.clamp(x, min=500., max=20000.)
    x -= 500.
    x /= 19500
    return x


def preprocessNegChannel(x):
    x[:, 1] = torch.clamp(x[:, 1], min=500., max=20000.)
    x[:, 1] -= 500.
    x[:, 1] /= 19500

    x[:, 0] = torch.clamp(x[:, 0], min=-20000., max=-500.)
    x[:, 0] += 500.
    x[:, 0] /= 19500

    return x


def xavier_init(m):
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_normal_(m.weight)


def main():

    if not os.path.exists(savedmodeldir):
        os.makedirs(savedmodeldir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        if os.listdir(logdir):
            ans = input("Empty log directory " + str(logdir) + "? (Y/N)")
            if ans == "y" or ans == "Y":
                shutil.rmtree(logdir)
                print("empty directory !")

    writer = SummaryWriter(logdir)

    #####################################################

    num_iteration = 0

    print("load dataset")
    dataset = Dataset(datadir)
    dataloader = DataLoader(dataset, shuffle=True,  batch_size=16)

    print("Initialize model")
    if torch.cuda.is_available():
        model = autoencoder().cuda()
    else:
        model = autoencoder()

    model.encoder.apply(xavier_init)
    model.decoder.apply(xavier_init)

    criterion1 = nn.MSELoss()
    criterion2 = Soft_cut_loss()
    criterion3 = Total_variation_loss()

    # model.load_state_dict(torch.load( os.path.join('savedModels/240718_1/', 'sim_autoencoder'+str(num_iteration)+'.pth')))
    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # print('parameters', model.parameters())
    # optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=1e-6)
    # optimizer_model = torch.optim.SGD(model.parameters(), lr=1e-1)

    cpt = 0

    for epoch in range(num_epochs):
        for data in dataloader:

            cpt += 1

            if torch.cuda.is_available():
                img = data.float().cuda()
            else:
                img = data.float()
            # ===================preprocess=====================
            # img = preprocess(img)
            img = preprocessNegChannel(img)
            img = Variable(img)
            # ===================forward=====================
            output, latent = model(img)
            reconstruction_loss = criterion1(output, img)
            soft_loss = criterion2(model.intermediate, model.soft_cut_kernel)
            TV_loss = criterion3(latent, img, 10)
            # loss = reconstruction_loss + Lambda * model.soft_cut_loss
            # loss = model.soft_cut_loss
            # if cpt < 4000:
            #     print('cpt<1200')
            #     loss = reconstruction_loss
            # elif cpt < 6000  and cpt > 4000:
            #     Lambda = l*(cpt-4000.)/2000
            #     print('lanbda:', Lambda)
            #     loss = reconstruction_loss + (Lambda*soft_loss)
            # else:
            loss = reconstruction_loss + l*soft_loss +l1*TV_loss
            # ===================get infos=====================
            writer.add_scalar('Train/Loss', loss, num_iteration)
            writer.add_scalar('Train/ReconstructionLoss', reconstruction_loss, num_iteration)
            #writer.add_scalar('Train/SoftCutLoss', soft_loss, num_iteration)
            writer.add_scalar('Train/TVLoss', TV_loss, num_iteration)
            writer.add_scalar('Train/Loss', loss, num_iteration)

            if num_iteration % 200 == 0:
                writer.add_image('Train/Input', img.data[0, 0, :, :, 20], num_iteration)
                writer.add_image('Train/Output', output.data[0, 0, :, :, 20], num_iteration)
                writer.add_image('Train/latent1', latent.data[0, 0, :, :, 20], num_iteration)
                writer.add_image('Train/latent2', latent.data[0, 1, :, :, 20], num_iteration)
                # writer.add_image('Train/latent3', latent.data[0, 2, :, :, 20], num_iteration)
                # writer.add_image('Train/latent4', latent.data[0, 3, :, :, 20], num_iteration)
                # writer.add_image('Train/latent5', latent.data[0, 4, :, :, 20], num_iteration)


                np.save(os.path.join(logdir, 'kernel_' + str(num_iteration)), model.soft_cut_kernel[0])
                np.save(os.path.join(logdir, 'output_' + str(num_iteration)), output.data[0, 1])
                np.save(os.path.join(logdir, 'input_' + str(num_iteration)), img.data[0, 1])
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


def test_on_bigger_data():
    datadir='/home/paul.bertin/Bureau/'
    savedir = '/home/paul.bertin/Bureau/datadir/'
    savedmodeldir = 'savedModels/240718_12/'
    patchShape = (200, 200, 200)
    if not os.path.exists(os.path.join(savedir)):
        os.makedirs(os.path.join(savedir))
        print('create directory : ',savedir)

    for file in os.listdir(datadir):
        if '.tif' in file:
            print(file)
            image = io.imread(os.path.join(datadir, file))
            subImage = image[100:100+patchShape[0], 100:100+patchShape[1], 100:100+patchShape[2]]
            np.save(os.path.join(savedir, file[:8]), subImage)

    dataset = Dataset(savedir)
    dataloader = DataLoader(dataset)
    num_iteration = 55000
    print("Initialize model")
    if torch.cuda.is_available():
        model = autoencoder().cuda()
    else:
        model = autoencoder()
    print(savedmodeldir, 'sim_autoencoder'+str(num_iteration)+'.pth')
    model.load_state_dict(torch.load( os.path.join(savedmodeldir, 'sim_autoencoder'+str(num_iteration)+'.pth')))
    #torch.load(model.state_dict(), os.path.join(savedmodeldir, 'sim_autoencoder'+str(num_iteration)+'.pth'))
    for data in dataloader:
        img=[]
        if torch.cuda.is_available():
            img = data.float().cuda()
        else:
            img = data.float()

        output, latent = model(Variable(img))
        print('save results')
        np.save(os.path.join(savedir, 'output.npy'), output.data[0, 0])
        np.save(os.path.join(savedir, 'latent1.npy'), latent.data[0, 0])
        np.save(os.path.join(savedir, 'latent2.npy'), latent.data[0, 1])
        vb1, vb2 = get_two_views()
        plot3d(np.array(latent.data[0, 0]).reshape((200, 200, 200)), view=vb1)
        plot3d(np.array(subImage).reshape((200, 200, 200)), view=vb2)
        show()

if __name__ == "__main__":
    main()
    # test_on_bigger_data()