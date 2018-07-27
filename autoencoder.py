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

date = "270718"
trainingNumber = "softcut5"

logdir = 'logs/training' + date + "_" + trainingNumber
savedmodeldir = 'savedModels/' + date + "_" + trainingNumber
sigma = 4
kernel_size = 11
Lambda = 0.6  # Used to weight relative importance of reconstruction loss and min cut loss
l = 5 * 1e-11
l1 = 5 * 1e-8


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin, kernel_size=3, padding=(1, 1, 1), groups=nin)
        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Wautoencoder(nn.Module):

    def __init__(self):
        super(Wautoencoder, self).__init__()
        self.soft_cut_kernel = get_gaussian_filter(sigma, kernel_size)
        self.soft_cut_loss = 0


        # encoder
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.upsample = nn.Upsample(mode="trilinear", scale_factor=2)

        self.conv1 = nn.Conv3d(2, 16, 3, stride=1, padding=(1, 1, 1), dilation=1)
        self.conv2 = nn.Conv3d(16, 16, 3, stride=1, padding=(1, 1, 1), dilation=1)

        self.desepconv1 = depthwise_separable_conv(16, 32)
        self.desepconv2 = depthwise_separable_conv(32, 64)
        self.desepconv3 = depthwise_separable_conv(64, 64)
        self.desepconv32 = depthwise_separable_conv(64, 32)

        self.deconv1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.desepconv4 = depthwise_separable_conv(64, 32)
        self.desepconv42 = depthwise_separable_conv(32, 16)
        self.deconv2 = nn.ConvTranspose3d(32, 16, 4, stride=2, padding=(1, 1, 1))
        # self.desepconv5 = depthwise_separable_conv(32, 16)
        # self.deconv3 = nn.ConvTranspose3d(16, 8, kernel_size=2)
        self.convbin1 = nn.Conv3d(32, 2, 3, stride=1, padding=(1, 1, 1), dilation=1)

        # decoder
        self.conv3 = nn.Conv3d(2, 16, 3, stride=1, padding=(1, 1, 1), dilation=1)
        self.conv4 = nn.Conv3d(16, 16, 3, stride=1, padding=(1, 1, 1), dilation=1)

        self.desepconv5 = depthwise_separable_conv(16, 32)
        self.desepconv6 = depthwise_separable_conv(32, 64)
        self.desepconv7 = depthwise_separable_conv(64, 64)

        self.deconv3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.desepconv8 = depthwise_separable_conv(64, 32)
        self.deconv4 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        # self.desepconv9 = depthwise_separable_conv(32, 16)
        # self.deconv6= nn.ConvTranspose3d(16, 8, 3)
        self.convbin2 = nn.Conv3d(32, 2, 3, stride=1, padding=(1, 1, 1), dilation=1)


    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        skip1 = x

        x = self.maxpool(skip1)
        x = self.desepconv1(x)
        skip2 = x

        x = self.maxpool(skip2)
        x = self.desepconv2(x)
        x = self.desepconv3(x)
        # x = self.deconv1(x)
        x = self.upsample(x)
        x = self.desepconv32(x)

        x = torch.cat((skip2, x), 1)
        x = self.desepconv4(x)
        # x = self.deconv2(x)
        x = self.upsample(x)
        x = self.desepconv42(x)
        x = torch.cat((skip1, x), 1)
        x = self.relu(self.convbin1(x))

        x = self.softmax(x)
        # print('X :', X.shape)
        return x

    def decode(self, x):
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        skip3 = x
        # print('skip3 :', skip3.shape)

        x = self.maxpool(skip3)
        x = self.desepconv5(x)
        skip4 = x
        # print('skip4 :', skip4.shape)

        x = self.maxpool(skip4)
        # print('x :', x.shape)
        x = self.desepconv6(x)
        # print('x :', x.shape)
        x = self.desepconv7(x)
        # print('x :', x.shape)
        x = self.deconv3(x)
        # print('xdeconv3 :', x.shape)

        x = torch.cat((skip4, x), 1)
        x = self.desepconv8(x)
        # print('x :', x.shape)
        x = self.deconv4(x)

        # print('xdeconv4 :', x.shape)
        x = torch.cat((skip3, x), 1)
        x = self.relu(self.convbin2(x))
        # print('x :', x.shape)

        return x

    def forward(self, x):
        x = self.encode(x)
        self.nonbinintermediate = x
        # print(np.unique(x.data[:, 0, :, :, :]+x.data[:, , :, :, :]))
        thresh = Variable(torch.Tensor([1/x.data.shape[1]])).float().cuda()
        # thresh = Variable(torch.Tensor([0.51])).float().cuda()
        x[:, 0] = (x[:, 0] >= thresh).float().cuda()
        x[:, 1] = (x[:, 1] > thresh).float().cuda()
        x = Variable(x).float().cuda()
        self.intermediate = x
        x = self.decode(x)
        return x, self.intermediate, self.nonbinintermediate



# autoencoder test
class autoencoder(nn.Module):

    def __init__(self):

        super(autoencoder, self).__init__()

        self.soft_cut_kernel = get_gaussian_filter(sigma, kernel_size)
        self.soft_cut_loss = 0

        self.conv1 = nn.Conv3d(2, 128, 3, stride=1, padding=(1, 1, 1), dilation=1)
        self.conv2 = nn.Conv3d(128, 64, 3, stride=1, padding=(1, 1, 1), dilation=1)
        self.conv3 = nn.Conv3d(64, 32, 3, stride=1, padding=(1, 1, 1), dilation=1)
        self.conv4 = nn.Conv3d(32, 16, 3, stride=1, padding=(1, 1, 1), dilation=1)
        self.conv5 = nn.Conv3d(16, 2, 3, stride=1, padding=(1, 1, 1), dilation=1)
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(dim=1)

        self.conv6 = nn.Conv3d(2, 16, 3, stride=1, padding=(1, 1, 1))
        self.conv7 = nn.Conv3d(16, 32, 3, stride=1, padding=(1, 1, 1))
        self.conv8 = nn.Conv3d(32, 2, 3, stride=1, padding=(1, 1, 1))


    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.softmax(self.conv5(x))

        return x

    def decode(self, x):
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.conv8(x)

        return x

    def forward(self, x):
        x = self.encode(x)
        self.nonbinintermediate = x
        # self.soft_cut_loss = soft_cut_loss(x, self.soft_cut_kernel)
        thresh = Variable(torch.Tensor([1/x.data.shape[1]])).float().cuda()
        x[:, 0] = (x[:, 0] >= thresh).float().cuda()
        x[:, 1] = (x[:, 1] > thresh).float().cuda()
        # x = Variable(x).float().cuda()
        self.intermediate = x
        x = self.decode(x)
        return x, self.intermediate, self.nonbinintermediate


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
        model = Wautoencoder().cuda()
    else:
        model = Wautoencoder()

    model.apply(xavier_init)
    # model.apply(xavier_init)

    criterion1 = nn.MSELoss()
    criterion2 = Soft_cut_loss()
    criterion3 = Total_variation_loss()

    # model.load_state_dict(torch.load( os.path.join('savedModels/240718_1/', 'sim_autoencoder'+str(num_iteration)+'.pth')))
    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)  # , weight_decay=1e-5)
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
            output, latent, nonbinlatent = model(img)
            reconstruction_loss = criterion1(output, img)
            soft_loss = criterion2(model.intermediate, model.soft_cut_kernel)
            TV_loss = criterion3(latent)

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

            loss = reconstruction_loss + l * soft_loss + l1 * TV_loss
            # ===================get infos=====================
            writer.add_scalar('Train/Loss', loss, num_iteration)
            writer.add_scalar('Train/ReconstructionLoss', reconstruction_loss, num_iteration)
            writer.add_scalar('Train/SoftCutLoss', soft_loss, num_iteration)
            writer.add_scalar('Train/TVLoss', TV_loss, num_iteration)
            writer.add_scalar('Train/Loss', loss, num_iteration)

            if num_iteration % 200 == 0:
                writer.add_image('Train/Input', img.data[0, 1, :, :, 20], num_iteration)
                writer.add_image('Train/Output', output.data[0, 1, :, :, 20], num_iteration)
                writer.add_image('Train/latent1', latent.data[0, 0, :, :, 20], num_iteration)
                writer.add_image('Train/latent2', latent.data[0, 1, :, :, 20], num_iteration)
                writer.add_image('Train/nonbinlatent1', nonbinlatent.data[0, 0, :, :, 20], num_iteration)
                writer.add_image('Train/nonbinlatent2', nonbinlatent.data[0, 1, :, :, 20], num_iteration)
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
            print('loss', loss.data)
            # print('reconstruction loss:', reconstruction_loss)
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

            if num_iteration % 400 == 0:
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
    savedmodeldir = 'savedModels/260718_1/'
    num_iteration = 5000

    patchShape = (200, 200, 200)
    if not os.path.exists(os.path.join(savedir)):
        os.makedirs(os.path.join(savedir))
        print('create directory : ',savedir)

    for file in os.listdir(datadir):
        if '.tif' in file:
            print(file)
            image = io.imread(os.path.join(datadir, file))
            subImage = image[100:100+patchShape[0], 500:500+patchShape[1], 100:100+patchShape[2]]
            np.save(os.path.join(savedir, file[:8]), subImage)

    dataset = Dataset(savedir)
    dataloader = DataLoader(dataset)

    print("Initialize model")
    if torch.cuda.is_available():
        model = Wautoencoder().cuda()
    else:
        model = Wautoencoder()
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
        np.save(os.path.join(savedir, 'output.npy'), output.data[0, 1])
        np.save(os.path.join(savedir, 'latent1.npy'), latent.data[0, 0])
        np.save(os.path.join(savedir, 'latent2.npy'), latent.data[0, 1])
        vb1, vb2 = get_two_views()
        plot3d(np.array(latent.data[0, 0]).reshape((200, 200, 200)), view=vb1)
        plot3d(np.array(subImage).reshape((200, 200, 200)), view=vb2)
        show()

if __name__ == "__main__":
    main()
    #test_on_bigger_data()