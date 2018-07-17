import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Dataset import Dataset

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

num_epochs = 100
batch_size = 128
learning_rate = 1e-3
workdir = '/mnt/raid/UnsupSegment/180509_IgG_10-43-24'

#autoencoder test
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=1),
            nn.ReLU(True),
            nn.Conv3d(32, 16, 3, stride=1),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3),
            nn.Conv3d(16, 2, 3, stride=1),
            nn.Softmax(dim=1))

        self.decoder = nn.Sequential(
            nn.Conv3d(2, 16, 3, stride=1),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=1),
            nn.ReLU(True),
            nn.Conv3d(32, 1, 3, stride=1))

    def forward(self, x):
        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        return x



def main():
    dataset = Dataset(workdir)
    dataloader = DataLoader(dataset)
    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            # check the output
            print('save picture')
        torch.save(model.state_dict(), './sim_autoencoder.pth')