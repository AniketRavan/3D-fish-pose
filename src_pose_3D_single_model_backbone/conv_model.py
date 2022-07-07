import torch
import torch.nn as nn
import torch.nn.functional as F
intermediate_dim = 1028

class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64*12*12, zDim=2):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv0_bn = nn.BatchNorm2d(imgChannels)
        self.encConv1 = nn.Conv2d(in_channels=imgChannels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.encConv1_bn = nn.BatchNorm2d(8)
        self.encConv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.encConv2_bn = nn.BatchNorm2d(16)
        self.encConv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.encConv3_bn = nn.BatchNorm2d(16)

        self.maxpool2d1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.encConv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.encConv4_bn = nn.BatchNorm2d(32)
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.encConv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.encConv5_bn = nn.BatchNorm2d(64)
        self.encFC1 = nn.Linear(featureDim, intermediate_dim)
        self.encFC2 = nn.Linear(intermediate_dim, zDim)
        self.encFC3 = nn.Linear(intermediate_dim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.decConv1_bn = nn.BatchNorm2d(32)
        self.decConv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.decConv2_bn = nn.BatchNorm2d(16)
        self.decConv3 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=0)
        self.decConv3_bn = nn.BatchNorm2d(8)
        self.decConv4 = nn.ConvTranspose2d(in_channels=8, out_channels=imgChannels, kernel_size=3, stride=2, padding=0)
        self.decSigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = self.leaky_relu(self.encConv0_bn(x))
        x = self.leaky_relu(self.encConv1(x))
        x = self.leaky_relu(self.encConv1_bn(x))
        x = self.leaky_relu(self.encConv2(x))
        x = self.leaky_relu(self.encConv2_bn(x))
        x = self.leaky_relu(self.encConv3(x))
        x = self.leaky_relu(self.encConv3_bn(x))
        x = self.leaky_relu(self.maxpool2d1(x))
        x = self.leaky_relu(self.encConv4(x))
        x = self.leaky_relu(self.encConv4_bn(x))
        x = self.leaky_relu(self.maxpool2d2(x))
        x = self.leaky_relu(self.encConv5(x))
        x = self.leaky_relu(self.encConv5_bn(x))
        x = x.view(-1, 64*12*12)
        x = self.leaky_relu(self.encFC1(x))
        #x = F.relu(self.maxPool2d3(x))
        mu = self.encFC2(x)
        logVar = self.encFC3(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = self.leaky_relu(self.decFC1(z))
        x = x.view(-1, 64, 12, 12)
        x = self.leaky_relu(self.decConv1(x))
        x = self.leaky_relu(self.decConv1_bn(x))
        x = self.leaky_relu(self.decConv2(x))
        x = self.leaky_relu(self.decConv2_bn(x))
        x = self.leaky_relu(self.decConv3(x))
        x = self.leaky_relu(self.decConv3_bn(x))
        x = self.decSigmoid(self.decConv4(x))
        #x = x.view(-1, 1, 119, 119)
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar
