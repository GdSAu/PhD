import torch
import torch.nn as nn
import torch.nn.functional as F

class Codificador(nn.Module):
    def __init__(self, input_channels, latent_space):
        super(Codificador, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, 5, stride=2)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=1)
        self.mpool = nn.MaxPool3d(2, 2, return_indices=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6912, 128)
        self.linear2 = nn.Linear(128, latent_space)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x, index = self.mpool(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x, index

class Decodificador(nn.Module):
    def __init__(self, latent_space, output_channels=1):
        super(Decodificador, self).__init__()
        self.linear3 = nn.Linear(latent_space, 128)
        self.linear4 = nn.Linear(128, 6912)
        self.uflatten = nn.Unflatten(1, (32, 6, 6, 6))
        self.umpool = nn.MaxUnpool3d(2, stride=2)
        self.tconv1 = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=1)
        self.tconv2 = nn.ConvTranspose3d(32, output_channels, kernel_size=5, stride=2, padding=0)
    
    def forward(self, x, index):
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.uflatten(x)
        x = self.umpool(x, index)
        x = F.leaky_relu(self.tconv1(x))
        x = F.leaky_relu(self.tconv2(x))
        return x

class VoxNetAutoencoder(nn.Module):
    def __init__(self, latent_space, input_channels=1, output_channels=1):
        super(VoxNetAutoencoder, self).__init__()
        self.latent_space = latent_space
        self.codificador = Codificador(input_channels, self.latent_space)
        self.decodificador = Decodificador(self.latent_space, output_channels)
    
    def forward(self, x):
        codificacion, indices = self.codificador(x)
        reconstruccion = self.decodificador(codificacion, indices)
        return reconstruccion