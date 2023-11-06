import torch
from torch import nn
import torch.nn.functional as F

class voxnet(nn.Module):
    def __init__(self, latent_space, input_shape=(31,31,31)):
        super(voxnet, self).__init__()
        self.latent_space = latent_space       
        
        self.conv1 = nn.Conv3d(1, 32, 5, stride=2)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=1)
        self.mpool = nn.MaxPool3d(2, 2, return_indices=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6912, 128)
        self.linear2 = nn.Linear(128, self.latent_space)
        self.linear3 = nn.Linear(self.latent_space, 128)
        self.linear4 = nn.Linear(128,6912)
        self.uflatten = nn.Unflatten(1, (32, 6, 6, 6))
        self.umpool = nn.MaxUnpool3d(2, stride=2)
        self.tconv1 = nn.ConvTranspose3d(32, 32, kernel_size= 3, stride=1)
        self.tconv2 = nn.ConvTranspose3d(32, 1, kernel_size= 5, stride=2, padding=0)
        
    
    def forward(self, x):
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x, index = self.mpool(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.uflatten(x)
        x = self.umpool(x, index)
        x = F.leaky_relu(self.tconv1(x))
        x = F.leaky_relu(self.tconv2(x))
        return x