import torch
from torch import nn
import torch.nn.functional as F

class autoencoder(nn.Module):
    def __init__(self, latent_space, input_shape=(31,31,31)):
        super(autoencoder, self).__init__()
        self.latent_space = latent_space       
        
        self.encoder = nn.Sequential(nn.Conv3d(1, 32, 5, stride=2),
                                     nn.LeakyReLU(),
                                     nn.Conv3d(32, 32, 3, stride=1),
                                     nn.LeakyReLU(),
                                     nn.MaxPool3d(2, 2),
                                     nn.Flatten(),
                                     nn.Linear(6912, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.latent_space))
        
        self.decoder1 = nn.Sequential(nn.Linear(self.latent_space, 128),
                                    nn.ReLU(),
                                    nn.Linear(128,6912),
                                    nn.Unflatten(1, (32, 6, 6, 6)),  
                                    nn.LeakyReLU(),
                                    nn.ConvTranspose3d(32, 32, kernel_size= 3, stride=2),
                                    nn.LeakyReLU(),
                                    nn.ConvTranspose3d(32, 16, kernel_size= 5, stride=2, padding=1),
                                    nn.LeakyReLU(),
                                    nn.ConvTranspose3d(16, 1, kernel_size= 5, stride=1, padding=1),
                                    nn.LeakyReLU(),
                                    nn.ConvTranspose3d(1, 1, kernel_size= 3, stride=1)
                                     )
        
        
    
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder1(x)
        return x