import torch
from torch import nn

"""Definir un modelo de MLP 16x2048x512x512x12 basado en cotinuity of rotations"""

class MLP(nn.Module):
    def __init__(self,input = 16, output = 3):
        super(MLP, self).__init__()
        self.input = input
        self.output = output
        self.mlp = nn.Sequential(nn.Linear(self.input,2048),
                                 nn.LeakyReLU(),
                                 nn.Linear(2048,512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512,512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, self.output),
                                 nn.Tanh())

    
    def forward(self,x):
        
        x = self.mlp(x)
        return x