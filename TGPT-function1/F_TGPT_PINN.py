import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch import pi
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPT(nn.Module):
    def __init__(self, layers, nu, P, nu_neurons, initial_c, x_data, u_exact):
        super().__init__()
        self.layers     = layers
        self.nu         = nu
        self.nu_neurons    = nu_neurons
        self.activation = P
        
        self.loss_function = nn.MSELoss(reduction='mean').to(device)
        self.linears = nn.ModuleList([nn.Linear(layers[0], layers[0]) for i in range(self.layers[1])]+[nn.Linear(layers[1], layers[2],bias=False)])

        self.x_data    = x_data.to(device)
        self.u_exact   = u_exact.to(device)
                
        for i in range(self.layers[1]):
            self.linears[i].weight.data = torch.eye(self.layers[0])
            self.linears[i].bias.data = torch.zeros(self.layers[0])

        self.linears[-1].weight.data = initial_c
        


    def forward(self):
        a = torch.Tensor().to(device)
        test_data=self.x_data.float()
        for i in range(0, self.layers[-2]): 
            z_data = self.linears[i](test_data)
            a = torch.cat((a, self.activation[i](z_data[:,:1])), 1)
        u_apr = self.linears[-1](a)

        return u_apr


    def loss(self):
        u = self.forward()
        loss_L2=self.loss_function(u, self.u_exact)

        return  loss_L2

    
