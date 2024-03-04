import matplotlib.pyplot as plt
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPT(nn.Module):
    def __init__(self, layers, nu, P, initial_c, initial_s, xy_data, u_exact):
        super().__init__()
        self.layers     = layers
        self.nu         = nu
        self.activation = P
        
        self.loss_function = nn.MSELoss(reduction='mean').to(device)
        self.linears       = nn.ModuleList([nn.Linear(layers[0], layers[1]),nn.Linear(layers[1], layers[2]),nn.Linear(layers[2], layers[3],bias=False)])
        self.xy_data       = xy_data.to(device)
        self.u_exact       = u_exact.to(device)
                
        self.linears[0].weight.data = torch.ones(self.layers[1], self.layers[0])
        self.linears[0].bias.data=initial_s
        self.linears[1].weight.data = torch.ones(self.layers[1], self.layers[0])
        self.linears[1].bias.data=initial_s
        self.linears[2].weight.data = initial_c

        
    def forward(self, data):
        a = torch.Tensor().to(device)
        x = self.linears[0](data[:,0].reshape(data[:,0].shape[0],1))
        y = self.linears[1](data[:,1].reshape(data[:,1].shape[0],1))
        xy=torch.cat((x,y),1)
        for i in range(0, self.layers[2]):
            a = torch.cat((a, self.activation[i](xy[:,0:2*(i+1):i+1]).reshape(data.shape[0],1)), 1)
        final_output = self.linears[-1](a)

        return final_output

    def loss(self):
        u  = self.forward(self.xy_data)
        loss_u = self.loss_function(u, self.u_exact)

        return  loss_u
    
