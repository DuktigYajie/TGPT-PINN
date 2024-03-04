import matplotlib.pyplot as plt
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPT(nn.Module):
    def __init__(self, layers, nu, P,nu_neurons, initial_c, x_data, u_exact):
        super().__init__()
        self.layers     = layers
        self.nu         = nu
        self.nu_neurons    = nu_neurons
        self.activation = P

        self.loss_function = nn.MSELoss().to(device)
        self.linears = nn.ModuleList([nn.Linear(layers[0], layers[0]) for i in range(self.layers[1])]+[nn.Linear(layers[1], layers[2],bias=False)])


        self.x_data    = x_data.to(device)
        self.u_exact   = u_exact.to(device)


        for i in range(self.layers[1]):
            self.linears[i].weight.data = torch.eye(self.layers[0])
            self.linears[i].bias.data = torch.zeros(self.layers[0])

        self.linears[-1].weight.data = initial_c

        self.cut_now   = torch.div(torch.add(self.nu_neurons,0.4),2).reshape([layers[1],1]).to(device)

    def forward(self,datatype):
        if datatype == 'u_loss':    
           a = torch.Tensor().to(device)
           for i in range(0, self.layers[1]):  
               z = self.linears[i](self.x_data).detach()
               a = torch.cat((a, self.activation[i](z[:,:1])), 1)
           output = self.linears[-1](a)

        if datatype == 'x_loss': 
            cut_data = torch.full([self.layers[1],1],(self.nu+0.4)/2).to(device)
            x_cut = torch.zeros(self.layers[1],1).to(device)
            for i in range(0, self.layers[1]):   
                x_cut[i] = self.linears[i](cut_data[i]).to(device)
            output= x_cut
        return output
    
    def loss_x(self):
        x_cut = self.forward(datatype = 'x_loss')
        loss_x=self.loss_function(x_cut, self.cut_now)
        return loss_x

    def loss_u(self):
        u = self.forward(datatype = 'u_loss')
        loss_u=self.loss_function(u, self.u_exact)
        return loss_u
