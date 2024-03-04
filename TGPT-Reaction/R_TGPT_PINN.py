import torch
import torch.nn as nn
from torch import sin, pi
import torch.autograd as autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPT(nn.Module):
    def __init__(self, layers, nu, P, c_initial, f_hat, IC_u, resid_data, IC_data, BC_bottom, BC_top):
        super().__init__()
        self.layers = layers
        self.num_neurons = self.layers[1]
        self.nu = nu
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[0], layers[0]) for i in range(self.layers[1])]+[nn.Linear(layers[1], layers[2],bias=False)])
        self.activation = P
        
        self.resid_data       = resid_data
        self.IC_data          = IC_data
        self.BC_bottom        = BC_bottom
        self.BC_top           = BC_top
        self.IC_u             = IC_u
        self.f_hat            = f_hat
        

        for i in range(self.layers[1]):
            self.linears[i].weight.data = torch.eye(self.layers[0])
            self.linears[i].bias.data=torch.zeros(self.layers[0])

        self.linears[-1].weight.data =  c_initial


    def forward(self, x_data):
        test_data    = x_data.float()
        u_data = torch.Tensor().to(device)
        for i in range(0, self.layers[-2]):
            shift_data  = self.linears[i](test_data)
            xshift_data = shift_data[:,:1]%(2*pi)
            tshift_data = shift_data[:,1:]
            u_data = torch.cat((u_data, self.activation[i](torch.cat((xshift_data,tshift_data),1))), 1)
        final_output = self.linears[-1](u_data)
        return final_output

    def lossR(self):
        """Residual loss function"""
        x    = self.resid_data.clone().requires_grad_()
        u    = self.forward(x)
        u_xt = autograd.grad(u, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        u_x, u_t  = u_xt[:,:1],u_xt[:,1:]
        f    = u_t-self.nu*u*(1-u)

        return self.loss_function(f, self.f_hat)
    
    def lossIC(self):
        """Initial loss function"""
        x = self.IC_data.clone().requires_grad_()
        return self.loss_function(self.forward(x), self.IC_u)


    def lossBC(self):
        """Periodic boundary condition loss function"""
        B1 = self.BC_bottom.clone().requires_grad_()
        B2 = self.BC_top.clone().requires_grad_()
        return self.loss_function(self.forward(B1), self.forward(B2))
 
    def loss(self):
        """Total Loss Function"""
        loss_R   = self.lossR()
        loss_IC  = self.lossIC()
        loss_BC  = self.lossBC()
        return loss_R + loss_IC + loss_BC