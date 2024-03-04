import torch
from torch import cos, sin
import torch.autograd as autograd
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WaveAct(nn.Module):
    """Full PINN Activation Function"""
    def __init__(self):
        super(WaveAct, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)
  
class NN(nn.Module):    
    def __init__(self, layers, nu, rho):
        super().__init__()
        self.layers = layers
        self.nu  = nu
        self.rho = rho
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)  
    
        self.activation = WaveAct()
    
    def forward(self, x):       
        a = x.float()
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def lossR(self, xt_residual, f_hat):
        """Residual loss function"""
        g = xt_residual.clone().requires_grad_()

        u = self.forward(g)
        u_xt = autograd.grad(u, g, torch.ones(g.shape[0], 1).to(device),retain_graph=True,create_graph=True)[0]    
        u_x = u_xt[:,[0]]
        u_t = u_xt[:,[1]] 

        u_xxtt = autograd.grad(u_x, g, torch.ones(g.shape[0],1).to(device),retain_graph=True,create_graph=True)[0]
        u_xx = u_xxtt[:,[0]]
        f = u_t-self.nu*u_xx-self.rho*u*(1-u)

        return self.loss_function(f, f_hat)
    
    def lossIC(self, IC_xt, IC_u):
        """Initial condition loss function"""

        loss_IC = self.loss_function(self.forward(IC_xt), IC_u)
        return loss_IC

    def lossBC(self, BC1, BC2):
        """Periodic boundary condition loss function"""

        loss_BC = self.loss_function(self.forward(BC1), self.forward(BC2))
        return loss_BC

    def loss(self, xt_resid, IC_xt, IC_u, BC1, BC2, f_hat):
        """Total loss function"""
        loss_R   = self.lossR(xt_resid, f_hat)
        loss_IC = self.lossIC(IC_xt, IC_u)
        loss_BC = self.lossBC(BC1, BC2)
        return loss_R + loss_IC +loss_BC