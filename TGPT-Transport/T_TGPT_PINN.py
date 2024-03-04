import torch
import torch.nn as nn
import torch.autograd as autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPT(nn.Module):
    def __init__(self, layers, nu, P, f_hat, IC_u, resid_data, IC_data, BC_bottom, BC_top):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1], bias=False) for i in range(len(layers)-1)])
        self.activation = P
        
        self.resid_data       = resid_data
        self.IC_data          = IC_data
        self.BC_bottom        = BC_bottom
        self.BC_top           = BC_top
        self.IC_u             = IC_u
        self.f_hat            = f_hat

        self.linears[0].weight.data = torch.tensor([1.0, 0.0]).reshape(self.layers[1],self.layers[0]).float()
        self.linears[1].weight.data = torch.ones(self.layers[2], self.layers[1])


    def forward(self, x_data):
        test_data    = x_data.float()
        xshift_data  = (self.linears[0](test_data) +1) % 2 -1
        u_data       = self.activation(torch.cat((xshift_data.reshape(test_data[:,0].shape[0],1),torch.zeros(test_data[:,0].shape[0],1).to(device)),1))
        return u_data

    def lossR(self):
        """Residual loss function"""
        x    = self.resid_data.clone().requires_grad_()
        u    = self.forward(x)
        u_xt = autograd.grad(u, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        u_x, u_t  = u_xt[:,:1],u_xt[:,1:]
        f    = u_t+self.nu*u_x
        d    = 0.1*abs(u_x)+1
        return self.loss_function(f/d, self.f_hat)

    
    def lossIC(self):
        """First initial loss function"""
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
